import torch
from einops import rearrange

from easyvolcap.engine import MODELS, NETWORKS, cfg, call_from_cfg, dotdict
from reactmotion.models.general_model import VQGeneralAgentModel, SparseControlAgentModel
from reactmotion.utils.agent import *
from reactmotion.utils.motion_repr_transform import MoReprTrans
from reactmotion.models.diffusion.create_diffusion import create_gaussian_diffusion
from reactmotion.models.diffusion.parser_util import train_args
from reactmotion.models.diffusion.resample import create_named_schedule_sampler

parser = train_args()
args, _ = parser.parse_known_args(["--save_dir", "debug"])


@MODELS.register_module()
class ReactiveARDiffDecModel(VQGeneralAgentModel):
    def __init__(self,
                 network_cfg: dotdict,
                 decoder_cfg: dotdict,
                 diffusion_steps: int = 1000,
                 sample_steps: int = 50,
                 diff_rec_weight: float = 1.,
                 predict_xstart: bool = True,
                 **kwargs):
        call_from_cfg(super().__init__, kwargs)
        # settings
        self.diffusion_steps = diffusion_steps
        self.skip_steps = diffusion_steps - sample_steps
        self.predict_xstart = predict_xstart
        # loss
        self.diff_rec_weight = diff_rec_weight
        self.dec_pose_rec_weight = 1.
        self.dec_root_rec_weight = 1.
        self.dec_oppo_rec_weight = 1
    
        self.network = NETWORKS.build(network_cfg)
        if decoder_cfg is not None:
            self.decoder = NETWORKS.build(decoder_cfg)
        # build diffusion
        args.diffusion_steps = self.diffusion_steps
        args.predict_xstart = self.predict_xstart
        self.diffusion = create_gaussian_diffusion(args)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)


    def load_norm_data(self, ):
        self.Xnorm = MoReprTrans.load_norm_data('', norm_file=cfg.norm_cfg.motoken_input_norm_file)
        self.Cnorm = MoReprTrans.load_norm_data("oppo_pose")
        self.Rnorm = MoReprTrans.load_norm_data("root_info")

    
    def normalize_batch_data(self, batch):
        batch.pose_series = MoReprTrans.Normalize(batch.pose_series, self.Xnorm)
        batch.oppo_pose = MoReprTrans.Normalize(batch.oppo_pose, self.Cnorm)
        batch.root_info = MoReprTrans.Normalize(batch.root_info, self.Rnorm)

    
    def condition_forward(self, past_target, batch, src_key_padding_mask):
        # condition
        if self.training:
            oppo_pose = batch.oppo_pose[:, :-U:U]
        else:
            oppo_pose = batch.oppo_pose
        conditions = self.network.conditon_forward(
            oppo_pose, 
            past_target, 
            src_key_padding_mask) # encode once
        return conditions
    

    def diffusion_forward(self, target, batch):
        '''
        target: whole sequence, [:, 1:] is the x_start, [:, :-1] is the input
        '''
        x_start = target[:, 1:]
        bs, nframes, device = x_start.shape[0], x_start.shape[1], x_start.device
        src_key_padding_mask = batch.mask[:, :-1]
        conditions = self.condition_forward(target[:, :-1], batch, src_key_padding_mask)
        # diffusion forward
        t, weights = self.schedule_sampler.sample(bs, device)
        t = t.unsqueeze(1).repeat(1, nframes)
        weights = weights.unsqueeze(1).repeat(1, nframes)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        diffusion_output = self.network(x_t, self.diffusion._scale_timesteps(t), conditions) # (b, t, c)
        # loss compute of diffusion
        if self.predict_xstart: target = x_start
        else: target = noise
        diff_recon_loss = ((diffusion_output - target) ** 2).mean(-1) * batch.mask[:, 1:]
        diff_recon_loss = (diff_recon_loss * weights).mean()
        return diffusion_output, diff_recon_loss * self.diff_rec_weight
    

    def decoder_forward(self, past_latents, future_latents, batch):
        bs = future_latents.shape[0]
        # decoder conditions
        if self.training:
            past_pose_series = rearrange(batch.pose_series[:, :-U], 'b (l t1) c -> (b l) t1 c', t1=U)
            past_root_info = rearrange(batch.root_info[:, :-U], 'b (l t1) c -> (b l) t1 c', t1=U)
            past_latents = rearrange(past_latents, 'b l c -> (b l) 1 c')
            future_latents = rearrange(future_latents, 'b l c -> (b l) 1 c')
        else:
            past_pose_series = batch.pose_series
            past_root_info = batch.root_info
        # decoder forward
        pred_pose_series, pred_root_info = self.decoder(past_pose_series, past_root_info, past_latents, future_latents)
        if self.training:
            pred_pose_series = rearrange(pred_pose_series, '(b l) t1 c -> b (l t1) c', t1=U, b=bs)
            pred_root_info = rearrange(pred_root_info, '(b l) t1 c -> b (l t1) c', t1=U, b=bs)
            # loss compute of decoder
            dec_pose_rec_loss = ((pred_pose_series - batch.pose_series[:, U:]) ** 2).mean(-1) * batch.pose_mask[:, U:]
            dec_root_rec_loss = ((pred_root_info - batch.root_info[:, U:]) ** 2).mean(-1) * batch.pose_mask[:, U:]
            return dec_pose_rec_loss, dec_root_rec_loss
        else:
            return pred_pose_series, pred_root_info


    def forward(self, batch, **kwargs):
        self.normalize_batch_data(batch)
        target = batch.latents
        diff_output, diff_recon_loss = self.diffusion_forward(target, batch)
        dec_pose_rec_loss, dec_root_rec_loss = self.decoder_forward(target[:, :-1], diff_output, batch)
        output = dotdict(
            loss = diff_recon_loss + dec_pose_rec_loss * self.dec_pose_rec_weight + dec_root_rec_loss * self.dec_root_rec_weight,
            scalar_stats = dotdict(
                diff_recon_loss = diff_recon_loss,
                dec_pose_rec_loss = dec_pose_rec_loss,
                dec_root_rec_loss = dec_root_rec_loss,
            ),
        )
        return output
    

    def diffusion_inference(self, past_latents, batch):
        conditions = self.condition_forward(past_latents, batch, None)
        # sample_fn = self.diffusion.p_sample_loop
        sample_fn = self.diffusion.ddim_sample_loop
        future_latents = sample_fn(
            self.network,
            past_latents.shape,
            clip_denoised=False,
            model_kwargs={'conditions': conditions, 'y': {}},
            skip_timesteps=self.skip_steps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        return future_latents
    

    def network_inference(self, batch):
        self.normalize_batch_data(batch)
        future_latents = self.diffusion_inference(batch.latents, batch)
        pred_pose_series, pred_root_info = self.decoder_forward(batch.latents[:, -1:], future_latents[:, -1:], batch)
        return future_latents[:, -1:], MoReprTrans.Renormalize(pred_pose_series, self.Xnorm), MoReprTrans.Renormalize(pred_root_info, self.Rnorm)
    

    def pred_next(self, agent1: Agent, agent2: Agent, ts, **kwargs):
        past_latents = agent1.get_curr('latents')
        seq_len = past_latents.size(1)
        pose_series = agent1.get_curr('ctrl_series', U)
        oppo_pose = MoReprTrans.cal_oppo_pose(agent1, agent2, list(range(ts-seq_len*U, ts, U)))
        root_info = MoReprTrans.cal_root_info(agent1, agent2, list(range(ts-U, ts)), 0)
        batch = dotdict(
            oppo_pose = oppo_pose,
            pose_series = pose_series,
            root_info = root_info,
            latents = past_latents,
        )
        pred_latents, pred_pose_series, pred_root_info = self.network_inference(batch)
        agent1.update('latents', pred_latents, replace=False)
        return pred_pose_series
    

    def update_agent(self, agent: Agent, next_motion, ts, i, agent2=None, gt_agent=None, **kwargs):
        pred_pose_series = next_motion[:, i:i+1]
        loc = MoReprTrans.split_pose(pred_pose_series)
        agent.update_pose_by_loc(loc)
        agent.update('ctrl_series', MoReprTrans.cal_pose_series(agent, [ts+i]), replace=False) # TODO
        agent.update('root2oppo_ctrl', MoReprTrans.cal_root2oppo_ctrl(agent, agent2, [ts+i]), replace=False) # TODO

   

@MODELS.register_module()
class SparseControlARDiffDecModel(SparseControlAgentModel, ReactiveARDiffDecModel):
    
    def load_norm_data(self):
        self.Vnorm = MoReprTrans.load_norm_data("ctrl_info", modify_std=True)
        return super().load_norm_data()
    

    def normalize_batch_data(self, batch):
        batch.ctrl_info = MoReprTrans.Normalize(batch.ctrl_info, self.Vnorm)
        return super().normalize_batch_data(batch)
        

    def condition_forward(self, past_target, batch, src_key_padding_mask):
        # condition
        if self.training:
            oppo_pose = batch.oppo_pose[:, :-U:U]
            ctrl_info = batch.ctrl_info[:, :-U:U]
        else:
            oppo_pose = batch.oppo_pose
            ctrl_info = batch.ctrl_info[:, ::U]
        conditions = self.network.conditon_forward(
            oppo_pose,
            past_target,
            ctrl_info,
            src_key_padding_mask) # encode once
        return conditions
    

    def decoder_forward(self, past_latents, future_latents, batch):
        bs = future_latents.shape[0]
        # decoder conditions
        if self.training:
            past_latents = rearrange(past_latents, 'b l c -> (b l) 1 c')
            future_latents = rearrange(future_latents, 'b l c -> (b l) 1 c')
            past_pose_series = rearrange(batch.pose_series[:, :-U], 'b (l t1) c -> (b l) t1 c', t1=U)
            past_root_info = rearrange(batch.root_info[:, :-U], 'b (l t1) c -> (b l) t1 c', t1=U)
            past_ctrl_info = rearrange(batch.ctrl_info[:, :-U], 'b (l t1) c -> (b l) t1 c', t1=U)
        else:
            past_pose_series = batch.pose_series
            past_root_info = batch.root_info
            past_ctrl_info = batch.ctrl_info[:, -U:]
        # decoder forward
        pred_pose_series, pred_root_info = self.decoder(past_pose_series, past_root_info, past_latents, future_latents, past_ctrl_info)
        if self.training:
            pred_pose_series = rearrange(pred_pose_series, '(b l) t1 c -> b (l t1) c', t1=U, b=bs)
            pred_root_info = rearrange(pred_root_info, '(b l) t1 c -> b (l t1) c', t1=U, b=bs)
            # loss compute of decoder
            dec_pose_rec_loss = ((pred_pose_series - batch.pose_series[:, U:]) ** 2).mean(-1) * batch.pose_mask[:, U:]
            dec_root_rec_loss = ((pred_root_info - batch.root_info[:, U:]) ** 2).mean(-1) * batch.pose_mask[:, U:]
            return dec_pose_rec_loss, dec_root_rec_loss
        else:
            return pred_pose_series, pred_root_info


    def pred_next(self, agent1: Agent, agent2: Agent, ts, gt_agent):
        past_latents = agent1.get_curr('latents')
        seq_len = past_latents.size(1)
        pose_series = agent1.get_curr('ctrl_series', U)
        oppo_pose = MoReprTrans.cal_oppo_pose(agent1, agent2, list(range(ts-seq_len*U, ts, U)))
        root_info = MoReprTrans.cal_root_info(agent1, agent2, list(range(ts-U, ts)), 0)
        ctrl_info = MoReprTrans.cal_ctrl_info(agent1, gt_agent, list(range(ts-seq_len*U, ts)))
        batch = dotdict(
            latents = past_latents,
            pose_series = pose_series,
            oppo_pose = oppo_pose,
            root_info = root_info,
            ctrl_info = ctrl_info,
        )
        # pred next ctrl
        pred_latents, pred_pose_series, pred_root_info = self.network_inference(batch)
        agent1.update('latents', pred_latents, replace=False)
        return pred_pose_series