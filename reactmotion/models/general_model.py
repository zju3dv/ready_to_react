# Default pipeline for volumetric videos
# This corresponds to the tranditional implementation's renderer
import torch
from torch import nn
from torch.nn import Parameter
import time

from easyvolcap.engine import cfg, call_from_cfg, log
from easyvolcap.engine import MODELS
from easyvolcap.utils.base_utils import dotdict
from reactmotion.utils.agent import *
from reactmotion.utils.motion_repr_transform import *
from reactmotion.utils.net_utils import load_other_model


@MODELS.register_module()
class GeneralAgentModel(nn.Module):
    def __init__(self,
                 inference_func: str, # required
                 pred_next_func: str = 'pred_next',
                 num_start_frames: int = 4,
                 num_new_frames: int = 0, 
                 use_gt_length: bool = True,
                 ts_step: int = 1,
                 agent_cfg = dotdict(
                                type="Agent",
                            ),
                 ) -> None:
        super().__init__()
        self.num_start_frames = num_start_frames
        self.num_new_frames = num_new_frames
        self.use_gt_length = use_gt_length
        if not use_gt_length:
            log(f"not using gt length, num_new_frames={num_new_frames}")
        self.ts_step = ts_step
        self.agent_cfg = agent_cfg
        self.inference_func = inference_func
        self.inference = getattr(self, f'inference_{inference_func}')
        self.apd_compute = cfg.runner_cfg.evaluator_cfg.get('apd_compute', False)
        self.test_speed = cfg.runner_cfg.evaluator_cfg.get('test_speed', False)
        if self.apd_compute:
            log('apd_compute, use_gt_length=True')
            self.use_gt_length = True
        self.pred_next = getattr(self, pred_next_func)
        self.load_norm_data()

    

    
    def forward(self, batch, **kwargs) -> dotdict:
        return dotdict(
            loss = 0., 
            scalar_stats = dotdict(),
        )
    

    def inference_reactive(self, batch, **kwargs) -> dotdict:
        agent1, gt_agent1 = self.prepare_agent(batch, 1)
        agent2, gt_agent2 = self.prepare_agent(batch, 2)

        # num_new_frames = gt_agent2.pose_vel.size(1) - self.num_start_frames
        if self.use_gt_length: num_new_frames = gt_agent2.pose_vel.size(1) - self.num_start_frames
        else: 
            assert self.num_new_frames != 0, f"num_new_frames is 0, should set a value"
            num_new_frames = self.num_new_frames
        time_list = []
        for ts in range(self.num_start_frames, self.num_start_frames + num_new_frames, self.ts_step):
            if self.test_speed:
                torch.cuda.synchronize()
                start = time.time()
                next_motion = self.pred_next(agent1, gt_agent2, ts, gt_agent=gt_agent1)
                torch.cuda.synchronize()
                end = time.time()
                time_list.append(end-start)
            else:
                next_motion = self.pred_next(agent1, gt_agent2, ts, gt_agent=gt_agent1)
            
            # update actor
            for i in range(self.ts_step):
                self.update_agent(agent1, next_motion, ts, i, agent2=gt_agent2, gt_agent=gt_agent1, **kwargs)
        if self.test_speed:
            batch.update({
                'time_network': time_list,
            })
        output = dotdict(
            agent1 = agent1,
            agent2 = gt_agent2, # reactive
        )
        batch.update({
            'agent1': gt_agent1,
            'agent2': gt_agent2,
        })
        return output

    
    def inference_twoagent(self, batch, **kwargs) -> dotdict:
        agent1, gt_agent1 = self.prepare_agent(batch, 1)
        agent2, gt_agent2 = self.prepare_agent(batch, 2)

        if self.use_gt_length: num_new_frames = gt_agent2.pose_vel.size(1) - self.num_start_frames
        else: 
            assert self.num_new_frames != 0, f"num_new_frames is 0, should set a value"
            num_new_frames = self.num_new_frames

        for ts in range(self.num_start_frames, self.num_start_frames + num_new_frames, self.ts_step):
            next_motion1 = self.pred_next(agent1, agent2, ts)
            next_motion2 = self.pred_next(agent2, agent1, ts)
            for i in range(self.ts_step):
                self.update_agent(agent1, next_motion1, ts, i, agent2=agent2, gt_agent=gt_agent1, **kwargs)
                self.update_agent(agent2, next_motion2, ts, i, agent2=agent1, gt_agent=gt_agent2, **kwargs)
            
        output = dotdict(
            agent1 = agent1,
            agent2 = agent2,
        )
        batch.update({
            'agent1': gt_agent1,
            'agent2': gt_agent2,
        })
        return output
        
    

    def pred_next(self, agent1: Agent, agent2: Agent, ts: int, **kwargs):
        '''
            agent1, agent2
            ts: current time step
        '''
        ...
    
    def update_agent(self, agent: Agent, next_motion: torch.Tensor, ts: int, i: int, **kwargs):
        '''
            agent
            next_motion
            ts: current time step
        '''
        next_motion = next_motion[:, i:i+1]
        agent.update_pose_by_loc(MoReprTrans.split_pose(next_motion))
        agent.update('ctrl_series', next_motion, replace=False)
    

    def prepare_agent(self, batch, agent_id):
        '''
            agent_id: 1 or 2
        '''
        agent_motion = batch.get(f'agent{agent_id}')
        cond_agent_motion = batch.get(f'cond_agent{agent_id}')
        device = agent_motion.pose_vel.device
        
        # init ground truth agent
        gt_agent = AGENTS.build(self.agent_cfg, device=device, block_size=cfg.window_cfg.get('block_size') * U)
        gt_agent.init_from_agent_motion(agent_motion)
        batch[f'agent{agent_id}'] = gt_agent

        # init predicted agent
        agent = AGENTS.build(self.agent_cfg, device=device, block_size=cfg.window_cfg.get('block_size') * U)
        agent.init_from_agent_motion(cond_agent_motion)

        # init ctrl
        ctrl_series = MoReprTrans.cal_pose_series(agent, list(range(agent.pose_vel.size(1))))
        agent.update('ctrl_series', ctrl_series, replace=True)
        return agent, gt_agent


@MODELS.register_module()
class VQGeneralAgentModel(GeneralAgentModel):
    def __init__(self, 
                 ### vq settings
                 vocab_size: int = cfg.window_cfg.get("vocab_size", 512),
                 block_size: int = cfg.window_cfg.get("block_size", 48),
                 infer_gpttoken_window_size: int = cfg.window_cfg.get("infer_gpttoken_window_size", 1),
                 motoken_type: str = 'vqvae',
                 **kwargs):
        call_from_cfg(super().__init__, kwargs)
        ### vq settings
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.infer_gpttoken_window_size = infer_gpttoken_window_size
        self.motoken_type = motoken_type
        
        self.ts_step = U

    
    def prepare_agent(self, batch, agent_id):
        '''
            agent_id: 1 or 2
        '''
        agent_motion = batch.get(f'agent{agent_id}')
        cond_agent_motion = batch.get(f'cond_agent{agent_id}')
        device = agent_motion.pose_vel.device
        
        # init ground truth agent
        gt_agent = AGENTS.build(self.agent_cfg, device=device, block_size=cfg.window_cfg.get('block_size'))
        gt_agent.init_from_agent_motion(agent_motion)
        batch[f'agent{agent_id}'] = gt_agent

        # init predicted agent
        agent = AGENTS.build(self.agent_cfg, device=device, block_size=cfg.window_cfg.get('block_size'))
        agent.init_from_agent_motion(cond_agent_motion)

        # init ctrl, tokens, latents
        ctrl_series = MoReprTrans.cal_pose_series(agent, list(range(agent.pose_vel.size(1))))
        agent.update('ctrl_series', ctrl_series, replace=True)
        init_tokens = cfg.motoken_net.encode(ctrl_series)
        agent.update('tokens', init_tokens, replace=True)
        init_latents = cfg.motoken_net.quantizer.dequantize(init_tokens)
        agent.update('latents', init_latents, replace=True)
        return agent, gt_agent
    

class SparseControlAgentModel(GeneralAgentModel):
    def __init__(self, 
                 **kwargs):
        call_from_cfg(super().__init__, kwargs)
    
        if self.inference_func == 'twoagent':
            reactive_cfg = parse_args_list(['-c', cfg.reactive_cfg_file])
            cfg.reactive_model = load_other_model(reactive_cfg)


    def inference_twoagent(self, batch, **kwargs) -> dotdict:
        agent1, gt_agent1 = self.prepare_agent(batch, 1)
        agent2, gt_agent2 = self.prepare_agent(batch, 2)

        assert self.use_gt_length
        num_new_frames = gt_agent2.pose_vel.size(1) - self.num_start_frames

        for ts in range(self.num_start_frames, self.num_start_frames + num_new_frames, self.ts_step):
            next_motion1 = self.pred_next(agent1, agent2, ts, gt_agent=gt_agent1)
            next_motion2 = cfg.reactive_model.pred_next(agent2, agent1, ts)
            for i in range(self.ts_step):
                self.update_agent(agent1, next_motion1, ts, i, agent2=agent2, gt_agent=gt_agent1, **kwargs)
                self.update_agent(agent2, next_motion2, ts, i, agent2=agent1, gt_agent=gt_agent2, **kwargs)
            
        output = dotdict(
            agent1 = agent1,
            agent2 = agent2,
        )
        batch.update({
            'agent1': gt_agent1,
            'agent2': gt_agent2,
        })
        return output
