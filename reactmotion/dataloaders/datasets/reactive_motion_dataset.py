import torch
import random

from easyvolcap.engine import DATASETS, cfg, dotdict, call_from_cfg
from reactmotion.dataloaders.datasets.motion_base_dataset import ReactivePoseTokenVQMotionDataset
from reactmotion.utils.motion_repr_transform import *


@DATASETS.register_module()
class BoxingReactDataset(ReactivePoseTokenVQMotionDataset):
    def __init__(self,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs)


    def load_react_motoken(self, meta, motion1, motion2):
        frames = list(range(motion1.pose_vel.size(1)))
        root_info = MoReprTrans.cal_root_info(motion1, motion2, frames, frames[0])[0]
        pose_series = MoReprTrans.cal_pose_series(motion1, frames)[0]
        oppo_pose = MoReprTrans.cal_oppo_pose(motion1, motion2, frames)[0]
        tokens = cfg.motoken_net.encode(pose_series[None].cuda())[0].detach().cpu()
        latents = cfg.motoken_net.quantizer.dequantize(tokens[None].cuda())[0].detach().cpu()
        
        return dotdict(
            meta = meta,
            tokens = tokens,
            latents = latents,
            pose_series = pose_series,
            oppo_pose = oppo_pose,
            root_info = root_info,
        )
    

    def cal_norm_data(self):
        MoReprTrans.save_norm_data(self.train_data, 'oppo_pose')
        MoReprTrans.save_norm_data(self.train_data, 'root_info')


    def _get_train_data_item(self, index):
        index = index % len(self.preprocess_data)
        wholeseq = self.preprocess_data[index]
        meta = wholeseq.meta

        tokens, latents, pose_series, oppo_pose, root_info = \
            wholeseq.get('tokens', wholeseq.latents[:, 0]), wholeseq.latents, wholeseq.pose_series, wholeseq.oppo_pose, wholeseq.root_info
        token_sid = random.randint(0, len(tokens) - self.infer_gpttoken_window_size - 1)
        token_eid = token_sid + self.block_size
        pose_sid, pose_eid = token_sid * U, token_eid * U
        token_eid = min(len(tokens), token_eid)
        
        self.max_motion_length = self.block_size
        tokens = self.pad_tokens(tokens[token_sid:token_eid])
        latents = self.pad_motion(latents[token_sid:token_eid])
        
        self.max_motion_length = self.block_size * U
        pose_series = self.pad_motion(pose_series[pose_sid:pose_eid])
        oppo_pose = self.pad_motion(oppo_pose[pose_sid:pose_eid])
        root_info = self.pad_motion(root_info[pose_sid:pose_eid])

        mask = torch.ones(self.block_size)
        mask[token_eid - token_sid:] = 0
        pose_mask = mask[:, None].repeat(1, U).view(-1)
        
        return dotdict(
            meta = meta,
            mask = mask,
            pose_mask = pose_mask,
            tokens = tokens,
            latents = latents,
            pose_series = pose_series,
            oppo_pose = oppo_pose,
            root_info = root_info,
        )
    



@DATASETS.register_module()
class SparseControlDataset(ReactivePoseTokenVQMotionDataset):

    def load_react_motoken(self, meta, motion1, motion2):
        frames = list(range(motion1.pose_vel.size(1)))
        root_info = MoReprTrans.cal_root_info(motion1, motion2, frames, frames[0])[0]
        ctrl_info = MoReprTrans.cal_ctrl_info(motion1, motion1, frames)[0]
        pose_series = MoReprTrans.cal_pose_series(motion1, frames)[0]
        oppo_pose = MoReprTrans.cal_oppo_pose(motion1, motion2, frames)[0]
        tokens = cfg.motoken_net.encode(pose_series[None].cuda())[0].detach().cpu()
        latents = cfg.motoken_net.quantizer.dequantize(tokens[None].cuda())[0].detach().cpu()
        
        return dotdict(
            meta = meta,
            tokens = tokens,
            latents = latents,
            pose_series = pose_series,
            oppo_pose = oppo_pose,
            root_info = root_info,
            ctrl_info = ctrl_info,
        )
    

    def cal_norm_data(self):
        MoReprTrans.save_norm_data(self.train_data, 'oppo_pose')
        MoReprTrans.save_norm_data(self.train_data, 'root_info')
        MoReprTrans.save_norm_data(self.train_data, 'ctrl_info')


    def _get_train_data_item(self, index):
        index = index % len(self.preprocess_data)
        wholeseq = self.preprocess_data[index]
        meta = wholeseq.meta

        tokens, latents, pose_series, oppo_pose, root_info, ctrl_info = \
            wholeseq.tokens, wholeseq.latents, wholeseq.pose_series, wholeseq.oppo_pose, wholeseq.root_info, wholeseq.ctrl_info
        
        token_sid = random.randint(0, len(tokens) - self.infer_gpttoken_window_size - 1)
        token_eid = token_sid + self.block_size
        pose_sid, pose_eid = token_sid * U, token_eid * U
        
        self.max_motion_length = self.block_size
        latents = self.pad_motion(latents[token_sid:token_eid])
        
        self.max_motion_length = self.block_size * U
        pose_series = self.pad_motion(pose_series[pose_sid:pose_eid])
        oppo_pose = self.pad_motion(oppo_pose[pose_sid:pose_eid])
        root_info = self.pad_motion(root_info[pose_sid:pose_eid])
        ctrl_info = self.pad_motion(ctrl_info[pose_sid:pose_eid])
        token_eid = min(len(tokens), token_eid)
        mask = torch.ones(self.block_size)
        mask[token_eid - token_sid:] = 0
        pose_mask = mask[:, None].repeat(1, U).view(-1)
        
        return dotdict(
            meta = meta,
            mask = mask,
            pose_mask = pose_mask,
            latents = latents,
            pose_series = pose_series,
            oppo_pose = oppo_pose,
            root_info = root_info,
            ctrl_info = ctrl_info,
        )