import torch
import numpy as np
import pickle
from typing import List
from glob import glob
from os.path import join
from pathlib import Path
from torch.utils.data import Dataset
from wis3d import Wis3D
from tqdm import tqdm

from easyvolcap.engine import dotdict, call_from_cfg, log, cfg
from reactmotion.utils.net_utils import load_other_network
from reactmotion.utils.motion_repr_transform import *
from reactmotion.utils.geo_transform import apply_T_on_points

# T_z2y: zup to yup
T_z2y = torch.FloatTensor([[
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]]]) # (1, 4, 4)



class MotionBaseDataset(Dataset):
    def __init__(self,
                 split: str = "TRAIN",
                 vis_output: int = cfg.runner_cfg.evaluator_cfg.get('vis_output', 0),
                 test_speed: bool = cfg.runner_cfg.evaluator_cfg.get('test_speed', False),
                 max_motion_length: int = 300,
                 min_motion_length: int = 100,
                 # motion split
                 gen_motion_split: bool = False,
                 # preprocess
                 gen_preprocess: bool = False,
                 # dirs
                 root_dirs: List = [],
                 motion_dir_names: List = [], # root_dir / motion_dir_name / *.npy
                 ):
        
        self.split = split
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_token_length = 0
        self.pad_token = 0

        # vis
        self.vis_output : int = vis_output
        self.vis3d : Wis3D = None

        # motion split
        self.gen_motion_split = gen_motion_split
        self.skeleton_process = cfg.skeleton_process
        self.train_motion_file = f"data/{datasetname}/{skeleton_process}/motion/train_motion.pkl"
        self.test_motion_file = f"data/{datasetname}/{skeleton_process}/motion/test_motion.pkl"

        # preprocess
        self.gen_preprocess = gen_preprocess
        self.preprocess_file = f"data/{datasetname}/{skeleton_process}/{dataname}/{dataname}.pkl"
        self.load_prep_type = dataname
        self.get_prep_type = dataname

        # dirs
        self.root_dirs = root_dirs
        self.motion_dir_names = motion_dir_names

        # motion split: pure motion data, used for unify baselines
        self.motion = self.load_motion_split()

        # load_prep_data: preprocess motion representation
        self.preprocess_data = self.load_prep_data()
        
        if self.vis_output != 0: 
            self.preprocess_data = self.preprocess_data[1::2]
            self.motion = self.motion[1::2]
            self.preprocess_data = self.preprocess_data[-self.vis_output:]
            self.motion = self.motion[-self.vis_output:]
        if test_speed:
            self.preprocess_data = self.preprocess_data[:20]
            self.motion = self.motion[:20]
        
        assert len(self.motion) == len(self.preprocess_data)
        
        log(f'preprocess data length: {len(self.preprocess_data)}')

    
    def __len__(self):
        if self.split == "TRAIN":
            return len(self.preprocess_data) * 100
        else: return len(self.preprocess_data)
    

    def load_prep_data(self, ):
        # Load preprocess data if exists
        if not self.gen_preprocess:
            if self.split == "TRAIN":
                if self.preprocess_file is not None:
                    return pickle.load(open(self.preprocess_file, 'rb'))    
            return self.motion
        
        # Save preprocess data
        assert self.split == "TRAIN"
        self.train_data = []
        for motion in tqdm(self.motion):
            self.train_data.append(
                getattr(self, f'load_{self.load_prep_type}')(motion.meta, motion.motion1, motion.motion2)
            ) # append
        # Save preprocess data
        Path(self.preprocess_file).parent.mkdir(parents=True, exist_ok=True)
        log(self.preprocess_file)
        # import pdb; pdb.set_trace()
        pickle.dump(self.train_data, open(self.preprocess_file, 'wb'))
        self.cal_norm_data()
        exit()


    def cal_norm_data(self, ):
        pass # do nothing


    def load_motion_split(self, ):
        # Load motion split data if exists
        if not self.gen_motion_split:
            if self.split == 'TRAIN':
                return pickle.load(open(self.train_motion_file, 'rb'))
            else:
                return pickle.load(open(self.test_motion_file, 'rb'))[::downsample]
        
        return getattr(self, f'load_motion_split_{datasetname}')()
        
    def load_motion_split_boxing(self, ):
        # Load lowlevel motion data
        train_motions, test_motions = [], []
        for root_dir, motion_dir_name in (zip(self.root_dirs, self.motion_dir_names)):
            npy_files = sorted(glob(join(root_dir, motion_dir_name, '*.npy')))
            for npy_file in tqdm(npy_files):
                npy_data = np.load(npy_file, allow_pickle=True).item()
                motion_data = dotdict()
                sbj_names = list(set(key[:-9] for key in npy_data.keys()))
                oppo_names = sbj_names[::-1]
                for sbj_name in sbj_names:
                    pose_pos = torch.FloatTensor(npy_data[f'{sbj_name}_position'])
                    pose_rot = torch.FloatTensor(npy_data[f'{sbj_name}_rotation'])
                    motion_data[sbj_name] = getattr(self, f'get_motion_{self.skeleton_process}')(pose_pos[None], pose_rot[None])
                for sbj_name, oppo_name in zip(sbj_names, oppo_names):
                    motion_data[sbj_name] = getattr(self, f'get_motion2oppo_{self.skeleton_process}')(motion_data[sbj_name], motion_data[oppo_name])
                if self.skeleton_process == "reactive":
                    for sbj_name in sbj_names:
                        motion_data[sbj_name].update({
                            'pose_pos': motion_data[sbj_name].pose_pos[:, downsample:],
                            'pose_rot': motion_data[sbj_name].pose_rot[:, downsample:],
                            'root_mat': motion_data[sbj_name].root_mat[:, downsample:],
                            'root_rot': motion_data[sbj_name].root_rot[:, downsample:],
                            'root_pos': motion_data[sbj_name].root_pos[:, downsample:],
                        })
                elif self.skeleton_process == "vr3joints":
                    for sbj_name in sbj_names:
                        motion_data[sbj_name].update({
                            'pose_pos': motion_data[sbj_name].pose_pos[:, downsample:-U*downsample],
                            'pose_rot': motion_data[sbj_name].pose_rot[:, downsample:-U*downsample],
                            'root_mat': motion_data[sbj_name].root_mat[:, downsample:-U*downsample],
                            'root_rot': motion_data[sbj_name].root_rot[:, downsample:-U*downsample],
                            'root_pos': motion_data[sbj_name].root_pos[:, downsample:-U*downsample],
                        })
                else: raise NotImplementedError
                train_motion, test_motion = self.split_motion(npy_file, motion_data)
                train_motions.extend(train_motion)
                test_motions.extend(test_motion)
        
        # Save train motion
        log(self.train_motion_file)
        log(self.test_motion_file)
        # import pdb; pdb.set_trace()
        Path(self.train_motion_file).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(train_motions, open(self.train_motion_file, 'wb'))
        pickle.dump(test_motions, open(self.test_motion_file, 'wb'))
        exit()
    
   
    def split_motion(self, npy_file, motion_data):
        train_data, test_data = [], []
        
        sbj_names = list(motion_data.keys())
        oppo_names = sbj_names[::-1]
        data_length = motion_data[sbj_names[0]].pose_vel.size(1)
        for i in range(2): # two players
            sbj_name, oppo_name = sbj_names[i], oppo_names[i]
            motion, oppo_motion = motion_data[sbj_name], motion_data[oppo_name]
            # downsample
            for d in range(downsample):
                frames, FN, TRAIN_FN = self.get_train_test_frames(data_length, d)
                train_data.append(dotdict(
                    meta = self.get_meta(npy_file, sbj_name, oppo_name, d, 'train'),
                    motion1 = MoReprTrans.get_actor_motion(motion, frames[:TRAIN_FN], squeeze=False),
                    motion2 = MoReprTrans.get_actor_motion(oppo_motion, frames[:TRAIN_FN], squeeze=False),
                ))
                test_data.append(dotdict(
                    meta = self.get_meta(npy_file, sbj_name, oppo_name, d, 'test'),
                    motion1 = MoReprTrans.get_actor_motion(motion, frames[TRAIN_FN:], squeeze=False),
                    motion2 = MoReprTrans.get_actor_motion(oppo_motion, frames[TRAIN_FN:], squeeze=False),
                ))
        return train_data, test_data


    def get_meta(self, npy_file, sbj_name, oppo_name, downsample, split):
        # meta info
        return dotdict(
            npy_file = npy_file,
            sbj_name = sbj_name,
            oppo_name = oppo_name,
            downsample = downsample,
            split = split,
            skeleton = skeleton,
        )
    

    def get_train_test_frames(self, data_length, d):
        '''
            data_length: total frames
            d: downsample
        '''
        frames = list(range(d, data_length, downsample))
        FN = len(frames) // U * U
        frames = frames[:FN]
        TRAIN_FN = int(FN * 0.8) // U * U
        return frames, FN, TRAIN_FN
    

    def __getitem__(self, index) -> dotdict:
        if self.split == "TRAIN":
            return self._get_train_data_item(index)
        elif self.split == "VAL":
            return self._get_eval_data_item(index)
        
    
    def _get_train_data_item(self, index):
        return dotdict()

    def _get_eval_data_item(self, index):
        return self._get_final_eval_data_item(index)
    
    def _get_final_eval_data_item(self, index):
        index = index % len(self.preprocess_data)
        motion = self.motion[index]
        meta = motion.meta

        sid, eid = 0, motion.motion1.pose_vel.size(1)
        m_length = eid - sid
        cond_m_length = U
        
        return dotdict(
            meta = meta,
            agent1 = MoReprTrans.get_actor_motion(motion.motion1, list(range(m_length)), squeeze=True),
            agent2 = MoReprTrans.get_actor_motion(motion.motion2, list(range(m_length)), squeeze=True),
            cond_agent1 = MoReprTrans.get_actor_motion(motion.motion1, list(range(cond_m_length)), squeeze=True),
            cond_agent2 = MoReprTrans.get_actor_motion(motion.motion2, list(range(cond_m_length)), squeeze=True),
        )


    def pad_motion(self, motion):
        S = len(motion)
        if S < self.max_motion_length:
            pad = torch.zeros((self.max_motion_length - S, *motion.shape[1:]), dtype=motion.dtype)
            motion = torch.cat((motion, pad), dim=0)
        return motion


    def get_motion_reactive(self, pose_pos, pose_rot):
        # transform from zup to yup
        pose_pos = apply_T_on_points(pose_pos, T_z2y)
        pose_rot[:, :, 0] = T_z2y[..., :3, :3] @ pose_rot[:, :, 0]
        # transform rel2parents to global rotations
        pose_rot = matrix.forward_kinematics(pose_rot, parents)
        # get root pos
        root_pos = pose_pos[:, :, 0].clone()
        root_pos[..., 1] = 0 # y = 0
        # get pose vel
        pose_vel = (pose_pos[:, downsample:] - pose_pos[:, :-downsample]) * train_fps
        # get root mat
        root_mat, root_rot = MoReprTrans.get_root_transmat(pose_pos=pose_pos, root_pos=root_pos)
        # get root ctrl
        root_off = matrix.get_relative_position_to(root_pos[:, downsample:][:, :, None], root_mat[:, :-downsample]) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(root_rot[:, :-downsample], root_rot[:, downsample:]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
        
        return dotdict(
            pose_pos = pose_pos,
            pose_rot = pose_rot,
            pose_vel = pose_vel,
            root_mat = root_mat,
            root_rot = root_rot,
            root_pos = root_pos,
            root_off = root_off_2d,
            root_dir = root_dir_2d,
            root_ctrl = root_ctrl,
        )

    def get_motion2oppo_reactive(self, motion1: dotdict, motion2):
        # get root2oppo ctrl
        root_off = matrix.get_relative_position_to(motion1.root_pos[:, downsample:][:, :, None], motion2.root_mat[:, :-downsample]) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(motion2.root_rot[:, :-downsample], motion1.root_rot[:, downsample:]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
        motion1.update({
            'root2oppo_ctrl': root_ctrl,
        })
        return motion1
    
    def get_motion_vr3joints(self, pose_pos, pose_rot):
        # transform from zup to yup
        pose_pos = apply_T_on_points(pose_pos, T_z2y)
        pose_rot[:, :, 0] = T_z2y[..., :3, :3] @ pose_rot[:, :, 0]
        # transform rel2parents to global rotations
        pose_rot = matrix.forward_kinematics(pose_rot, parents)
        # pad U*downsample
        pose_pos = torch.cat([pose_pos, torch.repeat_interleave(pose_pos[:, -1:], U*downsample, 1)], dim=1)
        pose_rot = torch.cat([pose_rot, torch.repeat_interleave(pose_rot[:, -1:], U*downsample, 1)], dim=1)
        # get root pos
        root_pos = pose_pos[:, :, 0].clone()
        root_pos[..., 1] = 0 # y = 0
        # get pose vel
        pose_vel = (pose_pos[:, downsample:-U*downsample] - pose_pos[:, :-(1+U)*downsample]) * train_fps
        # get root mat
        root_mat, root_rot = MoReprTrans.get_root_transmat(pose_pos=pose_pos, root_pos=root_pos)
        # get root ctrl
        root_off = matrix.get_relative_position_to(root_pos[:, downsample:-U*downsample][:, :, None], root_mat[:, :-(1+U)*downsample]) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(root_rot[:, :-(1+U)*downsample], root_rot[:, downsample:-U*downsample]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
        # get vr3joints ctrl
        pose_pos_vr3j = pose_pos[:, (1+U)*downsample:, VR3JOINTS_INDEX]
        pose_rot_vr3j = pose_rot[:, (1+U)*downsample:, VR3JOINTS_INDEX]
        
        return dotdict(
            pose_pos = pose_pos,
            pose_rot = pose_rot,
            pose_vel = pose_vel,
            pose_pos_vr3j = pose_pos_vr3j,
            pose_rot_vr3j = pose_rot_vr3j,
            root_mat = root_mat,
            root_rot = root_rot,
            root_pos = root_pos,
            root_off = root_off_2d,
            root_dir = root_dir_2d,
            root_ctrl = root_ctrl,
        )
    

    def get_motion2oppo_vr3joints(self, motion1: dotdict, motion2):
        # get root2oppo ctrl
        root_off = matrix.get_relative_position_to(motion1.root_pos[:, downsample:-U*downsample][:, :, None], motion2.root_mat[:, :-(1+U)*downsample]) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(motion2.root_rot[:, :-(1+U)*downsample], motion1.root_rot[:, downsample:-U*downsample]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
        motion1.update({
            'root2oppo_ctrl': root_ctrl,
        })
        return motion1


class ReactivePoseTokenVQMotionDataset(MotionBaseDataset):
    def __init__(self,
                # vq settings
                vocab_size: int = cfg.window_cfg.get("vocab_size", 512),
                vqmotion_window_size: int = cfg.window_cfg.get("vqmotion_window_size", 64),
                block_size = cfg.window_cfg.get("block_size", 48),
                infer_gpttoken_window_size = cfg.window_cfg.get("infer_gpttoken_window_size", 1),
                **kwargs,
                ):
        
        cfg.norm_cfg.motoken_input_norm_file = motoken_cfg.norm_cfg.motoken_input_norm_file
        if kwargs.get('gen_preprocess'):
            cfg.motoken_net = load_other_network(motoken_cfg)
        call_from_cfg(super().__init__, kwargs)
        # vq settings
        self.vocab_size = vocab_size
        self.vqmotion_window_size = vqmotion_window_size
        self.block_size = block_size
        self.infer_gpttoken_window_size = infer_gpttoken_window_size

        self.pad_token = self.vocab_size # 513
        self.max_motion_length = self.block_size * U
        self.max_token_length = self.block_size
        
    
    def pad_tokens(self, tokens):
        S = len(tokens)
        if S < self.max_token_length:
            pad = self.pad_token * torch.ones((self.max_token_length - S), dtype=tokens.dtype)
            tokens = torch.cat((tokens, pad), dim=0)
        return tokens
    