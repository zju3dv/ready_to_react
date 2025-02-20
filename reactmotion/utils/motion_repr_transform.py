import torch
from torch.nn.parameter import Parameter
import numpy as np
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d
from easyvolcap.engine import *
from easyvolcap.utils.data_utils import dotdict
import reactmotion.utils.matrix as matrix
from reactmotion.utils.skeleton import SG
from reactmotion.utils.geo_transform import compute_root_quaternion_ay
from reactmotion.utils.engine_utils import parse_args_list


### rarely changed
datasetname = cfg.get('datasetname')
dataname = cfg.get('dataname')
skeleton_process = cfg.get('skeleton_process')
skeleton = cfg.get('skeleton', 'motive')
data_fps = cfg.get('data_fps', 120)
train_fps = 30
downsample = data_fps // train_fps
dpose = 12

njoints = SG[skeleton]["njoints"]
parents = SG[skeleton]["parents"]
D = njoints * dpose

ALIGNMENT = SG[skeleton]["alignment"]
JOINTS_NAMES = SG[skeleton]["names"]
VR3JOINTS_NAMES = SG[skeleton]["vr3joints"]
VR3JOINTS_INDEX = [JOINTS_NAMES.index(j) for j in VR3JOINTS_NAMES]
### rarely changed


if cfg.get('motoken_cfg_file', None) is not None:
    motoken_cfg = parse_args_list(['-c', cfg.motoken_cfg_file])
    U = motoken_cfg.runner_cfg.get("unit_length", 4)
    cfg.norm_cfg.motoken_input_norm_file = motoken_cfg.norm_cfg.motoken_input_norm_file
else:
    motoken_cfg = None
    U = cfg.runner_cfg.get("unit_length", 4) # need to check when using other models


class MoReprTrans:

    @staticmethod
    def split_pose(ctrl):
        '''
            Split ctrl series to pose_pos, pose_rot, pose_vel, root_off, root_dir, root_ctrl
        '''
        B, T, C = ctrl.shape
        pose = ctrl[..., :D].reshape(B, T, njoints, dpose)

        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = rotation_6d_to_matrix(pose[..., 3:9]),
            pose_vel = pose[..., 9:12],
            root_off = ctrl[..., D:D+2],
            root_dir = ctrl[..., D+2:D+4],
            root_ctrl = ctrl[..., D:D+4],
        )
    

    @staticmethod
    def get_actor_motion(motion, frames, squeeze):
        actor_motion = dotdict()
        if squeeze:
            for key, item in motion.items():
                if isinstance(item, torch.Tensor):
                    actor_motion[key] = item[0, frames]
            return actor_motion
        
        for key, item in motion.items():
            if isinstance(item, torch.Tensor):
                actor_motion[key] = item[:, frames]
        return actor_motion
    

    @staticmethod
    def get_apd_motion(motion, K):
        for key, item in motion.items():
            if isinstance(item, torch.Tensor):
                motion[key] = torch.repeat_interleave(motion[key], K, dim=0)
        return motion


    @staticmethod
    def get_root_transmat(pose_pos: torch.Tensor, root_pos: torch.Tensor):
        '''
            Inputs:
                pose_pos: (B, J, 3) in world
                root_pos: (B, 1, 3) in world
            Outputs:
                root_transmat: rootmat
                root_rotmat: rootrot, root axis
        '''
        root_y_quat = compute_root_quaternion_ay(pose_pos, skeleton_type=skeleton)
        root_rotmat = quaternion_to_matrix(root_y_quat)
        root_transmat = matrix.get_TRS(root_rotmat, root_pos)
        return root_transmat, root_transmat[..., :3, :3]
    
    
    @staticmethod
    def cal_pose(motion, frames):
        B = motion.root_mat.size(0)
        root_mat, root_rot = motion.root_mat[:, frames], motion.root_rot[:, frames]
        pos = matrix.get_relative_position_to(motion.pose_pos[:, frames], root_mat)
        rot = matrix.get_mat_BtoA(root_rot[:, :, None], motion.pose_rot[:, frames])
        vel = matrix.get_relative_direction_to(motion.pose_vel[:, frames], root_mat)
        pose = torch.cat((pos, matrix_to_rotation_6d(rot), vel), dim=-1).view(B, len(frames), -1)
        return pose
    

    @staticmethod
    def cal_root_ctrl(motion, frames):
        return motion.root_ctrl[:, frames]
    
    
    @staticmethod
    def cal_root_info(motion, oppo, frames, f0):
        root2oppo_ctrl = motion.root2oppo_ctrl[:, frames]
        # cal root_pos to the center of the first frame
        center = (motion.root_pos[:, [f0]] + oppo.root_pos[:, [f0]]) / 2
        root2center = motion.root_pos[:, frames] - center
        root2center = (root2center[..., [0, 2]] ** 2).sum(-1, keepdim=True) ** 0.5
        root_info = torch.cat([root2center, root2oppo_ctrl], dim=-1)
        return root_info
    
    
    @staticmethod
    def cal_ctrl_info(motion, gt_motion, frames):
        B = motion.root_mat.size(0)
        root_mat, root_rot = motion.root_mat[:, frames], motion.root_rot[:, frames]
        keypos = matrix.get_relative_position_to(gt_motion.pose_pos_vr3j[:, frames], root_mat)
        keyrot = matrix.get_mat_BtoA(root_rot[:, :, None], gt_motion.pose_rot_vr3j[:, frames])
        ctrl_info = torch.cat((keypos, matrix_to_rotation_6d(keyrot)), dim=-1).view(B, len(frames), -1)
        return ctrl_info
    

    @staticmethod
    def cal_root2oppo_ctrl(motion, oppo, frames):
        frames_prev = [f - 1 for f in frames]
        root_off = matrix.get_relative_position_to(motion.root_pos[:, frames][:, :, None], oppo.root_mat[:, frames_prev]) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(oppo.root_rot[:, frames_prev], motion.root_rot[:, frames]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
        return root_ctrl


    @staticmethod
    def cal_oppo_pose(motion, oppo, frames):
        B = motion.root_mat.size(0)
        root_mat, root_rot = motion.root_mat[:, frames], motion.root_rot[:, frames]
        keypos = matrix.get_relative_position_to(oppo.pose_pos[:, frames], root_mat)
        keyrot = matrix.get_mat_BtoA(root_rot[:, :, None], oppo.pose_rot[:, frames])
        keyvel = matrix.get_relative_direction_to(oppo.pose_vel[:, frames], root_mat)
        oppo_pose = torch.cat((keypos, matrix_to_rotation_6d(keyrot), keyvel), dim=-1).view(B, len(frames), -1)
        return oppo_pose
    

    @staticmethod
    def cal_pose_series(motion, frames):
        # pose series: pose + root_ctrl
        pose = MoReprTrans.cal_pose(motion, frames)
        root_ctrl = MoReprTrans.cal_root_ctrl(motion, frames)
        pose_series = torch.cat([pose, root_ctrl], dim=-1)
        return pose_series

    
    @staticmethod
    def _get_norm_file(data_key, norm_file=None):
        if norm_file is None:
            return f"data/{datasetname}/{skeleton_process}/{dataname}/{data_key}.npy"
        else: return norm_file

    @staticmethod
    def save_norm_data(data_dict, data_key, norm_file=None):
        shape = data_dict[0][data_key].shape
        all_data = [data_dict[i][data_key].view(-1, shape[-1]) for i in range(len(data_dict))]
        all_data = torch.cat(all_data, dim=0)
        norm_data = all_data.mean(dim=0), all_data.std(dim=0)
        norm_data = torch.stack(norm_data, dim=0)

        norm_file = MoReprTrans._get_norm_file(data_key, norm_file=norm_file)
        log(norm_file)
        Path(norm_file).parent.mkdir(parents=True, exist_ok=True)
        # import pdb; pdb.set_trace()
        np.save(norm_file, norm_data.numpy())

    @staticmethod
    def load_norm_data(data_key, modify_std=True, norm_file=None):
        norm_file = MoReprTrans._get_norm_file(data_key, norm_file=norm_file)
        log("load norm file", blue(norm_file))
        norm_data = np.load(norm_file)
        if modify_std:
            for i in range(norm_data.shape[1]): # avoid std too small
                if norm_data[1, i] < 0.0001: norm_data[1, i] = 1
        return Parameter(torch.from_numpy(norm_data), requires_grad=False)
    
    @staticmethod
    def Normalize(X, N):
        mean = N[0]
        std = N[1]
        return (X - mean) / std

    @staticmethod
    def Renormalize(X, N):
        mean = N[0]
        std = N[1]
        return (X * std) + mean