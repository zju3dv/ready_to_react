import torch
import numpy as np

from easyvolcap.engine import EVALUATORS, cfg
from reactmotion.runners.evaluators.motion_evaluator import MotionEvaluator
from easyvolcap.utils.console_utils import *
from reactmotion.utils.motion_repr_transform import *
import reactmotion.utils.matrix as matrix
from reactmotion.utils.wis3d_utils import wis3d_add_skeleton, make_wis3d
from reactmotion.utils.data_utils import Lerp


def error_xy(x, y):
    return ((x - y) ** 2).sum(-1).sqrt().mean()


@EVALUATORS.register_module()
class MoTokenEvaluator(MotionEvaluator):
    def __init__(self, 
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)


    def compute_pose_error(self, output, batch):
        error = dotdict()
        error.pose_pos_mpj_error = error_xy(output.pose_pos, batch.pose_pos).item()
        error.pose_rot_mpj_error = error_xy(output.pose_rot, batch.pose_rot).item()
        error.pose_vel_mpj_error = error_xy(output.pose_vel, batch.pose_vel).item()
        return error

    def compute_root_error(self, output, batch):
        error = dotdict()
        error.root_off_error = error_xy(output.root_off, batch.root_off).item()
        error.root_dir_error = error_xy(output.root_dir, batch.root_dir).item()
        return error


    def post_process(self, output, batch):
        B, W = batch.get('supervise_MoToken').shape[:2]
        device = batch.get('supervise_MoToken').device

        batch.update(MoReprTrans.split_pose(batch['supervise_MoToken']))
        output.update(MoReprTrans.split_pose(output['netout_x']))

        def post_process_global(dict_obj: dotdict, interpolate=False):
            glb = dotdict()
            glb.root_pos = torch.zeros((B, W, 3)).to(device)
            glb.root_rot = torch.zeros((B, W, 3, 3)).to(device)
            for b in range(B):
                glb.root_rot[b, 0] = torch.eye(3).to(device)
                for t in range(1, W):
                    glb.root_pos[b, t] = glb.root_pos[b, t-1] + \
                        matrix.get_position_from_rotmat(matrix.xz2xyz(dict_obj.root_off[[b], t-1]),
                                                        glb.root_rot[b, t-1])
                    glb.root_rot[b, t] = matrix.get_mat_BfromA(glb.root_rot[b, t-1],
                                                               matrix.xzdir2rotmat(dict_obj.root_dir[b, t-1]))
            
            glb.root_mat = matrix.get_TRS(glb.root_rot, glb.root_pos)
            glb.pose_pos = torch.zeros_like(dict_obj.pose_pos)
            for b in range(B):
                for t in range(W):
                    glb.pose_pos[b, t] = matrix.get_position_from(dict_obj.pose_pos[b, t], glb.root_mat[b, t])
                    
                    if interpolate and t > 0:
                        pose_vel = matrix.get_direction_from(dict_obj.pose_vel[b, t-1], glb.root_mat[b, t-1])
                        glb.pose_pos[b, t] = Lerp(glb.pose_pos[b, t], 
                                                  glb.pose_pos[b, t-1] + pose_vel / train_fps, 0.5)
                    
            dict_obj.glb_pose_pos = glb.pose_pos

        post_process_global(batch) # gt donot need interpolate
        post_process_global(output, interpolate=False)


    def visualize_pose_pos(self, ):
        errors_list = self.metrics.pose_pos_mpj_error
        sort_idx = np.argsort(errors_list)[::-1]
        N = len(sort_idx)
        for i in range(int(self.lastn * N)): self.vis_pose_i(sort_idx[i], i)
        for i in range(int(self.bestn * N)): self.vis_pose_i(sort_idx[-i], int(self.lastn * N)+i)
        log(f"lastn: {int(self.lastn * N)}")
        log(f"bestn: {int(self.bestn * N)}")


    def vis_pose_i(self, idx, vis3d_id):
        gt = self.vis_stats.gt[idx]
        pd = self.vis_stats.pd[idx]
        name = gt.meta.npy_file[0].split('/')[-1].split('_')[0]
        sbj = gt.meta.sbj_name[0]
        self.vis3d = make_wis3d(f"{cfg.exp_name}-epoch{self.epoch_name}-sort{vis3d_id:02d}-file{name}-{sbj}")
        for f in range(gt.glb_pose_pos.shape[1]):
            wis3d_add_skeleton(self.vis3d, f, gt.glb_pose_pos[0, f], parents, f"pose-pos-gt-{name}")
            wis3d_add_skeleton(self.vis3d, f, pd.glb_pose_pos[0, f], parents, f"pose-pos-pd")