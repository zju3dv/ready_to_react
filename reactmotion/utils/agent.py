import torch

from reactmotion.utils.engine_utils import AGENTS
from reactmotion.utils.data_utils import Lerp
from reactmotion.utils.geo_transform import transform_mat, quaternion_to_matrix
from reactmotion.utils.quat_utils import qbetween
from reactmotion.utils.motion_repr_transform import *


@AGENTS.register_module()
class Agent():
    def __init__(self, 
                 device, 
                 block_size: int,
                 interpolate: bool = False, # interpolate pos and vel
                 align: bool = False, # align rot and pos
                ):
        # settings
        self.device = device
        self.block_size = block_size
        self.interpolate = interpolate
        self.align = align
        self.apd_compute = cfg.runner_cfg.evaluator_cfg.get('apd_compute', False)
        # glb
        self.pose_pos = None
        self.pose_rot = None
        self.pose_vel = None
        self.root_mat = None
        # loc
        self.loc_pos = None
        

    @property
    def root_rot(self, ):
        return self.root_mat[..., :3, :3]
    
    @property
    def root_pos(self, ):
        return self.root_mat[..., :3, -1]

    def loc_pos_mag(self, j: int):
        return torch.norm(self.loc_pos[j])
    
    def get_curr(self, key, block_size=None):
        value = getattr(self, key)
        block_size = self.block_size if block_size is None else block_size
        block_size = min(value.size(1), block_size)
        return value[:, -block_size:]
    
    def update(self, key: str, value: torch.Tensor, replace: bool = False, **kwargs):
        if key not in self.__dict__ or getattr(self, key) is None or replace: setattr(self, key, value)
        else: setattr(self, key, torch.cat((getattr(self, key), value), dim=1))
    
    def update_root(self, wld, **kwargs):
        self.update('root_mat', wld.root_mat, **kwargs)
        self.update('root_ctrl', wld.root_ctrl, **kwargs)
        if 'root2oppo_ctrl' in wld:
            self.update('root2oppo_ctrl', wld.root2oppo_ctrl, **kwargs)

    def update_pose(self, wld, **kwargs):
        self.update('pose_pos', wld.pose_pos, **kwargs)
        self.update('pose_rot', wld.pose_rot, **kwargs)
        self.update('pose_vel', wld.pose_vel, **kwargs)

    def update_vr3j(self, wld, **kwargs):
        if 'pose_pos_vr3j' in wld:
            self.update('pose_pos_vr3j', wld.pose_pos_vr3j, **kwargs)
            self.update('pose_rot_vr3j', wld.pose_rot_vr3j, **kwargs)

    def init_from_agent_motion(self, agent_motion):
        if self.apd_compute:
            agent_motion = MoReprTrans.get_apd_motion(agent_motion, K=10)
        self.update_root(agent_motion, replace=True)
        self.update_pose(agent_motion, replace=True)
        self.update_vr3j(agent_motion, replace=True)
        # t-pose
        if self.align: # record t-pose loc_pos
            glb_pos = self.pose_pos[0, -1]
            glb_mat = transform_mat(self.pose_rot[0, -1], self.pose_pos[0, -1])
            loc_pos = torch.zeros((njoints, 3)).to(self.device)
            for j in range(1, njoints):
                loc_pos[j:j+1] = matrix.get_relative_position_to(glb_pos[j:j+1], glb_mat[parents[j]])
            self.loc_pos = loc_pos


    def align_pose(self, glb):
        '''
            Align pose_pos and pose_rot
        '''
        for j in range(njoints):
            if ALIGNMENT[j]:
                par_pos = glb.pose_pos[:, 0, parents[j]]
                par_rot = glb.pose_rot[:, 0, parents[j]]
                pos = glb.pose_pos[:, 0, j]
                target = pos - par_pos
                aligned = matrix.get_direction_from(self.loc_pos[j][None], transform_mat(par_rot, par_pos)[0])
                glb.pose_rot[:, 0, parents[j]] = new_par_rot = quaternion_to_matrix(qbetween(aligned, target)) @ par_rot
                glb.pose_pos[:, 0, j] = new_pos = par_pos + self.loc_pos_mag(j) * matrix.normalized(target)

        return glb


    def get_glb_root_by_loc(self, loc, glb=None):
        glb = dotdict() if glb is None else glb
        B, T, J, C = loc.pose_pos.shape
        glb.root_pos = torch.zeros((B, T, 3)).to(self.device)
        glb.root_rot = torch.zeros((B, T, 3, 3)).to(self.device)
        for t in range(T):
            if t == 0:
                glb.root_pos[:, [0]] = self.root_pos[:, [-1]] + \
                    matrix.get_position_from_rotmat(matrix.xz2xyz(loc.root_off[:, t])[:, None],
                                                    self.root_rot[:, -1])
                glb.root_rot[:, 0] = matrix.get_mat_BfromA(self.root_rot[:, -1],
                                                           matrix.xzdir2rotmat(loc.root_dir[:, t]))
            else:
                glb.root_pos[:, t] = glb.root_pos[:, t-1] + \
                    matrix.get_position_from_rotmat(matrix.xz2xyz(loc.root_off[:, t])[:, None],
                                                    glb.root_rot[:, t-1])
                glb.root_rot[:, t] = matrix.get_mat_BfromA(glb.root_rot[:, t-1],
                                                           matrix.xzdir2rotmat(loc.root_dir[:, t]))
        glb.root_mat = matrix.get_TRS(glb.root_rot, glb.root_pos)
        # TODO: move it out
        glb.root_ctrl = loc.root_ctrl
        return glb
        

    def get_glb_pose_by_loc(self, loc, glb=None):
        glb = dotdict() if glb is None else glb
        B, T, J, C = loc.pose_pos.shape
        glb.pose_pos = torch.zeros((B, T, J, 3)).to(self.device)
        glb.pose_rot = torch.zeros((B, T, J, 3, 3)).to(self.device)
        glb.pose_vel = torch.zeros((B, T, J, 3)).to(self.device)
        for t in range(T):
            glb.pose_pos[:, t] = matrix.get_position_from(loc.pose_pos[:, t], glb.root_mat[:, t])
            glb.pose_rot[:, t] = matrix.get_mat_BfromA(glb.root_rot[:, t][:, None], loc.pose_rot[:, t])
            glb.pose_vel[:, t] = matrix.get_direction_from(loc.pose_vel[:, t], glb.root_mat[:, t])
            if self.interpolate:
                if t == 0:
                    glb.pose_pos[:, t] = Lerp(glb.pose_pos[:, t], 
                                              self.pose_pos[:, -1] + glb.pose_vel[:, t] / train_fps, 0.5)
                else:
                    glb.pose_pos[:, t] = Lerp(glb.pose_pos[:, t], 
                                              glb.pose_pos[:, t-1] + glb.pose_vel[:, t] / train_fps, 0.5)
        if self.align:
            for j in range(njoints):
                if ALIGNMENT[j]:
                    par_pos = glb.pose_pos[:, 0, parents[j]]
                    par_rot = glb.pose_rot[:, 0, parents[j]]
                    pos = glb.pose_pos[:, 0, j]
                    target = pos - par_pos
                    aligned = matrix.get_direction_from(self.loc_pos[j][None], transform_mat(par_rot, par_pos)[0])
                    glb.pose_rot[:, 0, parents[j]] = new_par_rot = quaternion_to_matrix(qbetween(aligned, target)) @ par_rot
                    glb.pose_pos[:, 0, j] = new_pos = par_pos + self.loc_pos_mag(j) * matrix.normalized(target)
        return glb


    def update_pose_by_loc(self, loc, **kwargs):
        glb = self.get_glb_root_by_loc(loc)
        glb = self.get_glb_pose_by_loc(loc, glb=glb)
        self.update_root(glb, **kwargs)
        self.update_pose(glb, **kwargs)
