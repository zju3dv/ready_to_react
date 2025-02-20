import torch
import torch.nn.functional as F
from pytorch3d.transforms import *
from reactmotion.utils.quat_utils import qbetween


def apply_T_on_points(points, T):
    """
    Args:
        points: (..., N, 3)
        T: (..., 4, 4)
    Returns: (..., N, 3)
    """
    points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T


def transform_mat(R, t):
    """
    Args:
        R: Bx3x3 array of a batch of rotation matrices
        t: Bx3x(1) array of a batch of translation vectors
    Returns:
        T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def compute_root_quaternion_ay(joints, skeleton_type='smplx'):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        root_quat: (B, 4) from z-axis to fz
    """
    joints_shape = joints.shape
    joints = joints.reshape((-1,) + joints_shape[-2:])
    t_ayfz2ay = joints[:, 0, :].clone()
    t_ayfz2ay[:, 1] = 0  # do not modify y

    if skeleton_type == 'smplx':
        RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side
        RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side
    elif skeleton_type == 'motive':
        RL_xz_h = joints[:, 13, [0, 2]] - joints[:, 17, [0, 2]]
        RL_xz_s = joints[:, 5, [0, 2]] - joints[:, 9, [0, 2]]
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    if I_mask.sum() > 0:
        # log("{} samples can't decide the face direction".format(I_mask.sum()))
        for i in range(I_mask.shape[0]):
            if I_mask[i]:
                RL_xz[i] = RL_xz[i-1]

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)

    z_dir[..., 2] += 1e-9
    pos_z_vec = torch.tensor([0, 0, 1]).to(joints.device).float()  # (3,)
    root_quat = qbetween(pos_z_vec.repeat(joints.shape[0]).reshape(-1, 3), z_dir)  # (B, 4)
    root_quat = root_quat.reshape(joints_shape[:-2] + (4,))
    return root_quat