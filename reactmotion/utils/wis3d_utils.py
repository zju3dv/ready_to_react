from wis3d import Wis3D
from pathlib import Path
from datetime import datetime
import torch
import os
import numpy as np
from einops import einsum
from pytorch3d.transforms import axis_angle_to_matrix
from easyvolcap.engine import *
from easyvolcap.utils.data_utils import to_numpy


def make_wis3d(name="debug", output_dir="data/wis3d", time_postfix=False) -> Wis3D:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if time_postfix:
        time_str = datetime.now().strftime("%m%d-%H%M-%S")
        name = f"{name}_{time_str}"
    log_dir = output_dir / name
    if log_dir.exists():
        log(f'remove contents of directory {log_dir}')
        os.system(f"rm -rf {log_dir}")
    log(f"Creating Wis3D {log_dir}")
    wis3d = Wis3D(output_dir.absolute(), name)
    return wis3d


def wis3d_add_skeleton(vis3d: Wis3D, t: int, joints, parents: list, name: str):
    # joints: (J, 3)
    vis3d.set_scene_id(t)
    joints = to_numpy(joints)
    start_points = joints[1:]
    end_points = [joints[parents[i]] for i in range(1, len(joints))]
    end_points = np.stack(end_points, axis=0)
    vis3d.add_lines(start_points=start_points, end_points=end_points, name=name)


color_schemes = {
    "red": ([255, 168, 154], [153, 17, 1]),
    "green": ([183, 255, 191], [0, 171, 8]),
    "blue": ([183, 255, 255], [0, 0, 255]),
    "cyan": ([183, 255, 255], [0, 255, 255]),
    "magenta": ([255, 183, 255], [255, 0, 255]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "orange": ([255, 183, 0], [255, 128, 0]),
    "grey": ([203, 203, 203], [203, 203, 203]),
}


### get skeleton mesh


def convert_motion_as_line_mesh(motion, skeleton_type="smpl22", const_color=None):
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    motion = motion.detach().cpu()
    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    length = motion.shape[0]
    device = motion.device
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, chain[:-1]])
        e_points.append(motion[:, chain[1:]])
        # if const_color is not None:
        #     color_name = const_color
        # color_ = get_const_colors(color_name, partial_shape=(length, num_line), alpha=1.0).to(device)  # (L, 4, 4)
        # m_colors.append(color_[..., :3] * 255)  # (L, 4, 3)

    s_points = torch.cat(s_points, dim=1)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=1)
    # m_colors = torch.cat(m_colors, dim=1)

    vertices = []
    for f in range(length):
        # vertices_, faces, vertex_colors = create_skeleton_mesh(s_points[f], e_points[f], radius=0.02, color=m_colors[f])
        vertices_, faces, vertex_colors = create_skeleton_mesh(s_points[f], e_points[f], radius=0.05, color=None)
        vertices.append(vertices_)
    vertices = torch.stack(vertices, dim=0)
    return vertices, faces, vertex_colors


def create_skeleton_mesh(p1, p2, radius, color=None, resolution=4, return_merged=True):
    """
    Create mesh between p1 and p2.
    Args:
        p1 (torch.Tensor): (N, 3),
        p2 (torch.Tensor): (N, 3),
        radius (float): radius,
        color (torch.Tensor): (N, 3)
        resolution (int): number of vertices in one circle, denoted as Q
    Returns:
        vertices (torch.Tensor): (N * 2Q, 3), if return_merged is False (N, 2Q, 3)
        faces (torch.Tensor): (M', 3), if return_merged is False (N, M, 3)
        vertex_colors (torch.Tensor): (N * 2Q, 3), if return_merged is False (N, 2Q, 3)
    """
    N = p1.shape[0]

    # Calculate segment direction
    seg_dir = p2 - p1  # (N, 3)
    unit_seg_dir = seg_dir / seg_dir.norm(dim=-1, keepdim=True)  # (N, 3)

    # Compute an orthogonal vector
    x_vec = torch.tensor([1, 0, 0], device=p1.device).float().unsqueeze(0).repeat(N, 1)  # (N, 3)
    y_vec = torch.tensor([0, 1, 0], device=p1.device).float().unsqueeze(0).repeat(N, 1)
    ortho_vec = torch.cross(unit_seg_dir, x_vec, dim=-1)  # (N, 3)
    ortho_vec_ = torch.cross(unit_seg_dir, y_vec, dim=-1)  # (N, 3)  backup
    ortho_vec = torch.where(ortho_vec.norm(dim=-1, keepdim=True) > 1e-3, ortho_vec, ortho_vec_)

    # Get circle points on two ends
    unit_ortho_vec = ortho_vec / ortho_vec.norm(dim=-1, keepdim=True)  # (N, 3)
    theta = torch.linspace(0, 2 * np.pi, resolution, device=p1.device)
    rotation_matrix = axis_angle_to_matrix(unit_seg_dir[:, None] * theta[None, :, None])  # (N, Q, 3, 3)
    rotated_points = einsum(rotation_matrix, unit_ortho_vec, "n q i j, n i -> n q j") * radius  # (N, Q, 3)
    bottom_points = rotated_points + p1.unsqueeze(1)  # (N, Q, 3)
    top_points = rotated_points + p2.unsqueeze(1)  # (N, Q, 3)

    # Combine bottom and top points
    vertices = torch.cat([bottom_points, top_points], dim=1)  # (N, 2Q, 3)
    vertices = vertices.round(decimals=3)

    # Generate face
    indices = torch.arange(0, resolution, device=p1.device)
    bottom_indices = indices
    top_indices = indices + resolution

    # wis3d = make_vis3d(None, 'check-face', 'data/vis3d')
    # for j in range(len(vertices[0])):
    #     wis3d.add_point_cloud(vertices[0, j:j+1], name=f'v{j}')

    # outside face
    # face_bottom = torch.stack([bottom_indices[:-2], bottom_indices[1:-1], bottom_indices[-1].repeat(resolution - 2)], 1)
    face_bottom = torch.stack([bottom_indices[1:-2], bottom_indices[2:-1], bottom_indices[-1].repeat(resolution - 3)], 1)
    # face_top = torch.stack([top_indices[1:-1], top_indices[:-2], top_indices[-1].repeat(resolution - 2)], 1)
    face_top = torch.stack([top_indices[2:-1], top_indices[1:-2], top_indices[-1].repeat(resolution - 3)], 1)
    faces = torch.cat(
        [
            torch.stack([bottom_indices[1:], bottom_indices[:-1], top_indices[:-1]], 1),  # out face
            torch.stack([bottom_indices[1:], top_indices[:-1], top_indices[1:]], 1),  # out face
            face_bottom,
            face_top,
        ]
    )
    faces = faces.unsqueeze(0).repeat(p1.shape[0], 1, 1)  # (N, M, 3)

    # Assign colors
    if color is not None:
        vertex_colors = color.unsqueeze(1).repeat(1, resolution * 2, 1)

    if return_merged:
        # manully adjust face ids
        N, V = vertices.shape[:2]
        faces = faces + torch.arange(0, N, device=p1.device).unsqueeze(1).unsqueeze(1) * V
        faces = faces.reshape(-1, 3)
        vertices = vertices.reshape(-1, 3)
        if color is not None:
            vertex_colors = vertex_colors.reshape(-1, 3)

    if color is not None:
        return vertices, faces, vertex_colors
    else:
        return vertices, faces, None



def get_gradient_colors(scheme="red", num_points=120, alpha=1.0):
    """
    Return a list of colors that are gradient from start to end.
    """
    start_rgba = torch.tensor(color_schemes[scheme][0] + [255 * alpha]) / 255
    end_rgba = torch.tensor(color_schemes[scheme][1] + [255 * alpha]) / 255
    colors = torch.stack([torch.linspace(s, e, steps=num_points) for s, e in zip(start_rgba, end_rgba)], dim=-1)
    return colors


def get_const_colors(name="red", partial_shape=(120, 5), alpha=1.0):
    """
    Return colors (partial_shape, 4)
    """
    rgba = torch.tensor(color_schemes[name][1] + [255 * alpha]) / 255
    partial_shape = tuple(partial_shape)
    colors = rgba[None].repeat(*partial_shape, 1)
    return colors


def get_colors_by_conf(conf, low="red", high="green"):
    colors = torch.stack([conf] * 3, dim=-1)
    colors = colors * torch.tensor(color_schemes[high][1]) + (1 - colors) * torch.tensor(color_schemes[low][1])
    return colors


# ================== Colored Motion Sequence ================== #


KINEMATIC_CHAINS = {
    "motive21": [
        [0, 1, 2, 3, 4], # body
        [2, 5, 6, 7, 8], # left arm
        [2, 9, 10, 11, 12], # right arm
        [0, 13, 14, 15, 16], # left leg
        [0, 17, 18, 19, 20], # right leg
    ],
    "smpl22": [
        [0, 2, 5, 8, 11],  # right-leg
        [0, 1, 4, 7, 10],  # left-leg
        [0, 3, 6, 9, 12, 15],  # spine
        [9, 14, 17, 19, 21],  # right-arm
        [9, 13, 16, 18, 20],  # left-arm
    ],
    "h36m17": [
        [0, 1, 2, 3],  # right-leg
        [0, 4, 5, 6],  # left-leg
        [0, 7, 8, 9, 10],  # spine
        [8, 14, 15, 16],  # right-arm
        [8, 11, 12, 13],  # left-arm
    ],
    "coco17": [
        [12, 14, 16],  # right-leg
        [11, 13, 15],  # left-leg
        [4, 2, 0, 1, 3],  # replace spine with head
        [6, 8, 10],  # right-arm
        [5, 7, 9],  # left-arm
    ],
}


def add_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22", const_color=None, offset=0):
    """
    Args:
        motion (tensor): (L, J, 3)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    motion = motion.detach().cpu()
    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    length = motion.shape[0]
    device = motion.device
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, chain[:-1]])
        e_points.append(motion[:, chain[1:]])
        if const_color is not None:
            color_name = const_color
        color_ = get_const_colors(color_name, partial_shape=(length, num_line), alpha=1.0).to(device)  # (L, 4, 4)
        m_colors.append(color_[..., :3] * 255)  # (L, 4, 3)

    s_points = torch.cat(s_points, dim=1)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=1)
    m_colors = torch.cat(m_colors, dim=1)

    for f in range(length):
        wis3d.set_scene_id(f + offset)

        # Add skeleton as cylinders
        vertices, faces, vertex_colors = create_skeleton_mesh(s_points[f], e_points[f], radius=0.02, color=m_colors[f])
        wis3d.add_mesh(vertices, faces, vertex_colors, name=name)

        # Old way to add lines, this may cause problems when the number of lines is large
        # wis3d.add_lines(s_points[f], e_points[f], m_colors[f], name=name)


def add_prog_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22"):
    """
    Args:
        motion (tensor): (P, L, J, 3)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    P, L, J, _ = motion.shape
    device = motion.device

    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, :, chain[:-1]])
        e_points.append(motion[:, :, chain[1:]])
        color_ = get_gradient_colors(color_name, L, alpha=1.0).to(device)  # (L, 4)
        color_ = color_[None, :, None, :].repeat(P, 1, num_line, 1)  # (P, L, num_line, 4)
        m_colors.append(color_[..., :3] * 255)  # (P, L, num_line, 3)
    s_points = torch.cat(s_points, dim=-2)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=-2)
    m_colors = torch.cat(m_colors, dim=-2)

    s_points = s_points.reshape(P, -1, 3)
    e_points = e_points.reshape(P, -1, 3)
    m_colors = m_colors.reshape(P, -1, 3)

    for p in range(P):
        wis3d.set_scene_id(p)
        wis3d.add_lines(s_points[p], e_points[p], m_colors[p], name=name)


def add_joints_motion_as_spheres(joints, wis3d, radius=0.05, name="joints", label_each_joint=False):
    """Visualize skeleton as spheres to explore the skeleton.
    Args:
        joints: (NF, NJ, 3)
        wis3d
        radius: radius of the spheres
        name
        label_each_joint: if True, each joints will have a label in wis3d (then you can interact with it, but it's slower)
    """
    colors = torch.zeros_like(joints).float()
    n_frames = joints.shape[0]
    n_joints = joints.shape[1]
    for i in range(n_joints):
        colors[:, i, 1] = 255 / n_joints * i
        colors[:, i, 2] = 255 / n_joints * (n_joints - i)
    for f in range(n_frames):
        wis3d.set_scene_id(f)
        if label_each_joint:
            for i in range(n_joints):
                wis3d.add_spheres(
                    joints[f, i].float(),
                    radius=radius,
                    colors=colors[f, i],
                    name=f"{name}-j{i}",
                )
        else:
            wis3d.add_spheres(
                joints[f].float(),
                radius=radius,
                colors=colors[f],
                name=f"{name}",
            )