from typing import Union, List, Dict
import numpy as np
import torch
from pytorch3d.transforms import *


def Lerp(a, b, t):
    '''
        t: weight of b
    '''
    return a + (b-a) * t


def Lerp_rotmat(a, b, t):
    qa = matrix_to_quaternion(a)
    qb = matrix_to_quaternion(b)
    q = qa + (qb - qa) * t
    return quaternion_to_matrix(q)


def to_numpy(batch, non_blocking=False, ignore_list: bool = False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch