# Default pipeline for volumetric videos
# This corresponds to the tranditional implementation's renderer
import torch
from torch import nn
from typing import Union
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_x
from easyvolcap.engine import MODELS, NETWORKS, SUPERVISORS
from reactmotion.utils.motion_repr_transform import *


@MODELS.register_module()
class MoTokenModel(nn.Module):
    def __init__(self,
                 network_cfg: dotdict,
                 supervisor_cfg: dotdict,

                 dtype: Union[str, torch.dtype] = torch.float,
                 ):
        super().__init__()

        self.network = NETWORKS.build(network_cfg)
        self.supervisor = SUPERVISORS.build(supervisor_cfg, network=self.network)
        self.inference = self.forward
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def extract_input(self, batch: dotdict):
        return batch.inputs, batch.iter

    def forward(self, batch: dotdict):
        # B, P, C
        input, iter = self.extract_input(batch) # iter for init_codebook
        input = to_x(input, self.dtype)

        x_tilde, z_e_x, z_q_x, perplexity, usage = self.network(input, iter)
        output = dotdict()
        output.netout_x = x_tilde
        output.z_e_x = z_e_x
        output.z_q_x = z_q_x
        output.perplexity = perplexity
        output.usage = usage

        # Loss computing part of the network
        if self.training:
            # Supervisor
            loss, scalar_stats = self.supervisor(output, batch)
            output.loss = loss
            output.scalar_stats = scalar_stats

        return output