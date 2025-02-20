import torch
from torch import nn

from easyvolcap.engine import SUPERVISORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.loss_utils import mse
from reactmotion.utils.motion_repr_transform import *


@SUPERVISORS.register_module()
class VQVAEMoTokenSupervisor(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 x_recon_l2_weight: float = 0.,
                 x_recon_l1smooth_weight: float = 0.,
                 vel_weight: float = 0.,
                 q_latent_weight: float = 0.,
                 e_latent_weight: float = 0.,
                 ):
        super().__init__()
        self.network = network
        self.x_recon_l2_weight = x_recon_l2_weight
        self.x_recon_l1smooth_weight = x_recon_l1smooth_weight
        self.vel_weight = vel_weight
        self.q_latent_weight = q_latent_weight
        self.e_latent_weight = e_latent_weight

        self.forward = self.supervise

    def supervise(self, output: dotdict, batch: dotdict):

        loss = 0  # accumulated final loss
        scalar_stats = dotdict()

        if 'motion_mask' in batch:
            motion_mask = batch.motion_mask[:, :, None]
        else:
            motion_mask = torch.ones_like(batch.supervise_MoToken[:, :, 0:1])
        
        if self.x_recon_l2_weight > 0.:
            x_recon_l2_loss = mse(MoReprTrans.Normalize(output.netout_x, self.network.Xnorm) * motion_mask, 
                                  MoReprTrans.Normalize(batch.supervise_MoToken, self.network.Xnorm) * motion_mask)
            loss += self.x_recon_l2_weight * x_recon_l2_loss
            scalar_stats.x_recon_l2_loss = x_recon_l2_loss

        if self.x_recon_l1smooth_weight > 0.:
            l1smooth_loss = nn.SmoothL1Loss()
            x_recon_l1smooth_loss = l1smooth_loss(MoReprTrans.Normalize(output.netout_x, self.network.Xnorm) * motion_mask, 
                                                  MoReprTrans.Normalize(batch.supervise_MoToken, self.network.Xnorm) * motion_mask)
            loss += self.x_recon_l1smooth_weight * x_recon_l1smooth_loss
            scalar_stats.x_recon_l1smooth_loss = x_recon_l1smooth_loss
        
        if self.q_latent_weight > 0.:
            q_latent_loss = mse(output.z_q_x * motion_mask[:, ::U], output.z_e_x.detach() * motion_mask[:, ::U])
            loss += self.q_latent_weight * q_latent_loss
            scalar_stats.q_latent_loss = q_latent_loss

        if self.e_latent_weight > 0.:
            e_latent_loss = mse(output.z_q_x.detach() * motion_mask[:, ::U], output.z_e_x * motion_mask[:, ::U]) # commit loss
            loss += self.e_latent_weight * e_latent_loss
            scalar_stats.e_latent_loss = e_latent_loss

        if 'usage' in output.keys(): scalar_stats.usage = output.usage
        return loss, scalar_stats
