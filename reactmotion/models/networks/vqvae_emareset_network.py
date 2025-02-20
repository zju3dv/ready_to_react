import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from easyvolcap.engine import NETWORKS, cfg, call_from_cfg
from easyvolcap.utils.data_utils import dotdict
from reactmotion.utils.engine_utils import ENCODERS, DECODERS
from reactmotion.utils.motion_repr_transform import MoReprTrans


@NETWORKS.register_module()
class VQVAEEMAResetMoTokenNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 dim: int,
                 K: int = 512, # num_embeding
                 ):
        
        super().__init__()
        
        self.quantizer = QuantizeEMAReset(K, dim)
        self.encoder = None
        self.decoder = None

        input_norm = np.load(cfg.norm_cfg.motoken_input_norm_file)
        for i in range(input_norm.shape[1]):
            if input_norm[1, i] < 0.0001: input_norm[1, i] = 1
            
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)

    def preprocess(self, x):
        x = MoReprTrans.Normalize(x, self.Xnorm)
        return x

    def postprocess(self, x):
        x = MoReprTrans.Renormalize(x, self.Xnorm)
        return x

    def encode(self, x):
        x = self.preprocess(x)
        z_e_x = self.encoder(x)
        latents = self.quantizer.quantize(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.quantizer.dequantize(latents)
        x_tilde = self.decoder(z_q_x)
        x_tilde = self.postprocess(x_tilde)
        return x_tilde
    
    def forward(self, x, iter):
        x = self.preprocess(x)
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, perplexity, usage = self.quantizer(z_e_x, iter)
        x_tilde = self.decoder(z_q_x_st)
        x_tilde = self.postprocess(x_tilde)
        return x_tilde, z_e_x, z_q_x, perplexity, usage


@NETWORKS.register_module()
class VQVAEEMAResetT2MGPTMoTokenNetwork(VQVAEEMAResetMoTokenNetwork):
    def __init__(self,
                 input_dim: int,
                 dim: int,
                 encoder_cfg: dotdict = dotdict(),
                 decoder_cfg: dotdict = dotdict(),
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, input_dim=input_dim, dim=dim)

        self.encoder = ENCODERS.build(encoder_cfg, input_emb_width=input_dim, output_emb_width=dim)
        self.decoder = DECODERS.build(decoder_cfg, input_emb_width=input_dim, output_emb_width=dim)

    def encode(self, x):
        # encode
        x = self.preprocess(x)
        z_e_x = self.encoder(x)
        # quantize
        B, T, C = z_e_x.shape
        latents = self.quantizer.quantize(z_e_x.view(-1, C))
        latents = latents.view(B, T)
        return latents
    
    def encode2latent(self, x):
        # encode
        x = self.preprocess(x)
        z_e_x = self.encoder(x)
        return z_e_x


    def latent2decode(self, z_q_x):
        # decode
        x_tilde = self.decoder(z_q_x)
        x_tilde = self.postprocess(x_tilde)
        return x_tilde
    

    def decode(self, latents):
        # dequantize
        z_q_x = self.quantizer.dequantize(latents) # (B, T, C)
        # decode
        x_tilde = self.decoder(z_q_x)
        x_tilde = self.postprocess(x_tilde)
        return x_tilde

    def forward(self, x, iter):
        # encode
        x = self.preprocess(x)
        z_e_x = self.encoder(x)
        # forward quantize
        B, T, C = z_e_x.shape
        z_q_x_st, z_q_x, perplexity, usage = self.quantizer(z_e_x.view(-1, C), iter)
        # decode
        x_tilde = self.decoder(z_q_x_st.view(B, T, C))
        x_tilde = self.postprocess(x_tilde)
        return x_tilde, z_e_x, z_q_x.view(B, T, C), perplexity, usage
    

class QuantizeEMAReset(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.nb_code = K
        self.code_dim = dim
        self.mu = 0.99
        self.reset_codebook()
        
    def reset_codebook(self):
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code).cuda()

    def init_codebook(self, x):
        assert len(x) > self.nb_code
        self.codebook = x[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code).cuda()
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        # calculate perplexity and usage
        code_count = code_onehot.sum(dim=-1)  # nb_code
        usage = (code_count.view(self.nb_code, 1) >= 1.0).float()
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity, torch.sum(usage)
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        # update code sum
        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        # update code count
        code_count = code_onehot.sum(dim=-1)  # nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code
        # update codebook
        code_rand = x[:self.nb_code]
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * code_rand
        # compute perplexity
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity, torch.sum(usage)

    def quantize(self, z_e_x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(z_e_x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(z_e_x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, z_e_x, iter):
        # Init codebook if not inited
        if self.training and iter == 0: self.init_codebook(z_e_x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(z_e_x)
        z_q_x = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity, usage = self.update_codebook(z_e_x, code_idx)
        else:
            perplexity, usage = self.compute_perplexity(code_idx)
        
        z_q_x_bar = z_e_x + (z_q_x - z_e_x).detach()
        return z_q_x_bar, z_q_x, perplexity, usage