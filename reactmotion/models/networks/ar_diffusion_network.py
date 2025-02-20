from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from easyvolcap.engine import NETWORKS, call_from_cfg


@NETWORKS.register_module()
class ReactiveARDiffDecNetwork(nn.Module):
    def __init__(self, 
                 input_feats,
                 oppo_feats, 
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu", 
                 arch='trans_enc', 
                 cond_input_feats=None,
                 **kwargs):
        super().__init__()
        self.input_feats = input_feats
        if cond_input_feats is None:
            self.cond_input_feats = input_feats
        else:
            self.cond_input_feats = cond_input_feats
        self.oppo_feats = oppo_feats
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_latent = EmbedLatent(self.cond_input_feats, self.latent_dim)
        self.embed_oppo = EmbedLatent(self.oppo_feats, self.latent_dim)
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.output_process = OutputProcess(self.input_feats, self.latent_dim)

        self.mlps = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
        )

        self.arch = arch
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)

        
    def conditon_forward(self, cond_oppo, cond_latents, mask):
        latents = self.embed_latent(cond_latents)
        oppo = self.embed_oppo(cond_oppo)
        xseq = latents + oppo
        if self.arch == 'trans_enc':
            xseq = self.sequence_pos_encoder(xseq)  # [l, b, d]
            causal_mask = subsequent_mask(xseq.size(0))[0].to(xseq.device)
            mask = None if mask is None else (mask == 0)
            conditions = self.seqTransEncoder(xseq, mask=~causal_mask, src_key_padding_mask=mask)  # [l, b, d]
            return rearrange(conditions, 't b c -> b t c')
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, conditions, **kwargs):
        """
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)
        conditions = conditions + emb
        x = torch.cat((x, conditions), dim=-1)
        x = self.mlps(x)
        output = self.output_process(x)
        return output
    


@NETWORKS.register_module()
class SparseControlARDiffDecNetwork(ReactiveARDiffDecNetwork):
    def __init__(self, 
                 ctrl_feats: int,
                 **kwargs):
        call_from_cfg(super().__init__, kwargs)
        self.embed_ctrl = EmbedLatent(ctrl_feats, self.latent_dim)
        
        
    def conditon_forward(self, cond_oppo, cond_latents, ctrl_info, mask):
        latents = self.embed_latent(cond_latents)
        oppo = self.embed_oppo(cond_oppo)
        ctrl = self.embed_ctrl(ctrl_info)
        xseq = latents + oppo + ctrl
        if self.arch == 'trans_enc':
            xseq = self.sequence_pos_encoder(xseq)  # [l, b, d]
            causal_mask = subsequent_mask(xseq.size(0))[0].to(xseq.device)
            mask = None if mask is None else (mask == 0)
            conditions = self.seqTransEncoder(xseq, mask=~causal_mask, src_key_padding_mask=mask)  # [l, b, d]
            return rearrange(conditions, 't b c -> b t c')
        else:
            raise NotImplementedError


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        emb = self.time_embed(self.sequence_pos_encoder.pe[timesteps])
        return emb.squeeze(2)


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    # Matrix containing 1s in its upper triangle, excluding the diagonal. The
    # rest of the matrix contains 0s.
    subsequent_mask = torch.triu(torch.ones(attn_shape),
                                 diagonal=1).type(torch.uint8)
    # Matrix with True in its diagonal and lower triangle, and False in
    # its upper triangle.
    return subsequent_mask == 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module   


def modulate(x, shift, scale):
    assert len(x.shape) == len(shift.shape) == len(scale.shape)
    return x * (1 + scale) + shift


class EmbedLatent(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = rearrange(x, 'b t c -> t b c')
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)
        return output