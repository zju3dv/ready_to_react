
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easyvolcap.engine import NETWORKS, call_from_cfg


@NETWORKS.register_module()
class ReactiveARDecoder(nn.Module):
    def __init__(self, 
                 pose_feats: int,
                 root_feats: int,
                 latents_feats: int, 
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 arch='trans_enc',
                 **kwargs):
        super().__init__()
        self.pose_feats = pose_feats
        self.root_feats = root_feats
        self.latents_feats = latents_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.embed_pose = InputProcess(self.pose_feats, self.latent_dim)
        self.embed_root2oppo = InputProcess(self.root_feats, self.latent_dim)
        self.embed_latents = InputProcess(self.latents_feats, self.latent_dim)
        self.pose_head = OutputProcess(self.pose_feats, self.latent_dim)
        self.root2oppo_head = OutputProcess(self.root_feats, self.latent_dim)

        self.arch = arch
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)



    def forward(self, past_pose_series, past_root_info, past_latents, future_latents):       
        pose_emb = self.embed_pose(past_pose_series)  # [seqlen, bs, d]
        root_emb = self.embed_root2oppo(past_root_info)
        past_latents_emb = self.embed_latents(past_latents)
        future_latents_emb = self.embed_latents(future_latents)
        
        x = torch.cat((pose_emb, root_emb, past_latents_emb, future_latents_emb), axis=0)
        x = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(x)  # [seqlen, bs, d]

        n_pose = pose_emb.shape[0]
        n_root = root_emb.shape[0]
        pred_pose = self.pose_head(output[0:n_pose])
        pred_root = self.root2oppo_head(output[n_pose:n_pose+n_root])
        return pred_pose, pred_root


@NETWORKS.register_module()
class SparseControlARDecoder(ReactiveARDecoder):
    def __init__(self, 
                 ctrl_feats: int, 
                 **kwargs):
        call_from_cfg(super().__init__, kwargs)
        self.embed_ctrl = InputProcess(ctrl_feats, self.latent_dim)


    def forward(self, past_pose_series, past_root_info, past_latents, future_latents, ctrl_info):
        pose_emb = self.embed_pose(past_pose_series)  # [seqlen, bs, d]
        root_emb = self.embed_root2oppo(past_root_info)
        ctrl_emb = self.embed_ctrl(ctrl_info)
        past_latents_emb = self.embed_latents(past_latents)
        future_latents_emb = self.embed_latents(future_latents)
        
        x = torch.cat((pose_emb, root_emb, ctrl_emb, past_latents_emb, future_latents_emb), axis=0)
        x = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(x)  # [seqlen, bs, d]

        n_pose = pose_emb.shape[0]
        n_root = root_emb.shape[0]
        pred_pose = self.pose_head(output[0:n_pose])
        pred_root = self.root2oppo_head(output[n_pose:n_pose+n_root])
        return pred_pose, pred_root


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = rearrange(x, 'b t c -> t b c')
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
        return rearrange(output, 't b c -> b t c')
    
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