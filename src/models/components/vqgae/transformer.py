import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(2*max_len)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cluster_number: int,
        nhead: int = 8,
        layer_num: int = 4,
        **kwargs
    ):
        super().__init__()

        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.attn_encoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=2*hidden_dim, batch_first=True)
            for _ in range(layer_num)
        ])

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=in_dim)

    def forward(self, edge_mtx):
        node_feat = self.node_proj(edge_mtx)
        node_feat = self.pos_encoder(node_feat)
        latent = self.attn_encoder(node_feat)

        return latent
    

class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cluster_number: int,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, node_repr):
        reconstruct = torch.bmm(node_repr, node_repr.permute(0, 2, 1))

        return node_repr, reconstruct