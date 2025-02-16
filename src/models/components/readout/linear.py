import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.Softmax()
        )
    
    def forward(self, edge: torch.Tensor, **kwargs):
        return self.readout(edge)
    

class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.readout = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )
    
    def get_flattened_edge(self, edge_mtx: torch.Tensor):
        # Flatten the edge matrix
        b, n, _ = edge_mtx.shape
        mask = torch.tril(torch.ones(n, n), diagonal=-1).bool()
        edge = edge_mtx[:, mask].reshape(b, -1)

        return edge
    
    def forward(self, edge_mtx: torch.Tensor, **kwargs):
        edge = edge_mtx if len(edge_mtx.shape) == 2 else self.get_flattened_edge(edge_mtx)
        return self.readout(edge)