from __future__ import annotations

import torch
from torch import nn

from src.rl.actor import mlp


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1], nn.ReLU)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)
