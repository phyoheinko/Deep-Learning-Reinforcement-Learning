"""Q-network architecture shared by the DQN agents."""

import torch
from torch import nn


class DQN(nn.Module):
    """Two-hidden-layer MLP mapping observations to action-values Q(s, a)."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
