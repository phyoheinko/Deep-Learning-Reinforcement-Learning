from torch import nn


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)

