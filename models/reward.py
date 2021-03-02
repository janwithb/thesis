import torch
import torch.nn as nn
import torch.distributions as td


class RewardModel(nn.Module):
    """MLP for reward model."""
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self.output_dim)
        return td.independent.Independent(td.Normal(reshaped_inputs, 1), len(self.output_dim))
