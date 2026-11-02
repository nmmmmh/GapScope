"""Small shared fusion model with separate aggregate and report outputs."""

import torch
from torch import nn


class DualTaskFusion(nn.Module):
    def __init__(self, input_dim: int = 582, width: int = 64):
        super().__init__()
        self.shared = nn.Linear(input_dim, width)
        self.activation = nn.ReLU()
        self.aggregate_head = nn.Linear(width, 1)
        self.report_head = nn.Linear(width, 2)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.shared(inputs))

    def aggregate(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.aggregate_head(self.encode(inputs)).squeeze(-1)

    def report(self, inputs: torch.Tensor, scale_bounds=(0.05, 2.5)):
        output = self.report_head(self.encode(inputs))
        log_mu = output[..., 0]
        low, high = scale_bounds
        sigma = low + (high - low) * torch.sigmoid(output[..., 1])
        return log_mu, sigma
