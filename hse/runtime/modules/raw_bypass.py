"""One-parameter frozen raw-formula location bypass for the T17 report head."""

from __future__ import annotations

import torch
from torch import nn


class RawBypassReportHead(nn.Linear):
    def __init__(self, in_features: int, out_features: int, beta: torch.Tensor,
                 feature_mean: float, feature_std: float):
        super().__init__(in_features, out_features)
        if beta.shape != (184,):
            raise ValueError(f"expected_raw_ols_beta_184:{tuple(beta.shape)}")
        if not torch.isfinite(beta).all() or not torch.isfinite(torch.tensor(feature_mean)):
            raise ValueError("nonfinite_raw_bypass_feature")
        if not torch.isfinite(torch.tensor(feature_std)) or feature_std <= 0:
            raise ValueError("invalid_raw_bypass_feature_std")
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer("raw_beta", beta.detach().to(dtype=torch.float64).clone())
        self.register_buffer("raw_feature_mean", torch.tensor(float(feature_mean), dtype=torch.float64))
        self.register_buffer("raw_feature_std", torch.tensor(float(feature_std), dtype=torch.float64))

    def normalized_raw_score(self, composition: torch.Tensor) -> torch.Tensor:
        if composition.shape[-1] != 183:
            raise ValueError(f"expected_raw_composition_width_183:{composition.shape[-1]}")
        score = composition.to(torch.float64) @ self.raw_beta[:183] + self.raw_beta[183]
        return (score - self.raw_feature_mean) / self.raw_feature_std


def make_model_class(base_class):
    """Create a T12-compatible class with the scalar bypass added to log-mu only."""

    class RawBypassModalFusion(base_class):
        def __init__(self, arm: str, beta: torch.Tensor,
                     feature_mean: float, feature_std: float):
            super().__init__(arm)
            original = self.report_head
            self.report_head = RawBypassReportHead(
                original.in_features, original.out_features, beta, feature_mean, feature_std)

        def report(self, inputs: torch.Tensor, scale_bounds=(0.05, 2.5)):
            output = self.report_head(self.encode(inputs))
            raw_z = self.report_head.normalized_raw_score(inputs[..., :183])
            log_mu = output[..., 0] + self.report_head.alpha * raw_z.to(output.dtype)
            low, high = scale_bounds
            sigma = low + (high - low) * torch.sigmoid(output[..., 1])
            return log_mu, sigma

    return RawBypassModalFusion
