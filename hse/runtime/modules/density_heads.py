"""Report-density heads and losses for the T13 frozen-fusion comparison."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def output_count(family: str) -> int:
    if family == "single_lognormal":
        return 2
    if family == "mixture_lognormal":
        return 5
    raise ValueError(f"unknown_density_family:{family}")


def replace_report_head(model, family: str) -> None:
    """Replace the already-loaded T12 report head without changing its base."""
    model.report_head = torch.nn.Linear(model.WIDTH, output_count(family)).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )


def raw_report_output(model, inputs: torch.Tensor) -> torch.Tensor:
    return model.report_head(model.encode(inputs))


def component_parameters(raw: torch.Tensor, family: str, bounds=(0.05, 2.5)):
    low, high = (float(value) for value in bounds)
    if family == "single_lognormal":
        mus = raw[..., 0:1]
        raw_sigmas = raw[..., 1:2]
        weights = torch.ones_like(mus)
    elif family == "mixture_lognormal":
        logits = raw[..., 0]
        mus = raw[..., 1:3]
        raw_sigmas = raw[..., 3:5]
        first_weight = torch.sigmoid(logits)
        weights = torch.stack((first_weight, 1.0 - first_weight), dim=-1)
    else:
        raise ValueError(f"unknown_density_family:{family}")
    sigmas = low + (high - low) * torch.sigmoid(raw_sigmas)
    return weights, mus, sigmas


def nll_per_record(raw: torch.Tensor, targets: torch.Tensor, family: str,
                   bounds=(0.05, 2.5)) -> torch.Tensor:
    """Return positive-log-gap density NLL, including the log-target Jacobian."""
    weights, mus, sigmas = component_parameters(raw, family, bounds)
    log_y = torch.log(targets).to(dtype=raw.dtype).unsqueeze(-1)
    if family == "single_lognormal":
        log_weights = torch.zeros_like(weights)
    else:
        logits = raw[..., 0]
        log_weights = torch.stack((F.logsigmoid(logits), F.logsigmoid(-logits)), dim=-1)
    log_density = (
        log_weights - log_y - torch.log(sigmas) - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((log_y - mus) / sigmas) ** 2
    )
    return -torch.logsumexp(log_density, dim=-1)


def report_loss(model, views: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                family: str, bounds=(0.05, 2.5)) -> torch.Tensor:
    count = targets.numel()
    raw = raw_report_output(model, views.reshape(-1, views.shape[-1]))
    raw = raw.reshape(4, count, output_count(family))
    per_record = nll_per_record(raw, targets.unsqueeze(0), family, bounds).mean(dim=0)
    return torch.sum(weights * per_record)


def validation_loss(model, formula_view: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor, family: str, bounds=(0.05, 2.5)) -> float:
    raw = raw_report_output(model, formula_view)
    return float(torch.sum(weights * nll_per_record(raw, targets, family, bounds)).item())


def _raw_scale(scale: torch.Tensor, bounds, fraction_epsilon: float = 1e-6) -> torch.Tensor:
    low, high = (float(value) for value in bounds)
    clipped = scale.clamp(min=low + 1e-6, max=high - 1e-6)
    fraction = ((clipped - low) / (high - low)).clamp(
        fraction_epsilon, 1.0 - fraction_epsilon,
    )
    return torch.log(fraction / (1.0 - fraction))


def initialize_report_head(model, family: str, targets: torch.Tensor,
                           weights: torch.Tensor, bounds=(0.05, 2.5)) -> dict:
    """Initialize to the fold's weighted log-target moments with zero head weights."""
    low, high = (float(value) for value in bounds)
    log_y = torch.log(targets.to(dtype=torch.float64))
    mean = torch.sum(weights * log_y)
    variance = torch.sum(weights * (log_y - mean) ** 2)
    raw_scale = torch.sqrt(torch.clamp_min(variance, 0.0))
    scale = raw_scale.clamp(min=low + 1e-6, max=high - 1e-6)
    with torch.no_grad():
        model.report_head.weight.zero_()
        if family == "single_lognormal":
            model.report_head.bias[0] = mean.to(model.report_head.bias.dtype)
            model.report_head.bias[1] = _raw_scale(scale, bounds).to(model.report_head.bias.dtype)
            info = {
                "weighted_log_gap_mean": float(mean.item()),
                "weighted_log_gap_std_unclamped": float(raw_scale.item()),
                "weighted_log_gap_std": float(scale.item()),
                "raw_scale_bias": float(_raw_scale(scale, bounds).item()),
            }
        elif family == "mixture_lognormal":
            offset = min(
                0.5 * float(scale.item()),
                math.sqrt(max(float(scale.item()) ** 2 - (low + 1e-6) ** 2, 0.0)),
            )
            component_scale = math.sqrt(max(float(scale.item()) ** 2 - offset ** 2, 0.0))
            component_scale_tensor = torch.as_tensor(component_scale, dtype=scale.dtype,
                                                     device=scale.device)
            sigma_bias = _raw_scale(
                component_scale_tensor, bounds, fraction_epsilon=1e-12,
            ).to(model.report_head.bias.dtype)
            model.report_head.bias[0] = 0.0
            model.report_head.bias[1] = (mean - offset).to(model.report_head.bias.dtype)
            model.report_head.bias[2] = (mean + offset).to(model.report_head.bias.dtype)
            model.report_head.bias[3] = sigma_bias
            model.report_head.bias[4] = sigma_bias
            info = {
                "weighted_log_gap_mean": float(mean.item()),
                "weighted_log_gap_std_unclamped": float(raw_scale.item()),
                "weighted_log_gap_std": float(scale.item()),
                "initial_weight_1": 0.5,
                "initial_weight_2": 0.5,
                "initial_component_offset_log": offset,
                "initial_component_sigma": component_scale,
                "raw_scale_bias": float(sigma_bias.item()),
            }
        else:
            raise ValueError(f"unknown_density_family:{family}")
    return info
