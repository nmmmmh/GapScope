"""Rank-four conditional residuals around a frozen multimodal report model."""

from __future__ import annotations

import torch
from torch import nn

from inference import _restore_model


ADAPTERS = {
    "sg_projection": (290, 32, False),
    "structure_projection": (160, 32, False),
    "fusion": (98, 64, True),
}
RANK = 4


class ResidualLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = RANK, gate_with_masks: bool = False):
        super().__init__()
        if gate_with_masks and base.in_features != 98:
            raise ValueError("mask_gated_residual_requires_98_inputs")
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False,
                              device=base.weight.device, dtype=base.weight.dtype)
        self.up = nn.Linear(rank, base.out_features, bias=False,
                            device=base.weight.device, dtype=base.weight.dtype)
        nn.init.zeros_(self.up.weight)
        self.gate_with_masks = gate_with_masks

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.down(inputs))
        if self.gate_with_masks:
            available = inputs[..., -2:].gt(0.5).any(dim=-1, keepdim=True)
            residual = residual * available.to(dtype=residual.dtype)
        return self.base(inputs) + residual


class ConditionalAdapter(nn.Module):
    """Wrap a restored multimodal model while training only its low-rank paths."""

    def __init__(self, base: nn.Module):
        super().__init__()
        base.requires_grad_(False)
        for name, (in_width, out_width, gate) in ADAPTERS.items():
            linear = getattr(base, name)
            if (linear.in_features, linear.out_features) != (in_width, out_width):
                raise ValueError(f"unexpected_adapter_projection_shape:{name}")
            setattr(base, name, ResidualLinear(linear, gate_with_masks=gate))
        self.base = base
        for name in ADAPTERS:
            layer = getattr(self.base, name)
            layer.down.requires_grad_(True)
            layer.up.requires_grad_(True)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep all restored model layers and buffers in inference mode.
        self.base.eval()
        return self

    def raw_report(self, inputs: torch.Tensor):
        return self.base.raw_report(inputs)

    def components(self, inputs: torch.Tensor, bounds=(0.05, 2.5)):
        return self.base.components(inputs, bounds)

    def per_record_nll(self, inputs: torch.Tensor, targets: torch.Tensor,
                       bounds=(0.05, 2.5)):
        return self.base.per_record_nll(inputs, targets, bounds)


class FrozenHeadStudent(nn.Module):
    """Keep a formula model in eval mode while optimizing its report head."""

    def __init__(self, base: nn.Module):
        super().__init__()
        base.requires_grad_(False)
        base.report_head.requires_grad_(True)
        self.base = base
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def raw_report(self, inputs: torch.Tensor):
        return self.base.raw_report(inputs)

    def components(self, inputs: torch.Tensor, bounds=(0.05, 2.5)):
        return self.base.components(inputs, bounds)

    def per_record_nll(self, inputs: torch.Tensor, targets: torch.Tensor,
                       bounds=(0.05, 2.5)):
        return self.base.per_record_nll(inputs, targets, bounds)


def adapter_state(model: ConditionalAdapter):
    return {
        f"{name}.{part}.weight": getattr(getattr(model.base, name), part).weight.detach().cpu().clone()
        for name in ADAPTERS
        for part in ("down", "up")
    }


def load_adapter_state(model: ConditionalAdapter, state):
    expected = set(adapter_state(model))
    if set(state) != expected:
        raise ValueError("conditional_adapter_state_keys_mismatch")
    with torch.no_grad():
        for key, value in state.items():
            name, part, _ = key.split(".")
            parameter = getattr(getattr(model.base, name), part).weight
            if parameter.shape != value.shape:
                raise ValueError(f"conditional_adapter_state_shape_mismatch:{key}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return model


def restore_adapter(modules, parent_multiview_state, adapter_parameters, device):
    """Restore a frozen parent state, then apply a separately saved adapter state."""
    base = _restore_model(modules, parent_multiview_state, "M", device)
    model = ConditionalAdapter(base).to(device)
    load_adapter_state(model, adapter_parameters)
    return model.eval()


def original_state(model: ConditionalAdapter):
    """Return frozen parent parameters and buffers with their original key names."""
    state = {}
    prefixes = tuple(f"{name}." for name in ADAPTERS)
    for key, value in model.base.state_dict().items():
        if any(key.startswith(prefix + part + ".")
               for prefix in prefixes for part in ("down", "up")):
            continue
        for prefix in prefixes:
            if key.startswith(prefix + "base."):
                key = prefix + key[len(prefix + "base."):]
                break
        state[key] = value.detach().cpu().clone()
    return state


def state_matches(actual, expected):
    if set(actual) != set(expected):
        return False
    return all(torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu())
               for key in actual)
