"""Small matched-token fusion models for the T12 modality interaction study."""

from __future__ import annotations

import hashlib

import torch
from torch import nn


class ModalFusion(nn.Module):
    COMPOSITION_DIM = 183
    SG_DIM = 237
    STRUCTURE_DIM = 160
    TOKEN_DIM = 32
    WIDTH = 64

    def __init__(self, arm: str):
        super().__init__()
        if arm not in {"simple_concat", "self_attention", "cross_attention"}:
            raise ValueError(f"unknown_fusion_arm:{arm}")
        self.arm = arm
        self.composition_projection = nn.Linear(self.COMPOSITION_DIM, self.TOKEN_DIM)
        self.sg_projection = nn.Linear(self.SG_DIM, self.TOKEN_DIM)
        self.structure_projection = nn.Linear(self.STRUCTURE_DIM, self.TOKEN_DIM)
        if arm == "simple_concat":
            self.attention = None
            self.attention_norm = None
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.TOKEN_DIM, num_heads=4, dropout=0.0, batch_first=True)
            self.attention_norm = nn.LayerNorm(self.TOKEN_DIM)
        self.fusion = nn.Linear(3 * self.TOKEN_DIM + 2, self.WIDTH)
        self.activation = nn.ReLU()
        self.aggregate_head = nn.Linear(self.WIDTH, 1)
        self.report_head = nn.Linear(self.WIDTH, 2)

    @staticmethod
    def _split_inputs(inputs: torch.Tensor):
        if inputs.shape[-1] != 582:
            raise ValueError(f"expected_t12_input_width_582:{inputs.shape[-1]}")
        composition = inputs[..., :183]
        sg = inputs[..., 183:420]
        structure = inputs[..., 420:580]
        sg_mask = inputs[..., 580:581]
        structure_mask = inputs[..., 581:582]
        return composition, sg, structure, sg_mask, structure_mask

    def encode(self, inputs: torch.Tensor, return_attention: bool = False):
        composition, sg, structure, sg_mask, structure_mask = self._split_inputs(inputs)
        composition_token = self.activation(self.composition_projection(composition))
        sg_token = self.activation(self.sg_projection(sg)) * sg_mask
        structure_token = self.activation(self.structure_projection(structure)) * structure_mask
        tokens = torch.stack((composition_token, sg_token, structure_token), dim=-2)
        attention_weights = None

        if self.arm == "self_attention":
            key_padding_mask = torch.cat(
                (torch.ones_like(sg_mask, dtype=torch.bool), sg_mask > 0.5,
                 structure_mask > 0.5), dim=-1)
            update, attention_weights = self.attention(
                tokens, tokens, tokens, key_padding_mask=~key_padding_mask,
                need_weights=return_attention, average_attn_weights=False)
            tokens = self.attention_norm(tokens + update)
            tokens = torch.stack((
                tokens[..., 0, :],
                tokens[..., 1, :] * sg_mask,
                tokens[..., 2, :] * structure_mask,
            ), dim=-2)
        elif self.arm == "cross_attention":
            key_padding_mask = torch.cat(
                (torch.ones_like(sg_mask, dtype=torch.bool), sg_mask > 0.5,
                 structure_mask > 0.5), dim=-1)
            update, attention_weights = self.attention(
                tokens[..., 0:1, :], tokens, tokens,
                key_padding_mask=~key_padding_mask, need_weights=return_attention,
                average_attn_weights=False)
            formula_token = self.attention_norm(tokens[..., 0, :] + update[..., 0, :])
            tokens = torch.stack((formula_token, sg_token, structure_token), dim=-2)

        fused_input = torch.cat((tokens.flatten(start_dim=-2), sg_mask, structure_mask), dim=-1)
        encoded = self.activation(self.fusion(fused_input))
        if return_attention:
            return encoded, attention_weights
        return encoded

    def aggregate(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.aggregate_head(self.encode(inputs)).squeeze(-1)

    def report(self, inputs: torch.Tensor, scale_bounds=(0.05, 2.5)):
        output = self.report_head(self.encode(inputs))
        log_mu = output[..., 0]
        low, high = scale_bounds
        sigma = low + (high - low) * torch.sigmoid(output[..., 1])
        return log_mu, sigma

    def copy_common_from(self, other: "ModalFusion") -> None:
        for name in ("composition_projection", "sg_projection", "structure_projection",
                     "fusion", "aggregate_head", "report_head"):
            getattr(self, name).load_state_dict(getattr(other, name).state_dict())

    def copy_interaction_from(self, other: "ModalFusion") -> None:
        if self.attention is None or self.attention_norm is None:
            raise ValueError("simple_concat_has_no_interaction_parameters")
        if other.attention is None or other.attention_norm is None:
            raise ValueError("source_model_has_no_interaction_parameters")
        self.attention.load_state_dict(other.attention.state_dict())
        self.attention_norm.load_state_dict(other.attention_norm.state_dict())


def matched_initial_models(seed: int, fold_id: int, device: torch.device | str = "cpu"):
    """Build one matched common initialization and a separate shared attention init."""
    fold_number = int.from_bytes(hashlib.sha256(str(fold_id).encode("utf-8")).digest()[:4], "big")
    common_seed = (int(seed) * 1_000_003 + fold_number * 97_409 + 11) % (2**31 - 1)
    interaction_seed = (int(seed) * 1_000_033 + fold_number * 65_537 + 29) % (2**31 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(common_seed)
        common = ModalFusion("simple_concat")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(common_seed)
        self_attention = ModalFusion("self_attention")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(common_seed)
        cross_attention = ModalFusion("cross_attention")
    self_attention.copy_common_from(common)
    cross_attention.copy_common_from(common)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(interaction_seed)
        interaction = nn.MultiheadAttention(
            embed_dim=ModalFusion.TOKEN_DIM, num_heads=4, dropout=0.0, batch_first=True)
    interaction_state = {key: value.detach().clone() for key, value in interaction.state_dict().items()}
    norm_state = {"weight": torch.ones(ModalFusion.TOKEN_DIM),
                  "bias": torch.zeros(ModalFusion.TOKEN_DIM)}
    self_attention.attention.load_state_dict(interaction_state)
    cross_attention.attention.load_state_dict(interaction_state)
    self_attention.attention_norm.load_state_dict(norm_state)
    cross_attention.attention_norm.load_state_dict(norm_state)
    return {
        "simple_concat": common.to(device),
        "self_attention": self_attention.to(device),
        "cross_attention": cross_attention.to(device),
    }
