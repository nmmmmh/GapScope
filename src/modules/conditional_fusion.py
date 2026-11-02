"""Learn a conditional correction around a preserved, trainable formula density."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CenteredConditionalFusion(nn.Module):
    def __init__(self, base, interaction, centered=True):
        super().__init__()
        self.base = base
        self.base.mode = 'F'
        self.interaction = interaction
        self.interaction.report_head = nn.Identity()
        self.correction = nn.Linear(64, 5, bias=False)
        nn.init.zeros_(self.correction.weight)
        self.centered = centered
        self.interaction.requires_grad_(True)
        self.interaction.aggregate_head.requires_grad_(False)

    @staticmethod
    def formula_input(inputs):
        return torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 452)), -1)

    def conditional_delta(self, inputs):
        hidden = self.interaction.encode(inputs)
        if self.centered:
            hidden = hidden - self.interaction.encode(self.formula_input(inputs))
        return self.correction(hidden)

    def density_parameters(self, inputs, bounds=(.05, 2.5)):
        formula = self.formula_input(inputs)
        base_delta = self.base.correction(self.base.interaction.encode(formula))
        delta = self.conditional_delta(inputs)
        mus, sigmas = [], []
        for index, branch in enumerate(self.base.branches):
            source = branch.source
            branch_formula = branch.formula_input(inputs)
            raw = source.report_head(source.encode(branch_formula))
            raw_z = source.report_head.normalized_raw_score(branch_formula[..., :183])
            mu = raw[..., 0] + source.report_head.alpha * raw_z.to(raw.dtype)
            mus.append(((mu + base_delta[..., 1 + index]) + delta[..., 1 + index]).double())
            scale_raw = (raw[..., 1] + base_delta[..., 3 + index]) + delta[..., 3 + index]
            sigmas.append((bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(scale_raw)).double())
        return ((base_delta[..., 0] + delta[..., 0]).double(),
                torch.stack(mus, -1), torch.stack(sigmas, -1))

    def components(self, inputs, bounds=(.05, 2.5)):
        gate, mus, sigmas = self.density_parameters(inputs, bounds)
        first = torch.sigmoid(gate)
        return torch.stack((first, 1 - first), -1), mus, sigmas

    def per_record_nll(self, inputs, targets, bounds=(.05, 2.5)):
        gate, mus, sigmas = self.density_parameters(inputs, bounds)
        log_y = targets.double().log().unsqueeze(-1)
        log_weights = torch.stack((F.logsigmoid(gate), F.logsigmoid(-gate)), -1)
        density = (log_weights - log_y - sigmas.log() - .5 * math.log(2 * math.pi)
                   - .5 * ((log_y - mus) / sigmas).square())
        return -torch.logsumexp(density, -1)

    def replay_loss(self, aggregate_views, standardized_targets, fit_indices):
        views = aggregate_views[:, fit_indices]
        target = standardized_targets[fit_indices]
        base_replay = self.base.replay_loss(aggregate_views, standardized_targets, fit_indices)
        optional_replay = (self.interaction.aggregate(views).double() - target).square().mean()
        return (3 * base_replay + optional_replay) / 4
