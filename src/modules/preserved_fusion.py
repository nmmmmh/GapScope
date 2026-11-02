"""Preserve both learned formula densities and train an interaction correction."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PreservedFusion(nn.Module):
    def __init__(self, branches, interaction, mode='M'):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.interaction = interaction
        self.interaction.report_head = nn.Identity()
        self.correction = nn.Linear(64, 5)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.mode = mode
        for branch in self.branches:
            branch.requires_grad_(False)
            for name in ('composition_projection', 'fusion', 'report_head'):
                getattr(branch.source, name).requires_grad_(True)
        self.interaction.requires_grad_(True)
        self.interaction.aggregate_head.requires_grad_(False)

    @staticmethod
    def formula_input(inputs):
        return torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 452)), -1)

    def density_parameters(self, inputs, bounds=(.05, 2.5)):
        selected = self.formula_input(inputs) if self.mode == 'F' else inputs
        delta = self.correction(self.interaction.encode(selected))
        mus, sigmas = [], []
        for index, branch in enumerate(self.branches):
            formula = branch.formula_input(inputs)
            source = branch.source
            raw = source.report_head(source.encode(formula))
            raw_z = source.report_head.normalized_raw_score(formula[..., :183])
            mu = raw[..., 0] + source.report_head.alpha * raw_z.to(raw.dtype)
            mus.append((mu + delta[..., 1 + index]).double())
            sigmas.append((bounds[0] + (bounds[1] - bounds[0]) *
                           torch.sigmoid(raw[..., 1] + delta[..., 3 + index])).double())
        return delta[..., 0].double(), torch.stack(mus, -1), torch.stack(sigmas, -1)

    def components(self, inputs, bounds=(.05, 2.5)):
        gate, mus, sigmas = self.density_parameters(inputs, bounds)
        first = torch.sigmoid(gate)
        return (torch.stack((first, 1 - first), -1),
                mus, sigmas)

    def per_record_nll(self, inputs, targets, bounds=(.05, 2.5)):
        gate, mus, sigmas = self.density_parameters(inputs, bounds)
        log_y = targets.double().log().unsqueeze(-1)
        # The correction's gate has stable log weights even at saturated logits.
        log_weights = torch.stack((F.logsigmoid(gate), F.logsigmoid(-gate)), -1)
        density = (log_weights - log_y - sigmas.log() - .5 * math.log(2 * math.pi)
                   - .5 * ((log_y - mus) / sigmas).square())
        return -torch.logsumexp(density, -1)

    def replay_loss(self, aggregate_views, standardized_targets, fit_indices):
        views = aggregate_views[:, fit_indices]
        target = standardized_targets[fit_indices]
        terms = [(branch.aggregate(views[0]).double() - target).square().mean()
                 for branch in self.branches]
        terms.append((self.interaction.aggregate(views).double() - target).square().mean())
        return torch.stack(terms).mean()
