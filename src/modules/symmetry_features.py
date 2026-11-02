"""Shared symmetry features with the original concatenation backbone."""
import hashlib
import math
import torch
from fusion_model import ModalFusion


class SymmetryFusion(ModalFusion):
    def __init__(self, arm):
        self.SG_DIM = 290 if arm == 'hybrid' else 237
        super().__init__('simple_concat')
        self.arm = arm

    def _split_inputs(self, inputs):
        assert inputs.shape[-1] == 183 + self.SG_DIM + 162
        end = 183 + self.SG_DIM
        return inputs[..., :183], inputs[..., 183:end], inputs[..., end:end + 160], inputs[..., -2:-1], inputs[..., -1:]


def initial_model(arm, common, seed, fold, device):
    init_seed = int.from_bytes(hashlib.sha256(f't35_symmetry|{seed}|{fold}'.encode()).digest()[:4], 'big') % (2**31 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = SymmetryFusion(arm)
    for name in ('composition_projection', 'structure_projection', 'fusion', 'aggregate_head', 'report_head'):
        getattr(model, name).load_state_dict(getattr(common, name).state_dict())
    if arm == 'identity_fallback':
        model.sg_projection.load_state_dict(common.sg_projection.state_dict())
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed + 1)
            physical_weights = torch.empty(32, 60).uniform_(-1 / math.sqrt(290), 1 / math.sqrt(290))
        physical_weights[:, 39:46].copy_(common.sg_projection.weight[:, 230:237].detach().cpu())
        with torch.no_grad():
            if arm == 'hybrid':
                model.sg_projection.weight[:, :230].copy_(common.sg_projection.weight[:, :230])
                model.sg_projection.weight[:, 230:].copy_(physical_weights)
            else:
                model.sg_projection.weight[:, :60].copy_(physical_weights)
            model.sg_projection.bias.copy_(common.sg_projection.bias)
    return model.to(device)


def transform(features, arm, table, seen):
    result = dict(features)
    old = features['sg_features']
    identities = old[:, :230]
    supported = features['sg_mask'][:, 0] > .5
    assert torch.equal(identities.sum(-1) > .5, supported)
    numbers = identities.argmax(-1)
    if arm == 'identity_fallback':
        sg = old.clone()
        keep = torch.tensor([int(n) + 1 in seen for n in numbers], dtype=torch.bool)
        sg[~keep, :230] = 0
    else:
        physical = table[numbers].clone()
        physical[~supported] = 0
        sg = (torch.cat((physical, torch.zeros(len(old), 177)), dim=-1) if arm == 'physical_shared'
              else torch.cat((identities, physical), dim=-1))
    result['sg_features'] = sg
    return result


class CommonFormulaResidual(torch.nn.Module):
    def __init__(self, formula_base, modal_base, basis):
        super().__init__()
        self.formula_base, self.modal_base = formula_base, modal_base
        for module in (formula_base, modal_base):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.register_buffer('basis', basis.detach().double().clone())
        self.location_weights = torch.nn.Parameter(torch.zeros(len(basis)))
        self.scale_weights = torch.nn.Parameter(torch.zeros(len(basis)))

    @torch.no_grad()
    def fixed_features(self, inputs):
        formula = inputs.clone()
        formula[..., 183:] = 0
        source_formula = torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 399)), dim=-1)
        hidden = self.formula_base.encode(source_formula)
        logits = self.formula_base.report_head(hidden)
        raw = self.formula_base.report_head.normalized_raw_score(source_formula[..., :183])
        mu = logits[..., 0] + self.formula_base.report_head.alpha * raw.to(logits.dtype)
        shape, width = inputs.shape[:-1], inputs.shape[-1]
        h = self.modal_base.encode(inputs.reshape(-1, width)).reshape(*shape, 64)
        h0 = self.modal_base.encode(formula.reshape(-1, width)).reshape(*shape, 64)
        coordinates = ((h - h0).double() @ self.basis.T).float()
        available = (inputs[..., -2:-1] + inputs[..., -1:] > .5).float()
        return mu, logits[..., 1], coordinates * available

    def residuals(self, features):
        return features[2] @ self.location_weights, features[2] @ self.scale_weights

    def report_from_features(self, features, bounds=(.05, 2.5)):
        dmu, dscale = self.residuals(features)
        low, high = bounds
        return features[0] + dmu, low + (high - low) * torch.sigmoid(features[1] + dscale)

    def report(self, inputs, bounds=(.05, 2.5)):
        return self.report_from_features(self.fixed_features(inputs), bounds)
