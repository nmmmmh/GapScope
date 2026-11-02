"""Report residual adaptation with a complete, warm-preserving coordinate basis."""
import torch
from t35_model import CommonFormulaResidual


def complete_checkpoint(checkpoint):
    basis = checkpoint['basis'].double()
    rank = len(basis)
    assert basis.shape == (rank, 64)
    assert torch.allclose(basis @ basis.T, torch.eye(rank, dtype=torch.float64), atol=1e-10, rtol=0)
    if rank < 64:
        _, _, vh = torch.linalg.svd(basis, full_matrices=True)
        basis = torch.cat((basis, vh[rank:]), dim=0)
    assert torch.allclose(basis @ basis.T, torch.eye(64, dtype=torch.float64), atol=1e-10, rtol=0)
    state = dict(checkpoint['state_dict'])
    state['basis'] = basis
    for key in ('location_weights', 'scale_weights'):
        state[key] = torch.cat((state[key], state[key].new_zeros(64 - rank)))
    return {**checkpoint, 'state_dict': state, 'basis': basis, 'rank': 64, 'old_rank': rank}


class Adaptation(CommonFormulaResidual):
    def __init__(self, formula_base, modal_base, basis, arm):
        super().__init__(formula_base, modal_base, basis)
        self.arm = arm
        names = []
        if arm in ('structure_projection', 'structure_and_fusion'):
            names.append('structure_projection')
        if arm in ('fusion', 'structure_and_fusion'):
            names.append('fusion')
        self.shared_names = names
        for name in names:
            for parameter in getattr(self.modal_base, name).parameters():
                parameter.requires_grad_(True)

    def shared_parameters(self):
        return [p for name in self.shared_names for p in getattr(self.modal_base, name).parameters()]

    def fixed_features(self, inputs):
        with torch.no_grad():
            source = torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 399)), dim=-1)
            hidden = self.formula_base.encode(source)
            logits = self.formula_base.report_head(hidden)
            raw = self.formula_base.report_head.normalized_raw_score(source[..., :183])
            mu = logits[..., 0] + self.formula_base.report_head.alpha * raw.to(logits.dtype)
        absent = inputs.clone()
        absent[..., 183:] = 0
        shape, width = inputs.shape[:-1], inputs.shape[-1]
        h = self.modal_base.encode(inputs.reshape(-1, width)).reshape(*shape, 64)
        h0 = self.modal_base.encode(absent.reshape(-1, width)).reshape(*shape, 64)
        coordinates = ((h - h0).double() @ self.basis.T).float()
        available = (inputs[..., -2:-1] + inputs[..., -1:] > .5).float()
        return mu, logits[..., 1], coordinates * available
