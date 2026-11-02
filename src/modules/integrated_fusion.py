"""Direct trainable fusion and two-component report density; no frozen report span."""
import hashlib
import torch
from torch import nn
from fusion_model import ModalFusion
from model_raw import RawBypassReportHead
from mixture_model import component_parameters, nll_per_record


class IntegratedFusion(ModalFusion):
    SG_DIM = 290

    def __init__(self, architecture, beta, raw_mean, raw_std, mode='M'):
        super().__init__('cross_attention' if architecture == 'cross_attention' else 'simple_concat')
        self.arm = self.architecture = architecture
        self.mode = mode
        self.report_head = RawBypassReportHead(64, 5, beta, raw_mean, raw_std)
        if architecture == 'capacity_mlp':
            self.extra = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 64))
            self.extra_norm = nn.LayerNorm(64)

    @staticmethod
    def _split_inputs(inputs):
        assert inputs.shape[-1] == 635
        return inputs[..., :183], inputs[..., 183:473], inputs[..., 473:633], inputs[..., 633:634], inputs[..., 634:635]

    @staticmethod
    def formula_input(inputs):
        return torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 452)), dim=-1)

    def encode(self, inputs, return_attention=False):
        shape = inputs.shape[:-1]
        result = super().encode(inputs.reshape(-1, 635), return_attention=return_attention)
        hidden, attention = result if return_attention else (result, None)
        if self.architecture == 'capacity_mlp':
            hidden = self.activation(hidden + self.extra_norm(self.extra(hidden)))
        hidden = hidden.reshape(*shape, 64)
        if return_attention:
            return hidden, attention
        return hidden

    def encode_report(self, inputs, return_attention=False):
        return self.encode(self.formula_input(inputs) if self.mode == 'F' else inputs,
                           return_attention=return_attention)

    def raw_report(self, inputs):
        raw = self.report_head(self.encode_report(inputs)).double()
        shift = self.report_head.alpha.double() * self.report_head.normalized_raw_score(inputs[..., :183])
        return torch.cat((raw[..., :1], raw[..., 1:3] + shift[..., None], raw[..., 3:]), dim=-1)

    def components(self, inputs, bounds=(.05, 2.5)):
        return component_parameters(self.raw_report(inputs), 'mixture_lognormal', bounds)

    def per_record_nll(self, inputs, targets, bounds=(.05, 2.5)):
        return nll_per_record(self.raw_report(inputs), targets, 'mixture_lognormal', bounds)

    def aggregate(self, inputs):
        # Common four-view FIT replay in F/M/J; F restricts report inference only.
        return self.aggregate_head(self.encode(inputs)).squeeze(-1)


def matched_models(common, beta, raw_mean, raw_std, seed, fold, device):
    models = {}
    for architecture in ('simple_concat', 'capacity_mlp', 'cross_attention'):
        token = int.from_bytes(hashlib.sha256(f't41|{seed}|{fold}|{architecture}'.encode()).digest()[:4], 'big')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(token)
            model = IntegratedFusion(architecture, beta, raw_mean, raw_std)
        for name in ('composition_projection', 'sg_projection', 'structure_projection', 'fusion', 'aggregate_head'):
            getattr(model, name).load_state_dict(getattr(common, name).state_dict())
        if models:
            model.report_head.load_state_dict(models['simple_concat'].report_head.state_dict())
        models[architecture] = model.to(device)
    shared_names = ('composition_projection', 'sg_projection', 'structure_projection', 'fusion', 'aggregate_head')
    reference = models['simple_concat']
    assert all(torch.equal(v, getattr(model, name).state_dict()[key])
               for model in models.values() for name in shared_names
               for key, v in getattr(reference, name).state_dict().items())
    return models
