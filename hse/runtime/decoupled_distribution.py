"""Independent formula-marginal and phase-conditional experimental density heads."""
import math
import runtime as rt
from inference import _restore_model

torch=rt.torch

class DecoupledDistribution(torch.nn.Module):
    def __init__(self,modules,formula_state,conditional_state,device):
        super().__init__()
        self.formula=_restore_model(modules,formula_state,'F',device)
        self.conditional=_restore_model(modules,conditional_state,'M',device)
        self.requires_grad_(False)
        for branch in (self.formula,self.conditional):
            for p in branch.report_head.parameters():p.requires_grad_(True)

    def raw_report(self,inputs):
        available=inputs[...,633:].bool().any(-1,keepdim=True)
        return torch.where(available,self.conditional.raw_report(inputs),self.formula.raw_report(inputs))

    def components(self,inputs):
        from mixture_model import component_parameters
        return component_parameters(self.raw_report(inputs),'mixture_lognormal')

    def per_record_nll(self,inputs,targets):
        from mixture_model import nll_per_record
        return nll_per_record(self.raw_report(inputs),targets,'mixture_lognormal')

def restore(modules,state,device):
    model=DecoupledDistribution(modules,{k[8:]:v for k,v in state.items() if k.startswith('formula.')},
        {k[12:]:v for k,v in state.items() if k.startswith('conditional.')},device)
    model.load_state_dict(state);return model.eval()

def cdf(parts,grid):
    weights,mus,sigmas=parts
    z=(grid[None,:,None]-mus[:,None,:])/sigmas[:,None,:]
    return (weights[:,None,:]*.5*(1+torch.erf(z/math.sqrt(2)))).sum(-1)

def means(parts):
    w,mu,sd=parts;return (w*torch.exp(mu+.5*sd.square())).sum(-1)
