"""Shared helpers: release Roost code (read-only, from the reproduction project), canonical key, inference."""
import sys, os, math
W = "/data/czx/GapScope_cv_remote/rw"
S = W + '/sources/extracted_members'
EMB = S + '/data/embeddings/matscholar-embedding.json'
OUT = "/data/czx/GapScope_cv_remote/cv"
for p in (S, W):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from roost.roost.data import parse_roost, CompositionData, collate_batch
from roost.roost.model import Roost
from roost.core import Normalizer
from torch.utils.data import DataLoader
from original_graph_adapter import enable_adapter
enable_adapter()

def canonical(formula):
    """project's key (run_original_comparison.preflight): sorted element:fraction, 12 decimals"""
    el, w = parse_roost(formula)
    m = {}
    for e, x in zip(el, w): m[e] = m.get(e, 0) + x
    el, w = list(m), list(m.values()); t = sum(w)
    return '|'.join(f'{e}:{x/t:.12f}' for e, x in sorted(zip(el, w)))

def predict(ckpt, csv, device='cuda'):
    """returns ids, mu (eV), ale (eV): ale = exp(log_std)*normalizer.std as in the release."""
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = Roost(**st['model_params'], device=device); model.to(device)
    model.load_state_dict(st['state_dict']); model.eval()
    nz = Normalizer(); nz.load_state_dict(st['normalizer'])
    data = CompositionData(csv, EMB, 'regression')
    ids, mu, ale = [], [], []
    with torch.no_grad():
        for batch, target, formulas, bid in DataLoader(data, batch_size=1024, shuffle=False, collate_fn=collate_batch):
            out = model(*[t.to(device) for t in batch]).cpu()
            m, ls = out.chunk(2, dim=1)
            mu += (m[:, 0] * nz.std + nz.mean).tolist()
            ale += (torch.exp(ls[:, 0]) * nz.std).tolist()
            ids += list(bid)
    return ids, mu, ale
