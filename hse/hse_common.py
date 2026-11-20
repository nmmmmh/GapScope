import sys, json
from pathlib import Path
H = Path('/data/czx/GapScope_Revision_20261113/hse')
SRC = Path('/home/czx/2024/BandgapFusion_MeanTargetControl_20261023/source')
CONSO = Path('/home/czx/2024/BandgapFusion_Consolidated')
STAGE1 = Path('/home/czx/2024/BandgapFusion_Stage1_20261023')
STAGE2 = Path('/home/czx/2024/BandgapFusion_Stage2_20261023/training')
INPUTS = Path('/home/czx/2024/BandgapFusion_ExperimentalAugmentationWork/inputs')
sys.path.insert(0, str(H/'runtime'))
sys.dont_write_bytecode = True
import numpy as np
import runtime as rt
torch = rt.torch
SEEDS = (17, 29, 41)

def external_inputs(device):
    """Expanded 183 formula inputs (raw composition) and the 32 independent external records (raw 4-view inputs)."""
    from inference import _input_features
    f = torch.load(STAGE1/'challenge/features.pt', map_location='cpu', weights_only=True)
    cohort = rt.read_json(CONSO/'data/confirmation/cohort.json')['records']
    enc = rt.read_json(CONSO/'data/confirmation/encoded_features.json')
    ids = {r['measurement_id'] for r in cohort if r['cohort'] == 'independent'}
    feats = [r for r in enc if r['measurement_id'] in ids]
    raw = [_input_features(r) for r in feats]
    inputs = {k: torch.cat([r[k] for r in raw]) for k in raw[0]}
    route = []
    for r in feats:
        sg, st = r.get('space_group') is not None, r.get('structure') is not None
        route.append('full' if sg and st else 'formula_sg' if sg else 'formula_structure' if st else 'formula_only')
    return dict(exp_keys=f['keys'], exp_comp=f['composition'], ext_inputs=inputs, ext_ids=[r['measurement_id'] for r in feats], ext_route=route)

def formula_view(comp, scalers, device):
    c = (comp.to(device=device, dtype=torch.float32) - scalers['composition_mean'].to(device)) / scalers['composition_std'].to(device)
    return torch.cat((c, c.new_zeros((len(c), 452))), 1)

def predict_all(model, d, ext, m, device):
    """Components for core outer (4 views), expanded 183 (formula only), external 32 (4 views)."""
    sc = d['scalers']; out = {}
    model.eval()
    def arr(parts): return [p.detach().cpu().numpy().astype(np.float64) for p in parts]
    with torch.no_grad():
        for v in range(4):
            out[f'core{v}'] = np.stack(arr(model.components(d['views'][v, d['oi']])))
        out['core_idx'] = d['oi'].cpu().numpy()
        out['exp'] = np.stack(arr(model.components(formula_view(ext['exp_comp'], sc, device))))
        ev = m['t10'].four_views(ext['ext_inputs'], sc['composition_mean'].to(device=device, dtype=torch.float32),
            sc['composition_std'].to(device=device, dtype=torch.float32), sc['structure_mean'].to(device=device, dtype=torch.float32),
            sc['structure_std'].to(device=device, dtype=torch.float32), device)
        for v in range(4):
            out[f'ext{v}'] = np.stack(arr(model.components(ev[v])))
    return out

def save_pred(path, out):
    np.savez_compressed(path, **out)
