"""HSE pool: filter, dedupe, exclude core/expanded/external keys, encode with frozen Roost (batched, GPU)."""
import sys, json, hashlib, time
from pathlib import Path
H = Path('/data/czx/GapScope_Revision_20261113/hse')
C = Path('/home/czx/2024/BandgapFusion_Consolidated')
SRC = Path('/home/czx/2024/BandgapFusion_MeanTargetControl_20261023/source')
sys.path.insert(0, str(C/'runtime/roost_deps'))
sys.path.insert(0, str(C/'models/official/MatterVial-MP-gap'))
import numpy as np, pandas as pd, torch
from pymatgen.core import Composition
from mattervial.packages.roost.roost.model import Roost

def key(formula):
    a = Composition(formula).get_el_amt_dict(); t = sum(a.values())
    return '|'.join(f'{e}:{a[e]/t:.12f}' for e in sorted(a))

d = pd.read_csv('/home/czx/2024/spinels/data/mp_hse.csv')
counts = {'raw_rows': len(d)}
d = d[d.energy_above_hull <= 0.1].copy(); counts['ehull_le_0.1'] = len(d)
d['key'] = [key(f) for f in d.formula]
d = d.sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['key', 'spacegroup'], keep='first')
counts['unique_key_sg'] = len(d); counts['unique_keys_before_exclusion'] = d.key.nunique()
core = {r['composition_key'] for r in json.load(open(SRC/'data/reports/metadata.json'))}
exp = {l['composition_key'] for l in pd.read_csv('/home/czx/2024/BandgapFusion_Stage1_20261023/challenge_identities.csv').to_dict('records')}
ext = {r['composition_key'] for r in json.load(open(C/'data/confirmation/cohort.json'))['records'] if r['cohort']=='independent'}
counts.update(core_keys=len(core), expanded_keys=len(exp), external_keys=len(ext))
ex = core | exp | ext
counts['excluded_rows_hit'] = int(d.key.isin(ex).sum()); counts['excluded_keys_hit'] = int(d[d.key.isin(ex)].key.nunique())
d = d[~d.key.isin(ex)].reset_index(drop=True)
counts['final_rows'] = len(d); counts['final_keys'] = d.key.nunique(); counts['final_zero_gap_rows'] = int((d.target == 0).sum())
d = d[d.spacegroup.between(1, 230)].reset_index(drop=True); counts['final_rows_valid_sg'] = len(d)
dev = torch.device('cuda')
base = C/'models/official/MatterVial-MP-gap/mattervial/featurizers/custom_models'
ck = torch.load(base/'mpgap_checkpoint-r0.pth.tar', map_location='cpu', weights_only=False)
emb = json.loads((base/'matscholar-embedding.json').read_text())
enc = Roost(**ck['model_params'], device=dev).to(dev).eval(); enc.load_state_dict(ck['state_dict'])
nm = ck['normalizer_dict']['band_gap']
ukeys = d.drop_duplicates('key')[['key', 'formula']].reset_index(drop=True)
out = {}
def batch(formulas):
    w, f, si, ni, ci = [], [], [], [], []; off = 0; fr = []
    for b, formula in enumerate(formulas):
        comp = Composition(formula); a = comp.get_el_amt_dict(); el = list(a); t = sum(a.values()); n = len(el)
        w += [[a[e]/t] for e in el]; f += [emb[e] for e in el]
        si += [off + i for i in range(n) for _ in range(n)]; ni += [off + j for _ in range(n) for j in range(n)]
        ci += [b]*n; off += n
        v = [0.]*118
        for e in comp.elements: v[e.Z-1] = a[str(e)]/t
        fr.append(v)
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=dev)
    return (T(w, torch.float32), T(f, torch.float32), T(si, torch.long), T(ni, torch.long), T(ci, torch.long)), fr
t0 = time.time(); rows = []
with torch.no_grad():
    for s in range(0, len(ukeys), 1024):
        fm = list(ukeys.formula[s:s+1024]); inp, fr = batch(fm)
        hid = enc.material_nn(*inp)
        raw = list(enc(*inp))[0][:, 0]
        sc = (raw.double()*float(nm['std']) + float(nm['mean'])).cpu().numpy(); hid = hid.cpu().numpy()
        for i in range(len(fm)): rows.append(fr[i] + hid[i].tolist() + [float(sc[i])])
print('encoded', len(rows), time.time()-t0, flush=True)
comp_by_key = dict(zip(ukeys.key, rows))
comp = torch.tensor([comp_by_key[k] for k in d.key], dtype=torch.float32)
# verify against unbatched reference encoder on a sample
sys.path.insert(0, str(SRC/'runtime'))
import importlib.util
spec = importlib.util.spec_from_file_location('encode_ref', C/'runtime/encode.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng = np.random.default_rng(0); mx = 0
for i in rng.choice(len(d), 25, replace=False):
    ref = torch.tensor(m.encode_formula(d.formula[i], device='cpu')['composition'])
    mx = max(mx, float((ref - comp[i]).abs().max()))
counts['batched_vs_reference_encoder_max_abs_diff_25_samples'] = mx
sg = torch.zeros(len(d), 237)
for i, g in enumerate(d.spacegroup.astype(int)):
    sg[i, g-1] = 1
    s_ = 0 if g<=2 else 1 if g<=15 else 2 if g<=74 else 3 if g<=142 else 4 if g<=167 else 5 if g<=194 else 6
    sg[i, 230+s_] = 1
N = len(d)
torch.save(dict(composition=comp, sg_features=sg, structure=torch.zeros(N,160), sg_mask=torch.ones(N,1), structure_mask=torch.zeros(N,1),
                targets=torch.tensor(d.target.values, dtype=torch.float64), keys=list(d.key)), H/'hse_pool_features.pt')
d[['key','formula','spacegroup','energy_above_hull','target']].to_csv(H/'hse_pool.csv', index=False)
json.dump(counts, open(H/'hse_pool_counts.json', 'w'), indent=1); print(json.dumps(counts))
