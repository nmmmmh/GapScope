"""Score arms like the paper: core 205 OOF (3-seed equal mix, hierarchical weights), expanded 183 (15-model mix), external 32 (15-model mix)."""
import sys, json, math, argparse
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
from hse_common import *
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata
from training_data import hierarchical_weights
sys.path.insert(0, str(H / 'runtime'))
from density_score import score

p = argparse.ArgumentParser()
p.add_argument('--arms', default='H,H_MEAN,H_AUX,ARCH,ARCH_S2')
p.add_argument('--out', default=str(H / 'out'))
p.add_argument('--csv', default=str(H / 'metrics.csv'))
a = p.parse_args()
out = Path(a.out)
CAL = Path('/data/czx/GapScope_Revision_20261113/calib')

Y = np.linspace(1e-4, 16, 12801)
DY = Y[1] - Y[0]
LY = np.log(Y)
WIN = {'pv': (1.1, 1.7), 'visible': (1.8, 3.1)}


def cdf_grid(w, mu, sd):
    return (w[:, None] * ndtr((LY[None] - mu[:, None]) / sd[:, None])).sum(0)


def auc(lab, sc):
    lab = np.asarray(lab, bool)
    n = lab.sum()
    if n in (0, len(lab)):
        return np.nan
    return float((rankdata(sc)[lab].sum() - n * (n + 1) / 2) / (n * (len(lab) - n)))


def score_comp(G, ys):
    r = {}
    r['crps'] = np.mean([float(((G - (Y >= y)) ** 2).sum() * DY) for y in ys])
    q = lambda al: float(np.interp(al, G, Y))
    med, lo, hi = q(.5), q(.1), q(.9)
    r['cov80'] = np.mean([(lo <= y <= hi) for y in ys])
    r['width80'] = hi - lo
    for k, (lo_, hi_) in WIN.items():
        pr = float(np.interp(hi_, Y, G) - np.interp(lo_, Y, G))
        o = np.array([(lo_ <= y <= hi_) for y in ys], float)
        r[k + '_p'] = pr
        r[k + '_any'] = bool(o.any())
        r[k + '_brier'] = float(np.mean((pr - o) ** 2))
        r[k + '_hard'] = float(np.mean((float(lo_ <= med <= hi_) - o) ** 2))
        r[k + '_dist'] = -max(lo_ - med, med - hi_, 0)
    return r


def exp_summary(parts_by_key, targets):
    rows = [score_comp(cdf_grid(*parts_by_key[k]), targets[k]) for k in sorted(parts_by_key)]
    d = pd.DataFrame(rows)
    o = dict(exp_n=len(d))
    for mname in ['crps', 'cov80', 'width80', 'pv_brier', 'visible_brier', 'visible_hard', 'pv_hard']:
        o['exp_' + mname] = float(d[mname].mean())
    for k in WIN:
        o[f'exp_{k}_auc_p'] = auc(d[k + '_any'], d[k + '_p'])
        o[f'exp_{k}_auc_dist'] = auc(d[k + '_any'], d[k + '_dist'])
    return o


def mix(list_parts):
    w = np.concatenate([x[0] for x in list_parts]) / len(list_parts)
    return w, np.concatenate([x[1] for x in list_parts]), np.concatenate([x[2] for x in list_parts])


# targets
exp_t = pd.read_csv(CAL / 'in/R/expanded_recheck/recovered_target_occurrences.csv')
exp_targets = exp_t.groupby('composition_key').target_ev.apply(list).to_dict()
records = [json.loads(l) for l in (SRC / 'data/reports/records.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
cw = hierarchical_weights(records, list(range(len(records))), 'cpu').numpy()
cohort = {r['measurement_id']: r for r in rt.read_json(CONSO / 'data/confirmation/cohort.json')['records'] if r['cohort'] == 'independent'}
enc = rt.read_json(CONSO / 'data/confirmation/encoded_features.json')
ext_ids = [r['measurement_id'] for r in enc if r['measurement_id'] in cohort]
ext_recs = [cohort[i] for i in ext_ids]
ew = hierarchical_weights(ext_recs, list(range(len(ext_recs))), 'cpu').numpy()
route = []
for r in enc:
    if r['measurement_id'] in cohort:
        sg, st = r.get('space_group') is not None, r.get('structure') is not None
        route.append(3 if sg and st else 1 if sg else 2 if st else 0)
exp_keys = torch.load(STAGE1 / 'challenge/features.pt', map_location='cpu', weights_only=True)['keys']


def core_metrics(parts_fn, nviews=(0, 3)):
    """parts_fn(view, record_index) -> (w, mu, sd) mixture over seeds."""
    res = {}
    for v, name in ((0, 'formula_only'), (1, 'formula_sg'), (2, 'formula_structure'), (3, 'full')):
        sc = [score(float(records[i]['gap_ev']), parts_fn(v, i)) for i in range(len(records))]
        for mname, col in (('crps', 'crps_ev'), ('nll', 'nll'), ('mae', 'median_mae_ev'), ('cov80', 'coverage_80')):
            res[f'core_{name}_{mname}'] = float(sum(cw[i] * sc[i][col] for i in range(len(records))))
    return res


def ext_metrics(parts_fn):
    res = {}
    for rname in ('formula_only', 'all_available'):
        sc = []
        for j, rec in enumerate(ext_recs):
            v = 0 if rname == 'formula_only' else route[j]
            sc.append(score(float(rec['gap_ev']), parts_fn(v, j)))
        for mname, col in (('crps', 'crps_ev'), ('nll', 'nll')):
            res[f'ext_{rname}_{mname}'] = float(sum(ew[j] * sc[j][col] for j in range(len(sc))))
    return res


def from_npz(arm):
    files = {(s, f): np.load(out / f'{arm}_seed{s}_fold{f}_pred.npz') for s in SEEDS for f in range(1, 6)}
    core = {}  # (seed, record index, view) -> parts
    for (s, f), z in files.items():
        for v in range(4):
            for j, i in enumerate(z['core_idx']):
                core[(s, int(i), v)] = (z[f'core{v}'][0][j], z[f'core{v}'][1][j], z[f'core{v}'][2][j])
    assert len({k[1] for k in core}) == 205
    cparts = lambda v, i: mix([core[(s, i, v)] for s in SEEDS])
    exp = {}
    for kidx, k in enumerate(exp_keys):
        exp[k] = mix([tuple(z['exp'][t][kidx:kidx + 1][0] for t in range(3)) for z in files.values()])
    ext = lambda v, j: mix([tuple(z[f'ext{v}'][t][j] for t in range(3)) for z in files.values()])
    return cparts, exp, ext


def csvparts(frame):
    return mix([tuple(np.asarray(json.loads(r[c]), float) for c in ('component_weights', 'log_mus', 'sigmas')) for _, r in frame.iterrows()])


rows = []
# archived references
arch = {}
oof = pd.read_csv(SRC / 'results/formula_and_modalities/oof_densities.csv')
oof = oof[oof.model == 'FROZEN']
idx = {r['measurement_id']: i for i, r in enumerate(records)}
tab = {}
for (v, mid), g in oof.groupby(['view', 'measurement_id']):
    tab[(v, idx[mid])] = csvparts(g)
vn = {0: 'formula_only', 1: 'formula_sg', 2: 'formula_structure', 3: 'full'}
core_arch = core_metrics(lambda v, i: tab[(vn[v], i)])
print('archived FROZEN core formula_only CRPS (csv)', core_arch['core_formula_only_crps'], flush=True)
s1 = pd.read_csv(CAL / 'in/S/BandgapFusion_Stage1_20261023/challenge/predictions.csv')
raw_parts = {k: csvparts(g) for k, g in s1[s1.model == 'RAW'].groupby('composition_key')}
ref_raw = exp_summary(raw_parts, exp_targets)
print('archived RAW expanded', ref_raw['exp_crps'], ref_raw['exp_cov80'], flush=True)
s2 = pd.read_csv(STAGE2 / 'predictions.csv', usecols=['model', 'state_role', 'subset', 'composition_key', 'component_weights', 'log_mus', 'sigmas'])
s2 = s2[(s2.model == 'DIST') & (s2.state_role == 'selected') & (s2.subset == 'challenge')]
s2_parts = {k: csvparts(g) for k, g in s2.groupby('composition_key')}
ref_s2 = exp_summary(s2_parts, exp_targets)
print('archived Stage2 DIST expanded', ref_s2['exp_crps'], flush=True)
rows.append(dict(arm='FROZEN_archived_csv', **core_arch, **ref_raw))
rows.append(dict(arm='Stage2_DIST_archived_csv', **ref_s2))
for arm in a.arms.split(','):
    if not list(out.glob(f'{arm}_seed*_pred.npz')):
        continue
    cparts, exp, ext = from_npz(arm)
    r = dict(arm=arm, **core_metrics(cparts), **exp_summary(exp, exp_targets), **ext_metrics(ext))
    rows.append(r)
    print(arm, {k: round(v, 4) for k, v in r.items() if k in ('core_formula_only_crps', 'core_full_crps', 'exp_crps', 'exp_cov80', 'exp_visible_brier', 'exp_visible_auc_p', 'exp_pv_auc_p', 'ext_formula_only_crps')}, flush=True)
pd.DataFrame(rows).to_csv(a.csv, index=False)
