"""Descriptor models, round 1. Elemental statistics (pymatgen) +/- polymorph-ensemble descriptors from the MP HSE pool
(hse/hse_pool.csv: key, formula, spacegroup, energy_above_hull, target = HSE-level gap). Gradient boosting with absolute
loss, trained per fold on tier-A reports (composition-equal weights) and tier-B compiled values; composition and
chemical-family holdout, five fold assignments (cv_rep). Pooled MAE over 2552 units, as in cv_rep summaries
(tier A: composition-weighted records; tier B: mean over compiled values). Compared with ROOST / this work from the same
assignments' predictions_summary files."""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
import evaluate_cv as ev
from pymatgen.core import Composition, Element
from sklearn.ensemble import HistGradientBoostingRegressor
R = '/data/czx/GapScope_Revision_20261113/'
A, Akey, Bkeys, Bvals = ev.A, ev.Akey, ev.Bkeys, ev.Bvals
wA = np.asarray(ev.wA, float)
keys = sorted(set(Akey) | set(Bkeys))
# ---- elemental statistics
props = {}
def ep(el):
    if el in props: return props[el]
    e = Element(el); f = lambda v: float(v) if v is not None else np.nan
    ie = e.ionization_energies[0] if e.ionization_energies else np.nan
    props[el] = np.array([f(e.X), f(e.Z), f(e.row), f(e.group), f(e.atomic_radius), f(ie), f(e.electron_affinity) if hasattr(e, 'electron_affinity') else np.nan,
                          f(e.mendeleev_no), f(e.average_ionic_radius), f(e.max_oxidation_state), f(e.min_oxidation_state), float(e.is_metal)], float)
    return props[el]
def elem_feats(k):
    d = {e.split(':')[0]: float(e.split(':')[1]) for e in k.split('|')}
    w = np.array(list(d.values())); P = np.stack([ep(e) for e in d])
    mean = np.nansum(w[:, None] * P, 0); mx = np.nanmax(P, 0); mn = np.nanmin(P, 0)
    dev = np.nansum(w[:, None] * np.abs(P - mean), 0)
    return np.concatenate([mean, mx - mn, mn, mx, dev, [len(d), np.sqrt((w ** 2).sum()), w.max()]])
EF = {k: elem_feats(k) for k in keys}
# ---- polymorph ensemble from MP HSE pool
H = pd.read_csv(R + 'hse/hse_pool.csv')
def pkey(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
H['ck'] = [pkey(f) for f in H.formula]
kB = 8.617e-5
def wmed(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
PF = {}
for k, g in H[H.ck.isin(keys)].groupby('ck'):
    e = g.energy_above_hull.values - g.energy_above_hull.min(); t = g.target.values
    w = np.exp(-e / (kB * 300)); wm = np.sum(w * t) / w.sum(); med = wmed(t, w)
    PF[k] = np.array([1.0, len(g), t[np.argmin(e)], wm, t.min(), t.max(), t.max() - t.min(), np.sum(w * np.abs(t - med)) / w.sum(), g.energy_above_hull.min()])
nP = 9
print('MP-pool coverage: traced %d/85, compiled %d/2467' % (sum(k in PF for k in set(Akey)), sum(k in PF for k in Bkeys)))
def X(k, use_p):
    f = EF[k]
    if use_p: f = np.concatenate([f, PF.get(k, np.r_[0.0, np.full(nP - 1, np.nan)])])
    return f
# training rows
rows_k, rows_y, rows_w = [], [], []
for k in set(Akey):
    ii = [i for i, x in enumerate(Akey) if x == k]
    for i in ii: rows_k.append(k); rows_y.append(float(A[i]['gap_ev'])); rows_w.append(wA[i] / wA[ii].sum())
for k in Bkeys:
    v = list(map(float, Bvals[k])); 
    for y in v: rows_k.append(k); rows_y.append(y); rows_w.append(1.0 / len(v))
rows_k = np.array(rows_k); rows_y = np.array(rows_y); rows_w = np.array(rows_w)
def pooled_mae(pred):
    ua = []
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]; ww = wA[ii] / wA[ii].sum()
        ua.append(np.sum(ww * np.abs(np.array([float(A[i]['gap_ev']) for i in ii]) - pred[k])))
    ub = [np.mean(np.abs(np.array(list(map(float, Bvals[k]))) - pred[k])) for k in Bkeys]
    return np.mean(ua + ub), np.mean(ua)
res = []
for d in sorted(glob.glob(R + 'cv_rep/s2026111*')):
    for v in ('comp', 'chem'):
        fd = pd.read_csv(f'{d}/folds_{v}.csv').set_index('composition_key').fold.to_dict()
        S = pd.read_csv(f'{d}/predictions_summary_rep_{v}.csv')
        for m in ('ROOST', 'GS_SELCAL_PRE_NZ_SC'):
            g = S[S.model == m]
            a = g[g.tier == 'A'].groupby('key').apply(lambda x: np.sum(x.w * x.mae) / x.w.sum()); b = g[g.tier == 'B'].mae
            res.append(dict(rep=d[-8:], holdout=v, model=m, pooled=np.mean(list(a) + list(b)), traced=a.mean()))
        for lab, use_p in (('GBM elemental', False), ('GBM elemental+polymorph', True)):
            pred = {}
            fr = np.array([fd[k] for k in rows_k])
            for f in range(1, 6):
                tr = fr != f
                Xtr = np.stack([X(k, use_p) for k in rows_k[tr]])
                mdl = HistGradientBoostingRegressor(loss='absolute_error', max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=0)
                mdl.fit(Xtr, rows_y[tr], sample_weight=rows_w[tr])
                tk = [k for k in keys if fd.get(k) == f]
                p = mdl.predict(np.stack([X(k, use_p) for k in tk]))
                pred.update(dict(zip(tk, np.maximum(p, 0.0))))
            pm, tm = pooled_mae(pred)
            res.append(dict(rep=d[-8:], holdout=v, model=lab, pooled=pm, traced=tm))
        print(d[-8:], v, 'done', flush=True)
T = pd.DataFrame(res); T.to_csv(R + 'refcheck/k6_table.csv', index=False)
print(T.groupby(['holdout', 'model'])[['pooled', 'traced']].agg(['mean', 'std']).round(3).to_string())
