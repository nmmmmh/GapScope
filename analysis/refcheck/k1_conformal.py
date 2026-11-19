"""(1) Split-conformal baselines on the Roost out-of-fold predictions, cross-conformal over the five folds: for test fold f,
the 80% quantile of conformity scores on the tier-A records of the other four folds (each scored by the Roost ensemble that
did not train on it), composition-equal weights; interval = median +/- q (absolute) or median +/- q*b (normalized by the
Roost scale). Coverage/width on fold f, composition-equal weights, as in Table S11. Both holdouts, original assignment.
(2) Unenriched compositions (origin == 'external', the six further sources): floor, Roost and this-work MAE, D_c AUC."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
import evaluate_cv as ev
CV = ev.CV; A, Akey = ev.A, ev.Akey; wA = np.asarray(ev.wA, float)
FOLDS = {v: pd.read_csv(CV / f'folds_{v}.csv').set_index('composition_key').fold.to_dict() for v in ('comp', 'chem')}
y = np.array([float(r['gap_ev']) for r in A])
def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum(); return x[np.searchsorted(c, q)]
def lap_median(mu, b, xs=np.linspace(-3, 12, 6001)):
    F = (0.5 + 0.5 * np.sign(xs[:, None] - mu[None]) * (1 - np.exp(-np.abs(xs[:, None] - mu[None]) / b[None]))).mean(1)
    return xs[np.searchsorted(F, 0.5)]
for variant in ('comp', 'chem'):
    rA = ev.load_roost(variant)[0][0]
    m = np.array([lap_median(rA[i][1], rA[i][2]) for i in range(len(A))]); b = np.array([rA[i][2].mean() for i in range(len(A))])
    fd = np.array([FOLDS[variant][k] for k in Akey])
    nk = {}
    for k in Akey: nk[k] = nk.get(k, 0) + 1
    # composition-equal weights (wA already sums to 1 within composition in ev? normalise to be safe)
    wc = np.zeros(len(A))
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]; wc[ii] = wA[ii] / wA[ii].sum()
    for lab, score, half in (('absolute', np.abs(y - m), lambda q, i: q * np.ones(len(i))), ('normalized', np.abs(y - m) / b, lambda q, i: q * b[i])):
        cov = np.zeros(len(A)); wid = np.zeros(len(A))
        for f in range(1, 6):
            tr = fd != f; te = np.where(fd == f)[0]
            q = wq(score[tr], wc[tr], 0.80)
            h = half(q, te); cov[te] = np.abs(y[te] - m[te]) <= h; wid[te] = 2 * h
        ck = pd.DataFrame(dict(k=Akey, c=cov * wc, w=wid * wc)).groupby('k').sum()
        print('%-5s conformal %-10s coverage %.3f width %.2f eV' % (variant, lab, ck.c.mean(), ck.w.mean()))
# (2) unenriched
import torch
T = torch.load(CV / 'data/tierA.pt', weights_only=False); R = T['records']
orig = {}
for r in R: orig[r['composition_key']] = r.get('origin', '?')
P = pd.read_csv(CV / 'predictions_summary_v4t1.csv')
P = P[(P.tier == 'A')]
print('models in file:', sorted(P.model.unique())[:12])
def wm(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
g0 = P[(P.model == 'ROOST') & (P.variant == 'comp')]
fl = {k: np.sum(g.w.values * np.abs(g.y.values - wm(g.y.values, g.w.values))) / g.w.sum() for k, g in g0.groupby('key')}
for o in ('core', 'external'):
    ks = [k for k in fl if orig.get(k) == o]
    print('origin %-8s compositions %2d | floor %.3f | multi-report %d' % (o, len(ks), np.mean([fl[k] for k in ks]), sum(len(g0[g0.key == k]) > 1 for k in ks)))
    for mdl in ('ROOST', 'ROOST_CAL', 'GS_SELCAL_PRE_NZ_SC', 'EASYUQ_KS'):
        for v in ('comp', 'chem'):
            g = P[(P.model == mdl) & (P.variant == v) & P.key.isin(ks)]
            if len(g): print('     %-22s %-4s MAE %.3f' % (mdl, v, g.groupby('key').apply(lambda x: np.sum(x.w * x.mae) / x.w.sum()).mean()))
