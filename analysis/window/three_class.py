import numpy as np, pandas as pd
from scipy.stats import rankdata
D = '/data/czx/GapScope_Revision_20261113/window/'
T = pd.read_csv(D + 'par2/window_units_all.csv')
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
MODELS = ('GapScope', 'ROOST_CAL', 'EASYUQ_KS')
rng = np.random.default_rng(5)


def fauc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(P, n):
    pin, pout = P ** n, (1 - P) ** n
    return pin, pout, 1 - pin - pout, 4 * P * (1 - P)


def evaluate(lab, preds, tag):
    """lab: DataFrame index key with cols n, cls (0 in, 1 out, 2 straddle) ; preds: {model: Series P indexed by key}"""
    keys = lab.index; y = lab.cls.values; n = lab.n.values; st = y == 2
    S = {}
    for m, P in preds.items():
        p = np.clip(P.loc[keys].values, 0, 1); pin, pout, pst, q = scores(p, n)
        O = np.stack([y == 0, y == 1, y == 2], 1).astype(float)
        b3 = ((np.stack([pin, pout, pst], 1) - O) ** 2).sum(1)
        S[m] = dict(b3=b3, pst=pst, q=q)
    # one-value baseline: class from median-in-window, straddle never predicted
    I = rng.integers(0, len(keys), (2000, len(keys)))
    rows = []
    for m in preds:
        r = dict(layer=tag, model=m, n_comp=len(keys), n_straddle=int(st.sum()), Brier3=S[m]['b3'].mean(),
                 AUC_straddle=fauc(st, S[m]['pst']), AUC_straddle_q=fauc(st, S[m]['q']), mean_pst_straddle=S[m]['pst'][st].mean(), mean_pst_other=S[m]['pst'][~st].mean())
        if m != 'GapScope':
            d = S['GapScope']['b3'] - S[m]['b3']; bs = d[I].mean(1)
            r.update(dBrier3=d.mean(), dB_lo=np.percentile(bs, 2.5), dB_hi=np.percentile(bs, 97.5))
            da = np.array([fauc(st[i], S['GapScope']['pst'][i]) - fauc(st[i], S[m]['pst'][i]) for i in I])
            r.update(dAUC=r['AUC_straddle'] - fauc(st, S[m]['pst']) if False else fauc(st, S['GapScope']['pst']) - fauc(st, S[m]['pst']),
                     dAUC_lo=np.nanpercentile(da, 2.5), dAUC_hi=np.nanpercentile(da, 97.5))
        rows.append(r)
    return rows


def cls_of(y, a, b):
    o = (y >= a) & (y <= b)
    return 0 if o.all() else (1 if (~o).all() else 2)


out = []
# experimental layer: compositions with >=2 reports (traced + compiled)
import sys
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import evaluate_cv_v4 as E
ev = E.ev
Y = {}
for i, k in enumerate(ev.Akey): Y.setdefault(('A', k), []).append(float(ev.A[i]['gap_ev']))
for k, v in ev.Bvals.items(): Y[('B', k)] = list(map(float, v))
for V in ('comp', 'chem'):
    for wn, (a, b) in W.items():
        for tier in ('A', 'B', 'A+B'):
            rows = [(t + '|' + k, len(y), cls_of(np.array(y), a, b)) for (t, k), y in Y.items() if len(y) >= 2 and (tier == 'A+B' or t == tier)]
            lab = pd.DataFrame(rows, columns=['key', 'n', 'cls']).set_index('key')
            preds = {}
            for m in MODELS:
                t = T[(T.holdout == V) & (T.model == m)]
                preds[m] = pd.Series(t['P_' + wn].values, index=t.tier + '|' + t.key)
            for r in evaluate(lab, preds, 'exp'):
                r.update(holdout=V, window=wn, tier=tier); out.append(r)
# HSE layer: SNUMAT polymorphs of compositions with out-of-fold predictions (compiled or traced)
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
for mode in ('accessible', 'all'):
    H = {}
    for k, t in S.groupby('ckey'):
        if mode == 'accessible':
            t = t.dropna(subset=['Free_energy']); t = t[t.Free_energy <= t.Free_energy.min() + 0.1] if len(t) else t
        if len(t) >= 2: H[k] = t.hse.values
    for V in ('comp', 'chem'):
        for wn, (a, b) in W.items():
            preds, keys = {}, None
            for m in MODELS:
                t = T[(T.holdout == V) & (T.model == m)]
                t = t[t.tier == 'B'].set_index('key')['P_' + wn]
                ta = T[(T.holdout == V) & (T.model == m) & (T.tier == 'A')].set_index('key')['P_' + wn]
                preds[m] = pd.concat([t, ta[~ta.index.isin(t.index)]])
            ks = [k for k in H if k in preds['GapScope'].index]
            lab = pd.DataFrame([(k, len(H[k]), cls_of(H[k], a, b)) for k in ks], columns=['key', 'n', 'cls']).set_index('key')
            for r in evaluate(lab, preds, 'hse_' + mode):
                r.update(holdout=V, window=wn, tier='overlap'); out.append(r)
O = pd.DataFrame(out)
cols = ['layer', 'holdout', 'window', 'tier', 'model', 'n_comp', 'n_straddle', 'Brier3', 'dBrier3', 'dB_lo', 'dB_hi', 'AUC_straddle', 'dAUC', 'dAUC_lo', 'dAUC_hi', 'AUC_straddle_q', 'mean_pst_straddle', 'mean_pst_other']
O = O[cols].round(4); O.to_csv(D + 'par2/three_class.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 500)
print(O.to_string(index=False))
