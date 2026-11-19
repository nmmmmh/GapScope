import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import evaluate_cv_v4 as E
ev = E.ev
T = pd.read_csv('/data/czx/GapScope_Revision_20261113/window/par2/window_units_all.csv')
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
rng = np.random.default_rng(7)
def fauc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def strad(y, a, b):
    o = (y >= a) & (y <= b); return bool(o.any() and (~o).any())
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
Dacc = {}
for k, t in S.groupby('ckey'):
    te = t.dropna(subset=['Free_energy'])
    if len(te): te = te[te.Free_energy <= te.Free_energy.min() + 0.1]
    u = te if len(te) else t
    Dacc[k] = float(u.hse.max() - u.hse.min())
Y = {}
for i, k in enumerate(ev.Akey): Y.setdefault('A|' + k, []).append(float(ev.A[i]['gap_ev']))
for k, v in ev.Bvals.items(): Y['B|' + k] = list(map(float, v))
for V in ('comp', 'chem'):
    for m in ('GapScope', 'ROOST_CAL'):
        t = T[(T.holdout == V) & (T.model == m)].copy(); t['uk'] = t.tier + '|' + t.key; t = t.set_index('uk')
        for wn, (a, b) in W.items():
            ks = [k for k, y in Y.items() if len(y) >= 2 and k[2:] in Dacc]
            lab = np.array([strad(np.array(Y[k]), a, b) for k in ks])
            med = t.loc[ks, 'median'].values; P = t.loc[ks, 'P_' + wn].values; Dc = np.array([Dacc[k[2:]] for k in ks])
            e = np.minimum(np.abs(med - a), np.abs(med - b))
            sc_edge, sc_dc = -e, Dc / 2 - e
            near = e <= 0.4
            I = rng.integers(0, len(ks), (2000, len(ks)))
            dd = np.array([fauc(lab[i], sc_dc[i]) - fauc(lab[i], sc_edge[i]) for i in I])
            print(V, m, wn, 'n %d str %d (Dc>0: %d) | AUC edge %.3f | edge+Dc %.3f (d %.3f, %.3f to %.3f) | P(1-P) %.3f | near n %d str %d: AUC Dc %.3f' % (
                len(ks), lab.sum(), (Dc > 0).sum(), fauc(lab, sc_edge), fauc(lab, sc_dc), fauc(lab, sc_dc) - fauc(lab, sc_edge), np.nanpercentile(dd, 2.5), np.nanpercentile(dd, 97.5),
                fauc(lab, P * (1 - P)), near.sum(), lab[near].sum(), fauc(lab[near], Dc[near])))
