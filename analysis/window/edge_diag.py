import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import evaluate_cv_v4 as E
ev = E.ev
T = pd.read_csv('/data/czx/GapScope_Revision_20261113/window/par2/window_units_all.csv')
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
def fauc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def strad(y, a, b):
    o = (y >= a) & (y <= b); return bool(o.any() and (~o).any())
Y = {}
for i, k in enumerate(ev.Akey): Y.setdefault('A|' + k, []).append(float(ev.A[i]['gap_ev']))
for k, v in ev.Bvals.items(): Y['B|' + k] = list(map(float, v))
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
H = {k: t.hse.values for k, t in S.groupby('ckey') if len(t) >= 2}
for V in ('comp', 'chem'):
    for m in ('GapScope', 'ROOST_CAL'):
        t = T[(T.holdout == V) & (T.model == m)].copy(); t['uk'] = t.tier + '|' + t.key; t = t.set_index('uk')
        for layer in ('exp', 'hse_all'):
            for wn, (a, b) in W.items():
                if layer == 'exp':
                    ks = [k for k, y in Y.items() if len(y) >= 2]; lab = np.array([strad(np.array(Y[k]), a, b) for k in ks])
                else:
                    ks = ['B|' + k for k in H if 'B|' + k in t.index]; lab = np.array([strad(H[k[2:]], a, b) for k in ks])
                med = t.loc[ks, 'median'].values; wid = t.loc[ks, 'width80'].values; P = t.loc[ks, 'P_' + wn].values
                edge = -np.minimum(np.abs(med - a), np.abs(med - b))
                pst = 4 * P * (1 - P)
                near = np.abs(edge) <= 0.4
                print(V, m, layer, wn, 'n %d str %d | AUC edge-dist %.3f  P(1-P) %.3f  width %.3f | near-edge n %d str %d: AUC width %.3f  P(1-P) %.3f | width+edge logit-free: %.3f' % (
                    len(ks), lab.sum(), fauc(lab, edge), fauc(lab, pst), fauc(lab, wid), near.sum(), lab[near].sum(), fauc(lab[near], wid[near]), fauc(lab[near], pst[near]),
                    fauc(lab, rankdata(edge) + rankdata(wid))))
