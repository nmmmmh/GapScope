"""Compiled set (2,467 compositions): model error on compositions the 0.1 eV agreement rule removes vs keeps.
Error of a composition = within-composition-weighted MAE of the predicted median against its compiled values."""
import numpy as np, pandas as pd
R0 = '/data/czx/GapScope_Revision_20261113/'
B = pd.read_csv(R0 + 'cv/data/tierB_rows.csv')
def wm(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return y[np.searchsorted(c, 0.5)]
G = {k: (g.target_ev.values, g.within_composition_weight.values) for k, g in B.groupby('composition_key')}
rm = pd.Series({k: (y.max() - y.min()) > 0.1 + 1e-9 for k, (y, w) in G.items()})
fl = pd.Series({k: np.sum(w * np.abs(y - wm(y, w))) / w.sum() for k, (y, w) in G.items()})
multi = pd.Series({k: len(y) > 1 for k, (y, w) in G.items()})
print('compositions %d, with >=2 values %d, removed %d, floor share removed %.1f%%' % (len(G), multi.sum(), rm.sum(), 100 * fl[rm].sum() / fl.sum()))
for f, var in (('predictions_summary_v4t1.csv', 'comp'), ('predictions_summary_v4chem_NZ.csv', 'chem')):
    a = pd.read_csv(R0 + 'cv/' + f)
    for m in ('ROOST', 'ROOST_CAL', 'GS_SELCAL_PRE_NZ_SC'):
        med = a[(a.model == m) & (a.tier == 'B')].groupby('key')['median'].first()
        e = pd.Series({k: np.sum(w * np.abs(y - med[k])) / w.sum() for k, (y, w) in G.items() if k in med.index})
        r = rm[e.index]; mu = multi[e.index]
        print('%s %-20s removed %.3f (floor %.3f) | kept multi-valued %.3f | kept single-valued %.3f | all %.3f'
              % (var, m, e[r].mean(), fl[e.index][r].mean(), e[~r & mu].mean(), e[~r & ~mu].mean(), e.mean()))
