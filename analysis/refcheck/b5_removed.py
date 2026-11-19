"""MAE on the compositions removed by the 0.1 eV agreement rule vs those kept, same construction as ledger_checks.py.
Run on c137-13: OMP_NUM_THREADS=1 <venv>/python -I b5_removed.py"""
import numpy as np, pandas as pd
R0 = '/data/czx/GapScope_Revision_20261113/'
a = pd.read_csv(R0 + 'cv/predictions_summary_v4a.csv'); t = pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv')
M = [("ROOST", a), ("ROOST_CAL", a), ("ROOST_CAL_CV", a), ("EASYUQ_KS", a), ("EASYUQ_IDR", a), ("GS_SELCAL_PRE_NZ_SC", t)]
def wm(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return y[np.searchsorted(c, 0.5)]
for m, d in M:
    g = d[(d.model == m) & (d.tier == 'A')]; R = []
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values
        R.append((np.sum(w * np.abs(y - gg['median'].values)) / w.sum(), np.sum(w * abs(y - wm(y, w))) / w.sum(), round(y.max() - y.min(), 6) > 0.1 + 1e-9))
    R = pd.DataFrame(R, columns=['mae', 'fl', 'rm'])
    print('%-20s removed (n %d) MAE %.3f floor %.3f | kept (n %d) MAE %.3f floor %.3f | ratio %.2f'
          % (m, R.rm.sum(), R[R.rm].mae.mean(), R[R.rm].fl.mean(), (~R.rm).sum(), R[~R.rm].mae.mean(), R[~R.rm].fl.mean(), R[R.rm].mae.mean() / R[~R.rm].mae.mean()))
