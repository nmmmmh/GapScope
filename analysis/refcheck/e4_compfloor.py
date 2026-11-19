"""Computed floor: Boltzmann-weighted mean absolute deviation of polymorph HSE06 gaps about their weighted median, same
functional form as the measured floor (eq 2), energy-only cells. Compare magnitudes with the measured floor on the 50."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv').rename(columns={'ckey': 'composition_key'})
S = S[S.composition_key.isin(keys) & S.Free_energy.notna() & S.hse.notna()]
kB = 8.617e-5; pos = F > 0.05
def wmed(v, w): o = np.argsort(v); v, w = v[o], w[o]; return v[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)]
for T in (300, 600, 1000):
    x = []
    for k in keys:
        c = S[S.composition_key == k]
        if len(c) < 2: x.append(0.0); continue
        w = np.exp(-(c.Free_energy.values - c.Free_energy.min()) / (kB * T)); g = c.hse.values; m = wmed(g, w)
        x.append(np.sum(w * np.abs(g - m)) / w.sum())
    x = np.array(x)
    print('T %4d | AUC %.3f rho %.2f | mean computed %.3f vs measured %.3f | multi-phase(F>0.05) %.3f vs %.3f | nonzero %d'
          % (T, roc_auc_score(pos, x), spearmanr(x, F).statistic, x.mean(), F.mean(), x[pos].mean(), F[pos].mean(), (x > 0.001).sum()))
