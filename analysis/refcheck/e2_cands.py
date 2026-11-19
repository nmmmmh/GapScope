"""Candidate spread descriptors on the 50 traced compositions (floor > 0.05 eV positive), no fitting.
Energy-only polymorph cells (median per composition x space group). Candidates: D_c (energy-only, 0.1 window), Boltzmann-weighted
SD and weighted range at T, gap-energy coupling, GGA and HSE-optical spreads, Harrison bandwidth spread (CN x d^-2)."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
Dd = pd.read_csv(R + 'struct/descriptors.csv', usecols=['composition_key', 'sg_number', 'cn_cat_mean', 'bond_cat_an_mean', 'mean_shared_vertices', 'Band_gap_HSE'])
c2 = Dd.groupby(['composition_key', 'sg_number']).median().reset_index()
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv').rename(columns={'ckey': 'composition_key', 'sg': 'sg_number'})
c2 = c2.merge(S[['composition_key', 'sg_number', 'Free_energy', 'hse_opt', 'gga']], on=['composition_key', 'sg_number'], how='left')
c2 = c2[c2.composition_key.isin(keys) & c2.Free_energy.notna()]
c2['W'] = c2.cn_cat_mean / c2.bond_cat_an_mean ** 2
pos = F > 0.05; kB = 8.617e-5
def sp(s): s = s.dropna(); return s.max() - s.min() if len(s) >= 2 else 0.0
def feats(c):
    c = c.assign(dE=c.Free_energy - c.Free_energy.min()); a = c[c.dE <= 0.1]; o = {}
    o['D_c energy-only'] = sp(a.Band_gap_HSE)
    o['D_c, HSE optical'] = sp(a.hse_opt); o['D_c, GGA'] = sp(a.gga)
    for T in (300, 1000, 2000):
        w = np.exp(-c.dE / (kB * T)); m = (w * c.Band_gap_HSE).sum() / w.sum()
        o[f'Boltzmann SD {T} K'] = np.sqrt((w * (c.Band_gap_HSE - m) ** 2).sum() / w.sum())
        o[f'Boltzmann mean |dev| {T} K'] = (w * (c.Band_gap_HSE - m).abs()).sum() / w.sum()
    o['D_c, no window'] = sp(c.Band_gap_HSE)
    o['gap-energy slope x 0.1'] = abs(np.polyfit(c.dE, c.Band_gap_HSE, 1)[0]) * 0.1 if len(c) >= 2 and c.dE.std() > 0 else 0.0
    o['Harrison W rel. spread'] = sp(a.W) / a.W.mean() if len(a) >= 2 else 0.0
    o['D_struct'] = sp(a.cn_cat_mean) + sp(a.mean_shared_vertices)
    return o
G = {k: feats(g) for k, g in c2.groupby('composition_key')}
X = pd.DataFrame([G.get(k, {}) for k in keys]).fillna(0.0)
rng = np.random.default_rng(1)
def ci(x):
    b = [roc_auc_score(pos[i], x[i]) for i in (rng.integers(0, 50, 50) for _ in range(2000)) if pos[i].min() != pos[i].max()]
    return np.percentile(b, [2.5, 97.5])
for col in X:
    x = X[col].values; lo, hi = ci(x)
    print('%-28s AUC %.3f [%.2f, %.2f] rho %.2f' % (col, roc_auc_score(pos, x), lo, hi, spearmanr(x, F).statistic))
from scipy.stats import rankdata
for a_, b_ in (('D_c energy-only', 'Harrison W rel. spread'), ('D_c energy-only', 'D_struct'), ('Boltzmann SD 1000 K', 'D_struct')):
    x = (rankdata(X[a_]) + rankdata(X[b_])) / 2; print('rank mean %-18s + %-22s AUC %.3f' % (a_, b_, roc_auc_score(pos, x)))
