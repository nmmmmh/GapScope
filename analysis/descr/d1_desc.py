"""Two closed-form polymorph descriptors, scored without any fitting against the floor and range of the 50 SNUMAT-matched traced compositions.
D_struct = spread over accessible polymorphs of mean cation coordination number + spread of mean shared vertices.
D_elec   = spread over accessible polymorphs of the HSE06 gap.
Accessible = polymorph cell whose free energy lies within W eV/atom of the lowest cell of the composition; cells without an energy are kept."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
R = '/data/czx/GapScope_Revision_20261113/'
src = open(R + 'refs/width_family.py').read(); exec(src.split(r"print('\nheld-out")[0])   # gives q (50 compositions), y (range), Xr, G, gbm, oof_random
a = pd.read_csv(R + 'cv/predictions_summary_v4a.csv'); g0 = a[(a.model == 'ROOST') & (a.tier == 'A')]
def wm(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
fl = {k: np.sum(gg.w.values * np.abs(gg.y.values - wm(gg.y.values, gg.w.values))) / gg.w.sum() for k, gg in g0.groupby('key')}
keys = q.composition_key.values; F = np.array([fl[k] for k in keys]); RNG = q.exp_range.values
Dd = pd.read_csv(R + 'struct/descriptors.csv', usecols=['composition_key', 'sg_number', 'cn_cat_mean', 'mean_shared_vertices', 'Band_gap_HSE'])
cell = Dd.groupby(['composition_key', 'sg_number']).median().reset_index()
E = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')[['ckey', 'sg', 'Free_energy']].rename(columns={'ckey': 'composition_key', 'sg': 'sg_number'})
cell = cell.merge(E, on=['composition_key', 'sg_number'], how='left')
sub = cell[cell.composition_key.isin(keys)]
print('cells for the 50 compositions: %d, with energy %d' % (len(sub), sub.Free_energy.notna().sum()))
def desc(W):
    out = {}
    for k, c in sub.groupby('composition_key'):
        if W is not None and c.Free_energy.notna().any():
            e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= W)]
        sp = lambda s: (s.max() - s.min()) if s.notna().sum() >= 2 else 0.0
        out[k] = (sp(c.cn_cat_mean) + sp(c.mean_shared_vertices), sp(c.Band_gap_HSE), len(c))
    return np.array([out.get(k, (np.nan, np.nan, 0)) for k in keys])
pos_r, pos_f = RNG > 0.2, F > 0.05
rows = []
for W in (None, 0.30, 0.20, 0.10, 0.05):
    d = desc(W)
    for j, name in ((0, 'D_struct'), (1, 'D_elec')):
        x = d[:, j]
        rows.append(dict(window=('all' if W is None else W), descriptor=name, rho_floor=spearmanr(x, F).statistic, P_floor=spearmanr(x, F).pvalue,
                         rho_range=spearmanr(x, RNG).statistic, AUC_range=roc_auc_score(pos_r, x), AUC_floor=roc_auc_score(pos_f, x),
                         mean_cells=d[:, 2].mean()))
T = pd.DataFrame(rows); pd.set_option('display.width', 200); print(T.round(3).to_string(index=False))
T.to_csv(R + 'descr/d1_scores.csv', index=False)
# references on the same 50: existing flag, existing raw spreads, and the fitted model under cation-set holdout
print('var_coordination  AUC_range %.3f AUC_floor %.3f' % (roc_auc_score(pos_r, q.var_coordination), roc_auc_score(pos_f, q.var_coordination)))
print('gapspr_hse (all cells) AUC_range %.3f' % roc_auc_score(pos_r, q.gapspr_hse.fillna(0)))
print('positives: range>0.2 %d, floor>0.05 %d of %d' % (pos_r.sum(), pos_f.sum(), len(keys)))
