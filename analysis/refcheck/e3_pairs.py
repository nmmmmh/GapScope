"""(1) Madelung gap spread on the 50 compositions; (2) polymorph pairs: does the Madelung gap M or the Harrison bandwidth W
order the HSE06 gaps where six local descriptors explain 6%? Cells = median per composition x space group; pairs within a
composition, both orders, equal weight per composition, ridge without intercept, compositions held out in 5 folds."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pymatgen.core import Composition
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
M = pd.read_csv(R + 'refcheck/m1_madelung.csv'); M = M[M.M.notna()]
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
M['composition_key'] = [key(f) for f in M.formula]
Mc = M.groupby(['composition_key', 'sg']).agg(M=('M', 'median'), hse=('hse', 'median')).reset_index().rename(columns={'sg': 'sg_number'})
D6 = ['cn_cat_mean', 'mean_shared_vertices', 'dim_cat_an', 'bond_cat_an_mean', 'distortion_index', 'vol_per_atom']
Dd = pd.read_csv(R + 'struct/descriptors.csv', usecols=['composition_key', 'sg_number', 'Band_gap_HSE'] + D6)
C = Dd.groupby(['composition_key', 'sg_number']).median().reset_index().merge(Mc[['composition_key', 'sg_number', 'M']], on=['composition_key', 'sg_number'], how='inner')
C['W'] = C.cn_cat_mean / C.bond_cat_an_mean ** 2
E = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv').rename(columns={'ckey': 'composition_key', 'sg': 'sg_number'})[['composition_key', 'sg_number', 'Free_energy']]
C = C.merge(E, on=['composition_key', 'sg_number'], how='left').dropna(subset=D6 + ['M', 'Band_gap_HSE'])
print('cells', len(C), 'compositions', C.composition_key.nunique())
# (1) 50 compositions, energy-only 0.1 window
pos = F > 0.05
def sp(s): return s.max() - s.min() if len(s) >= 2 else 0.0
x = []
for k in keys:
    c = C[(C.composition_key == k) & C.Free_energy.notna()]
    c = c[c.Free_energy - c.Free_energy.min() <= 0.1] if len(c) else c
    x.append(sp(c.M))
x = np.array(x); print('(1) Madelung spread  AUC %.3f rho %.2f (n with >=2 cells %d)' % (roc_auc_score(pos, x), spearmanr(x, F).statistic, (x > 0).sum()))
# (2) pairs
rows = []
for k, g in C.groupby('composition_key'):
    if len(g) < 2: continue
    v = g.reset_index(drop=True); n = len(v); w = 1.0 / (n * (n - 1))
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append([k, w, v.Band_gap_HSE[i] - v.Band_gap_HSE[j]] + [v[f][i] - v[f][j] for f in D6 + ['M', 'W']])
P = pd.DataFrame(rows, columns=['k', 'w', 'dg'] + D6 + ['M', 'W'])
print('(2) pairs (both orders)', len(P), 'compositions', P.k.nunique())
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
def cv(cols):
    pred = np.zeros(len(P)); Xs = P[cols].values / P[cols].values.std(0)
    for tr, te in GroupKFold(5).split(Xs, groups=P.k):
        m = Ridge(alpha=1.0, fit_intercept=False).fit(Xs[tr], P.dg.values[tr], sample_weight=P.w.values[tr]); pred[te] = m.predict(Xs[te])
    r2 = 1 - np.sum(P.w * (P.dg - pred) ** 2) / np.sum(P.w * P.dg ** 2)
    nz = P.dg != 0; auc = roc_auc_score(P.dg[nz] > 0, pred[nz], sample_weight=P.w[nz])
    big = P.dg.abs() > 0.5; auc5 = roc_auc_score(P.dg[big] > 0, pred[big], sample_weight=P.w[big])
    return r2, auc, auc5
for lab, cols in (('six local', D6), ('Madelung M alone', ['M']), ('Harrison W alone', ['W']), ('six + M', D6 + ['M']), ('six + M + W', D6 + ['M', 'W'])):
    r2, a, a5 = cv(cols); print('%-18s R2 %.3f | sign AUC %.3f | |dg|>0.5 AUC %.3f' % (lab, r2, a, a5))
print('raw sign agreement dM vs dg (weighted): %.3f' % (np.sum(P.w * ((P.M * P.dg) > 0)) / np.sum(P.w * ((P.M * P.dg) != 0))))
