"""Data for Fig. 3: out-of-fold width predictions (cation families held out), elemental repertoires."""
import sys, collections, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
D0 = '/data/czx/GapScope_Revision_20261113/struct/'
SPREAD_OF = ['cn_cat_mean','cn_cat_max','cn_all_mean','mean_shared_vertices','frac_corner','frac_edge','frac_face','dim_cat_an','bond_cat_an_mean','distortion_index','vol_per_atom','packing_fraction','n_symops','frac_isolated_poly']
CHEMF = ['n_elements','anion_frac','en_an_mean','en_cat_mean','en_diff','ionicity','en_std_all','nval_per_atom','radius_ratio','mean_atomic_radius','is_halide','is_oxide','is_chalcogenide','is_pnictide','is_intermetallic','dom_anion_Z']
STRUCF = ['n_polymorphs','n_crystal_systems','any_noncentro','frac_direct','mixed_direct_indirect','var_coordination','var_dimensionality','var_connectivity'] + ['spr_'+k for k in SPREAD_OF] + ['mean_'+k for k in SPREAD_OF] + ['hse_mean']
ANIONS = {'O','S','Se','Te','F','Cl','Br','I'}
W = pd.read_csv(D0 + 'width.csv')
q = W[(W.set=='provenance') & W.exp_range.notna() & (W.in_snumat==1)].reset_index(drop=True)
feats = [f for f in CHEMF+STRUCF if f in q.columns]
X = q[feats].values.astype(float); y = q.exp_range.values
X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X); X = np.where(np.isnan(X), 0, X)
grp = np.array([','.join(sorted(set(s.split(',')) - ANIONS)) for s in q.elem_system.fillna('?')])
P = np.zeros(len(q)); PM = np.zeros((len(q), 20)); ug = np.unique(grp)
for s in range(20):
    r = np.random.default_rng(s); fo = dict(zip(ug, r.permutation(len(ug)) % 5)); f = np.array([fo[v] for v in grp])
    for k in range(5):
        te = f == k
        PM[te, s] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0).fit(X[~te], y[~te]).predict(X[te])
P = PM.mean(1)
E = pd.read_csv(D0 + 'elem_cn.csv'); E = E[E.set == 'provenance'][['key', 'ent_max', 'n_cn_max']].rename(columns={'key': 'composition_key'})
out = q[['composition_key', 'exp_range', 'var_coordination', 'n_reports']].assign(pred=P).merge(E, on='composition_key', how='left')
out = pd.concat([out, pd.DataFrame(PM, columns=['p%02d' % i for i in range(20)])], axis=1); out.to_csv(D0 + 'tables/fig3_width.csv', index=False)
print('mean single-assignment AUC', round(np.mean([roc_auc_score(out.exp_range > 0.2, PM[:, i]) for i in range(20)]), 3))
from sklearn.metrics import roc_auc_score
lab = out.exp_range > 0.2
for c in ('pred', 'var_coordination', 'n_reports', 'ent_max'):
    v = out[c].fillna(out[c].median()); print(c, round(roc_auc_score(lab, v), 3), 'n', int(out[c].notna().sum()))
# elemental repertoires over the single-cation pool
Dd = pd.read_csv(D0 + 'descriptors.csv'); Dd = Dd[Dd.dom_cn_cat.notna() & Dd.cations.notna()]
Dd['catlist'] = Dd.cations.map(lambda s: [e.strip() for e in str(s).replace(';', ',').split(',') if e.strip()])
sg = Dd[Dd.catlist.map(len) == 1]
rep = collections.defaultdict(collections.Counter)
for c, cn in zip(sg.catlist.map(lambda c: c[0]), sg.dom_cn_cat): rep[c][int(round(float(cn)))] += 1
rows = []
for e, c in rep.items():
    tot = sum(c.values()); keep = [k for k, v in c.items() if v / tot >= 0.10] if tot >= 4 else []
    rows.append((e, tot, len(keep), len(c), ';'.join('%d:%d' % kv for kv in sorted(c.items()))))
R = pd.DataFrame(rows, columns=['element', 'records', 'n_cn', 'n_cn_any', 'counts']); R.to_csv(D0 + 'tables/fig3_elements.csv', index=False)
Rk = R[R.records >= 4]; print('cations', len(Rk), 'versatile 10pct', int((Rk.n_cn >= 2).sum()), 'any', int((R.n_cn_any >= 2).sum()), 'of', len(R))
