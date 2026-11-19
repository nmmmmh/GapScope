"""Does adding computed polymorph gaps help or hurt the width prediction? Cation-family holdout, 20 assignments."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
D0 = '/data/czx/GapScope_Revision_20261113/struct/'
SPREAD_OF = ['cn_cat_mean','cn_cat_max','cn_all_mean','mean_shared_vertices','frac_corner','frac_edge','frac_face','dim_cat_an','bond_cat_an_mean','distortion_index','vol_per_atom','packing_fraction','n_symops','frac_isolated_poly']
CHEMF = ['n_elements','anion_frac','en_an_mean','en_cat_mean','en_diff','ionicity','en_std_all','nval_per_atom','radius_ratio','mean_atomic_radius','is_halide','is_oxide','is_chalcogenide','is_pnictide','is_intermetallic','dom_anion_Z']
STRUCF = ['n_polymorphs','n_crystal_systems','any_noncentro','frac_direct','mixed_direct_indirect','var_coordination','var_dimensionality','var_connectivity'] + ['spr_'+k for k in SPREAD_OF] + ['mean_'+k for k in SPREAD_OF] + ['hse_mean']
GAPF = ['gapspr_hse','gapsd_hse','gapspr_hseopt','gapspr_gga']
ANIONS = {'O','S','Se','Te','F','Cl','Br','I'}
W = pd.read_csv(D0 + 'width.csv'); q = W[(W.set=='provenance') & W.exp_range.notna() & (W.in_snumat==1)].reset_index(drop=True)
y = q.exp_range.values; lab = y > 0.2
grp = np.array([','.join(sorted(set(s.split(',')) - ANIONS)) for s in q.elem_system.fillna('?')]); ug = np.unique(grp)
def run(feats):
    feats = [f for f in feats if f in q.columns]; X = q[feats].values.astype(float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X); X = np.where(np.isnan(X), 0, X)
    A, R = [], []
    for s in range(20):
        r = np.random.default_rng(s); fo = dict(zip(ug, r.permutation(len(ug)) % 5)); f = np.array([fo[v] for v in grp]); p = np.zeros(len(y))
        for k in range(5):
            te = f == k
            p[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0).fit(X[~te], y[~te]).predict(X[te])
        A.append(roc_auc_score(lab, p)); R.append(spearmanr(p, y).statistic)
    return np.mean(A), np.std(A), np.mean(R), len(feats)
for name, fs in (('chemistry only', CHEMF), ('chemistry + structure (main)', CHEMF + STRUCF), ('chemistry + structure + computed gap spread', CHEMF + STRUCF + GAPF), ('computed gap spread only', GAPF)):
    a, sd, r, n = run(fs); print('%-46s n_feat=%2d  AUC %.3f (sd %.3f)  rho %.3f' % (name, n, a, sd, r))
