"""Width prediction with element-family holdout (review round 1, attack 6).
Same features/model/target as width2.py; only the CV grouping changes.
Primary grouping chosen BEFORE scoring: the strictest grouping whose largest group
holds <= 25% of compositions (so 5-fold GroupKFold is meaningful).
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/struct')
D = Path('/data/czx/GapScope_Revision_20261113/struct')
SPREAD_OF = ['cn_cat_mean','cn_cat_max','cn_all_mean','mean_shared_vertices','frac_corner','frac_edge','frac_face',
             'dim_cat_an','bond_cat_an_mean','distortion_index','vol_per_atom','packing_fraction','n_symops','frac_isolated_poly']
CHEMF = ['n_elements','anion_frac','en_an_mean','en_cat_mean','en_diff','ionicity','en_std_all','nval_per_atom',
         'radius_ratio','mean_atomic_radius','is_halide','is_oxide','is_chalcogenide','is_pnictide','is_intermetallic','dom_anion_Z']
STRUCF = (['n_polymorphs','n_crystal_systems','any_noncentro','frac_direct','mixed_direct_indirect',
           'var_coordination','var_dimensionality','var_connectivity'] +
          ['spr_'+k for k in SPREAD_OF] + ['mean_'+k for k in SPREAD_OF] + ['hse_mean'])
ANIONS = {'O','S','Se','Te','F','Cl','Br','I'}
W = pd.read_csv(D / 'width.csv')
q = W[(W.set=='provenance') & W.exp_range.notna() & (W.in_snumat==1)].reset_index(drop=True)
feats = [f for f in CHEMF+STRUCF if f in q.columns]
X = q[feats].values.astype(float); y = q.exp_range.values; n = len(q)
els = [set(s.split(',')) for s in q.elem_system.fillna('?')]
cats = [e - ANIONS for e in els]

def components(link):
    par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i+1, n):
            if link(i, j): par[f(i)] = f(j)
    return np.array([f(i) for i in range(n)])

G = {'element system (original)': np.array(q.elem_system.fillna('?')),
     'cation set': np.array([','.join(sorted(c)) for c in cats]),
     'share >=2 elements (components)': components(lambda i, j: len(els[i] & els[j]) >= 2),
     'share any cation (components)': components(lambda i, j: len(cats[i] & cats[j]) >= 1)}
order = ['share any cation (components)', 'share >=2 elements (components)', 'cation set', 'element system (original)']
print('n=%d' % n)
for k in order:
    g = G[k]; _, c = np.unique(g, return_counts=True)
    print('%-34s groups=%2d largest=%2d (%.0f%%)' % (k, len(c), c.max(), 100*c.max()/n))
primary = next(k for k in order if np.unique(G[k], return_counts=True)[1].max() <= 0.25*n)
print('PRIMARY (chosen before scoring):', primary)

def gbm(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0)
def ridge(): return Ridge(alpha=3.0)
Xr = np.where(np.isnan(X), np.nanmedian(X, axis=0), X); Xr = np.where(np.isnan(Xr), 0, Xr)
Xs = StandardScaler().fit_transform(Xr)
rng = np.random.default_rng(7)

def oof_random(g, mk, Xu, seed):
    # random assignment of whole groups to 5 folds (repeated to average out fold luck)
    ug = np.unique(g); r = np.random.default_rng(seed); fold_of = dict(zip(ug, r.permutation(len(ug)) % 5))
    f = np.array([fold_of[v] for v in g]); p = np.full(n, np.nan)
    for k in range(5):
        te = f == k
        if te.sum() == 0 or (~te).sum() == 0: continue
        p[te] = mk().fit(Xu[~te], y[~te]).predict(Xu[te])
    return p

def evaluate(g, mk, Xu, reps=20):
    rho, auc = [], {t: [] for t in (0.15, 0.2, 0.25, 0.3)}
    for s in range(reps):
        p = oof_random(g, mk, Xu, s); rho.append(sps.spearmanr(p, y).statistic)
        for t in auc: auc[t].append(roc_auc_score(y > t, p))
    return np.mean(rho), {t: np.mean(v) for t, v in auc.items()}, np.array(auc[0.2])

print('\nheld-out results (mean over 20 random group-to-fold assignments)')
print('%-34s %-5s %6s  AUC@0.15 0.20 0.25 0.30' % ('grouping', 'model', 'rho'))
store = {}
for k in order:
    for mn, mk, Xu in (('gbm', gbm, Xr), ('ridge', ridge, Xs)):
        r, a, a20 = evaluate(G[k], mk, Xu); store[(k, mn)] = a20
        print('%-34s %-5s %+.3f   %.3f %.3f %.3f %.3f' % (k, mn, r, a[0.15], a[0.2], a[0.25], a[0.3]))
print('\nvar_coordination alone (no fitting): AUC@0.20 = %.3f' % roc_auc_score(y > 0.2, q.var_coordination.values))
# permutation null for the primary grouping, gbm, AUC@0.2 (one fold assignment per permutation)
obs = store[(primary, 'gbm')].mean(); nul = []
for b in range(200):
    ys = rng.permutation(y)
    # refit on permuted target
    ug = np.unique(G[primary]); r = np.random.default_rng(b); fo = dict(zip(ug, r.permutation(len(ug)) % 5))
    f = np.array([fo[v] for v in G[primary]]); pp = np.full(n, np.nan)
    for k in range(5):
        te = f == k
        if te.sum() and (~te).sum(): pp[te] = gbm().fit(Xr[~te], ys[~te]).predict(Xr[te])
    nul.append(roc_auc_score(ys > 0.2, pp))
nul = np.array(nul)
print('primary gbm AUC@0.20 = %.3f; permutation null mean %.3f, 95th pct %.3f, p = %.3f'
      % (obs, nul.mean(), np.percentile(nul, 95), (nul >= obs).mean()))
