"""Why H3 (shorter bond -> smaller gap) comes out with the opposite sign: at fixed composition
the mean bond length is not independent of the coordination number.  Partial effects and the
correlation structure of the six descriptors.  Also the dimensionality-change census.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sps

D = Path('/data/czx/GapScope_Revision_20261113/struct')
p = pd.read_csv(D / 'pairs.csv')
y = p.d_Band_gap_HSE.values
w = p.weight.values
six = ['d_dim_cat_an', 'd_mean_shared_vertices', 'd_bond_cat_an_mean', 'd_cn_cat_mean',
       'd_distortion_index', 'd_vol_per_atom', 'd_packing_fraction', 'd_frac_face', 'd_frac_corner']
X = p[six].values.astype(float)
ok = np.isfinite(X).all(1) & np.isfinite(y)
print('pairs with all six descriptors', int(ok.sum()), 'of', len(p))
C = pd.DataFrame(np.corrcoef(X[ok].T), index=six, columns=six).round(3)
print('\ncorrelation of the descriptor differences (symmetric set, both orientations identical):')
print(C.to_string())
print('\ncorrelation of each with d_gap_HSE:')
print(pd.Series({f: sps.pearsonr(X[ok][:, i], y[ok]).statistic for i, f in enumerate(six)}).round(3).to_string())

# partial slope of the gap difference on the bond-length difference, controlling the CN difference
def oslope(x, yy, ww):
    return float(np.sum(ww * x * yy) / np.sum(ww * x * x))

xb = X[ok][:, six.index('d_bond_cat_an_mean')]
xc = X[ok][:, six.index('d_cn_cat_mean')]
ww = w[ok]
yy = y[ok]
print('\nraw slope  d_gap / d_bond  = %+.3f eV/A' % oslope(xb, yy, ww))
print('raw slope  d_gap / d_cn    = %+.3f eV per CN unit' % oslope(xc, yy, ww))
print('corr(d_bond, d_cn) = %+.3f' % sps.pearsonr(xb, xc).statistic)
# two-feature no-intercept weighted least squares
A = np.vstack([xb, xc]).T
Wm = np.diag(ww)
beta = np.linalg.solve(A.T @ Wm @ A, A.T @ Wm @ yy)
print('joint fit: d_gap = %+.3f * d_bond %+.3f * d_cn' % (beta[0], beta[1]))
# bond-length effect inside bins of (nearly) equal coordination number
m = np.abs(xc) < 0.1
print('pairs with |d_cn_cat_mean| < 0.1: n=%d, slope d_gap/d_bond = %+.3f eV/A (corr %+.3f)'
      % (m.sum(), oslope(xb[m], yy[m], ww[m]), sps.pearsonr(xb[m], yy[m]).statistic))
m2 = np.abs(xb) < 0.02
print('pairs with |d_bond| < 0.02 A:    n=%d, slope d_gap/d_cn   = %+.3f eV (corr %+.3f)'
      % (m2.sum(), oslope(xc[m2], yy[m2], ww[m2]), sps.pearsonr(xc[m2], yy[m2]).statistic))

# dimensionality census
print('\ndimensionality of the cation-anion network, pair members:')
print('pairs with equal dim_cat_an: %.3f' % float((p.d_dim_cat_an == 0).mean()))
print(p.d_dim_cat_an.value_counts(dropna=False).to_string())
print('\nmean |d_gap| by |d_dim|:')
print(p.assign(ad=p.d_dim_cat_an.abs()).groupby('ad').d_Band_gap_HSE.agg(
    ['size', lambda s: s.abs().mean(), 'mean']).round(3).to_string())
print('\nmean |d_gap| when nothing structural changes much (|d_cn|<0.1, |d_bond|<0.02, d_dim=0): ')
q = (p.d_cn_cat_mean.abs() < 0.1) & (p.d_bond_cat_an_mean.abs() < 0.02) & (p.d_dim_cat_an == 0)
print('n=%d mean|d_gap|=%.3f eV  median=%.3f' % (q.sum(), p.d_Band_gap_HSE[q].abs().mean(),
                                                 p.d_Band_gap_HSE[q].abs().median()))
