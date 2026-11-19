"""Borlido et al. 2019 benchmark: are functional errors larger for compositions with a large polymorph gap spread D_c?
Pre-specified: primary = |MBJ - exp| and |HSE06 - exp|, D_c >= 0.3 eV vs rest (in SNUMAT), one-sided Mann-Whitney,
Spearman rho(D_c, |err|) and partial rho controlling the experimental gap.
Secondary = best-functional error (min over 12 non-SOC functionals) and the shared residual
(mean over functionals of the residual after a linear fit of each functional to experiment).
Run on c137-13: OMP_NUM_THREADS=1 <venv>/python -I b3_errors.py"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, rankdata
R = '/data/czx/GapScope_Revision_20261113/borlido/'
raw = pd.read_csv(R + 'raw/sheet1.csv', header=None, encoding='utf-8-sig')
cols = ['Material', 'MP_ID', 'ICSD_ID'] + [str(c) for c in raw.iloc[1, 3:19]]
B = raw.iloc[2:, :19].copy(); B.columns = cols
B = B[B.Material.notna()].reset_index(drop=True)
FUN = ['LDA', 'PBE', 'PBE_SOL', 'HLE16', 'BJ', 'MBJ', 'SCAN', 'HSE06', 'HSE14', 'HSE06_mix', 'PBE0', 'PBE0_mix']
for c in FUN + ['Experimental']: B[c] = pd.to_numeric(B[c], errors='coerce')
D = pd.read_csv(R + 'b1_borlido_Dc.csv').drop_duplicates('formula').set_index('formula')
B['D_c'] = B.Material.map(D.D_c); B['in_snumat'] = B.Material.map(D.in_snumat).fillna(False).astype(bool)
B['n_acc'] = B.Material.map(D.n_acc)
print('entries', len(B), 'distinct materials', B.Material.nunique(), 'unmatched names', B.D_c.isna().sum() - (~B.in_snumat).sum())
E = B.Experimental
for f in FUN: B['ae_' + f] = (B[f] - E).abs()
B['ae_best'] = B[['ae_' + f for f in FUN]].min(axis=1)
res = []
for f in FUN:
    ok = B[f].notna() & E.notna(); a, b = np.polyfit(E[ok], B.loc[ok, f], 1)
    res.append((B[f] - (a * E + b)) / a)          # residual expressed on the experimental scale
B['shared'] = pd.concat(res, axis=1).mean(axis=1, skipna=True).abs()

def partial_rho(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]

M = B[B.in_snumat & B.Experimental.notna()].copy()
hi = M.D_c >= 0.3
print('matched entries', len(M), 'D_c>=0.3:', hi.sum(), 'rest:', (~hi).sum())
rows = []
for m in ['ae_MBJ', 'ae_HSE06', 'ae_best', 'shared', 'ae_PBE', 'ae_SCAN', 'ae_HLE16']:
    q = M[M[m].notna()]; h = q.D_c >= 0.3
    p = mannwhitneyu(q.loc[h, m], q.loc[~h, m], alternative='greater').pvalue
    r = spearmanr(q.D_c, q[m]).statistic; pr = partial_rho(q.D_c.values, q[m].values, q.Experimental.values)
    rows.append(dict(metric=m, n_hi=int(h.sum()), n_lo=int((~h).sum()), mean_hi=q.loc[h, m].mean(), mean_lo=q.loc[~h, m].mean(),
                     med_hi=q.loc[h, m].median(), med_lo=q.loc[~h, m].median(), P_MW_greater=p, rho=r, partial_rho_expgap=pr))
T = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(T.round(4).to_string(index=False))
# composition-level check (duplicate entries averaged)
C = M.groupby('Material').agg(D_c=('D_c', 'first'), ae_MBJ=('ae_MBJ', 'mean'), ae_HSE06=('ae_HSE06', 'mean'), ae_best=('ae_best', 'mean'),
                             shared=('shared', 'mean'), Experimental=('Experimental', 'mean'))
for m in ['ae_MBJ', 'ae_HSE06', 'ae_best', 'shared']:
    q = C[C[m].notna()]; h = q.D_c >= 0.3
    print('composition-level %-9s n %d/%d mean %.3f/%.3f P %.4f rho %.3f' % (m, h.sum(), (~h).sum(), q.loc[h, m].mean(), q.loc[~h, m].mean(),
          mannwhitneyu(q.loc[h, m], q.loc[~h, m], alternative='greater').pvalue, spearmanr(q.D_c, q[m]).statistic))
# among compositions with >=2 accessible polymorphs only (D_c defined beyond 0)
q = M[M.n_acc >= 2]; h = q.D_c >= 0.3
print('>=2 accessible: n %d/%d MBJ mean %.3f/%.3f P %.4f | HSE06 %.3f/%.3f P %.4f' % (h.sum(), (~h).sum(), q.loc[h, 'ae_MBJ'].mean(), q.loc[~h, 'ae_MBJ'].mean(),
      mannwhitneyu(q.loc[h, 'ae_MBJ'].dropna(), q.loc[~h, 'ae_MBJ'].dropna(), alternative='greater').pvalue,
      q.loc[h, 'ae_HSE06'].mean(), q.loc[~h, 'ae_HSE06'].mean(),
      mannwhitneyu(q.loc[h, 'ae_HSE06'].dropna(), q.loc[~h, 'ae_HSE06'].dropna(), alternative='greater').pvalue))
print('largest shared residuals among D_c>=0.3:')
print(M[hi].sort_values('shared', ascending=False)[['Material', 'D_c', 'Experimental', 'MBJ', 'HSE06', 'shared']].head(12).round(2).to_string(index=False))
T.to_csv(R + 'b3_errors.csv', index=False); M.to_csv(R + 'b3_entries.csv', index=False)
