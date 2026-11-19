"""Gain from knowing the phase (err_blind - err_matched) for the 68 reports of compositions with >=2 computed polymorphs,
split by the composition's D_c (energy-only, 0.1 eV/atom, main-text definition) and by the spread of all its polymorph gaps.
Bootstrap over compositions, 4000 draws."""
import numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
M = pd.read_csv(R + 'struct/bench_split.csv'); M = M[M.n_poly >= 2].copy()
M['gain'] = M.err_blind - M.err_matched
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def dc(c, need_e=True, W=0.10):
    c = c[c.hse.notna()]
    if need_e: c = c[c.Free_energy.notna()]
    if W is not None and c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= W)]
    return (c.hse.max() - c.hse.min()) if len(c) >= 2 else 0.0
G = {k: (dc(c), dc(c, False, None)) for k, c in S.groupby('ckey')}
M['Dc'] = [G.get(k, (np.nan, np.nan))[0] for k in M.key]; M['spread_all'] = [G.get(k, (np.nan, np.nan))[1] for k in M.key]
rng = np.random.default_rng(20261118); ks = M.key.unique()
def boot(sub):
    kk = sub.key.unique(); b = []
    for _ in range(4000):
        s = rng.choice(kk, len(kk)); b.append(np.concatenate([sub.gain[sub.key == k].values for k in s]).mean())
    return np.percentile(b, [2.5, 97.5])
print('all: n %d reports, %d compositions, gain %.3f, err_matched %.3f, err_blind %.3f' % (len(M), M.key.nunique(), M.gain.mean(), M.err_matched.mean(), M.err_blind.mean()))
for lab, col, thr in (('D_c energy-only', 'Dc', 0.3), ('spread of all polymorph gaps', 'spread_all', 0.3), ('spread of all polymorph gaps', 'spread_all', 1.0)):
    for name, sub in (('>= %.1f' % thr, M[M[col] >= thr]), ('<  %.1f' % thr, M[M[col] < thr])):
        if sub.key.nunique() < 3: print(lab, name, 'too few', len(sub)); continue
        lo, hi = boot(sub)
        print('%-30s %-6s n %2d (%2d comp) | gain %.3f [%.3f, %.3f] | matched %.3f blind %.3f' % (lab, name, len(sub), sub.key.nunique(), sub.gain.mean(), lo, hi, sub.err_matched.mean(), sub.err_blind.mean()))
from scipy.stats import spearmanr
print('rho(gain, spread_all) %.2f P %.2g | rho(gain, Dc) %.2f P %.2g' % (*spearmanr(M.gain, M.spread_all), *spearmanr(M.gain, M.Dc)))
print(M.groupby('formula').agg(n=('gain', 'size'), gain=('gain', 'mean'), Dc=('Dc', 'first'), spread=('spread_all', 'first')).sort_values('spread', ascending=False).round(2).head(12).to_string())
