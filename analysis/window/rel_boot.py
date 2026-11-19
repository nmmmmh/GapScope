import numpy as np, pandas as pd
T = pd.read_csv('/data/czx/GapScope_Revision_20261113/window/par2/window_units_all.csv')
W = ('PV', 'VIS'); rng = np.random.default_rng(11)
def rel(p, o):
    b = np.clip((p * 10).astype(int), 0, 9); r = 0
    for k in range(10):
        q = b == k
        if q.any(): r += q.mean() * (p[q].mean() - o[q].mean()) ** 2
    return r
rows = []
for V in ('comp', 'chem'):
    t = {m: T[(T.holdout == V) & (T.model == m)].sort_values(['tier', 'key']).reset_index(drop=True) for m in ('GapScope', 'ROOST_CAL', 'EASYUQ_KS')}
    n = len(t['GapScope']); I = rng.integers(0, n, (1000, n))
    for wn in W:
        R = {m: rel(t[m]['P_' + wn].values, t[m]['frac_' + wn].values) for m in t}
        bs = {m: np.array([rel(t[m]['P_' + wn].values[i], t[m]['frac_' + wn].values[i]) for i in I]) for m in t}
        # reliability curve for figure
        for m in t:
            p, o = t[m]['P_' + wn].values, t[m]['frac_' + wn].values; b = np.clip((p * 10).astype(int), 0, 9)
            curve = ' '.join('%.2f>%.2f(%d)' % (p[b == k].mean(), o[b == k].mean(), (b == k).sum()) for k in range(10) if (b == k).any())
            print(V, wn, m, 'REL %.4f [%.4f, %.4f]' % (R[m], *np.percentile(bs[m], [2.5, 97.5])), '| curve', curve)
        for ref in ('ROOST_CAL', 'EASYUQ_KS'):
            d = bs['GapScope'] - bs[ref]; print('   dREL vs', ref, '%.4f [%.4f, %.4f]' % (R['GapScope'] - R[ref], *np.percentile(d, [2.5, 97.5])))
