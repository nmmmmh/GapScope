import numpy as np, pandas as pd
exec(open('c10_rev.py').read().split("D = pd.read_csv")[0].split("print('all floor")[0])
P['am'] = P.ph.str.contains('amorph')
am = set(P[P.am].composition_key)
T['am'] = T.k.isin(am)
M = T[T.nph >= 2]
print('multi all %.3f (n %d) | without amorphous comps %.3f (n %d) | amorphous comps %.3f (n %d)' % (M.F.mean(), len(M), M[~M.am].F.mean(), (~M.am).sum(), M[M.am].F.mean(), M.am.sum()))
print('share of total floor from amorphous comps %.1f%%' % (100 * T[T.am].F.sum() / T.F.sum()))
# floor recomputed dropping amorphous reports
P2 = P[~P.am]; rows = []
for k, g in P2.groupby('composition_key'): rows.append((k, wmed_floor(g), g[g.ph != ''].ph.nunique()))
T2 = pd.DataFrame(rows, columns=['k', 'F', 'nph'])
print('dropping amorphous reports: all %.3f | multi %.3f (n %d)' % (T2.F.mean(), T2[T2.nph >= 2].F.mean(), (T2.nph >= 2).sum()))
