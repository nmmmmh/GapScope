import numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
exec(open(R + 'window/rep_summary.py').read().split('rows = []')[0].split("F = {")[0])
def rel(p, o):
    b = np.clip((p * 10).astype(int), 0, 9); r = 0
    for k in range(10):
        q = b == k
        if q.any(): r += q.mean() * (p[q].mean() - o[q].mean()) ** 2
    return r
rows = []
for s in ('20261113', '20261114', '20261115', '20261116', '20261117'):
    T = pd.read_csv(R + 'cv_rep/s%s/window/window_units_rcs.csv' % s)
    for V, t in T.groupby('holdout'):
        for wn in ('PV', 'VIS'):
            p, o = t['P_' + wn].values, t['frac_' + wn].values; q = p >= 0.5
            rows.append(dict(partition=s, holdout=V, model='ROOST_CAL_CV', window=wn, REL=rel(p, o), Brier=t['bp_' + wn].mean(), n=int(q.sum()), promised=p[q].mean(), delivered=o[q].mean()))
X = pd.DataFrame(rows); X.to_csv(R + 'window/par2/window_partitions_rcs.csv', index=False)
print(X.round(4).to_string(index=False))
print(X.groupby(['holdout', 'window'])[['REL', 'Brier', 'promised', 'delivered']].agg(['mean', 'std']).round(4).to_string())
P = pd.read_csv(R + 'window/par2/window_partitions.csv')
print(P[P.holdout == 'chem'].groupby(['model', 'window'])[['REL', 'promised_0.5', 'delivered_0.5']].agg(['mean', 'std']).round(4).to_string())
