import numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
F = {'20261113': R + 'window/par2/window_units_all.csv'}
for s in ('20261114', '20261115', '20261116', '20261117'): F[s] = R + 'cv_rep/s%s/window/window_units_all.csv' % s
W = ('PV', 'VIS')


def rel(p, o):
    b = np.clip((p * 10).astype(int), 0, 9); r = 0
    for k in range(10):
        q = b == k
        if q.any(): r += q.mean() * (p[q].mean() - o[q].mean()) ** 2
    return r


rows = []
for s, f in F.items():
    T = pd.read_csv(f)
    for (V, m), t in T.groupby(['holdout', 'model']):
        for wn in W:
            p, o = t['P_' + wn].values, t['frac_' + wn].values
            r = dict(partition=s, holdout=V, model=m, window=wn, REL=rel(p, o), Brier=t['bp_' + wn].mean())
            for thr in (0.5, 0.7):
                q = p >= thr
                r['n_%g' % thr] = int(q.sum()); r['promised_%g' % thr] = p[q].mean() if q.any() else np.nan; r['delivered_%g' % thr] = o[q].mean() if q.any() else np.nan
                r['gap_%g' % thr] = r['promised_%g' % thr] - r['delivered_%g' % thr]
            rows.append(r)
P = pd.DataFrame(rows); P.to_csv(R + 'window/par2/window_partitions.csv', index=False)
num = [c for c in P.columns if c not in ('partition', 'holdout', 'model', 'window')]
M = P.groupby(['holdout', 'window', 'model'])[num].agg(['mean', 'std']).round(4)
M.columns = ['%s_%s' % c for c in M.columns]; M.to_csv(R + 'window/par2/window_partitions_mean_sd.csv')
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40)
print(M[[c for c in M.columns if c.split('_')[0] in ('REL', 'Brier', 'n', 'promised', 'delivered', 'gap')]].to_string())
print(P[(P.holdout == 'chem')][['partition', 'model', 'window', 'REL', 'promised_0.7', 'delivered_0.7', 'n_0.7']].round(3).to_string(index=False))
