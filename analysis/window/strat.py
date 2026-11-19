import numpy as np, pandas as pd
T = pd.read_csv('/data/czx/GapScope_Revision_20261113/window/par2/window_units_all.csv')
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
def strata(t):
    return {'all': t.index == t.index,
            'single report': t.n_reports == 1,
            'reports agree (<0.1 eV)': (t.n_reports >= 2) & (t.spread < 0.1),
            'reports disagree (>=0.3 eV)': (t.n_reports >= 2) & (t.spread >= 0.3),
            'straddle window': None}
rng = np.random.default_rng(3); out = []
for V in ('comp', 'chem'):
    for wn in W:
        P = {m: T[(T.holdout == V) & (T.model == m)].set_index(['tier', 'key']) for m in ('GapScope', 'ROOST_CAL', 'EASYUQ_KS')}
        g = P['GapScope']; idx = g.index
        for tier in ('A', 'B'):
            gt = g.loc[tier]
            for sn, mask in strata(gt).items():
                if mask is None: mask = (gt['frac_' + wn] > 0) & (gt['frac_' + wn] < 1)
                keys = gt.index[np.asarray(mask)]
                if len(keys) < 5: continue
                bp = {m: P[m].loc[tier].loc[keys, 'bp_' + wn].values for m in P}
                bm = P['GapScope'].loc[tier].loc[keys, 'bm_' + wn].values
                I = rng.integers(0, len(keys), (2000, len(keys)))
                r = dict(holdout=V, window=wn, tier=tier, stratum=sn, n=len(keys), median_call=bm.mean(),
                         GapScope=bp['GapScope'].mean(), RoostCal=bp['ROOST_CAL'].mean(), EasyUQ=bp['EASYUQ_KS'].mean())
                for ref, rn in (('ROOST_CAL', 'vsRC'), ('EASYUQ_KS', 'vsEQ')):
                    d = bp['GapScope'] - bp[ref]; bs = d[I].mean(1)
                    r[rn] = d.mean(); r[rn + '_lo'] = np.percentile(bs, 2.5); r[rn + '_hi'] = np.percentile(bs, 97.5)
                out.append(r)
O = pd.DataFrame(out).round(4); O.to_csv('/data/czx/GapScope_Revision_20261113/window/par2/brier_strata.csv', index=False)
pd.set_option('display.width', 250); print(O.to_string(index=False))
# reliability: pooled comp/chem, PV/VIS, 10 bins of P, weighted outcome = frac
for V in ('comp', 'chem'):
    for m in ('GapScope', 'ROOST_CAL', 'EASYUQ_KS'):
        t = T[(T.holdout == V) & (T.model == m)]
        s = []
        for wn in W:
            p, o = t['P_' + wn].values, t['frac_' + wn].values
            b = np.clip((p * 10).astype(int), 0, 9); rel = res = 0; ob = o.mean()
            for k in range(10):
                q = b == k
                if q.any(): rel += q.mean() * (p[q].mean() - o[q].mean()) ** 2; res += q.mean() * (o[q].mean() - ob) ** 2
            s.append('%s REL %.4f RES %.4f UNC %.4f' % (wn, rel, res, ob * (1 - ob)))
        print(V, m, ' | '.join(s))
