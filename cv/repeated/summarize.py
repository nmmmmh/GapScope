"""Partition summary: aggregation exactly as cv/paired_bootstrap_final.py (tier A per composition with hierarchical weights normalised
within composition; pooled = A compositions + B units, equal weight). Reads cv_rep/s{seed}/predictions_summary_rep_{holdout}.csv;
for the original seed also checks against the published predictions_summary files."""
import os, glob, numpy as np, pandas as pd
Q = '/data/czx/GapScope_Revision_20261113/cv_rep/'; C = '/data/czx/GapScope_Revision_20261113/cv/'
PUB = {('comp', 'GS_SELCAL_PRE_NZ_SC'): 'predictions_summary_v4t1.csv', ('chem', 'GS_SELCAL_PRE_NZ_SC'): 'predictions_summary_v6chem.csv',
       ('comp', 'ROOST_CAL'): 'predictions_summary_v4a.csv', ('chem', 'ROOST_CAL'): 'predictions_summary_v4eqchem.csv',
       ('comp', 'EASYUQ_KS'): 'predictions_summary_v4a.csv', ('chem', 'EASYUQ_KS'): 'predictions_summary_v4eqchem.csv',
       ('comp', 'ROOST'): 'predictions_summary_v4a.csv', ('chem', 'ROOST'): 'predictions_summary_v4eqchem.csv'}
MODELS = ('GS_SELCAL_PRE_NZ_SC', 'ROOST_CAL', 'EASYUQ_KS', 'ROOST'); M = ('crps', 'mae', 'cov80', 'width80')
NAME = {'GS_SELCAL_PRE_NZ_SC': 'this_work', 'ROOST_CAL': 'roost_cal', 'EASYUQ_KS': 'easyuq_ks', 'ROOST': 'roost'}
def units(d, hold, m):
    d = d[(d.model == m) & (d.variant == hold)]
    a = d[d.tier == 'A'].copy(); a['wn'] = a.w / a.groupby('key').w.transform('sum')
    A = a.groupby('key').apply(lambda g: pd.Series({k: (g[k] * g.wn).sum() for k in M})).reset_index(); A['tier'] = 'A'
    B = d[d.tier == 'B'][['key'] + list(M)].copy(); B['tier'] = 'B'
    return pd.concat([A, B]).sort_values(['tier', 'key']).reset_index(drop=True)
rows, prs, chk = [], [], []
for p in sorted(glob.glob(Q + 's2026111*')):
    seed = int(p.split('/s')[-1])
    for hold in ('comp', 'chem'):
        f = f'{p}/predictions_summary_rep_{hold}.csv'
        if not os.path.exists(f): continue
        d = pd.read_csv(f); U = {}
        for m in MODELS:
            if not (d.model == m).any(): continue
            U[m] = u = units(d, hold, m); a = u[u.tier == 'A']
            rows.append(dict(partition_seed=seed, holdout=hold, model=NAME[m], n_units=len(u), n_A=len(a), crps=u.crps.mean(), cov80=u.cov80.mean(), width80=u.width80.mean(),
                             mae_pooled=u.mae.mean(), mae_tierA=a.mae.mean(), crps_tierA=a.crps.mean()))
            if seed == 20261113:
                up = units(pd.read_csv(C + PUB[(hold, m)]), hold, m)
                chk.append(dict(holdout=hold, model=NAME[m], **{f'{k}_rep': u[k].mean() for k in ('crps', 'cov80', 'width80')}, **{f'{k}_pub': up[k].mean() for k in ('crps', 'cov80', 'width80')},
                                max_abs_unit_crps_diff=float(np.abs(u.crps.values - up.crps.values).max()) if len(u) == len(up) else np.nan))
        g = 'GS_SELCAL_PRE_NZ_SC'
        if g not in U: continue
        rng = np.random.default_rng(20261113)
        for r in ('ROOST_CAL', 'EASYUQ_KS', 'ROOST'):
            if r not in U: continue
            ua, ub = U[g], U[r]; assert (ua.key.values == ub.key.values).all() and (ua.tier.values == ub.tier.values).all()
            uk, inv = np.unique(ua.key.values, return_inverse=True); n = np.bincount(inv).astype(float)
            cnt = np.stack([np.bincount(x, minlength=len(uk)) for x in rng.integers(0, len(uk), (4000, len(uk)))])
            rec = dict(partition_seed=seed, holdout=hold, reference=NAME[r])
            for k in ('crps', 'cov80', 'width80'):
                sa, sb = np.bincount(inv, ua[k].values), np.bincount(inv, ub[k].values); dd = (cnt @ sa - cnt @ sb) / (cnt @ n)
                rec[f'd_{k}'] = ua[k].mean() - ub[k].mean(); rec[f'd_{k}_lo'] = np.percentile(dd, 2.5); rec[f'd_{k}_hi'] = np.percentile(dd, 97.5)
            prs.append(rec)
R = pd.DataFrame(rows); R.to_csv(Q + 'summary_partitions.csv', index=False)
P = pd.DataFrame(prs); P.to_csv(Q + 'summary_paired_partitions.csv', index=False)
ms = []
for (h, m), g in R.groupby(['holdout', 'model']):
    ms.append(dict(holdout=h, model=m, n_partitions=len(g), **{f'{k}_{s}': getattr(g[k], s)() for k in ('crps', 'cov80', 'width80', 'mae_pooled', 'mae_tierA') for s in ('mean', 'std')}))
for (h, r), g in P.groupby(['holdout', 'reference']):
    ms.append(dict(holdout=h, model=f'this_work_minus_{r}', n_partitions=len(g), **{f'{k}_{s}': getattr(g['d_' + k], s)() for k in ('crps', 'cov80', 'width80') for s in ('mean', 'std')}))
pd.DataFrame(ms).to_csv(Q + 'summary_mean_sd.csv', index=False)
if chk: pd.DataFrame(chk).to_csv(Q + 'check_original_partition.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(chk).round(4).to_string(index=False) if chk else 'no check')
print(pd.DataFrame(ms).round(4).to_string(index=False))
print(P[['partition_seed', 'holdout', 'reference', 'd_crps', 'd_crps_lo', 'd_crps_hi', 'd_cov80', 'd_width80']].round(4).to_string(index=False))
