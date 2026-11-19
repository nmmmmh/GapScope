"""Paired composition-cluster bootstrap for the model differences in the main text and Table S11, aggregated exactly as evaluate_cv (tier A per composition, pooled = A compositions + B units, equal weight)."""
import numpy as np, pandas as pd
C = '/data/czx/GapScope_Revision_20261113/cv/'
SRC = {('comp', 'GS_SELCAL_PRE_NZ_SC'): 'predictions_summary_v4t1.csv', ('chem', 'GS_SELCAL_PRE_NZ_SC'): 'predictions_summary_v6chem.csv',
       ('comp', 'ROOST_CAL'): 'predictions_summary_v4a.csv', ('chem', 'ROOST_CAL'): 'predictions_summary_v4eqchem.csv',
       ('comp', 'EASYUQ_KS'): 'predictions_summary_v4a.csv', ('chem', 'EASYUQ_KS'): 'predictions_summary_v4eqchem.csv',
       ('comp', 'ROOST'): 'predictions_summary_v4a.csv', ('chem', 'ROOST'): 'predictions_summary_v4eqchem.csv'}
M = ('crps', 'mae', 'cov80', 'width80')
def units(fold, m, tier):
    d = pd.read_csv(C + SRC[(fold, m)]); d = d[(d.model == m) & (d.variant == fold)]
    a = d[d.tier == 'A'].copy(); a['wn'] = a.w / a.groupby('key').w.transform('sum')
    A = a.groupby('key').apply(lambda g: pd.Series({k: (g[k] * g.wn).sum() for k in M})).reset_index(); A['tier'] = 'A'
    B = d[d.tier == 'B'][['key'] + list(M)].copy(); B['tier'] = 'B'
    U = A if tier == 'A' else B if tier == 'B' else pd.concat([A, B])
    return U.sort_values(['tier', 'key']).reset_index(drop=True)
rng = np.random.default_rng(20261113); out = []
def comp(fold, a, b, tier, metric, nb=4000):
    ua, ub = units(fold, a, tier), units(fold, b, tier)
    assert (ua.key.values == ub.key.values).all()
    uk, inv = np.unique(ua.key.values, return_inverse=True)
    n = np.bincount(inv).astype(float); sa = np.bincount(inv, ua[metric].values); sb = np.bincount(inv, ub[metric].values)
    cnt = np.stack([np.bincount(r, minlength=len(uk)) for r in rng.integers(0, len(uk), (nb, len(uk)))])
    diff = (cnt @ sa - cnt @ sb) / (cnt @ n)
    out.append(dict(fold=fold, a=a, b=b, tier=tier, metric=metric, n_units=len(ua), a_val=ua[metric].mean(), b_val=ub[metric].mean(),
                    diff=ua[metric].mean() - ub[metric].mean(), lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
for fold in ('comp', 'chem'):
    for b in ('ROOST_CAL', 'EASYUQ_KS', 'ROOST'):
        for metric in ('crps', 'cov80', 'width80'): comp(fold, 'GS_SELCAL_PRE_NZ_SC', b, 'pooled', metric)
    for b in ('ROOST', 'ROOST_CAL'): comp(fold, 'GS_SELCAL_PRE_NZ_SC', b, 'A', 'mae')
T = pd.DataFrame(out); T.to_csv(C + 'paired_bootstrap_final.csv', index=False)
pd.set_option('display.width', 220); print(T.round(4).to_string(index=False))
