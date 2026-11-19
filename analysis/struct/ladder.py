"""Within-study ladder with composition-clustered bootstrap (review round 1, attacks 3 and 13)."""
import numpy as np, pandas as pd, itertools
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
rows = []
for c, g in S.groupby('composition_key'):
    r = g.to_dict('records')
    for a, b in itertools.combinations(r, 2):
        if not a['ph'] or not b['ph']:
            continue
        ss = a['source_group_id'] == b['source_group_id']; sp = a['specimen_group_id'] == b['specimen_group_id']; sph = a['ph'] == b['ph']
        k = ('repeat' if ss and sp and sph else 'specimen' if ss and sph else 'phase_same_study' if ss and not sph
             else 'phase_cross_study' if not sph else 'same_phase_cross_study')
        rows.append(dict(comp=c, kind=k, d=abs(a['bandgap_ev'] - b['bandgap_ev'])))
P = pd.DataFrame(rows); P.to_csv('/data/czx/GapScope_Revision_20261113/struct/tables/ladder_pairs.csv', index=False)
rng = np.random.default_rng(20261114)
comps = P.comp.unique(); by = {c: P[P.comp == c] for c in comps}
order = ['repeat', 'specimen', 'phase_same_study', 'phase_cross_study', 'same_phase_cross_study']
boots = {k: [] for k in order}
for _ in range(4000):
    s = pd.concat([by[c] for c in rng.choice(comps, len(comps))])
    for k in order:
        v = s[s.kind == k].d
        if len(v): boots[k].append(v.mean())
print('kind                    pairs comps  mean   [95% CI, compositions resampled]  median')
for k in order:
    v = P[P.kind == k]
    print('%-22s %5d %5d  %.3f  [%.3f, %.3f]  %.3f' % (k, len(v), v.comp.nunique(), v.d.mean(),
          np.percentile(boots[k], 2.5), np.percentile(boots[k], 97.5), v.d.median()))
# composition-matched: compositions with both same-study and cross-study phase pairs; per-composition means
m = P[P.kind.isin(['phase_same_study', 'phase_cross_study'])].groupby(['comp', 'kind']).d.mean().unstack().dropna()
print('\ncomposition-matched (n=%d comps): same-study %.3f, cross-study %.3f (per-composition means, then averaged)'
      % (len(m), m.phase_same_study.mean(), m.phase_cross_study.mean()))
print('compositions with same-study phase pairs: %d; multi-phase compositions: %d'
      % (P[P.kind == 'phase_same_study'].comp.nunique(), S[S.ph != ''].groupby('composition_key').ph.nunique().ge(2).sum()))
