"""Ranking of the four models by MAE of the predicted median on the compositions the 0.1 eV rule keeps vs removes,
for both holdouts and all five fold assignments (original + cv_rep). Composition-weighted, report weights within."""
import numpy as np, pandas as pd, glob
R0 = '/data/czx/GapScope_Revision_20261113/'
M = {'GS_SELCAL_PRE_NZ_SC': 'This work', 'ROOST': 'Roost', 'ROOST_CAL': 'Roost-cal'}
def per_comp(d, m):
    g = d[(d.model == m) & (d.tier == 'A')]; out = {}
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values
        out[k] = (np.sum(w * np.abs(y - gg['median'].values)) / w.sum(), round(y.max() - y.min(), 6) > 0.1 + 1e-9)
    return pd.DataFrame(out, index=['mae', 'rm']).T
files = [('comp', 'orig', R0 + 'cv/predictions_summary_v4t1.csv'), ('chem', 'orig', R0 + 'cv/predictions_summary_v4chem_NZ.csv')]
files += [(f.split('_rep_')[1][:4], f.split('cv_rep/')[1][:9], f) for f in sorted(glob.glob(R0 + 'cv_rep/s2026111*/predictions_summary_rep_*.csv'))]
rows = []
for ho, tag, f in files:
    d = pd.read_csv(f); ms = sorted(set(d.model))
    if tag == 'orig' and ho == 'comp': print('models available:', ms)
    eu = [m for m in ms if m.startswith('EASYUQ')]
    use = dict(M); use.update({eu[0]: 'EasyUQ'} if eu else {})
    P = {lab: per_comp(d, m) for m, lab in use.items() if m in ms}
    idx = P['This work'].index; rm = P['This work'].rm.astype(bool)
    for sub, sel in (('kept', ~rm), ('removed', rm), ('all', rm | ~rm)):
        r = {lab: p.loc[idx].mae[sel].astype(float).mean() for lab, p in P.items()}
        rows.append(dict(holdout=ho, assign=tag, subset=sub, n=int(sel.sum()), **r))
T = pd.DataFrame(rows)
labs = [c for c in ('This work', 'Roost', 'Roost-cal', 'EasyUQ') if c in T]
T['first'] = T[labs].idxmin(axis=1)
T['rank_this'] = T[labs].rank(axis=1)['This work']
pd.set_option('display.width', 200)
print(T.round(3).to_string(index=False))
print('\nwins of This work (rank 1) out of assignments; mean rank; mean MAE')
for (ho, sub), g in T.groupby(['holdout', 'subset']):
    print('%-5s %-8s first %d/%d | mean rank %.1f | %s | first-place counts %s' % (ho, sub, (g['first'] == 'This work').sum(), len(g), g.rank_this.mean(),
          ' '.join('%s %.3f' % (l, g[l].mean()) for l in labs), g['first'].value_counts().to_dict()))
T.to_csv(R0 + 'refcheck/c9_rank.csv', index=False)
