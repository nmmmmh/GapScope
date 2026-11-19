"""Paired composition bootstrap: this work minus Roost (and Roost-cal) MAE on removed (0.1 eV rule) vs kept compositions,
composition holdout (v4t1) and chemical-family holdout (v4chem_NZ). Also the five fold assignments (cv_rep) if present."""
import numpy as np, pandas as pd, glob
R0 = '/data/czx/GapScope_Revision_20261113/'
def per_comp(d, m):
    g = d[(d.model == m) & (d.tier == 'A')]; out = {}
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values
        out[k] = (np.sum(w * np.abs(y - gg['median'].values)) / w.sum(), round(y.max() - y.min(), 6) > 0.1 + 1e-9)
    return pd.DataFrame(out, index=['mae', 'rm']).T
def report(tag, d):
    G = per_comp(d, 'GS_SELCAL_PRE_NZ_SC')
    for ref in ('ROOST', 'ROOST_CAL'):
        Rf = per_comp(d, ref).loc[G.index]; diff = (G.mae - Rf.mae).astype(float); rm = G.rm.astype(bool)
        rng = np.random.default_rng(7)
        for lab, sel in (('removed', rm), ('kept', ~rm), ('all', rm | ~rm)):
            v = diff[sel].values; b = [rng.choice(v, len(v)).mean() for _ in range(4000)]
            print('%-28s %-9s %-8s n %2d  GS %.3f ref %.3f diff %+.3f [%+.3f, %+.3f]' % (tag, ref, lab, sel.sum(), G.mae[sel].mean(), Rf.mae[sel].mean(), v.mean(), *np.percentile(b, [2.5, 97.5])))
report('comp (original)', pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv'))
report('chem (original)', pd.read_csv(R0 + 'cv/predictions_summary_v4chem_NZ.csv'))
for f in sorted(glob.glob(R0 + 'cv_rep/s2026111*/predictions_summary_rep_*.csv')):
    d = pd.read_csv(f)
    if 'GS_SELCAL_PRE_NZ_SC' in set(d.model) and 'ROOST' in set(d.model):
        G = per_comp(d, 'GS_SELCAL_PRE_NZ_SC'); Rf = per_comp(d, 'ROOST').loc[G.index]; rm = G.rm.astype(bool)
        print('%-60s removed GS %.3f Roost %.3f | kept GS %.3f Roost %.3f' % (f.split('cv_rep/')[1], G.mae[rm].mean(), Rf.mae[rm].mean(), G.mae[~rm].mean(), Rf.mae[~rm].mean()))
