"""Coverage of the central 80% interval on the traced reports, split by multi-phase vs single-phase composition,
composition and chemical-family holdout; composition-weighted (each composition equal weight, reports by w)."""
import numpy as np, pandas as pd, glob
R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
nph = S[S.ph != ''].groupby('composition_key').ph.nunique(); multi = set(nph[nph >= 2].index)
def cov(d, m):
    g = d[(d.model == m) & (d.tier == 'A')]
    c = g.groupby('key').apply(lambda q: np.sum(q.w * q.cov80) / q.w.sum(), include_groups=False)
    wdt = g.groupby('key').apply(lambda q: np.sum(q.w * q.width80) / q.w.sum(), include_groups=False)
    mp = c.index.isin(multi)
    return c[mp].mean(), c[~mp].mean(), c.mean(), wdt[mp].mean(), wdt[~mp].mean()
files = [('comp', R0 + 'cv/predictions_summary_v4t1.csv'), ('chem', R0 + 'cv/predictions_summary_v4chem_NZ.csv')]
files += [(('comp' if 'comp' in f else 'chem') + ' ' + f.split('cv_rep/')[1][:9], f) for f in sorted(glob.glob(R0 + 'cv_rep/s2026111*/predictions_summary_rep_*.csv'))]
for tag, f in files:
    d = pd.read_csv(f)
    for m in ('ROOST_CAL', 'EASYUQ_KS', 'GS_SELCAL_PRE_NZ_SC'):
        if m in set(d.model):
            a, b, c, wa, wb = cov(d, m)
            print('%-15s %-20s multi %.2f single %.2f all %.2f | width multi %.2f single %.2f' % (tag, m, a, b, c, wa, wb))
