import numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
P['ph'] = P.phase_label.fillna('').astype(str).str.strip().str.lower()
def wmed_floor(g):
    # equal weight per source, then specimen, then report
    w = []
    ns = g.source_group_id.nunique()
    for s, gs in g.groupby('source_group_id'):
        nsp = gs.specimen_group_id.nunique()
        for sp, gg in gs.groupby('specimen_group_id'):
            w += [(i, 1 / ns / nsp / len(gg)) for i in gg.index]
    idx = [i for i, _ in w]; ww = np.array([x for _, x in w]); y = g.loc[idx, 'bandgap_ev'].values
    o = np.argsort(y); cw = np.cumsum(ww[o]) / ww.sum(); m = y[o][np.searchsorted(cw, 0.5)]
    return np.sum(ww * np.abs(y - m)) / ww.sum()
rows = []
for k, g in P.groupby('composition_key'):
    rows.append(dict(k=k, F=wmed_floor(g), nph=g[g.ph != ''].ph.nunique(), nrep=len(g), nspec=g.specimen_group_id.nunique(), nval=g.bandgap_ev.nunique()))
T = pd.DataFrame(rows)
print('all floor %.3f (n %d) | multi %.3f (%d) | single %.3f (%d)' % (T.F.mean(), len(T), T[T.nph >= 2].F.mean(), (T.nph >= 2).sum(), T[T.nph < 2].F.mean(), (T.nph < 2).sum()))
S1 = T[T.nph < 2]
for lab, s in (('single, 1 report', S1[S1.nrep == 1]), ('single, >=2 reports', S1[S1.nrep >= 2]), ('single, >=2 specimens', S1[S1.nspec >= 2]), ('single, >=2 distinct values', S1[S1.nval >= 2])):
    print('%-28s n %2d floor mean %.3f median %.3f max %.3f' % (lab, len(s), s.F.mean(), s.F.median(), s.F.max() if len(s) else np.nan))
# SUPERSEDED below this line (2026-11-19): the D_c part reads the earlier all-polymorph descr/d2_per_composition.csv;
# the paper's D_c numbers come from refcheck/c14_energy.py. The floor part above is current (used by c12_amorph.py).
D = pd.read_csv(R + 'descr/d2_per_composition.csv')
y = D.floor >= 0.05; x = D.D_elec >= 0.3
tp = (x & y).sum(); print('D_c>=0.3 on 50: flagged %d, precision %d/%d = %.2f, recall %d/%d = %.2f, base rate %.2f' % (x.sum(), tp, x.sum(), tp / x.sum(), tp, y.sum(), tp / y.sum(), y.mean()))
print('median floor flagged %.3f vs not %.3f' % (D.floor[x].median(), D.floor[~x].median()))
