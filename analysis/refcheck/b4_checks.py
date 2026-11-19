"""Referee checks from existing data (2026-11-18):
(1) median predicted 80% width, multi- vs single-phase, with the ledger's multi-phase definition;
(2) floor and within-study phase shift by collection (first collection vs the six further sources);
(3) within-study phase pairs that share measurement method, gap definition and sample form.
Run on c137-13: OMP_NUM_THREADS=1 <venv>/python -I b4_checks.py"""
import numpy as np, pandas as pd, itertools
from scipy.stats import mannwhitneyu, spearmanr
R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
nph = S[S.ph != ''].groupby('composition_key').ph.nunique()
multi = set(nph[nph >= 2].index); print('multi-phase compositions', len(multi), 'record groups', S.record_group.value_counts().to_dict())

a = pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv')
def wm(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return y[np.searchsorted(c, 0.5)]
g = a[(a.model == 'GS_SELCAL_PRE_NZ_SC') & (a.tier == 'A')]
W = g.groupby('key').apply(lambda q: np.sum(q.w * q.width80) / q.w.sum(), include_groups=False)
fl = {k: np.sum(q.w.values * np.abs(q.y.values - wm(q.y.values, q.w.values))) / q.w.sum() for k, q in g.groupby('key')}
F = pd.Series(fl)[W.index]; mp = W.index.isin(multi)
print('(1) width multi/single median %.3f/%.3f (n %d/%d) P %.1e | rho(width, floor) %.3f P %.1e'
      % (W[mp].median(), W[~mp].median(), mp.sum(), (~mp).sum(), mannwhitneyu(W[mp], W[~mp]).pvalue, *spearmanr(W, F)))
Wm = g.groupby('key').width80.mean()
print('    unweighted mean per composition: %.3f/%.3f P %.1e' % (Wm[mp].median(), Wm[~mp].median(), mannwhitneyu(Wm[mp], Wm[~mp]).pvalue))

grp = S.groupby('composition_key').record_group.first()
r0 = a[(a.model == 'ROOST') & (a.tier == 'A')]
fl0 = pd.Series({k: np.sum(q.w.values * np.abs(q.y.values - wm(q.y.values, q.w.values))) / q.w.sum() for k, q in r0.groupby('key')})
for gname in grp.unique():
    ks = grp[grp == gname].index; f = fl0[fl0.index.isin(ks)]
    print('(2) %-20s compositions %d floor mean %.3f | multi-phase %d | floor share in multi-phase %.0f%%'
          % (gname, len(f), f.mean(), len(set(ks) & multi), 100 * f[f.index.isin(multi)].sum() / f.sum()))

rows = []
for c, gg in S.groupby('composition_key'):
    for x, y in itertools.combinations(gg.to_dict('records'), 2):
        if not x['ph'] or not y['ph'] or x['source_group_id'] != y['source_group_id']: continue
        if x['ph'] == y['ph']: continue
        rows.append(dict(comp=c, grp=x['record_group'], d=abs(x['bandgap_ev'] - y['bandgap_ev']),
                         same_def=str(x['gap_definition']) == str(y['gap_definition']),
                         same_meth=str(x['measurement_method']) == str(y['measurement_method']),
                         same_form=str(x['sample_form']) == str(y['sample_form'])))
P = pd.DataFrame(rows); rng = np.random.default_rng(20261118)
def boot(q):
    cs = q.comp.unique(); by = {c: q[q.comp == c].d.values for c in cs}
    b = [np.concatenate([by[c] for c in rng.choice(cs, len(cs))]).mean() for _ in range(4000)]
    return np.percentile(b, [2.5, 97.5])
for lab, q in [('all same-study phase pairs', P), ('first collection', P[P.grp == 'original_collection']),
               ('six further sources', P[P.grp != 'original_collection']),
               ('same method and definition', P[P.same_def & P.same_meth]),
               ('same method, definition and sample form', P[P.same_def & P.same_meth & P.same_form])]:
    if len(q) == 0: print('(3) %-40s none' % lab); continue
    lo, hi = boot(q) if q.comp.nunique() > 1 else (np.nan, np.nan)
    print('(3) %-40s pairs %3d comps %2d mean %.3f [%.3f, %.3f] median %.3f' % (lab, len(q), q.comp.nunique(), q.d.mean(), lo, hi, q.d.median()))
