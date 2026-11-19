"""Reproduces the numbers that were computed ad hoc during writing (floor tables, curation, width robustness).
Run on c137-13:  OMP_NUM_THREADS=1 <venv>/python -I ledger_checks.py  > ledger_checks.txt"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
nph = S[S.ph != ''].groupby('composition_key').ph.nunique()
same = S[S.ph != ''].groupby(['composition_key', 'source_group_id']).ph.nunique()
print('== data')
print('reports %d compositions %d sources %d specimens %d phase-labelled %d' % (len(S), S.composition_key.nunique(),
      S.source_group_id.nunique(), S.specimen_group_id.nunique(), (S.ph != '').sum()))
rng_ = S.groupby('composition_key').bandgap_ev.agg(lambda x: x.max() - x.min())
print('distinct-valued compositions %d; median range %.2f; max range %.2f' % ((rng_ > 0).sum(), rng_[rng_ > 0].median(), rng_.max()))
print('multi-phase compositions %d; of which in one study %d' % ((nph >= 2).sum(), same[same >= 2].reset_index().composition_key.nunique()))
print('reports with space group %d (compositions %d)' % (S.space_group.notna().sum(), S[S.space_group.notna()].composition_key.nunique()))

a = pd.read_csv(R0 + 'cv/predictions_summary_v4a.csv'); t = pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv')
M = [('ROOST', a), ('ROOST_CAL', a), ('ROOST_CAL_CV', a), ('EASYUQ_KS', a), ('EASYUQ_IDR', a), ('GS_SELCAL_PRE_NZ_SC', t)]
def wm(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return y[np.searchsorted(c, 0.5)]
print('\n== floor and model MAE (composition weight unless stated)')
for m, d in M:
    g = d[(d.model == m) & (d.tier == 'A')]; R = []
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values; e = np.abs(y - gg['median'].values)
        R.append((k, np.sum(w * e) / w.sum(), np.sum(w * abs(y - wm(y, w))) / w.sum(), round(y.max() - y.min(), 6) > 0.1 + 1e-9,
                  nph.get(k, 0) >= 2, e.sum(), len(y), np.abs(y - np.median(y)).sum(), y.max() - y.min()))
    R = pd.DataFrame(R, columns=['k', 'mae', 'fl', 'rm', 'multi', 'esum', 'n', 'flsum', 'rng'])
    print('%-20s all %.3f | report-weighted %.3f | multi %.3f single %.3f | kept %.3f | floor share %.3f / multi %.3f'
          % (m, R.mae.mean(), R.esum.sum() / R.n.sum(), R[R.multi].mae.mean(), R[~R.multi].mae.mean(), R[~R.rm].mae.mean(),
             R.fl.mean() / R.mae.mean(), R[R.multi].fl.mean() / R[R.multi].mae.mean()))
print('floor all %.4f | report-weighted %.4f | multi %.4f single %.4f | kept %.4f (n kept %d)'
      % (R.fl.mean(), R.flsum.sum() / R.n.sum(), R[R.multi].fl.mean(), R[~R.multi].fl.mean(), R[~R.rm].fl.mean(), (~R.rm).sum()))
print('max single-composition floor %.3f; compositions with zero floor %d' % (R.fl.max(), (R.fl == 0).sum()))
Rs = R.sort_values('rng', ascending=False); c = 100 * Rs.fl.cumsum() / Rs.fl.sum()
print('curation (>0.1 eV): removes %d, floor share %.1f%%; widest 10 %.0f%%, widest 20 %.0f%%; multi-phase share %.1f%%'
      % (R.rm.sum(), 100 * R.fl[R.rm].sum() / R.fl.sum(), c.iloc[9], c.iloc[19], 100 * R.fl[R.multi].sum() / R.fl.sum()))
ga = R[R.k == S[S.formula == 'Ga2O3'].composition_key.iloc[0]]
print('Ga2O3 floor %.3f' % ga.fl.iloc[0])

print('\n== width robustness (50 compositions)')
B = pd.read_csv(R0 + 'fig4_box.csv'); g = S.groupby('composition_key')
st = pd.DataFrame({'mad': g.bandgap_ev.apply(lambda x: np.mean(np.abs(x - x.median()))), 'nph': nph, 'n': g.size()})
d = B.merge(st, left_on='composition_key', right_index=True, how='left'); d['nph'] = d.nph.fillna(0)
for lab, x in (('all', d), ('multi-phase', d[d.nph >= 2]), ('n<=3', d[d.n <= 3])):
    v, f = x[x.var_coordination == 1], x[x.var_coordination == 0]
    for col in ('exp_range', 'mad'):
        print('%-12s %-9s var %2d fixed %2d medians %.3f / %.3f  P=%.4f' % (lab, col, len(v), len(f), v[col].median(), f[col].median(), mannwhitneyu(v[col], f[col]).pvalue))
print('rho(range,n) %.3f rho(mad,n) %.3f rho(varCN,n) %.3f' % (spearmanr(d.exp_range, d.n)[0], spearmanr(d.mad, d.n)[0], spearmanr(d.var_coordination, d.n)[0]))
