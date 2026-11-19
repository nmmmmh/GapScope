"""Scope checks (2026-11-18). Run on c137-13: OMP_NUM_THREADS=1 <venv>/python -I c4_scope.py
(1) Are well-studied compositions more often polymorphic? Share with >=2 accessible SNUMAT polymorphs and with D_c >= 0.3 eV,
    for all SNUMAT compositions, compiled compositions by number of distinct reported values, the Borlido benchmark, the traced set.
(2) Do compiled values correspond to any measured phase? For compositions in both the traced and compiled sets.
(3) Named compositions of the photovoltaic computed screen (lowest-energy polymorph inside 1.1-1.7 eV, an accessible one outside)."""
import numpy as np, pandas as pd, torch
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def acc(c, need_energy=False):
    if need_energy: c = c[c.Free_energy.notna()]
    if len(c) == 0: return c
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= 0.10)]
    return c[c.hse.notna()]
stat = {}
for k, c in S.groupby('ckey'):
    a = acc(c); stat[k] = (len(a), (a.hse.max() - a.hse.min()) if len(a) >= 2 else 0.0)
ST = pd.DataFrame(stat, index=['n_acc', 'Dc']).T
def share(keys, lab):
    k = [x for x in keys if x in ST.index]; t = ST.loc[k]
    print('(1) %-38s n %5d | >=2 accessible %5.1f%% | D_c>=0.3 eV %5.1f%% | D_c>=0.1 eV %5.1f%%' % (lab, len(t), 100 * (t.n_acc >= 2).mean(), 100 * (t.Dc >= 0.3).mean(), 100 * (t.Dc >= 0.1).mean()))
share(ST.index, 'all SNUMAT compositions')
A = pd.read_csv(R + 'cv/data/tierB_all_rows.csv').drop_duplicates(['key', 'Eg'])
nv = A.groupby('key').Eg.size()
share(nv.index, 'compiled, all')
for lo, hi, lab in ((1, 1, 'compiled, 1 value'), (2, 2, 'compiled, 2 values'), (3, 99, 'compiled, >=3 values')):
    share(nv[(nv >= lo) & (nv <= hi)].index, lab)
Bd = pd.read_csv(R + 'borlido/b1_borlido_Dc.csv')
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
share([key(f) for f in Bd.formula.unique()], 'Borlido benchmark')
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
share(P.composition_key.unique(), 'traced 85 (selected for phases)')

# (2) compiled values vs traced reports
P['ph'] = P.phase_label.fillna('').astype(str).str.strip().str.lower()
ov = sorted(set(P.composition_key) & set(A.key)); rows = []
for k in ov:
    tr = P[P.composition_key == k]; cv = A[A.key == k].Eg.values; tv = tr.bandgap_ev.values
    for v in cv:
        d = np.abs(tv - v).min()
        rows.append(dict(key=k, formula=tr.formula.iloc[0], nph=tr[tr.ph != ''].ph.nunique(), comp_val=v, nearest_traced=d,
                         outside=(v < tv.min() - 0.1) or (v > tv.max() + 0.1), t_lo=tv.min(), t_hi=tv.max()))
O = pd.DataFrame(rows)
for lab, o in (('all overlap', O), ('multi-phase traced', O[O.nph >= 2])):
    print('(2) %-20s compositions %d compiled values %d | >0.1 eV from every traced report %d (%.0f%%) | outside traced range by >0.1 eV %d (%.0f%%)'
          % (lab, o.key.nunique(), len(o), (o.nearest_traced > 0.1).sum(), 100 * (o.nearest_traced > 0.1).mean(), o.outside.sum(), 100 * o.outside.mean()))
print(O[O.nearest_traced > 0.1].sort_values('nearest_traced', ascending=False)[['formula', 'nph', 'comp_val', 't_lo', 't_hi', 'nearest_traced']].round(2).to_string(index=False))

# (3) photovoltaic computed screen, named
rows = []
for k, c in S.groupby('ckey'):
    a = acc(c, need_energy=True)
    if len(a) < 2: continue
    gs = a.loc[a.Free_energy.idxmin()]
    if 1.1 <= gs.hse <= 1.7:
        out = a[(a.hse < 1.1) | (a.hse > 1.7)]
        rows.append(dict(formula=Composition(k.replace('|', ' ').replace(':', '')).reduced_formula if False else None, key=k, gs_sg=int(gs.sg), gs_hse=round(gs.hse, 2),
                         n_acc=len(a), outside=len(out), out_hse=';'.join('%d:%.2f' % (int(r.sg), r.hse) for r in out.itertuples()),
                         dE=';'.join('%.3f' % (r.Free_energy - gs.Free_energy) for r in out.itertuples())))
W = pd.DataFrame(rows)
def red(k):
    d = {e.split(':')[0]: float(e.split(':')[1]) for e in k.split('|')}; return Composition(d).reduced_formula
W['formula'] = W.key.map(red)
print('(3) lowest-energy polymorph inside PV window: %d, with an accessible polymorph outside: %d' % (len(W), (W.outside > 0).sum()))
print(W[W.outside > 0][['formula', 'gs_sg', 'gs_hse', 'n_acc', 'out_hse', 'dE']].to_string(index=False))
