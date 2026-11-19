"""Supplementary Data 2: every SNUMAT composition with two or more accessible polymorphs (main-text definition: within
0.1 eV/atom of the lowest free energy, polymorphs without an energy kept), with D_c (eq 1) and screen flags.
Also writes the revised computed-screen counts (window/par2/hse_screen_v2.csv) and prints the SNUMAT baselines."""
import numpy as np, pandas as pd
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
OPEN = {'V','Cr','Mn','Fe','Co','Ni','Ru','Rh','Os','Ir','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','U','Np','Pu'}
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}; RES = 0.005
def formula(k):
    return Composition({e.split(':')[0]: float(e.split(':')[1]) for e in k.split('|')}).get_integer_formula_and_factor()[0]
Bd = pd.read_csv(R + 'borlido/b1_borlido_Dc.csv')
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
BK = {key(f) for f in Bd.formula.unique()}
TK = set(pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv').composition_key)
rows, base = [], []
for k, c in S.groupby('ckey'):
    c = c[c.hse.notna()]
    if len(c) == 0: continue
    allE = bool(c.Free_energy.notna().all()); c = c[c.Free_energy.notna()]   # energy-only definition (2026-11-18)
    if len(c) == 0: base.append((0.0, allE, 0)); continue
    a = c
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); a = c[c.Free_energy - e0 <= 0.10]
    dc = a.hse.max() - a.hse.min() if len(a) >= 2 else 0.0
    base.append((dc, allE, len(a)))
    if len(a) < 2: continue
    a = a.assign(dE=a.Free_energy - a.Free_energy.min()).sort_values(['dE', 'sg'], na_position='last')
    r = dict(formula=formula(k), composition_key=k, n_accessible=len(a), n_with_energy=int(a.Free_energy.notna().sum()),
             space_groups=';'.join(str(int(s)) for s in a.sg), hse06_gaps_ev=';'.join('%.3f' % g for g in a.hse),
             energy_above_lowest_ev_per_atom=';'.join('' if np.isnan(x) else '%.4f' % x for x in a.dE),
             D_c_ev=round(dc, 3), D_c_ge_0p3=dc >= 0.3)
    e = a[a.Free_energy.notna()]
    if len(e) >= 2:
        g, de = e.hse.values, e.dE.values
        amb = bool(((de[1:] < RES) & (np.abs(g[1:] - g[0]) > 0.1)).any())
        r.update(lowest_energy_sg=int(e.sg.iloc[0]), lowest_energy_gap_ev=round(g[0], 3), ground_state_resolved=not amb)
        for wn, (lo, hi) in W.items():
            d = 0.1; ins = (g >= lo + d) & (g <= hi - d); out = (g < lo - d) | (g > hi + d)
            r[f'{wn}_membership_changes'] = bool(ins.any() and out.any())
            r[f'{wn}_lowest_inside_other_outside'] = bool((not amb) and ins[0] and (out[1:] & (de[1:] >= RES)).any())
    r.update(open_shell_element=bool({x.split(':')[0] for x in k.split('|')} & OPEN), in_borlido_benchmark=k in BK, in_traced_set=k in TK)
    rows.append(r)
T = pd.DataFrame(rows).sort_values('D_c_ev', ascending=False)
T['lowest_energy_sg'] = T.lowest_energy_sg.astype('Int64')
T.to_csv(R + 'suppdata/Supplementary_Data_2_GapScope_polymorph_gap_spread.csv', index=False)
B = pd.DataFrame(base, columns=['dc', 'allE', 'n'])
print('SNUMAT compositions', len(B), '| >=2 accessible', int((B.n >= 2).sum()), '| D_c>=0.3 %.1f%% (%d)' % (100 * (B.dc >= 0.3).mean(), (B.dc >= 0.3).sum()),
      '| all-energy subset n %d D_c>=0.3 %.1f%%' % (B.allE.sum(), 100 * (B[B.allE].dc >= 0.3).mean()))
print('table rows', len(T), '| D_c>=0.3', int(T.D_c_ge_0p3.sum()), '| with >=2 energies', int(T.lowest_energy_sg.notna().sum()),
      '| GS unresolved', int((T.ground_state_resolved == False).sum()), '| in Borlido', int(T.in_borlido_benchmark.sum()), '| traced', int(T.in_traced_set.sum()))
for wn in W:
    n_in = int(((T.ground_state_resolved == True) & T.lowest_energy_gap_ev.between(W[wn][0] + 0.1, W[wn][1] - 0.1)).sum())
    print(wn, 'changes', int(T[f'{wn}_membership_changes'].sum()), '| GS inside (resolved, d=0.1)', n_in, '| with outside', int(T[f'{wn}_lowest_inside_other_outside'].sum()))
pd.DataFrame([dict(window=wn, n_gs_in=int(((T.ground_state_resolved == True) & T.lowest_energy_gap_ev.between(W[wn][0] + 0.1, W[wn][1] - 0.1)).sum()),
                   n_gs_in_with_outside=int(T[f'{wn}_lowest_inside_other_outside'].sum())) for wn in W]).to_csv(R + 'window/par2/hse_screen_v2.csv', index=False)
print(T.head(8)[['formula', 'n_accessible', 'D_c_ev', 'ground_state_resolved']].to_string(index=False))
