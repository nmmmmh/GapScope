"""PV computed screen (lowest-energy polymorph 1.1-1.7 eV, an accessible polymorph outside) under artefact filters."""
import numpy as np, pandas as pd
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
OPEN = {'V','Cr','Mn','Fe','Co','Ni','Ru','Rh','Os','Ir','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','U','Np','Pu'}
rows = []
for k, c in S.groupby('ckey'):
    c = c[c.Free_energy.notna() & c.hse.notna()]
    if len(c) < 2: continue
    e0 = c.Free_energy.min(); a = c[c.Free_energy - e0 <= 0.10].sort_values('Free_energy')
    if len(a) < 2: continue
    gs = a.iloc[0]
    if not (1.1 <= gs.hse <= 1.7): continue
    els = {e.split(':')[0] for e in k.split('|')}
    oth = a.iloc[1:]; dE = oth.Free_energy - e0
    marg = np.maximum(1.1 - oth.hse, oth.hse - 1.7)
    rows.append(dict(f=Composition({e.split(':')[0]: float(e.split(':')[1]) for e in k.split('|')}).reduced_formula,
        gs=gs.hse, open_=bool(els & OPEN), amb=bool((dE < 0.005).any() and (np.abs(oth.hse[dE < 0.005] - gs.hse) > 0.1).any()),
        out01=(marg > 0).any(), out_m1=(marg > 0.1).any(), out_m2=(marg > 0.2).any(),
        out_m2_dE=((marg > 0.2) & (dE >= 0.005)).any()))
W = pd.DataFrame(rows)
def rep(lab, w): print('%-40s inside %3d | outside>0 %3d | >0.1 %3d | >0.2 %3d | >0.2 & dE>=5meV %3d' % (lab, len(w), w.out01.sum(), w.out_m1.sum(), w.out_m2.sum(), w.out_m2_dE.sum()))
rep('all', W); rep('no open-shell TM/f', W[~W.open_]); rep('no open-shell, GS unambiguous', W[~W.open_ & ~W.amb])
print(W[W.out01].sort_values('f')[['f','gs','open_','amb','out_m1','out_m2','out_m2_dE']].round(2).to_string(index=False))
