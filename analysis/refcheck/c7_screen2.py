"""Both windows, same filters as hse_exp.py part A plus: symmetric margin d, energy resolution 5 meV/atom, open-shell flag.
n_change: accessible polymorphs straddle (one inside by d, one outside by d), with the pair >= 5 meV/atom apart when 'res'.
gs screen: lowest-energy polymorph inside by d, another accessible polymorph outside by d; 'res' also requires the ground state
to be resolved (no other accessible polymorph within 5 meV/atom), removing it from the denominator otherwise."""
import numpy as np, pandas as pd
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
OPEN = {'V','Cr','Mn','Fe','Co','Ni','Ru','Rh','Os','Ir','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','U','Np','Pu'}
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}; RES = 0.005
P = []
for k, t in S.groupby('ckey'):
    te = t.dropna(subset=['Free_energy'])
    if len(te) < 2: continue
    a = te[te.Free_energy <= te.Free_energy.min() + 0.1].sort_values('Free_energy')
    if len(a) < 2: continue
    P.append(dict(e=a.Free_energy.values - a.Free_energy.values[0], g=a.hse.values, open_=bool({x.split(':')[0] for x in k.split('|')} & OPEN)))
print('GS with near-degenerate twin', sum(bool((p['e'][1:] < RES).any()) for p in P), '| of which other gap >0.1 eV', sum(bool(((p['e'][1:] < RES) & (np.abs(p['g'][1:] - p['g'][0]) > 0.1)).any()) for p in P))
print('compositions with >=2 accessible polymorphs', len(P), '| open-shell', sum(p['open_'] for p in P))
def ev(p, a, b, d, res):
    g, e = p['g'], p['e']; ins = (g >= a + d) & (g <= b - d); out = (g < a - d) | (g > b + d)
    ch = bool(ins.any() and out.any())
    amb = bool(((e[1:] < RES) & (np.abs(g[1:] - g[0]) > 0.1)).any())   # near-degenerate twin with another gap
    gin = bool(((not res) or (not amb)) and ins[0]); gout = gin and bool((out[1:] & (e[1:] >= (RES if res else -1))).any())
    return ch, gin, gout
for sub, lab in ((lambda p: True, 'all'), (lambda p: not p['open_'], 'closed-shell')):
    Q = [p for p in P if sub(p)]
    for wn, (a, b) in W.items():
        for res in (False, True):
            for d in (0.0, 0.1, 0.2):
                r = np.array([ev(p, a, b, d, res) for p in Q])
                print('%-12s %-3s res=%d d=%.1f | n %3d change %3d (%4.1f%%) | GS inside %3d with outside %3d (%4.1f%%)'
                      % (lab, wn, res, d, len(Q), r[:, 0].sum(), 100 * r[:, 0].mean(), r[:, 1].sum(), r[:, 2].sum(), 100 * r[:, 2].sum() / max(r[:, 1].sum(), 1)))
