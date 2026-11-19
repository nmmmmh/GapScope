"""Robustness of the Matbench computed floor (nonmetals).
(a) SNUMAT pool, windows 0.05/0.10/0.20 eV/atom x T 300/600/1000 K.
(b) MP pool (mp_hse.csv) with three filters: no GGA+U entries, no zero-gap polymorphs, lowest present entry ehull <= 0.01.
Reports: n covered, n with >=2 accessible, mean floor, n flagged (D_c >= 0.3), share of floor on flagged, mean on flagged."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
kB, SM = 8.617e-5, 0.005
M = pd.read_csv('/home/czx/spinels/data/matbench/expt_gap.csv'); M = M[M.target > 0]
def ck(f):
    try:
        c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.6f}' for el in sorted(c.as_dict()))
    except Exception: return None
M['ck'] = [ck(f) for f in M.formula]
xs = np.linspace(-1, 12, 6501)
def floor(g, w):
    w = w / w.sum(); pdf = sum(wi * np.exp(-0.5 * ((xs - gi) / SM) ** 2) for gi, wi in zip(g, w)); pdf /= pdf.sum()
    c = np.cumsum(pdf); med = xs[np.searchsorted(c, 0.5)]; return float(np.sum(pdf * np.abs(xs - med)))
def stats(P, win, T, tag):
    G = {k: g for k, g in P.groupby('ck')}; F, D, cov = [], [], 0
    for k in M.ck:
        g = G.get(k)
        if g is None: continue
        cov += 1; e = g.e.values - g.e.min(); m = e <= win; t = g.g.values[m]; e = e[m]
        if len(t) < 2: continue
        F.append(floor(t, np.exp(-e / (kB * T)))); D.append(t.max() - t.min())
    F, D = np.array(F), np.array(D); f = D >= 0.3
    print(f'{tag:28s} win {win:.2f} T {T:4.0f} | covered {cov:4d} | >=2 {len(F):3d} | mean {F.mean():.3f} | flagged {f.sum():3d} | share {F[f].sum() / F.sum():.2f} | mean flagged {F[f].mean():.3f}')
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv'); S = S[S.hse.notna() & S.Free_energy.notna()]
S = pd.DataFrame(dict(ck=['|'.join(p.split(':')[0] + ':%.6f' % float(p.split(':')[1]) for p in k.split('|')) for k in S.ckey], g=S.hse.clip(lower=0).values, e=S.Free_energy.values))
for win in (0.05, 0.10, 0.20):
    for T in (300, 600, 1000):
        stats(S, win, T, 'SNUMAT')
H = pd.read_csv('/home/czx/2024/spinels/data/mp_hse.csv')
H = H[H.target.notna() & H.spacegroup.between(1, 230)].sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['formula', 'spacegroup'])
H['ck'] = [ck(f) for f in H.formula]
gs = H.groupby('ck').energy_above_hull.transform('min')
Hf = H[(H.is_GGA_U == 0) & (H.target > 0) & (gs <= 0.01) & (H.energy_above_hull <= 0.10)]
stats(pd.DataFrame(dict(ck=Hf.ck, g=Hf.target, e=Hf.energy_above_hull)), 0.10, 300, 'MP, three filters')
