# SUPERSEDED (2026-11-19): computed floor of the Matbench task on the Materials Project pretraining pool, where the GGA+U gaps are kept as computed and some ground-state polymorphs are missing.
# The paper uses the SNUMAT version, analysis/refcheck/e8s_sd3_snumat.py (robustness and a filtered Materials Project check in e9_robust.py).
# Kept for comparison only; no number, table or figure of the paper comes from this script.
"""Computed floor of the Matbench experimental band-gap task (4,604 compositions, one value each).
Floor of a composition = E|G - med G| for the Boltzmann(300 K) mixture of its MP polymorph HSE-level gaps
(ehull <= 0.1 eV/atom, one entry per space group), each broadened by measurement scatter sigma_m = 0.038 eV
(same recipe as k5/e4 computed floor). Compositions with one or no computed polymorph get the noise term only.
Metals (target 0) are reported separately. Also D_c (max - min) flags."""
import numpy as np, pandas as pd
from pymatgen.core import Composition
kB, T, SM = 8.617e-5, 300.0, 0.038
NOISE = SM * np.sqrt(2 / np.pi)
M = pd.read_csv('/home/czx/spinels/data/matbench/expt_gap.csv')
H = pd.read_csv('/home/czx/2024/spinels/data/mp_hse.csv')
H = H[(H.energy_above_hull <= 0.10) & H.target.notna() & H.spacegroup.between(1, 230)].sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['formula', 'spacegroup'])
def ck(f):
    try:
        c = Composition(f).fractional_composition
        return '|'.join(f'{el}:{c[el]:.6f}' for el in sorted(c.as_dict()))
    except Exception:
        return None
H['ck'] = [ck(f) for f in H.formula]; M['ck'] = [ck(f) for f in M.formula]
G = {k: g for k, g in H.groupby('ck')}
xs = np.linspace(-1, 12, 6501)
def floor(g, w):
    w = w / w.sum()
    pdf = sum(wi * np.exp(-0.5 * ((xs - gi) / SM) ** 2) for gi, wi in zip(g, w)); pdf /= pdf.sum()
    c = np.cumsum(pdf); med = xs[np.searchsorted(c, 0.5)]
    return float(np.sum(pdf * np.abs(xs - med)))
rows = []
for k, y in zip(M.ck, M.target):
    g = G.get(k)
    if g is None or len(g) == 0:
        rows.append((y, 0, np.nan, NOISE)); continue
    e = g.energy_above_hull.values - g.energy_above_hull.min(); t = np.clip(g.target.values, 0, None)
    w = np.exp(-e / (kB * T))
    rows.append((y, len(g), t.max() - t.min(), floor(t, w) if len(g) > 1 else NOISE))
R = pd.DataFrame(rows, columns=['y', 'n_poly', 'Dc', 'F'])
R.to_csv('/data/czx/GapScope_Revision_20261113/refcheck/e6_matbench_floor.csv', index=False)
for lab, s in (('all', R.index == R.index), ('nonmetal (y>0)', R.y > 0), ('metal (y=0)', R.y == 0)):
    r = R[s]; cov = r.n_poly > 0; mp = r.n_poly > 1
    print(f'{lab:15s} n {len(r)} | in MP {cov.sum()} | >=2 polymorphs {mp.sum()} | Dc>=0.3 {(r.Dc >= 0.3).sum()} | '
          f'mean floor (metals counted with noise term) {r.F.mean():.3f} | mean floor on MP-covered {r.F[cov].mean():.3f} | '
          f'on >=2 polymorphs {r.F[mp].mean():.3f} | median {r.F[mp].median():.3f}')
nm = R[R.y > 0]
print('nonmetal floor quantiles (>=2 poly):', nm.F[nm.n_poly > 1].quantile([.5, .75, .9, .95]).round(3).to_dict())
print('nonmetal floor without metals noise, metals floor set 0: all-task mean %.3f' % (np.where(R.y > 0, R.F, 0).mean()))
