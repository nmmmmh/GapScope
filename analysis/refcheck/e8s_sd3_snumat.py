"""Supplementary Data 3: computed floor and polymorph gap spread D_c for every composition of the Matbench
experimental band-gap task (matbench_expt_gap, 4,604 compositions). Same recipe as e6b (main text 2.3, SI S4):
MP HSE-corrected gaps of the pretraining pool, ehull <= 0.1 eV/atom, one entry per composition and space group,
accessible = within 0.1 eV/atom of the lowest entry of the composition, Boltzmann weights at 300 K,
floor = E|G - med G| of the weighted mixture (numerical, 0.005 eV broadening), no measurement-noise term."""
import json, numpy as np, pandas as pd
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
kB, T, SM = 8.617e-5, 300.0, 0.005
M = pd.read_csv('/home/czx/spinels/data/matbench/expt_gap.csv')
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv'); S = S[S.hse.notna() & S.Free_energy.notna()]
H = pd.DataFrame(dict(formula=[Composition({p.split(':')[0]: float(p.split(':')[1]) for p in k.split('|')}).reduced_formula for k in S.ckey], spacegroup=S.sg.values, energy_above_hull=(S.Free_energy - S.groupby('ckey').Free_energy.transform('min')).values, target=S.hse.values))
H = H[(H.energy_above_hull <= 0.10) & H.target.notna() & H.spacegroup.between(1, 230)].sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['formula', 'spacegroup'])
def ck(f):
    try:
        c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.6f}' for el in sorted(c.as_dict()))
    except Exception:
        return None
H['ck'] = [ck(f) for f in H.formula]; M['ck'] = [ck(f) for f in M.formula]
G = {k: g for k, g in H.groupby('ck')}
xs = np.linspace(-1, 12, 6501)
def floor(g, w):
    w = w / w.sum(); pdf = sum(wi * np.exp(-0.5 * ((xs - gi) / SM) ** 2) for gi, wi in zip(g, w)); pdf /= pdf.sum()
    c = np.cumsum(pdf); med = xs[np.searchsorted(c, 0.5)]; return float(np.sum(pdf * np.abs(xs - med)))
rows = []
for f, k, y in zip(M.formula, M.ck, M.target):
    g = G.get(k)
    r = dict(formula=f, reduced_formula=Composition(f).reduced_formula, matbench_gap_ev=y, nonmetal=bool(y > 0),
             in_snumat=g is not None, n_accessible_polymorphs=0, space_groups='', hse_gaps_ev='', e_above_lowest_ev_per_atom='',
             boltzmann_weights_300K='', D_c_ev=np.nan, computed_floor_300K_ev=np.nan, flag_Dc_ge_0p3=False)
    if g is not None:
        e = g.energy_above_hull.values - g.energy_above_hull.min(); keep = e <= 0.10
        gg = g[keep]; e = e[keep]; t = np.clip(gg.target.values, 0, None); w = np.exp(-e / (kB * T)); o = np.argsort(e, kind='mergesort')
        r.update(n_accessible_polymorphs=int(len(gg)), space_groups=';'.join(str(int(s)) for s in gg.spacegroup.values[o]),
                 hse_gaps_ev=';'.join(f'{v:.3f}' for v in t[o]), e_above_lowest_ev_per_atom=';'.join(f'{v:.4f}' for v in e[o]),
                 boltzmann_weights_300K=';'.join(f'{v:.4f}' for v in (w / w.sum())[o]),
                 D_c_ev=round(float(t.max() - t.min()), 4), computed_floor_300K_ev=round(floor(t, w), 4) if len(gg) > 1 else 0.0)
        r['flag_Dc_ge_0p3'] = bool(r['D_c_ev'] >= 0.3)
    rows.append(r)
D = pd.DataFrame(rows)
out = R + 'suppdata/Supplementary_Data_3_Matbench_expt_gap_computed_floor.csv'
D.to_csv(out, index=False, encoding='utf-8')
# reproduce the numbers in the text
nm = D[D.nonmetal]; mp2 = nm[nm.n_accessible_polymorphs >= 2]; fl = mp2[mp2.flag_Dc_ge_0p3]
print('rows', len(D), 'nonmetal', len(nm), 'in pool', int(nm.in_snumat.sum()), '>=2 poly', len(mp2))
print('floor mean %.3f median %.3f P90 %.3f n>=0.05 %d' % (mp2.computed_floor_300K_ev.mean(), mp2.computed_floor_300K_ev.median(), mp2.computed_floor_300K_ev.quantile(.9), (mp2.computed_floor_300K_ev >= 0.05).sum()))
print('flagged %d share %.3f mean %.3f' % (len(fl), fl.computed_floor_300K_ev.sum() / mp2.computed_floor_300K_ev.sum(), fl.computed_floor_300K_ev.mean()))
print('mean over all nonmetals %.3f' % nm.computed_floor_300K_ev.fillna(0).mean())
print('all compositions flagged', int(D.flag_Dc_ge_0p3.sum()))
print(out)
