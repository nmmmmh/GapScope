"""Stage 1 step 5: the Ga2O3 and FAPbI3 sanity checks.

Ga2O3 polymorphs are taken from SNUMAT itself (computed HSE gaps) and, for the experimental
comparison, from the provenance-linked experimental records.  FAPbI3 is not in SNUMAT (organic
cation), so its two phases come from the two reference CIFs in the experimental set; the
disordered formamidinium cation is removed and the descriptors are those of the PbI3 framework
(Pb cation, I anion), which is stated in the README.

Models are refitted with the case chemistry held out, so the prediction is a transfer prediction.
Usage: python -I case_check.py
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/struct')
from desc_worker import descriptors
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

D = Path('/data/czx/GapScope_Revision_20261113/struct')
SIX = ['d_dim_cat_an', 'd_mean_shared_vertices', 'd_bond_cat_an_mean', 'd_cn_cat_mean',
       'd_distortion_index', 'd_vol_per_atom']
CIF = Path('/home/czx/2024/BandgapFusion_Consolidated/data/structures')
FA = {'alpha_sg215': '2a42341dc6027ab13c00ed626339127ce0954b60a555ad0bdc56fb1f4a2fe466',
      'delta_sg186': '63a8ea4c93d29840cb444a9843c0d654c35e765541de0200172ca3f3381be9f5'}
GA = {'alpha_sg167': 'f2165d98973ab0dd1d2e24f77be8ee8f57ce63a5d41a0a8d39f3eddb92c3c6c8',
      'beta_sg12': 'a4c08730575268f0e28ded9d1ac46caacbd3302586efd885f82ff89365bc2f49'}


def fit(p, feats, exclude_mask, alpha=1.0):
    q = p[~exclude_mask]
    X = q[feats].values.astype(float)
    X = np.where(np.isnan(X), 0.0, X)
    y = q.d_Band_gap_HSE.values
    w = q.weight.values
    sc = StandardScaler().fit(np.vstack([X, -X]))
    Z = (X - sc.mean_) / sc.scale_
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(np.vstack([Z, -Z]), np.concatenate([y, -y]), sample_weight=np.concatenate([w, w]))
    def pred(row):
        x = np.array([row[f] for f in feats], float)
        x = np.where(np.isnan(x), 0.0, x)
        return float(m.predict(((x - sc.mean_) / sc.scale_).reshape(1, -1))[0])
    return pred, len(q), m, sc


def framework(path, keep, merge_tol=0.6):
    """read a CIF with partial occupancies, keep only the `keep` elements, average split
    positions within `merge_tol` angstrom into one site and set every occupancy to 1.
    The alpha-FAPbI3 reference CIF has the iodine on a 4-fold split site of occupancy 0.25
    (x = 0.962 instead of 1.0, i.e. 0.24 A off the ideal octahedral position); averaging it
    recovers the ideal corner-sharing PbI3 framework.  For delta-FAPbI3 and both Ga2O3
    references the merge is a no-op."""
    st = CifParser(str(path), occupancy_tolerance=100).parse_structures(primitive=False)[0]
    sp, co = [], []
    for s in st:
        el = max(s.species, key=lambda e: s.species[e]).symbol
        if el in keep:
            sp.append(el)
            co.append(s.frac_coords)
    out = Structure(st.lattice, sp, co)
    n0 = len(out)
    out.merge_sites(tol=merge_tol, mode='average')
    out = Structure(out.lattice, [s.specie.symbol for s in out], [s.frac_coords for s in out])
    print('   %s: %d kept sites -> %d after merging split positions' % (Path(path).name[:8], n0, len(out)))
    return out


def desc_of(st, cat_an):
    return descriptors(st, {}, cat_an=cat_an)


def pairrow(a, b, STRUCT):
    r = {}
    for k in STRUCT:
        r['d_' + k] = float(b[k]) - float(a[k])
        r['m_' + k] = 0.5 * (float(b[k]) + float(a[k]))
    r['same_crystal_system'] = int(a['crystal_system'] == b['crystal_system'])
    return r


def main():
    import pairs_analysis as PA
    p = pd.read_csv(D / 'pairs.csv')
    feats_all = ['d_' + k for k in PA.STRUCT] + ['same_crystal_system']
    out = []

    # ---------- Ga2O3 inside SNUMAT
    d = pd.read_csv(D / 'descriptors.csv')
    g = d[d.formula == 'Ga2O3']
    print('SNUMAT Ga2O3 records', len(g))
    if len(g):
        print(g[['snumat_id', 'Space_group_rlx', 'sg_number', 'sg_symbol', 'Band_gap_HSE',
                 'Band_gap_HSE_optical', 'dim_cat_an', 'cn_cat_mean', 'mean_shared_vertices',
                 'frac_corner', 'frac_edge', 'frac_face', 'bond_cat_an_mean', 'distortion_index',
                 'vol_per_atom']].to_string(index=False))
    gp = p[p.formula == 'Ga2O3']
    print('\nSNUMAT Ga2O3 pairs', len(gp))
    excl = p.elem_system.str.contains('Ga') | p.elem_system.eq('Ga,O')
    for nm, feats, al in (('six', SIX, 1.0), ('all', feats_all, 10.0)):
        pr, n, _, _ = fit(p, feats, excl, al)
        for _, r in gp.iterrows():
            out.append(dict(case='Ga2O3 (SNUMAT)', model=nm, n_train=n, pair='%d->%d' % (r.sg_a, r.sg_b),
                            observed=r.d_Band_gap_HSE, predicted=pr(r)))
    # ---------- Ga2O3 experimental reference CIFs (alpha R-3c vs beta C2/m)
    gd = {k: desc_of(framework(CIF / (v + '.cif'), {'Ga', 'O'}), ({'Ga'}, {'O'})) for k, v in GA.items()}
    ra = pairrow(gd['beta_sg12'], gd['alpha_sg167'], PA.STRUCT)   # a = beta (sg 12), b = alpha (sg 167)
    print('\nGa2O3 experimental reference CIFs, beta -> alpha differences:')
    print({k: round(v, 4) for k, v in ra.items() if k.startswith('d_') and k[2:] in
           ('dim_cat_an', 'mean_shared_vertices', 'bond_cat_an_mean', 'cn_cat_mean', 'distortion_index',
            'vol_per_atom', 'frac_corner', 'frac_edge', 'frac_face')})
    for nm, feats, al in (('six', SIX, 1.0), ('all', feats_all, 10.0)):
        pr, n, _, _ = fit(p, feats, excl, al)
        out.append(dict(case='Ga2O3 (experimental CIFs)', model=nm, n_train=n, pair='beta(12)->alpha(167)',
                        observed=np.nan, predicted=pr(ra)))
    # ---------- FAPbI3
    fd = {k: desc_of(framework(CIF / (v + '.cif'), {'Pb', 'I'}), ({'Pb'}, {'I'})) for k, v in FA.items()}
    for k, v in fd.items():
        print('\nFAPbI3 %s framework: %s sites, sg %s %s, dim %s, CN %.2f, shared %.2f, corner %.2f edge %.2f face %.2f, bond %.3f A, distortion %.4f, V/atom %.2f'
              % (k, v['n_sites'], v['sg_number'], v['sg_symbol'], v['dim_cat_an'], v['cn_cat_mean'],
                 v['mean_shared_vertices'], v['frac_corner'], v['frac_edge'], v['frac_face'],
                 v['bond_cat_an_mean'], v['distortion_index'], v['vol_per_atom']))
    rf = pairrow(fd['delta_sg186'], fd['alpha_sg215'], PA.STRUCT)  # a = delta (186), b = alpha (215)
    exclf = p.elem_system.str.contains('Pb') | p.elem_system.str.contains('I')
    for nm, feats, al in (('six', SIX, 1.0), ('all', feats_all, 10.0)):
        pr, n, _, _ = fit(p, feats, exclf, al)
        out.append(dict(case='FAPbI3 (PbI3 framework, experimental CIFs)', model=nm, n_train=n,
                        pair='delta(186)->alpha(215)', observed=1.47 - 2.43, predicted=pr(rf)))
    O = pd.DataFrame(out)
    O.to_csv(D / 'tables' / 'case_checks.csv', index=False)
    print('\n== case checks (observed and predicted gap difference b - a, eV) ==')
    print(O.to_string(index=False))


if __name__ == '__main__':
    main()
