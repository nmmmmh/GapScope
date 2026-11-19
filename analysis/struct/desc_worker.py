"""Stage 1: physically interpretable structure descriptors for every SNUMAT record.
Usage: python -I desc_worker.py <shard> <nshards> <out_csv> [limit]
Each shard takes records with index % nshards == shard. Writes one csv shard.
"""
import sys, os, json, math, signal, warnings, traceback
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter
from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis import bond_valence as bvmod
from pymatgen.core.sites import PeriodicSite

SNUMAT = '/home/czx/2024/doped/data/snumat.json'
LAUE = {'-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m', '6/m', '6/mmm', 'm-3', 'm-3m'}
CNN = CrystalNN()


class TO(Exception):
    pass


class _N(object):
    """minimal neighbour object for pymatgen.analysis.bond_valence.calculate_bv_sum,
    which needs only .specie and .nn_distance"""

    def __init__(self, specie, d):
        self.specie = specie
        self.nn_distance = d


def _alarm(s, f):
    raise TO()


signal.signal(signal.SIGALRM, _alarm)

POLY = {1: 'monomer', 2: 'linear', 3: 'triangular', 4: 'tetrahedral', 5: 'five-fold', 6: 'octahedral',
        7: 'seven-fold', 8: 'eight-fold', 9: 'nine-fold', 10: 'ten-fold', 11: 'eleven-fold', 12: 'twelve-fold'}


def nval(el):
    """valence electron count: electrons in the highest principal shell plus an incomplete
    (n-1)d and (n-2)f sub-shell."""
    fes = el.full_electronic_structure
    ns = max(n for n, o, e in fes)
    v = sum(e for n, o, e in fes if n == ns)
    v += sum(e for n, o, e in fes if n == ns - 1 and o == 'd' and e < 10)
    v += sum(e for n, o, e in fes if n == ns - 2 and o == 'f' and e < 14)
    return v


def rad(el):
    for a in ('atomic_radius', 'atomic_radius_calculated'):
        r = getattr(el, a, None)
        if r:
            return float(r)
    return 1.5


def ckey(comp):
    f = comp.fractional_composition.as_dict()
    return '|'.join('%s:%.12f' % (el, f[el]) for el in sorted(f))


def classify(els):
    """anion set = the most electronegative element plus every element within 0.4 Pauling
    units of it; cations = the rest.  Purely compositional, so it is identical for the two
    members of a polymorph pair."""
    X = {}
    for e in els:
        x = Element(e).X
        X[e] = float(x) if x is not None and not math.isnan(float(x)) else 0.0
    xm = max(X.values())
    an = set(e for e in els if xm - X[e] <= 0.4)
    cat = set(els) - an
    return cat, an, X


def stats(v, pre, out):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        for s in ('mean', 'min', 'max', 'std', 'spread'):
            out[pre + '_' + s] = np.nan
        return
    out[pre + '_mean'] = float(v.mean())
    out[pre + '_min'] = float(v.min())
    out[pre + '_max'] = float(v.max())
    out[pre + '_std'] = float(v.std())
    out[pre + '_spread'] = float(v.max() - v.min())


def empty_graph(st):
    try:
        return StructureGraph.from_empty_graph(st)
    except AttributeError:
        return StructureGraph.with_empty_graph(st)


def descriptors(st, rec, cat_an=None):
    o = {}
    comp = st.composition
    els = [str(s) for s in comp.elements]
    cat, an, X = classify(els)
    if cat_an is not None:      # explicit override, used only for the two manuscript cases
        cat, an = set(cat_an[0]), set(cat_an[1])
    o['n_sites'] = len(st)
    o['n_elements'] = len(els)
    o['formula'] = comp.reduced_formula
    o['composition_key'] = ckey(comp)
    o['cations'] = ','.join(sorted(cat))
    o['anions'] = ','.join(sorted(an))
    # ---- packing / density
    o['volume'] = float(st.volume)
    o['vol_per_atom'] = float(st.volume / len(st))
    o['density'] = float(st.density)
    o['packing_fraction'] = float(sum(4 / 3 * math.pi * rad(s.specie) ** 3 for s in st) / st.volume)
    # ---- chemistry (composition only)
    fr = comp.fractional_composition.as_dict()
    wc = sum(fr[e] for e in cat)
    o['anion_frac'] = float(sum(fr[e] for e in an))
    o['en_cat_mean'] = float(sum(fr[e] * X[e] for e in cat) / wc) if wc > 0 else np.nan
    o['en_an_mean'] = float(sum(fr[e] * X[e] for e in an) / o['anion_frac'])
    o['en_diff'] = o['en_an_mean'] - o['en_cat_mean'] if wc > 0 else np.nan
    o['ionicity'] = 1 - math.exp(-0.25 * o['en_diff'] ** 2) if wc > 0 else np.nan
    o['en_std_all'] = float(np.std([X[e] for e in els]))
    o['nval_per_atom'] = float(sum(fr[e] * nval(Element(e)) for e in els))
    o['radius_ratio'] = float((sum(fr[e] * rad(Element(e)) for e in cat) / wc) /
                              (sum(fr[e] * rad(Element(e)) for e in an) / o['anion_frac'])) if wc > 0 else np.nan
    o['mean_atomic_radius'] = float(sum(fr[e] * rad(Element(e)) for e in els))
    # ---- symmetry
    try:
        signal.alarm(120)
        sa = SpacegroupAnalyzer(st, symprec=0.1, angle_tolerance=5)
        o['sg_number'] = int(sa.get_space_group_number())
        o['sg_symbol'] = sa.get_space_group_symbol()
        o['crystal_system'] = sa.get_crystal_system()
        o['lattice_type'] = sa.get_lattice_type()
        o['point_group'] = sa.get_point_group_symbol()
        o['bravais'] = o['crystal_system'] + '-' + o['sg_symbol'][0]
        o['centrosymmetric'] = int(o['point_group'] in LAUE)
        o['n_symops'] = len(sa.get_symmetry_operations())
    except Exception:
        for k in ('sg_number', 'n_symops', 'centrosymmetric'):
            o[k] = np.nan
        for k in ('sg_symbol', 'crystal_system', 'lattice_type', 'point_group', 'bravais'):
            o[k] = ''
    finally:
        signal.alarm(0)
    # ---- CrystalNN
    signal.alarm(900)
    try:
        allnn = CNN.get_all_nn_info(st)
    finally:
        signal.alarm(0)
    spec = [str(s.specie) for s in st]
    iscat = [spec[i] in cat for i in range(len(st))]
    cn = [len(allnn[i]) for i in range(len(st))]
    stats([cn[i] for i in range(len(st)) if iscat[i]], 'cn_cat', o)
    stats([cn[i] for i in range(len(st)) if not iscat[i]], 'cn_an', o)
    stats(cn, 'cn_all', o)
    cc = [cn[i] for i in range(len(st)) if iscat[i]]
    if cc:
        dom = Counter(cc).most_common(1)[0][0]
        o['dom_cn_cat'] = int(dom)
        o['dom_poly_cat'] = POLY.get(dom, '%d-fold' % dom)
        o['dom_poly_frac'] = float(sum(1 for c in cc if c == dom) / len(cc))
    else:
        o['dom_cn_cat'] = np.nan
        o['dom_poly_cat'] = ''
        o['dom_poly_frac'] = np.nan
    # ---- bond geometry (cation-anion bonds only)
    bl, dist = [], []
    for i in range(len(st)):
        if not iscat[i]:
            continue
        ds = [float(np.linalg.norm(nn['site'].coords - st[i].coords))
              for nn in allnn[i] if not iscat[nn['site_index']]]
        if not ds:
            continue
        bl += ds
        m = float(np.mean(ds))
        dist.append(float(np.mean([abs(d - m) for d in ds]) / m))   # Baur distortion index
    o['bond_cat_an_mean'] = float(np.mean(bl)) if bl else np.nan
    o['bond_cat_an_std'] = float(np.std(bl)) if bl else np.nan
    o['bond_cat_an_rel_std'] = (o['bond_cat_an_std'] / o['bond_cat_an_mean']) if bl else np.nan
    o['distortion_index'] = float(np.mean(dist)) if dist else np.nan
    o['distortion_index_max'] = float(np.max(dist)) if dist else np.nan
    # ---- connectivity of cation-centred anion polyhedra
    V = {}
    for i in range(len(st)):
        if iscat[i]:
            V[i] = set((nn['site_index'], tuple(int(round(x)) for x in nn['image']))
                       for nn in allnn[i] if not iscat[nn['site_index']])
    conns = {}
    ks = sorted(V)
    for ai, i in enumerate(ks):
        for j in ks[ai:]:
            Ts = set()
            for k1, m1 in V[i]:
                for k2, m2 in V[j]:
                    if k1 != k2:
                        continue
                    T = (m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2])
                    if i == j and T == (0, 0, 0):
                        continue
                    Ts.add(T)
            for T in Ts:
                shn = len(V[i] & set((k, (m[0] + T[0], m[1] + T[1], m[2] + T[2])) for k, m in V[j]))
                if shn == 0:
                    continue
                key = (i, j, T)
                if i == j:
                    key = (i, j, max(T, (-T[0], -T[1], -T[2])))
                conns[key] = shn
    sh = list(conns.values())
    o['n_connections'] = len(sh)
    o['conn_per_cation'] = len(sh) / len(V) if V else np.nan
    if sh:
        o['frac_corner'] = float(np.mean([s == 1 for s in sh]))
        o['frac_edge'] = float(np.mean([s == 2 for s in sh]))
        o['frac_face'] = float(np.mean([s >= 3 for s in sh]))
        o['mean_shared_vertices'] = float(np.mean(sh))
        o['max_shared_vertices'] = int(max(sh))
    else:
        o['frac_corner'] = o['frac_edge'] = o['frac_face'] = 0.0
        o['mean_shared_vertices'] = 0.0
        o['max_shared_vertices'] = 0
    linked = set()
    for (i, j, T) in conns:
        linked.add(i)
        linked.add(j)
    o['frac_isolated_poly'] = float(1 - len(linked) / len(V)) if V else np.nan
    # ---- dimensionality
    try:
        g = empty_graph(st)
        gca = empty_graph(st)
        for i in range(len(st)):
            for nn in allnn[i]:
                jm = tuple(int(round(x)) for x in nn['image'])
                g.add_edge(i, nn['site_index'], from_jimage=(0, 0, 0), to_jimage=jm, warn_duplicates=False)
                if iscat[i] != iscat[nn['site_index']]:
                    gca.add_edge(i, nn['site_index'], from_jimage=(0, 0, 0), to_jimage=jm, warn_duplicates=False)
        o['dim_all'] = int(get_dimensionality_larsen(g))
        o['dim_cat_an'] = int(get_dimensionality_larsen(gca))
    except Exception:
        o['dim_all'] = o['dim_cat_an'] = np.nan
    # ---- bond valence sums (needs oxidation states; guessed from the composition)
    o['bvs_cat_mean'] = np.nan
    o['bvs_dev_mean'] = np.nan
    o['oxi_ok'] = 0
    try:
        signal.alarm(20)
        g = comp.oxi_state_guesses()
        signal.alarm(0)
        if g:
            ox = g[0]
            bvs, dev = [], []
            for i in range(len(st)):
                if not iscat[i]:
                    continue
                nl2 = []
                for nn in allnn[i]:
                    if iscat[nn['site_index']]:
                        continue
                    nl2.append(_N(nn['site'].specie,
                                  float(np.linalg.norm(nn['site'].coords - st[i].coords))))
                if not nl2:
                    continue
                v = bvmod.calculate_bv_sum(st[i], nl2)
                bvs.append(v)
                dev.append(abs(v - ox[spec[i]]))
            if bvs:
                o['bvs_cat_mean'] = float(np.mean(bvs))
                o['bvs_dev_mean'] = float(np.mean(dev))
                o['oxi_ok'] = 1
    except Exception:
        pass
    finally:
        signal.alarm(0)
    return o


def main():
    shard, nsh, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    data = json.load(open(SNUMAT))
    if isinstance(data, dict):
        data = list(data.values())
    rows, fails = [], []
    for idx, rec in enumerate(data):
        if lim and idx >= lim:
            break
        if idx % nsh != shard:
            continue
        rid = rec.get('SNUMAT_id', rec.get('ICSD_number', idx))
        try:
            a = rec['atoms']
            st = Structure(a['lattice_mat'], a['elements'], a['coords'], coords_are_cartesian=False)
            o = descriptors(st, rec)
        except Exception as e:
            fails.append(dict(idx=idx, snumat_id=rid, n_sites=len(rec.get('atoms', {}).get('elements', [])),
                              error=('%s: %s' % (type(e).__name__, e))[:200]))
            continue
        o['idx'] = idx
        o['snumat_id'] = rid
        for k in ('Band_gap_HSE', 'Band_gap_HSE_optical', 'Band_gap_GGA', 'Band_gap_GGA_optical',
                  'Direct_or_indirect', 'Direct_or_indirect_HSE', 'SOC', 'Space_group_rlx',
                  'Magnetic_ordering', 'ICSD_number'):
            o[k] = rec.get(k)
        rows.append(o)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(fails).to_csv(out.replace('.csv', '_fail.csv'), index=False)
    print('shard', shard, 'ok', len(rows), 'fail', len(fails), flush=True)


if __name__ == '__main__':
    main()
