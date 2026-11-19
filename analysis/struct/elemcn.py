"""Is the width readable from the elements alone?

For every element, collect the coordination numbers it adopts as the cation of a
single-cation compound, across the whole SNUMAT pool.  Build a composition-level
descriptor from those repertoires only -- never from the polymorphs of the
composition being scored -- and test it against the reported range.
"""
import json, collections, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, mannwhitneyu

D = pd.read_csv('/data/czx/GapScope_Revision_20261113/struct/descriptors.csv')
W = pd.read_csv('/data/czx/GapScope_Revision_20261113/struct/width.csv')
print('pool', len(D), 'width set', len(W), [c for c in W.columns][:14])

D = D[D.dom_cn_cat.notna() & D.cations.notna()]
D['catlist'] = D.cations.map(lambda s: [e for e in str(s).replace(';', ',').split(',') if e.strip()])
single = D[D.catlist.map(len) == 1].copy()
single['cat'] = single.catlist.map(lambda c: c[0].strip())
print('single-cation records', len(single), 'distinct cations', single.cat.nunique())

def repertoire(exclude_keys=(), exclude_elems=()):
    rep = collections.defaultdict(collections.Counter)
    s = single
    if exclude_keys:
        s = s[~s.composition_key.isin(exclude_keys)]
    for e, cn in zip(s.cat, s.dom_cn_cat):
        rep[e][int(round(float(cn)))] += 1
    return rep

def feats(cats, rep, minfrac=0.10, minrec=4):
    """per-composition features from the element repertoires only"""
    out = []
    for e in cats:
        c = rep.get(e)
        if not c:
            continue
        tot = sum(c.values())
        if tot < minrec:
            continue
        keep = [k for k, v in c.items() if v / tot >= minfrac]
        if not keep:
            keep = [max(c, key=c.get)]
        p = np.array([c[k] for k in keep], float); p /= p.sum()
        out.append(dict(n_cn=len(keep), cn_range=max(keep) - min(keep),
                        ent=float(-(p * np.log(p)).sum()), tot=tot))
    if not out:
        return None
    return dict(n_cn_max=max(o['n_cn'] for o in out),
                cn_range_max=max(o['cn_range'] for o in out),
                ent_max=max(o['ent'] for o in out),
                flex=int(any(o['n_cn'] >= 2 for o in out)))

from pymatgen.core.periodic_table import Element
def cations_of(key):
    els = [p.split(':')[0] for p in str(key).split('|') if p]
    if len(els) < 2:
        return []
    x = {e: float(Element(e).X or 2.0) for e in els}
    top = max(x.values())
    return [e for e in els if top - x[e] > 0.4]

keycol = 'composition_key'
rows = []
for _, r in W.iterrows():
    cats = cations_of(r[keycol])
    if not cats:
        continue
    rep = repertoire(exclude_keys={r[keycol]})       # no leakage from this composition
    f = feats(cats, rep)
    if f is None:
        continue
    f.update(key=r[keycol], formula=r.get('formula'), exp_range=float(r['exp_range']),
             n_reports=float(r.get('n_reports', np.nan)),
             set=r.get('set'),
             var_coordination=float(r['var_coordination']) if 'var_coordination' in W.columns and not pd.isna(r['var_coordination']) else np.nan)
    rows.append(f)
T0 = pd.DataFrame(rows)
print('scored compositions', len(T0), dict(T0.set.value_counts()))

def auc(lab, sc):
    lab = np.asarray(lab, bool); sc = np.asarray(sc, float)
    r = rankdata(sc); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    c = lambda a, b: np.corrcoef(a, b)[0, 1]
    return (c(rx, ry) - c(rx, rz) * c(ry, rz)) / np.sqrt((1 - c(rx, rz) ** 2) * (1 - c(ry, rz) ** 2))

T0['exp_range'] = pd.to_numeric(T0['exp_range'], errors='coerce')
T0 = T0[np.isfinite(T0.exp_range)]
print('with finite exp_range:', dict(T0.set.value_counts()))

for SET in ('provenance', 'compilation'):
    T = T0[T0.set == SET].reset_index(drop=True)
    if len(T) < 10:
        continue
    y = T.exp_range.values
    print()
    print('=============== %s  (n = %d)' % (SET, len(T)))
    for f in ('n_cn_max', 'cn_range_max', 'ent_max', 'flex'):
        v = T[f].values.astype(float)
        if len(set(v)) < 2:
            print('  %-13s constant' % f); continue
        rho, p = spearmanr(v, y)
        a = auc(y > 0.20, v)
        pr = partial(y, v, T.n_reports.values.astype(float))
        print('  %-13s rho=%+.3f p=%.4f   AUC(range>0.2)=%.3f   partial on n_reports=%+.3f'
              % (f, rho, p, a, pr))
    m = T.var_coordination.notna().values
    if m.sum() > 10:
        print('  agreement with the structure-derived descriptor: rho=%.3f (n=%d)'
              % (spearmanr(T.flex.values[m], T.var_coordination.values[m])[0], int(m.sum())))
        yy = y[m]; ff = T.flex.values[m]
        if len(set(ff)) > 1:
            print('  on those: elem-level rho=%.3f   structure-level rho=%.3f'
                  % (spearmanr(ff, yy)[0], spearmanr(T.var_coordination.values[m], yy)[0]))
    g0, g1 = y[T.flex == 0], y[T.flex == 1]
    if len(g0) > 2 and len(g1) > 2:
        print('  median reported range  rigid %.3f (n=%d)   flexible %.3f (n=%d)  P=%.4f'
              % (np.median(g0), len(g0), np.median(g1), len(g1), mannwhitneyu(g0, g1).pvalue))
T0.to_csv('/data/czx/GapScope_Revision_20261113/struct/elem_cn.csv', index=False)
