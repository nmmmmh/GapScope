"""D_c from polymorphs WITH a free energy only, against the main-text definition (no-energy polymorphs kept).
(a) the 50 traced compositions: AUC for floor > 0.05 eV with bootstrap CI, rho, tercile medians, which compositions change;
(b) Borlido benchmark and all SNUMAT compositions: share with D_c >= 0.3 eV."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
def dc(W, need_e):
    out = {}
    for k, c in sub.groupby('composition_key'):
        if need_e: c = c[c.Free_energy.notna()]
        if W is not None and c.Free_energy.notna().any():
            e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= W)]
        s = c.Band_gap_HSE; out[k] = ((s.max() - s.min()) if s.notna().sum() >= 2 else 0.0, int(s.notna().sum()))
    return np.array([out.get(k, (0.0, 0)) for k in keys])
pos = F > 0.05; rng = np.random.default_rng(20261118)
def ci(x):
    b = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        if pos[i].min() != pos[i].max(): b.append(roc_auc_score(pos[i], x[i]))
    return np.percentile(b, [2.5, 97.5])
def terc(x):
    r = pd.qcut(pd.Series(x).rank(method='first'), 3, labels=['low', 'mid', 'high']); return pd.Series(F).groupby(r).median().round(3).to_dict()
print('(a) 50 compositions, positives', pos.sum())
for W in (0.10, 0.05, None):
    for need_e in (False, True):
        d = dc(W, need_e); x = d[:, 0]; lo, hi = ci(x)
        print('window %-4s %-12s AUC %.3f [%.2f, %.2f] rho %.2f | D_c>0 %2d | polymorphs/comp %.2f | terciles %s'
              % (W, 'energy-only' if need_e else 'main', roc_auc_score(pos, x), lo, hi, spearmanr(x, F).statistic, (x > 0).sum(), d[:, 1].mean(), terc(x)))
A, E = dc(0.10, False)[:, 0], dc(0.10, True)[:, 0]
fm = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig').drop_duplicates('composition_key').set_index('composition_key').formula
X = pd.DataFrame(dict(formula=[fm.get(k, k) for k in keys], floor=F, Dc_main=A, Dc_energy=E)); X['chg'] = (X.Dc_main - X.Dc_energy).abs() > 1e-6
print('changed', X.chg.sum(), '| of which floor>0.05', int((X.chg & (X.floor > 0.05)).sum()))
print(X[X.chg].sort_values('floor', ascending=False).round(3).to_string(index=False))
# (b)
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def dcS(c, need_e):
    c = c[c.hse.notna()]
    if need_e: c = c[c.Free_energy.notna()]
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= 0.10)]
    return (c.hse.max() - c.hse.min()) if len(c) >= 2 else 0.0
G = {k: (dcS(c, False), dcS(c, True)) for k, c in S.groupby('ckey')}
from pymatgen.core import Composition
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
Bk = [k for k in {key(f) for f in pd.read_csv(R + 'borlido/b1_borlido_Dc.csv').formula.unique()} if k in G]
for lab, ks in (('Borlido', Bk), ('all SNUMAT', list(G))):
    m = np.array([G[k] for k in ks])
    print('(b) %-10s n %4d | D_c>=0.3 main %4d (%.1f%%) energy-only %4d (%.1f%%)' % (lab, len(ks), (m[:, 0] >= 0.3).sum(), 100 * (m[:, 0] >= 0.3).mean(), (m[:, 1] >= 0.3).sum(), 100 * (m[:, 1] >= 0.3).mean()))
