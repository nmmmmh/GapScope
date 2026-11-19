"""Energy-only D_c (polymorphs without a computed energy dropped before the window): all numbers the text and SI quote."""
import numpy as np, pandas as pd
from pymatgen.core import Composition
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0].replace("print('cells", "pass #"))
subE = sub[sub.Free_energy.notna()]
def descE(W):
    out = {}
    for k, c in subE.groupby('composition_key'):
        if W is not None:
            c = c[c.Free_energy - c.Free_energy.min() <= W]
        sp = lambda s: (s.max() - s.min()) if s.notna().sum() >= 2 else 0.0
        out[k] = (sp(c.cn_cat_mean) + sp(c.mean_shared_vertices), sp(c.Band_gap_HSE), len(c))
    return np.array([out.get(k, (0.0, 0.0, 0)) for k in keys])
pos = F >= 0.05 - 1e-9; rng = np.random.default_rng(20261118)
def ci(x):
    b = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        if pos[i].min() != pos[i].max(): b.append(roc_auc_score(pos[i], x[i]))
    return np.percentile(b, [2.5, 97.5])
print('positives', pos.sum(), 'of', len(F))
for W in (None, 0.30, 0.20, 0.10, 0.05):
    d = descE(W)
    for j, nm in ((1, 'D_c'), (0, 'D_struct')):
        x = d[:, j]; lo, hi = ci(x)
        print('W %-4s %-8s AUC %.3f [%.2f, %.2f] rho %.2f | polymorphs/comp %.2f' % (W, nm, roc_auc_score(pos, x), lo, hi, spearmanr(x, F).statistic, d[:, 2].mean()))
    rm = pd.Series(d[:, 1]).rank().values + pd.Series(d[:, 0]).rank().values; lo, hi = ci(rm)
    print('W %-4s rankmean AUC %.3f [%.2f, %.2f] rho %.2f' % (W, roc_auc_score(pos, rm), lo, hi, spearmanr(rm, F).statistic))
d = descE(0.10); Dc = d[:, 1]
r = pd.qcut(pd.Series(Dc).rank(method='first'), 3, labels=['low', 'mid', 'high']); print('terciles', pd.Series(F).groupby(r).median().round(3).to_dict())
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig')
fm = P.drop_duplicates('composition_key').set_index('composition_key').formula; nrep = P.groupby('composition_key').size()
X = pd.DataFrame(dict(formula=[fm.get(k, k) for k in keys], floor=F, Dc=Dc, nrep=[nrep.get(k, 0) for k in keys], ncell=d[:, 2]))
fl = X.Dc >= 0.3; tp = (fl & pos).sum()
print('threshold 0.3: flagged %d, TP %d, precision %.2f, recall %.2f (%d/%d), base %.2f; median floor flagged %.3f others %.3f'
      % (fl.sum(), tp, tp / fl.sum(), tp / pos.sum(), tp, pos.sum(), pos.mean(), X.floor[fl].median(), X.floor[~fl].median()))
for lab, s in (('2-3', X.nrep.between(2, 3)), ('>=3', X.nrep >= 3), ('>=4', X.nrep >= 4)):
    y = pos[s.values]; print('strata %s n %d AUC %.3f' % (lab, s.sum(), roc_auc_score(y, X.Dc[s])))
print('rho Dc nrep %.2f' % spearmanr(X.Dc, X.nrep).statistic)
print('MISSES'); print(X[pos & ~fl].sort_values('floor', ascending=False).round(3).to_string(index=False))
print('FALSE FLAGS'); print(X[~pos & fl].round(3).to_string(index=False))
# SNUMAT / Borlido
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
G = {}
for k, c in S.groupby('ckey'):
    c = c[c.hse.notna() & c.Free_energy.notna()]
    c = c[c.Free_energy - c.Free_energy.min() <= 0.10] if len(c) else c
    G[k] = ((c.hse.max() - c.hse.min()) if len(c) >= 2 else 0.0, len(c))
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
Bk = [k for k in {key(f) for f in pd.read_csv(R + 'borlido/b1_borlido_Dc.csv').formula.unique()} if k in G]
for lab, ks in (('Borlido', Bk), ('SNUMAT', list(G))):
    m = np.array([G[k] for k in ks]); n2 = m[:, 1] >= 2
    print('%-8s n %d | >=2 acc %d | Dc>=0.1 %d >=0.3 %d (%.1f%%) >=1.0 %d | given >=2 acc: %.0f%%' % (lab, len(ks), n2.sum(), (m[:, 0] >= 0.1).sum(), (m[:, 0] >= 0.3).sum(), 100 * (m[:, 0] >= 0.3).mean(), (m[:, 0] >= 1.0).sum(), 100 * (m[n2, 0] >= 0.3).mean()))
