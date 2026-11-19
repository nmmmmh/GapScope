"""D_c (eq 1 of the manuscript) for the compositions of the Borlido et al. 2019 functional benchmark, from the SNUMAT HSE06 table."""
import re, numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def key(f):
    c = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', f): c[el] = c.get(el, 0) + (float(n) if n else 1.0)
    t = sum(c.values()); return '|'.join('%s:%.12f' % (e, c[e] / t) for e in sorted(c))
def dc(c, W=0.10):
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= W)]
    return (c.hse.max() - c.hse.min()) if c.hse.notna().sum() >= 2 else 0.0, len(c)
G = {k: g for k, g in S.groupby('ckey')}
# consistency with the published D_c of three traced compositions
for f, ref in (('SiC', 0.6615), ('Ga2O3', 0.4743), ('MoS2', 0.0696)): print(f, round(dc(G[key(f)])[0], 4), ref)
F = open(R + 'borlido/borlido_formulas.txt').read().split()
rows = []
for f in F:
    k = key(f)
    if k in G:
        d, n = dc(G[k]); rows.append(dict(formula=f, in_snumat=True, n_sg=len(G[k]), n_acc=n, D_c=d))
    else: rows.append(dict(formula=f, in_snumat=False, n_sg=0, n_acc=0, D_c=np.nan))
T = pd.DataFrame(rows); T.to_csv(R + 'borlido/b1_borlido_Dc.csv', index=False)
m = T[T.in_snumat]
print('compositions', len(T), 'in SNUMAT', len(m), 'with >=2 accessible polymorphs', (m.n_acc >= 2).sum())
for th in (0.1, 0.2, 0.3, 0.5, 1.0): print('D_c >= %.1f eV: %d (%.1f%% of matched)' % (th, (m.D_c >= th).sum(), 100 * (m.D_c >= th).mean()))
print(m.sort_values('D_c', ascending=False).head(25).to_string(index=False))
