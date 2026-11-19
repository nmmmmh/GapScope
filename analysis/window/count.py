import sys
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, pandas as pd
import evaluate_cv_v4 as E
ev = E.ev
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
def strad(y, a, b):
    o = (y >= a) & (y <= b); return o.any() and (~o).any()
B = {k: np.array(v) for k, v in ev.Bvals.items()}
m = {k: v for k, v in B.items() if len(v) >= 2}
print('compiled formulas', len(B), 'with >=2 gaps', len(m), 'spread>=0.3', sum(v.max() - v.min() >= 0.3 for v in m.values()))
for wn, (a, b) in W.items(): print('  compiled straddle', wn, sum(strad(v, a, b) for v in m.values()))
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
print('SNUMAT rows', len(S), 'comps', S.ckey.nunique(), 'Free_energy non-null', S.Free_energy.notna().sum())
g = S.groupby('ckey')
multi = [k for k, t in g if len(t) >= 2]
print('SNUMAT comps with >=2 space groups', len(multi))
for wn, (a, b) in W.items(): print('  SNUMAT straddle (all polymorphs)', wn, sum(strad(g.get_group(k).hse.values, a, b) for k in multi))
# accessible within 0.1 eV/atom where energies known
acc = []
for k, t in g:
    t = t.dropna(subset=['Free_energy'])
    if len(t) >= 2:
        t = t[t.Free_energy <= t.Free_energy.min() + 0.1]
        if len(t) >= 2: acc.append((k, t.hse.values))
print('SNUMAT comps >=2 accessible (energy known)', len(acc))
for wn, (a, b) in W.items(): print('  accessible straddle', wn, sum(strad(v, a, b) for _, v in acc))
Bk = set(B); Ak = set(ev.Akey); Sk = set(S.ckey)
print('SNUMAT overlap compiled', len(Sk & Bk), 'traced', len(Sk & Ak), 'multi-sg overlap compiled', len(set(multi) & Bk))
