"""Controls for the 'well-studied compounds are more often polymorphic' trend.
(a) stratify by number of elements; (b) accessible polymorphs counted only among polymorphs WITH a free energy;
(c) within compiled compositions, logistic check of polymorphism vs number of values controlling for n_elements and n SNUMAT records."""
import numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def acc(c, need_energy):
    if need_energy: c = c[c.Free_energy.notna()]
    if len(c) == 0: return c
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= 0.10)]
    return c[c.hse.notna()]
rows = []
for k, c in S.groupby('ckey'):
    a, b = acc(c, False), acc(c, True)
    rows.append(dict(key=k, nel=k.count('|') + 1, nrec=int(c.n_rec.sum()), nsg=len(c),
                     poly=len(a) >= 2, dc=(a.hse.max() - a.hse.min()) if len(a) >= 2 else 0.0,
                     polyE=len(b) >= 2, dcE=(b.hse.max() - b.hse.min()) if len(b) >= 2 else 0.0))
T = pd.DataFrame(rows).set_index('key')
A = pd.read_csv(R + 'cv/data/tierB_all_rows.csv').drop_duplicates(['key', 'Eg'])
nv = A.groupby('key').Eg.size()
T['nv'] = nv.reindex(T.index)
T['grp'] = np.where(T.nv.isna(), 'SNUMAT only', np.where(T.nv >= 3, 'compiled >=3', np.where(T.nv == 2, 'compiled 2', 'compiled 1')))
order = ['SNUMAT only', 'compiled 1', 'compiled 2', 'compiled >=3']
print('(a) share >=2 accessible / D_c>=0.3, by number of elements')
for ne in (1, 2, 3, 4):
    t = T[T.nel == ne] if ne < 4 else T[T.nel >= 4]
    print('  n_el %s: ' % (str(ne) if ne < 4 else '>=4') + ' | '.join('%s n %d %.0f%%/%.0f%%' % (g, (t.grp == g).sum(), 100 * t[t.grp == g].poly.mean(), 100 * (t[t.grp == g].dc >= 0.3).mean()) for g in order if (t.grp == g).sum() >= 5))
print('(b) energy-required accessible polymorphs: ' + ' | '.join('%s %.0f%%/%.0f%%' % (g, 100 * T[T.grp == g].polyE.mean(), 100 * (T[T.grp == g].dcE >= 0.3).mean()) for g in order))
print('(c) mean SNUMAT records per composition: ' + ' | '.join('%s %.1f (median %d)' % (g, T[T.grp == g].nrec.mean(), T[T.grp == g].nrec.median()) for g in order))
# (d) at fixed number of SNUMAT space-group cells, is the D_c>=0.3 share higher for well-measured compositions?
print('(d) D_c>=0.3 share among compositions with >=2 accessible polymorphs: ' + ' | '.join('%s n %d %.0f%%' % (g, T[(T.grp == g) & T.poly].shape[0], 100 * (T[(T.grp == g) & T.poly].dc >= 0.3).mean()) for g in order))
import statsmodels.api as sm
C = T[T.nv.notna()].copy(); C['lnv'] = np.log(C.nv); C['lrec'] = np.log(C.nrec)
for y in ('poly', 'polyE'):
    X = sm.add_constant(pd.get_dummies(C.nel.clip(upper=4), prefix='nel', drop_first=True).astype(float).assign(lnv=C.lnv))
    m = sm.Logit(C[y].astype(float), X).fit(disp=0)
    print('(c2) %s ~ log(n values) + n_elements: coef %.2f P %.1e (OR per doubling %.2f)' % (y, m.params.lnv, m.pvalues.lnv, np.exp(m.params.lnv * np.log(2))))
