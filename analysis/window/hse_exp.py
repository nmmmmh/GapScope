import sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import evaluate_cv_v4 as E
ev = E.ev
OUT = '/data/czx/GapScope_Revision_20261113/window/par2/'
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
rng = np.random.default_rng(13)


def strad(y, a, b, d=0.0):
    i = (y >= a + d) & (y <= b - d); x = (y < a - d) | (y > b + d); return bool(i.any() and x.any())


S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
POLY = {}
for k, t in S.groupby('ckey'):
    te = t.dropna(subset=['Free_energy'])
    acc = te[te.Free_energy <= te.Free_energy.min() + 0.1] if len(te) >= 2 else te.iloc[:0]
    POLY[k] = dict(all=t.hse.values, acc=acc.hse.values if len(acc) >= 2 else None,
                   gs=te.sort_values('Free_energy').hse.values[0] if len(te) else np.nan)
# experimental reports per composition: traced first, compiled otherwise; record source tier
Y, SRC = {}, {}
for i, k in enumerate(ev.Akey): Y.setdefault(k, []).append(float(ev.A[i]['gap_ev'])); SRC[k] = 'traced'
for k, v in ev.Bvals.items():
    if k not in Y: Y[k] = list(map(float, v)); SRC[k] = 'compiled'

rows, lines = [], []
# A. computational screen
accd = {k: p for k, p in POLY.items() if p['acc'] is not None}
for wn, (a, b) in W.items():
    for d in (0.0, 0.1):
        n = sum(strad(p['acc'], a, b, d) for p in accd.values())
        gin = [p for p in accd.values() if a <= p['gs'] <= b]
        f = sum(strad(p['acc'], a, b, d) for p in gin)
        rows.append(dict(part='A_screen', window=wn, margin=d, n_comp=len(accd), n_change=n, frac_change=n / len(accd), n_gs_in=len(gin), n_gs_in_with_outside=f))
# B. concordance
for pset in ('acc', 'all'):
    for wn, (a, b) in W.items():
        for d in (0.0, 0.1):
            ks = [k for k, p in POLY.items() if p[pset] is not None and len(p[pset]) >= 2 and k in Y and len(Y[k]) >= 2]
            h = np.array([strad(POLY[k][pset], a, b, d) for k in ks]); e = np.array([strad(np.array(Y[k]), a, b, d) for k in ks])
            tr = np.array([SRC[k] == 'traced' for k in ks])
            t = [[int((h & e).sum()), int((h & ~e).sum())], [int((~h & e).sum()), int((~h & ~e).sum())]]
            o, p = fisher_exact(t)
            I = rng.integers(0, len(ks), (4000, len(ks))); diff = []
            for ii in I:
                hh, ee = h[ii], e[ii]
                if hh.any() and (~hh).any(): diff.append(ee[hh].mean() - ee[~hh].mean())
            rows.append(dict(part='B_concordance', polymorphs=pset, window=wn, margin=d, n_comp=len(ks), n_traced=int(tr.sum()),
                             hse_straddle=int(h.sum()), exp_straddle_given_hse=t[0][0], exp_straddle_given_not=t[1][0],
                             rate_hse=e[h].mean(), rate_not=e[~h].mean(), diff=e[h].mean() - e[~h].mean(),
                             diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5), odds_ratio=o, fisher_P=p))
            if pset == 'acc' and d == 0.0:
                for k in np.array(ks)[h]:
                    lines.append(dict(window=wn, key=k, source=SRC[k], hse=np.round(np.sort(POLY[k]['acc']), 2).tolist(), exp=np.round(sorted(Y[k]), 2).tolist(), exp_straddle=strad(np.array(Y[k]), a, b)))
R = pd.DataFrame(rows).round(4); R.to_csv(OUT + 'hse_vs_exp.csv', index=False)
pd.DataFrame(lines).to_csv(OUT + 'hse_vs_exp_cases.csv', index=False)
pd.set_option('display.width', 300); print(R.to_string(index=False)); print(pd.DataFrame(lines).to_string(index=False))
