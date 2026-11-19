import os, sys as _s; os.environ['CVDIR'] = _s.argv[1]
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, pickle
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv_rep/code'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import rankdata
import evaluate_cv, density_score as ds
from functools import partial
ds.brentq = partial(ds.brentq, maxiter=2000)
import evaluate_cv_v4 as E
ev = E.ev
OUT = os.environ['CVDIR'].rstrip('/') + '/window/'; os.makedirs(OUT, exist_ok=True)
ARM = 'GS_SELCAL_PRE_NZ'; NP = 20
WIN = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
A, Akey, wA, Bkeys, Bvals = ev.A, ev.Akey, ev.wA, ev.Bkeys, ev.Bvals
Aidx = {}
for i, k in enumerate(Akey): Aidx.setdefault(k, []).append(i)
PAPER = {('comp', 'GapScope'): 0.2523, ('chem', 'GapScope'): 0.2672, ('comp', 'ROOST_CAL'): 0.2611, ('chem', 'ROOST_CAL'): 0.2798, ('comp', 'EASYUQ_KS'): 0.2684, ('chem', 'EASYUQ_KS'): 0.2771}

SETS = {}
for V in ('comp', 'chem'):
    rA, rB = ev.load_roost(V); SETS[(V, 'ROOST_CAL_CV')] = E.calibrate(rA, rB, V, E.fit_cal(V, 'cv'))
print('sets', list(SETS), flush=True)

UNITS = []  # (V, name, tier, key, dist, y, w)
for (V, name), (gA, gB) in SETS.items():
    for k, ii in Aidx.items():
        UNITS.append((V, name, 'A', k, gA[0][ii[0]], np.array([float(A[i]['gap_ev']) for i in ii]), wA[ii] / wA[ii].sum()))
    for k in Bkeys:
        y = np.array(Bvals[k]); UNITS.append((V, name, 'B', k, gB[k], y, np.ones(len(y)) / len(y)))
print('units', len(UNITS), flush=True)


def row(j):
    V, name, tier, k, dist, y, w = UNITS[j]
    e = ev.unit_eval(dist, y[:1]); med = e['median']
    r = dict(holdout=V, model=name, key=k, tier=tier, median=med, width80=e['width80'], n_reports=len(y), spread=float(y.max()-y.min()), ymin=float(y.min()), ymax=float(y.max()))
    for wn, (a, b) in WIN.items():
        p = ev.win_p(dist, a, b); o = ((y >= a) & (y <= b)).astype(float)
        r['P_' + wn] = p; r['frac_' + wn] = float((o * w).sum()); r['any_' + wn] = bool(o.any())
        r['bp_' + wn] = float((w * (p - o) ** 2).sum())
        r['bm_' + wn] = float((w * (float(a <= med <= b) - o) ** 2).sum())
        r['dm_' + wn] = -max(a - med, med - b, 0.0)
    return r


def crps(key):
    V, name = key; gA, gB = SETS[key]
    ua, ub, _ = ev.evaluate(V, name, gA, gB); ca = ev.comp_level_A(ua)
    return key, float(pd.concat([ca, ub]).crps.mean())


def fauc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == '__main__':
    with Pool(NP) as pool:
        cr = pool.map_async(crps, list(SETS))
        T = pd.DataFrame(pool.map(row, range(len(UNITS)), chunksize=20))
        for key, c in cr.get(): print('CRPS', key, '%.4f' % c, 'paper', PAPER.get(key), flush=True)
    T.to_csv(OUT + 'window_units_rcs.csv', index=False)
    print("UNITS DONE", flush=True)
