import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, pickle
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import rankdata
import evaluate_cv, density_score as ds
from functools import partial
ds.brentq = partial(ds.brentq, maxiter=2000)
import evaluate_cv_v4 as E
ev = E.ev
OUT = '/data/czx/GapScope_Revision_20261113/window/par/'; os.makedirs(OUT, exist_ok=True)
ARM = 'GS_SELCAL_PRE_NZ'; NP = 64
WIN = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
A, Akey, wA, Bkeys, Bvals = ev.A, ev.Akey, ev.wA, ev.Bkeys, ev.Bvals
Aidx = {}
for i, k in enumerate(Akey): Aidx.setdefault(k, []).append(i)
PAPER = {('comp', 'GapScope'): 0.2523, ('chem', 'GapScope'): 0.2672, ('comp', 'ROOST_CAL'): 0.2611, ('chem', 'ROOST_CAL'): 0.2798}

SETS = {}
for V in ('comp', 'chem'):
    g = E.load_gs_dir(V, ARM); fac = E.fit_scale_oof(V, g[0]); SETS[(V, 'GapScope')] = E.apply_scale(V, *g, fac)
    rA, rB = ev.load_roost(V); SETS[(V, 'ROOST_CAL')] = E.calibrate(rA, rB, V, E.fit_cal(V, 'heldout'))
    try:
        eq = pickle.load(open(ev.CV / ('easyuq_%s_GS_SELCAL_PRE_NZ.pkl' % V), 'rb'))
        for nm, v in eq.items():
            if 'KS' in nm: SETS[(V, 'EASYUQ_ON_GS_KS')] = v
    except Exception as ex:
        print('easyuq fail', ex)
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
    r = dict(holdout=V, model=name, key=k, tier=tier, median=med, width80=e['width80'], n_reports=len(y))
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
        cr = pool.map_async(crps, [k for k in SETS if k in PAPER])
        T = pd.DataFrame(pool.map(row, range(len(UNITS)), chunksize=20))
        for key, c in cr.get(): print('CRPS', key, '%.4f' % c, 'paper', PAPER.get(key), flush=True)
    T.to_csv(OUT + 'window_units_all.csv', index=False)
    rows = []
    for (V, name), TT in T.groupby(['holdout', 'model']):
        for wn in WIN:
            for tier in ('A', 'B', 'pooled'):
                t = TT if tier == 'pooled' else TT[TT.tier == tier]
                yb = t['any_' + wn].values.astype(bool); P = t['P_' + wn].values; D = t['dm_' + wn].values
                bpv, bmv = t['bp_' + wn].values, t['bm_' + wn].values
                bp, bm = bpv.mean(), bmv.mean(); ap, am = fauc(yb, P), fauc(yb, D)
                if tier == 'A' and wn == 'PV': print('AUC check', V, name, ap, ev.auc(t['any_' + wn], t['P_' + wn]), flush=True)
                rng = np.random.default_rng(1); I = rng.integers(0, len(t), (2000, len(t)))
                b1, b2 = bpv[I].mean(1), bmv[I].mean(1); red = np.where(b2 > 0, 1 - b1 / b2, np.nan)
                dau = np.array([fauc(yb[i], P[i]) - fauc(yb[i], D[i]) for i in I])
                rows.append(dict(holdout=V, model=name, window=wn, tier=tier, Brier_prob=bp, Brier_median=bm, relative_reduction=1 - bp / bm,
                                 red_lo=np.nanpercentile(red, 2.5), red_hi=np.nanpercentile(red, 97.5), AUC_prob=ap, AUC_median=am, dAUC=ap - am,
                                 dAUC_lo=np.nanpercentile(dau, 2.5), dAUC_hi=np.nanpercentile(dau, 97.5),
                                 n_reports=int(t.n_reports.sum()), n_comp=len(t), n_positive=int(yb.sum())))
    M = pd.DataFrame(rows).round(4); M.to_csv(OUT + 'window_metrics.csv', index=False)
    print(M.to_string(), flush=True)
    G = T[T.model == 'GapScope']
    G[['holdout', 'key', 'tier', 'P_PV', 'P_VIS', 'median', 'width80', 'n_reports', 'frac_PV', 'frac_VIS']].to_csv(OUT + 'window_probs_per_composition.csv', index=False)

    def parse(k): return {p.split(':')[0]: float(p.split(':')[1]) for p in k.split('|')}
    TA = G[(G.holdout == 'comp') & (G.tier == 'A')].copy()
    fa = [k for k in TA.key if set(parse(k)) == {'C', 'H', 'N', 'Pb', 'I'}]
    print('FAPbI3-like:', [(k, int(TA[TA.key == k].n_reports.iloc[0])) for k in fa][:5])
    TA['strad'] = np.maximum(np.minimum(TA.frac_PV, 1 - TA.frac_PV), np.minimum(TA.frac_VIS, 1 - TA.frac_VIS))
    top = TA[TA.n_reports >= 4].sort_values('strad', ascending=False).head(6)
    print(top[['key', 'n_reports', 'P_PV', 'frac_PV', 'P_VIS', 'frac_VIS', 'strad']].to_string())
    exk = list(fa[:1]) + [k for k in top.key if k not in fa][:3]
    grid = np.round(np.arange(0, 4.0001, 0.01), 2); dr, er = [], []
    for V in ('comp', 'chem'):
        gA = SETS[(V, 'GapScope')][0]
        for k in exk:
            d = E.dens(gA[0][Aidx[k][0]], grid); dr += [dict(holdout=V, key=k, E_eV=x, density=y) for x, y in zip(grid, d)]
    for k in exk:
        ii = Aidx[k]; w = wA[ii] / wA[ii].sum()
        for i, wi in zip(ii, w): er.append(dict(key=k, gap_eV=float(A[i]['gap_ev']), weight=wi))
    pd.DataFrame(dr).to_csv(OUT + 'example_densities.csv', index=False); pd.DataFrame(er).to_csv(OUT + 'example_reports.csv', index=False)
    print('examples', exk, 'DONE', flush=True)
