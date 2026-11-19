import sys, pickle
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, pandas as pd
import evaluate_cv, density_score as ds
from functools import partial
ds.brentq = partial(ds.brentq, maxiter=2000)
import evaluate_cv_v4 as E
ev = E.ev
OUT = '/data/czx/GapScope_Revision_20261113/window/'
ARM = 'GS_SELCAL_PRE_NZ'
WIN = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
A, Akey, wA, Bkeys, Bvals = ev.A, ev.Akey, ev.wA, ev.Bkeys, ev.Bvals
Aidx = {}
for i, k in enumerate(Akey): Aidx.setdefault(k, []).append(i)
print('A comps', len(Aidx), 'B comps', len(Bkeys), 'overlap', len(set(Aidx) & set(Bkeys)))
PAPER = {('comp', 'GapScope'): 0.2523, ('chem', 'GapScope'): 0.2672, ('comp', 'ROOST_CAL'): 0.2611, ('chem', 'ROOST_CAL'): 0.2798, ('comp', 'EASYUQ_KS'): 0.2684, ('chem', 'EASYUQ_KS'): 0.2771}


def crps_pooled(V, name, gA, gB):
    ua, ub, _ = ev.evaluate(V, name, gA, gB)
    ca = ev.comp_level_A(ua)
    return float(pd.concat([ca, ub]).crps.mean()), float(ca.crps.mean()), float(ub.crps.mean())


def units(gA, gB, view=0):
    # per composition: formula-level dist, reports y, weights summing to 1 within composition, tier
    U = {}
    for k, ii in Aidx.items():
        w = wA[ii] / wA[ii].sum()
        U[('A',k)] = dict(tier='A', dist=gA[view][ii[0]], y=np.array([float(A[i]['gap_ev']) for i in ii]), w=w)
    for k in Bkeys:
        y = np.array(Bvals[k]); U[('B',k)] = dict(tier='B', dist=gB[k], y=y, w=np.ones(len(y)) / len(y))
    return U


def table(U):
    rows = []
    for (_t, k), u in U.items():
        e = ev.unit_eval(u['dist'], u['y'][:1]); med = e['median']
        r = dict(key=k, tier=u['tier'], median=med, width80=e['width80'], n_reports=len(u['y']))
        for wn, (a, b) in WIN.items():
            p = ev.win_p(u['dist'], a, b); o = ((u['y'] >= a) & (u['y'] <= b)).astype(float)
            r['P_' + wn] = p; r['frac_' + wn] = float((o * u['w']).sum()); r['any_' + wn] = bool(o.any())
            r['bp_' + wn] = float((u['w'] * (p - o) ** 2).sum())
            r['bm_' + wn] = float((u['w'] * (float(a <= med <= b) - o) ** 2).sum())
            r['dm_' + wn] = -max(a - med, med - b, 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


rows, perc, allT, GA_ex = [], [], {}, {}
for V in ('comp', 'chem'):
    g = E.load_gs_dir(V, ARM); fac = E.fit_scale_oof(V, g[0]); sets = {'GapScope': E.apply_scale(V, *g, fac)}
    rA, rB = ev.load_roost(V); sets['ROOST_CAL'] = E.calibrate(rA, rB, V, E.fit_cal(V, 'heldout'))
    try:
        eq = pickle.load(open(ev.CV / ('easyuq_%s.pkl' % V), 'rb')); print(V, 'easyuq keys', list(eq))
        for nm, v in eq.items():
            if 'KS' in nm: sets['EASYUQ_KS'] = v
    except Exception as ex:
        print('easyuq fail', ex)
    for name, (gA, gB) in sets.items():
        c = crps_pooled(V, name, gA, gB)
        print('CRPS', V, name, 'pooled %.4f A %.4f B %.4f' % c, 'paper', PAPER.get((V, name)), flush=True)
        T = table(units(gA, gB, 0)); allT[(V, name)] = T
        for wn in WIN:
            for tier, sel in (('A', T.tier == 'A'), ('B', T.tier == 'B'), ('pooled', None)):
                t = T[sel] if sel is not None else T
                bp, bm = t['bp_' + wn].mean(), t['bm_' + wn].mean()
                ap = ev.auc(t['any_' + wn], t['P_' + wn]); am = ev.auc(t['any_' + wn], t['dm_' + wn])
                r = dict(holdout=V, model=name, window=wn, tier=tier, Brier_prob=bp, Brier_median=bm, relative_reduction=1 - bp / bm, AUC_prob=ap, AUC_median=am,
                         n_reports=int(t.n_reports.sum()), n_comp=len(t), n_positive=int(t['any_' + wn].sum()))
                rng = np.random.default_rng(1); red, dau = [], []
                for _ in range(2000):
                    s = t.iloc[rng.integers(0, len(t), len(t))]
                    b1, b2 = s['bp_' + wn].mean(), s['bm_' + wn].mean()
                    red.append(1 - b1 / b2 if b2 > 0 else np.nan)
                    dau.append(ev.auc(s['any_' + wn], s['P_' + wn]) - ev.auc(s['any_' + wn], s['dm_' + wn]))
                r.update(red_lo=np.nanpercentile(red, 2.5), red_hi=np.nanpercentile(red, 97.5), dAUC=ap - am, dAUC_lo=np.nanpercentile(dau, 2.5), dAUC_hi=np.nanpercentile(dau, 97.5))
                rows.append(r)
        if name == 'GapScope':
            T2 = T[['key', 'tier', 'P_PV', 'P_VIS', 'median', 'width80', 'n_reports', 'frac_PV', 'frac_VIS']].copy(); T2.insert(0, 'holdout', V); perc.append(T2)
            GA_ex[V] = (gA, gB)
pd.DataFrame(rows).round(4).to_csv(OUT + 'window_metrics.csv', index=False)
pd.concat(perc).to_csv(OUT + 'window_probs_per_composition.csv', index=False)


def parse(k): return {p.split(':')[0]: float(p.split(':')[1]) for p in k.split('|')}


TA = allT[('comp', 'GapScope')]; TA = TA[TA.tier == 'A'].copy()
fa = [k for k in TA.key if set(parse(k)) == {'C', 'H', 'N', 'Pb', 'I'}]
print('FAPbI3-like candidates:', [(k, TA[TA.key == k].n_reports.iloc[0]) for k in fa][:5])
TA['strad'] = np.maximum(np.minimum(TA.frac_PV, 1 - TA.frac_PV), np.minimum(TA.frac_VIS, 1 - TA.frac_VIS))
top = TA[TA.n_reports >= 4].sort_values('strad', ascending=False).head(6)
print(top[['key', 'n_reports', 'frac_PV', 'frac_VIS', 'strad']].to_string())
exk = list(fa[:1]) + [k for k in top.key if k not in fa][:3]
grid = np.round(np.arange(0, 4.0001, 0.01), 2); dr, er = [], []
U = units(*GA_ex['comp'], 0)
for V in ('comp', 'chem'):
    Uv = units(*GA_ex[V], 0)
    for k in exk:
        d = E.dens(Uv[('A',k)]['dist'], grid)
        dr += [dict(holdout=V, key=k, E_eV=x, density=y) for x, y in zip(grid, d)]
for k in exk:
    for y, w in zip(U[('A',k)]['y'], U[('A',k)]['w']): er.append(dict(key=k, gap_eV=y, weight=w))
pd.DataFrame(dr).to_csv(OUT + 'example_densities.csv', index=False); pd.DataFrame(er).to_csv(OUT + 'example_reports.csv', index=False)
print('examples', exk)
