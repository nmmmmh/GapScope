"""Polymorph-anchored mixture on top of the final model (GS_SELCAL_PRE_NZ_SC), cross-fitted fold by fold.
For a composition with K>=2 accessible SNUMAT polymorphs (free energy within 0.1 eV/atom of the lowest; cells without an energy kept,
cells with HSE gap < 0.05 eV dropped), every lognormal component (w, mu, s) of the formula-level prediction becomes K components
(w*pi_k, mu + lam*(log Eg_k - sum_j pi_j log Eg_j), c*s), with pi_k proportional to exp(-dE_k/tau) (dE = 0 where missing).
Records with a reported space group keep their prediction. lam, tau, c are chosen per test fold by the NLL of the other four folds."""
import sys, json, math, itertools
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import evaluate_cv_v4 as e4
ev = e4.ev
sys.path.insert(0, R + 'descr') if False else sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/descr')
import ue_grid; ue_grid.install(ev, e4)
R = '/data/czx/GapScope_Revision_20261113/'
variant = sys.argv[1]
W = 0.10
# accessible polymorphs
Dd = pd.read_csv(R + 'struct/descriptors.csv', usecols=['composition_key', 'sg_number', 'Band_gap_HSE'])
cell = Dd.groupby(['composition_key', 'sg_number']).median().reset_index()
E = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')[['ckey', 'sg', 'Free_energy']].rename(columns={'ckey': 'composition_key', 'sg': 'sg_number'})
cell = cell.merge(E, on=['composition_key', 'sg_number'], how='left'); cell = cell[cell.Band_gap_HSE >= 0.05]
POLY = {}
for k, c in cell.groupby('composition_key'):
    if c.Free_energy.notna().any():
        e0 = c.Free_energy.min(); c = c[c.Free_energy.isna() | (c.Free_energy - e0 <= W)]
        dE = (c.Free_energy - e0).fillna(0.0).values
    else:
        dE = np.zeros(len(c))
    if len(c) >= 2: POLY[k] = (np.log(c.Band_gap_HSE.values), dE)
sgm = torch.load(R + 'cv/data/tierA.pt', weights_only=False)['features']['sg_mask']
sgm = np.asarray(sgm).reshape(len(ev.A), -1).max(1) > 0
fac = json.load(open(R + ('cv/scale_factors_v4t1.json' if variant == 'comp' else 'cv/scale_factors_v6chem.json')))[f'{variant}_GS_SELCAL_PRE_NZ_SC']
fac = {int(f): v for f, v in fac.items()}
gA, gB = e4.load_gs_dir(variant, 'GS_SELCAL_PRE_NZ')
gA, gB = e4.apply_scale(variant, gA, gB, fac)
V = 3
FD = e4.FOLDS[variant]
def pam(d, k, lam, tau, c):
    lg, dE = POLY[k]
    pi = np.exp(-dE / tau) if tau < 1e8 else np.ones(len(lg)); pi = pi / pi.sum()
    delta = lam * (lg - (pi * lg).sum())
    _, w, mu, s = d
    W_, M_, S_ = (w[:, None] * pi[None]).ravel(), (mu[:, None] + delta[None]).ravel(), np.repeat(s * c, len(lg)); keep = W_ > 1e-8
    return ('ln', W_[keep] / W_[keep].sum(), M_[keep], S_[keep])
# affected units: tier-A records without SG on a polymorph composition; tier-B compositions in POLY
unitsA = [i for i, r in enumerate(ev.A) if r['composition_key'] in POLY and not sgm[i]]
unitsB = [k for k in ev.Bkeys if k in POLY]
wc = {}
for k in {ev.Akey[i] for i in unitsA}:
    ii = [i for i in unitsA if ev.Akey[i] == k]; ws = ev.wA[ii]
    for i, x in zip(ii, ws): wc[i] = x / ws.sum()
GRID = list(itertools.product(np.round(np.linspace(0, 1.5, 16), 3), (0.01, 0.025, 0.05, 0.1, 1e9), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)))
def obj(args):
    f, (lam, tau, c) = args
    LA = [e4.dist_nll(pam(gA[V][i], ev.Akey[i], lam, tau, c), float(ev.A[i]['gap_ev'])) * wc[i] for i in unitsA if FD[ev.Akey[i]] != f]
    nA = len({ev.Akey[i] for i in unitsA if FD[ev.Akey[i]] != f})
    LB = [np.mean([e4.dist_nll(pam(gB[k], k, lam, tau, c), y) for y in ev.Bvals[k]]) for k in unitsB if FD.get(k, -1) != f]
    return (sum(LA) + sum(LB)) / (nA + len(LB))
if __name__ == '__main__':
    print(variant, 'polymorph compositions', len(POLY), 'affected A records', len(unitsA), 'A comps', len({ev.Akey[i] for i in unitsA}), 'B comps', len(unitsB), flush=True)
    with Pool(96) as P:
        res = P.map(obj, [(f, g) for f in range(1, 6) for g in GRID], chunksize=4)
    best = {}
    for f in range(1, 6):
        L = res[(f - 1) * len(GRID): f * len(GRID)]; j = int(np.argmin(L)); best[f] = GRID[j]
        ref = L[GRID.index((0.0, 1e9, 1.0))]
        print('fold', f, 'lam %.2f tau %g c %.1f' % best[f], 'train NLL %.4f vs base %.4f' % (L[j], ref), flush=True)
    json.dump({f: list(map(float, b)) for f, b in best.items()}, open(R + f'descr/pam_params_{variant}.json', 'w'))
    gA2 = {v: dict(gA[v]) for v in gA}; gB2 = dict(gB)
    for i in unitsA: gA2[V][i] = pam(gA[V][i], ev.Akey[i], *best[FD[ev.Akey[i]]])
    for k in unitsB: gB2[k] = pam(gB[k], k, *best[FD[k]])
    out = {}
    for name, (a_, b_) in (('BASE', (gA, gB)), ('PAM', (gA2, gB2))):
        ua, ub, _ = ev.evaluate(variant, name, a_, b_)
        ca = ev.comp_level_A(ua).assign(tier='A'); ub = ub.assign(tier='B')
        out[name] = pd.concat([ca[['key', 'tier', 'crps', 'nll', 'cov80', 'width80']], ub[['key', 'tier', 'crps', 'nll', 'cov80', 'width80']]], ignore_index=True)
        out[name].to_csv(R + f'descr/pam_units_{variant}_{name}.csv', index=False)
    b, p = out['BASE'], out['PAM']
    keys = sorted(set(b.key)); ci = {k: n for n, k in enumerate(keys)}; idx = np.array([ci[k] for k in b.key])
    rng = np.random.default_rng(20261115)
    aff = b.key.isin(POLY.keys()).values
    for lab, m in (('pooled', np.ones(len(b), bool)), ('tier A', (b.tier == 'A').values), ('polymorph units', aff), ('polymorph A', aff & (b.tier == 'A').values)):
        pt, iv = ev.boot_diff(p.crps.values[m], b.crps.values[m], idx[m], len(keys), rng)
        print('%-16s n=%4d CRPS base %.4f PAM %.4f diff %+.4f [%+.4f, %+.4f] | cov %.3f->%.3f width %.2f->%.2f' % (lab, m.sum(), b.crps[m].mean(), p.crps[m].mean(), pt, *iv,
              b.cov80[m].mean(), p.cov80[m].mean(), b.width80[m].mean(), p.width80[m].mean()), flush=True)
