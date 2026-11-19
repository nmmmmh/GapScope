"""Grouped-CV evaluation v2: evaluate_cv.py functions unchanged, extended to ROOST_CAL and the fine-tuned GS arms.
Writes *_v2 outputs only (metrics_v2.csv, bootstrap_v2.csv, pit_v2.csv, windows_v2.csv, windows_units_v2.csv,
phase_check_v2.csv, phase_summary_v2.json, predictions_summary_v2.csv, roost_cal_factors.json)."""
import sys, json, math, argparse
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv_rep/code')
import evaluate_cv as ev
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CV = ev.CV
FTO = CV / 'ft/out/gs_out_v2'
CAL = CV / 'ft/cal'
SEEDS = ev.SEEDS
A, Akey, Bkeys = ev.A, ev.Akey, ev.Bkeys
FOLDS = {v: pd.read_csv(CV / f'folds_{v}.csv').set_index('composition_key').fold.to_dict() for v in ('comp', 'chem')}
IV = {v: json.load(open(CV / f'data/inner_val_{v}.json')) for v in ('comp', 'chem')}


_cdf0, _ue0 = ev.cdf, ev.unit_eval


def dens(dist, ys):
    ys = np.asarray(ys, float)
    if dist[0] == 'mix':
        return dist[1] * dens(dist[2], ys) + (1 - dist[1]) * dens(dist[3], ys)
    if dist[0] == 'lap':
        _, mu, b = dist
        return (np.exp(-np.abs(ys[:, None] - mu[None]) / b[None]) / (2 * b[None])).mean(1)
    _, w, mu, s = dist
    if dist[0] == 'wlap':
        return (w[None] * np.exp(-np.abs(ys[:, None] - mu[None]) / s[None]) / (2 * s[None])).sum(1)
    ly = np.log(np.maximum(ys, 1e-12))[:, None]
    z = (ly - mu[None]) / s[None]
    return np.where(ys > 0, (w[None] * np.exp(-.5 * z * z) / (s[None] * np.sqrt(2 * np.pi))).sum(1) / np.maximum(ys, 1e-12), 0.0)


def cdf2(dist, x):
    if dist[0] == 'mix':
        return dist[1] * cdf2(dist[2], x) + (1 - dist[1]) * cdf2(dist[3], x)
    if dist[0] != 'wlap':
        return _cdf0(dist, x)
    _, w, mu, b = dist
    x = np.asarray(x, float)
    z = (x[:, None] - mu[None]) / b[None]
    return (w[None] * np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0)))).sum(1)


def unit_eval2(dist, ys):
    """weighted Laplace mixture ('wlap'): same grid procedure as the 'lap' branch of evaluate_cv.unit_eval."""
    if dist[0] not in ('wlap', 'mix'):
        return _ue0(dist, ys)
    ys = np.asarray(ys, float)
    G = cdf2(dist, ev.YL)
    q = lambda a: float(np.interp(a, G, ev.YL))
    med, lo, hi = q(.5), q(.1), q(.9)
    crps = np.array([float(((G - (ev.YL >= y)) ** 2).sum() * ev.DY) for y in ys])
    return dict(crps=crps, nll=-np.log(dens(dist, ys)), mae=np.abs(ys - med), cov80=((lo <= ys) & (ys <= hi)).astype(float), median=med, width80=hi - lo, pit=cdf2(dist, ys))


def fit_blend_oof(variant, gA1, gA2, view=3):
    """per test fold f: weight w of w*model1 + (1-w)*model2 minimising composition-mean hierarchically weighted tier-A NLL over
    the other 4 folds' out-of-fold predictions (same procedure as the width factor)."""
    fd = FOLDS[variant]
    grid = np.linspace(0, 1, 51)
    wc = {}
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]
        for i in ii:
            wc[i] = ev.wA[i] / ev.wA[ii].sum()
    d1 = np.array([dens(gA1[view][i], [float(A[i]['gap_ev'])])[0] for i in range(len(A))])
    d2 = np.array([dens(gA2[view][i], [float(A[i]['gap_ev'])])[0] for i in range(len(A))])
    out = {}
    for f in range(1, 6):
        idx = np.array([i for i in range(len(A)) if fd[Akey[i]] != f])
        nk = len({Akey[i] for i in idx})
        ww = np.array([wc[i] for i in idx]) / nk
        L = [float((-np.log(g * d1[idx] + (1 - g) * d2[idx] + 1e-300) * ww).sum()) for g in grid]
        out[f] = dict(w=float(grid[int(np.argmin(L))]), nll=float(min(L)))
    return out


def apply_blend(variant, s1, s2, fac):
    fd = FOLDS[variant]
    gA = {v: {i: ('mix', fac[fd[Akey[i]]]['w'], s1[0][v][i], s2[0][v][i]) for i in s1[0][v]} for v in range(4)}
    gB = {k: ('mix', fac[fd[k]]['w'], s1[1][k], s2[1][k]) for k in s1[1]}
    return gA, gB


ev.cdf, ev.unit_eval = cdf2, unit_eval2


def mix(parts, kind):
    d = ev.ln_mix(parts)
    return d if kind == 'ln' else ('wlap',) + d[1:]


def dist_nll(d, y):
    if d[0] == 'lap':
        return float(lap_nll(np.array([y]), d[1][None], d[2][None])[0])
    _, w, mu, s = d
    if d[0] == 'ln':
        z = (math.log(y) - mu) / s
        return -float(np.log((w * np.exp(-.5 * z * z) / (s * y * math.sqrt(2 * math.pi))).sum()))
    return -float(np.log((w * np.exp(-np.abs(y - mu) / s) / (2 * s)).sum()))


def scale(d, c):
    return (d[0], d[1], d[2] * c) if d[0] == 'lap' else (d[0], d[1], d[2], d[3] * c)


def fit_scale_oof(variant, gA, view=3):
    """per test fold f: one width factor (sd of every component, all seeds) minimising composition-mean hierarchically
    weighted tier-A NLL over the other 4 folds' out-of-fold predictions (same procedure as ROOST_CAL_CV)."""
    fd = FOLDS[variant]
    grid = np.exp(np.linspace(np.log(0.3), np.log(5), 300))
    wc = {}
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]
        for i in ii:
            wc[i] = ev.wA[i] / ev.wA[ii].sum()
    out = {}
    for f in range(1, 6):
        idx = [i for i in range(len(A)) if fd[Akey[i]] != f]
        nk = len({Akey[i] for i in idx})
        L = [sum(dist_nll(scale(gA[view][i], c), float(A[i]['gap_ev'])) * wc[i] / nk for i in idx) for c in grid]
        out[f] = dict(factor=float(grid[int(np.argmin(L))]), nll_cal=float(min(L)))
    return out


def apply_scale(variant, gA, gB, fac):
    fd = FOLDS[variant]
    return ({v: {i: scale(d, fac[fd[Akey[i]]]['factor']) for i, d in gA[v].items()} for v in gA},
            {k: scale(d, fac[fd[k]]['factor']) for k, d in gB.items()})


def load_gs_dir(variant, model, folds=range(1, 6), seeds=SEEDS):
    gsA = {v: {} for v in range(4)}
    gsB, per = {}, {}
    kind = 'ln'
    for s in seeds:
        for f in folds:
            z = dict(np.load(FTO / f'{variant}_{model}_seed{s}_fold{f}.npz'))
            kind = str(z.get('kind', 'ln'))
            for kk in ('A0', 'A1', 'A2', 'A3', 'B'):
                z[kk] = z[kk].copy(); z[kk][0] = np.clip(z[kk][0], 1e-12, None); z[kk][0] = z[kk][0] / z[kk][0].sum(-1, keepdims=True)
            for j, i in enumerate(z['A_idx']):
                for v in range(4):
                    per.setdefault(('A', v, int(i)), []).append(tuple(z[f'A{v}'][t][j] for t in range(3)))
            for j, k in enumerate(z['B_keys']):
                per.setdefault(('B', str(k)), []).append(tuple(z['B'][t][j] for t in range(3)))
    for key, parts in per.items():
        assert len(parts) == len(seeds), key
        d = mix(parts, kind)
        assert abs(d[1].sum() - 1) < 1e-6
        if key[0] == 'A':
            gsA[key[1]][key[2]] = d
        else:
            gsB[key[1]] = d
    return gsA, gsB


def lap_nll(y, mu, b):
    dens = np.exp(-np.abs(y[:, None] - mu) / b) / (2 * b)
    return -np.log(dens.mean(1) + 1e-300)


def load_roost_s(variant, seeds):
    """evaluate_cv.load_roost for any seed list (extra seeds 53/67/79 from run_roost_cv_r5.py)."""
    per = {}
    for s in seeds:
        for f in range(1, 6):
            for r in pd.read_csv(CV / 'roost_runs' / f'pred_{variant}_f{f}_s{s}.csv').itertuples():
                per.setdefault(int(r.id), []).append((r.mu, r.ale / math.sqrt(2)))
    rA, rB = {}, {}
    for i, lst in per.items():
        assert len(lst) == len(seeds), i
        d = ('lap', np.array([x[0] for x in lst]), np.array([x[1] for x in lst]))
        if i < 200000:
            rA[i - 100000] = d
        else:
            rB[Bkeys[i - 200000]] = d
    return {v: rA for v in range(4)}, rB


def combine(sets_):
    """equal-weight mixture of several (gA, gB) model sets (same distribution family)."""
    n = len(sets_)
    def mixd(ds):
        if ds[0][0] == 'lap':
            return ('lap', np.concatenate([d[1] for d in ds]), np.concatenate([d[2] for d in ds]))
        return (ds[0][0], np.concatenate([d[1] for d in ds]) / n, np.concatenate([d[2] for d in ds]), np.concatenate([d[3] for d in ds]))
    gA = {v: {i: mixd([s[0][v][i] for s in sets_]) for i in sets_[0][0][v]} for v in range(4)}
    gB = {k: mixd([s[1][k] for s in sets_]) for k in sets_[0][1]}
    return gA, gB


def fit_cal(variant, mode, rA_in=None, seeds=SEEDS):
    """per-fold scale factor c (b -> c*b, all 3 seeds) minimising composition-mean hierarchically weighted NLL of the
    3-seed ROOST mixture. mode 'heldout': tier-A training records in the release internal 20% validation split (never
    used for gradient steps); mode 'innerval': tier-A records of the inner-validation compositions (in ROOST's training
    data, so in-sample)."""
    wc = {}
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]
        s = ev.wA[ii].sum()
        for i in ii:
            wc[i] = ev.wA[i] / s
    grid = np.exp(np.linspace(np.log(0.3), np.log(30), 500))
    out = {}
    if mode == 'cv':
        # out-of-fold: tier-A records of the other 4 folds, each scored by the ROOST ensemble of its own fold (which did not train on it)
        rA = (rA_in if rA_in is not None else ev.load_roost(variant)[0])[0]
        fd = FOLDS[variant]
        for f in range(1, 6):
            idx = [i for i in range(len(A)) if fd[Akey[i]] != f]
            y = np.array([float(A[i]['gap_ev']) for i in idx])
            mu = np.stack([rA[i][1] for i in idx]); b = np.stack([rA[i][2] for i in idx])
            nk = len({Akey[i] for i in idx})
            ws = np.array([wc[i] for i in idx]) / nk
            L = np.array([(lap_nll(y, mu, c * b) * ws).sum() for c in grid])
            out[f] = dict(factor=float(grid[L.argmin()]), n_records=len(idx), n_comp=nk, nll_raw=float((lap_nll(y, mu, b) * ws).sum()), nll_cal=float(L.min()))
        return out
    for f in range(1, 6):
        dfs = [pd.read_csv(CAL / f'predtrain_{variant}_f{f}_s{s}.csv') for s in seeds]
        d0 = dfs[0]
        assert all((d.id.values == d0.id.values).all() and (d.internal_val.values == d0.internal_val.values).all() for d in dfs)
        isA = (d0.id >= 100000) & (d0.id < 200000)
        ivk = set(IV[variant][str(f)])
        sel = isA & (d0.internal_val if mode == 'heldout' else d0.id.map(lambda i: Akey[i - 100000] in ivk if 100000 <= i < 200000 else False))
        rows = np.where(sel.values)[0]
        idx = d0.id.values[rows] - 100000
        y = np.array([float(A[i]['gap_ev']) for i in idx])
        mu = np.stack([d.mu.values[rows] for d in dfs], 1)
        b = np.stack([d.ale.values[rows] / math.sqrt(2) for d in dfs], 1)
        keys = [Akey[i] for i in idx]
        nk = len(set(keys))
        ww = np.array([wc[i] for i in idx])
        ws = np.zeros(len(idx))
        for k in set(keys):          # renormalise within composition over the selected records, equal composition mass
            m_ = np.array([x == k for x in keys])
            ws[m_] = ww[m_] / ww[m_].sum() / nk
        L = np.array([(lap_nll(y, mu, c * b) * ws).sum() for c in grid])
        out[f] = dict(factor=float(grid[L.argmin()]), n_records=int(len(idx)), n_comp=nk, nll_raw=float((lap_nll(y, mu, b) * ws).sum()), nll_cal=float(L.min()))
    return out


def calibrate(rA, rB, variant, fac):
    fd = FOLDS[variant]
    cA = {i: ('lap', d[1], d[2] * fac[fd[Akey[i]]]['factor']) for i, d in rA[0].items()}
    cB = {k: ('lap', d[1], d[2] * fac[fd[k]]['factor']) for k, d in rB.items()}
    return {v: cA for v in range(4)}, cB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='comp')
    ap.add_argument('--new', default='GS_FT,GS_FT_HSE')
    ap.add_argument('--suffix', default='_v2')
    ap.add_argument('--scale', action='store_true')
    ap.add_argument('--seeds5', default='')
    ap.add_argument('--ens', default='')
    ap.add_argument('--roost_extra', action='store_true')
    ap.add_argument('--blend', default='')
    ap.add_argument('--only', default='', help='comma list: evaluate only these set names (plus ROOST_CAL_CV refs)')
    ap.add_argument('--seeds6', default='')
    ap.add_argument('--easyuq', action='store_true')
    ap.add_argument('--bands', default='', help='comma list of model names for per-record band units (or *)')
    ap.add_argument('--skip_old', action='store_true', help='skip GS / ROOST_CAL / ROOST_CAL_IV (faster per-arm runs)')
    a = ap.parse_args()
    met, pitrows, winrows, boots, allrows, f1rows = [], [], [], [], [], []
    phase, calinfo = {}, {}
    bandrows = []
    for variant in a.variants.split(','):
        sets = {}
        if not a.skip_old and (CV / 'gs_out' / f'{variant}_GS_seed41_fold5.npz').exists():
            sets['GS'] = ev.load_gs(variant, 'GS')
        rA, rB = ev.load_roost(variant)
        sets['ROOST'] = (rA, rB)
        for mode, name in (('heldout', 'ROOST_CAL'), ('innerval', 'ROOST_CAL_IV'), ('cv', 'ROOST_CAL_CV')):
            if a.skip_old and name != 'ROOST_CAL_CV':
                continue
            if all((CAL / f'predtrain_{variant}_f{f}_s{s}.csv').exists() for f in range(1, 6) for s in SEEDS):
                fac = fit_cal(variant, mode)
                calinfo[f'{variant}_{name}'] = fac
                sets[name] = calibrate(rA, rB, variant, fac)
        for nm, seeds in (('ROOST5', SEEDS + (53, 67)), ('ROOST6', SEEDS + (53, 67, 79)), ('ROOSTX', (53, 67, 79))):
            if a.roost_extra and all((CV / 'roost_runs' / f'pred_{variant}_f{f}_s{s}.csv').exists() for f in range(1, 6) for s in seeds):
                r_ = load_roost_s(variant, seeds)
                sets[nm] = r_
                fac = fit_cal(variant, 'cv', r_[0])
                calinfo[f'{variant}_{nm}_CAL_CV'] = fac
                sets[nm + '_CAL_CV'] = calibrate(r_[0], r_[1], variant, fac)
                if all((CAL / f'predtrain_{variant}_f{f}_s{s}.csv').exists() for f in range(1, 6) for s in seeds):
                    fac = fit_cal(variant, 'heldout', None, seeds)
                    calinfo[f'{variant}_{nm}_CAL'] = fac
                    sets[nm + '_CAL'] = calibrate(r_[0], r_[1], variant, fac)
        loaded = {}
        todo = [(mn, SEEDS, mn) for mn in a.new.split(',') if mn] + [(mn, SEEDS + (53, 67), mn + '_5S') for mn in a.seeds5.split(',') if mn] + [(mn, SEEDS + (53, 67, 79), mn + '_6S') for mn in a.seeds6.split(',') if mn]
        for mn, seeds, name in todo:
            if all((FTO / f'{variant}_{mn}_seed{s}_fold{f}.npz').exists() for f in range(1, 6) for s in seeds):
                loaded[name] = sets[name] = load_gs_dir(variant, mn, seeds=seeds)
        for spec in [x for x in a.ens.split(';') if x]:      # NAME=ARM1+ARM2 (each 3 seeds), equal-weight mixture
            name, parts = spec.split('=')
            loaded[name] = sets[name] = combine([load_gs_dir(variant, p_) for p_ in parts.split('+')])
        if a.scale:
            for name in list(loaded):
                fac = fit_scale_oof(variant, sets[name][0])
                calinfo[f'{variant}_{name}_SC'] = fac
                sets[name + '_SC'] = apply_scale(variant, *sets[name], fac)
        for spec in [x for x in a.blend.split(';') if x]:      # NAME=SET1|SET2 (names already in sets)
            name, parts = spec.split('=')
            p1, p2 = parts.split('|')
            fac = fit_blend_oof(variant, sets[p1][0], sets[p2][0])
            calinfo[f'{variant}_{name}_blend_w'] = fac
            sets[name] = apply_blend(variant, sets[p1], sets[p2], fac)
        if a.easyuq:
            import pickle
            import glob
            for pk_ in sorted(glob.glob(str(CV / __import__('os').environ.get('EQPAT', f'easyuq_{variant}*.pkl')))):
                for nm_, v_ in pickle.load(open(pk_, 'rb')).items():
                    sets[nm_] = v_
        if a.only:
            keep = set(a.only.split(',')) | {'ROOST', 'ROOST_CAL_CV', 'ROOST5_CAL_CV', 'ROOST6_CAL_CV', 'ROOST5_CAL', 'ROOST6_CAL'}
            sets = {k: v for k, v in sets.items() if k in keep}
        res = {}
        for mname, (gA, gB) in sets.items():
            ua, ub, pits = ev.evaluate(variant, mname, gA, gB)
            ca = ev.comp_level_A(ua)
            if a.bands and (a.bands == '*' or mname in a.bands.split(',')):
                wsc = ua.w / ua.groupby('key').w.transform('sum')
                for k_, y_, c_, n_, w_ in zip(ua.key, ua.y, ua.crps, ua.nll, wsc):
                    bandrows.append((variant, mname, 'A', k_, y_, c_, n_, w_))
                for k in Bkeys:
                    e_ = ev.unit_eval(gB[k], ev.Bvals[k])
                    for y_, c_, n_ in zip(ev.Bvals[k], e_['crps'], e_['nll']):
                        bandrows.append((variant, mname, 'B', k, y_, c_, n_, 1.0 / len(ev.Bvals[k])))
            res[mname] = (ua, ub, ca)
            view = 'composition' if mname.startswith('ROOST') else 'all_available'
            for tier, df in (('A', ca), ('B', ub), ('pooled', pd.concat([ca, ub]))):
                met.append(dict(variant=variant, model=mname, tier=tier, view=view, n_comp=len(df), **{m: float(df[m].mean()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            if variant == 'comp':
                for tier, df in (('A', ca), ('B', ub)):
                    g1 = df[df.key.map(FOLDS['comp']) == 1]
                    f1rows.append(dict(model=mname, tier=tier, n_comp=len(g1), **{m: float(g1[m].mean()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            met.append(dict(variant=variant, model=mname, tier='A_records_hier', view=view, n_comp=len(ua), **{m: float((ua[m] * ua.w).sum() / ua.w.sum()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            pa = np.histogram(ua.pit.values, bins=10, range=(0, 1))[0]
            pb = np.histogram([p for _, p in pits], bins=10, range=(0, 1))[0]
            for tier, h in (('A', pa), ('B', pb)):
                pitrows.append(dict(variant=variant, model=mname, tier=tier, **{f'bin{i + 1}': int(h[i]) for i in range(10)}))
            allrows += [ua.assign(variant=variant), ub.assign(variant=variant)]
            dcomp = {k: gB[k] for k in Bkeys}
            first = {}
            for i, k in enumerate(Akey):
                first.setdefault(k, i)
            for k, i in first.items():
                dcomp.setdefault(k, gA[0][i])
            wu = ev.window_units(mname, dcomp)
            wu['variant'] = variant
            winrows.append(wu)
            if mname not in ('GS',) and not mname.startswith('ROOST') and variant == 'comp':
                pc = ev.phase_check(gA)
                rho, pv = spearmanr(pc.omega2, pc.narrowing)
                phase[mname] = dict(n_comp=len(pc), spearman_rho=float(rho), p=float(pv), mean_omega2=float(pc.omega2.mean()), mean_narrowing=float(pc.narrowing.mean()))
                pc.assign(model=mname).to_csv(CV / f'phase_check_v2_{mname}.csv', index=False)
        cid = {k: j for j, k in enumerate(sorted(set(Akey) | set(Bkeys)))}
        new = [m_ for m_ in sets if m_ not in ('GS',) and not m_.startswith('ROOST')]
        refs = [r_ for r_ in ('ROOST', 'ROOST_CAL', 'ROOST6_CAL', 'GS_SELCAL_SC', 'EASYUQ_KS', 'ROOST_CAL_CV', 'ROOST5_CAL_CV', 'ROOST6_CAL_CV', 'ROOST_CAL_IV', 'GS') if r_ in sets]
        pairs = [(n_, r_) for n_ in new for r_ in refs if n_ != r_] + [('GS', r_) for r_ in ('ROOST_CAL',) if 'GS' in sets and r_ in sets]
        for x, y_ in pairs:
            for tier in ('A', 'B', 'pooled'):
                def units(mn):
                    ua, ub, ca = res[mn]
                    return ca if tier == 'A' else ub if tier == 'B' else pd.concat([ca, ub])
                da, db = units(x), units(y_)
                assert (da['key'].values == db['key'].values).all()
                ci = np.array([cid[k] for k in da['key'].values])
                for metric in ('crps', 'nll'):
                    rng = np.random.default_rng(20261113)
                    pt, (lo, hi) = ev.boot_diff(da[metric].values, db[metric].values, ci, len(cid), rng)
                    boots.append(dict(variant=variant, comparison=f'{x}_vs_{y_}', tier=tier, metric=metric, n_comp=len(set(da['key'])), diff=pt, ci95_lo=lo, ci95_hi=hi,
                                      a=float(da[metric].mean()), b=float(db[metric].mean())))
    sfx = a.suffix
    if bandrows:
        pd.DataFrame(bandrows, columns=['variant', 'model', 'tier', 'key', 'y', 'crps', 'nll', 'w']).to_csv(CV / f'bands_units{sfx}.csv', index=False)
    pd.DataFrame(met).to_csv(CV / f'metrics{sfx}.csv', index=False)
    if f1rows:
        pd.DataFrame(f1rows).to_csv(CV / f'fold1_quick{"" if sfx == "_v2_partial" else sfx}.csv', index=False)
    pd.DataFrame(boots).to_csv(CV / f'bootstrap{sfx}.csv', index=False)
    pd.DataFrame(pitrows).to_csv(CV / f'pit{sfx}.csv', index=False)
    wins = pd.concat(winrows)
    wins.to_csv(CV / f'windows_units{sfx}.csv', index=False)
    srows = [dict(variant=v, model=mn, n_comp=len(g), **ev.window_summary(g.reset_index(drop=True))) for (v, mn), g in wins.groupby(['variant', 'model'])]
    pd.DataFrame(srows).to_csv(CV / f'windows{sfx}.csv', index=False)
    pd.concat(allrows).to_csv(CV / f'predictions_summary{sfx}.csv', index=False)
    json.dump(phase, open(CV / f'phase_summary{sfx}.json', 'w'), indent=1)
    json.dump(calinfo, open(CV / ('roost_cal_factors.json' if sfx in ('_v2', '_v2_partial') else f'scale_factors{sfx}.json'), 'w'), indent=1)
    print('done', phase)


def quick(variant, model, fold):
    """sanity: one fold, one seed file set (any seeds present), CRPS by tier vs ROOST on the same test compositions."""
    fd = FOLDS[variant]
    import glob
    fs = sorted(glob.glob(str(FTO / f'{variant}_{model}_seed*_fold{fold}.npz')))
    rA, rB = ev.load_roost(variant)
    rows = []
    for p in fs:
        z = dict(np.load(p))
        for kk in ('A3', 'B'):
            z[kk] = z[kk].copy(); z[kk][0] = np.clip(z[kk][0], 1e-12, None); z[kk][0] = z[kk][0] / z[kk][0].sum(-1, keepdims=True)
        for name, getd in (('new', None), ('ROOST', None)):
            pass
        ca, cr = [], []
        for j, i in enumerate(z['A_idx']):
            d = ('ln', z['A3'][0][j], z['A3'][1][j], z['A3'][2][j])
            y = [float(A[i]['gap_ev'])]
            ca.append((Akey[i], ev.wA[i], ev.unit_eval(d, y)['crps'][0], ev.unit_eval(rA[0][int(i)], y)['crps'][0], ev.unit_eval(d, y)['width80']))
        cb = []
        for j, k in enumerate(z['B_keys']):
            d = ('ln', z['B'][0][j], z['B'][1][j], z['B'][2][j])
            e = ev.unit_eval(d, ev.Bvals[k]); r = ev.unit_eval(rB[k], ev.Bvals[k])
            cb.append((e['crps'].mean(), r['crps'].mean(), e['cov80'].mean(), e['width80']))
        df = pd.DataFrame(ca, columns=['k', 'w', 'n', 'r', 'wd'])
        g = df.groupby('k').apply(lambda x: pd.Series(dict(n=(x.n * x.w).sum() / x.w.sum(), r=(x.r * x.w).sum() / x.w.sum())))
        cb = np.array(cb)
        print(p.split('/')[-1], 'A crps new %.3f roost(3seed) %.3f | B crps new %.3f roost %.3f cov80 %.2f width80 %.2f' % (g.n.mean(), g.r.mean(), *cb.mean(0)))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
