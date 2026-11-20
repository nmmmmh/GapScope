"""Evaluate the grouped CV: metrics.csv, bootstrap.csv, pit.csv, windows.csv, phase_check.csv, predictions.csv."""
import sys, json, math
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse/runtime')
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import spearmanr
from scipy.special import ndtr, logsumexp
from density_score import score as ln_score
from training_data import hierarchical_weights
CV = Path('/data/czx/GapScope_Revision_20261113/cv')
SEEDS = (17, 29, 41)
WIN = {'pv': (1.1, 1.7), 'visible': (1.8, 3.1)}
YL = np.linspace(-3, 16, 19001)
DY = YL[1] - YL[0]
A = torch.load(CV / 'data/tierA.pt', weights_only=False)['records']
Bf = torch.load(CV / 'data/tierB_features.pt', weights_only=False)
rowsB = pd.read_csv(CV / 'data/tierB_rows.csv', dtype={'target_ev': str})
Bvals = {k: sorted(set(g.target_ev.astype(float))) for k, g in rowsB.groupby('composition_key')}
Bkeys = list(Bf['keys'])
Avals = {}
for r in A:
    Avals.setdefault(r['composition_key'], []).append(float(r['gap_ev']))
wA = hierarchical_weights(A, list(range(len(A))), 'cpu').numpy()   # equal composition mass, then source/specimen/report
Akey = [r['composition_key'] for r in A]


# ---- distributions: ('ln', w, mu, sd) or ('lap', mu[3], b[3])
def cdf(dist, x):
    x = np.asarray(x, float)
    if dist[0] == 'ln':
        _, w, mu, sd = dist
        xs = np.maximum(x, 1e-12)
        return np.where(x > 0, (w[None] * ndtr((np.log(xs)[:, None] - mu[None]) / sd[None])).sum(1), 0.0)
    _, mu, b = dist
    z = (x[:, None] - mu[None]) / b[None]
    return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0))).mean(1)


def unit_eval(dist, ys):
    """per-y arrays crps, nll, mae, cov80 + scalars median, q10, q90 (width80)."""
    ys = np.asarray(ys, float)
    if dist[0] == 'ln':
        _, w, mu, sd = dist
        rs = [ln_score(float(y), (w, mu, sd)) for y in ys]
        return dict(crps=np.array([r['crps_ev'] for r in rs]), nll=np.array([r['nll'] for r in rs]), mae=np.array([r['median_mae_ev'] for r in rs]),
                    cov80=np.array([r['coverage_80'] for r in rs]), median=rs[0]['median_ev'], width80=rs[0]['width_80_ev'], pit=cdf(dist, ys))
    _, mu, b = dist
    G = cdf(dist, YL)
    q = lambda a: float(np.interp(a, G, YL))
    med, lo, hi = q(.5), q(.1), q(.9)
    crps = np.array([float(((G - (YL >= y)) ** 2).sum() * DY) for y in ys])
    dens = np.exp(-np.abs(ys[:, None] - mu[None]) / b[None]) / (2 * b[None])
    nll = -np.log(dens.mean(1))
    return dict(crps=crps, nll=nll, mae=np.abs(ys - med), cov80=((lo <= ys) & (ys <= hi)).astype(float), median=med, width80=hi - lo, pit=cdf(dist, ys))


def win_p(dist, a, b):
    return float(cdf(dist, [b])[0] - cdf(dist, [a])[0])


def ln_mix(parts):
    return ('ln', np.concatenate([p[0] for p in parts]) / len(parts), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))


# ---- load predictions
def load_gs(variant, model):
    gsA = {v: {} for v in range(4)}
    gsB = {}
    per = {}
    for s in SEEDS:
        for f in range(1, 6):
            z = np.load(CV / 'gs_out' / f'{variant}_{model}_seed{s}_fold{f}.npz')
            for j, i in enumerate(z['A_idx']):
                for v in range(4):
                    per.setdefault(('A', v, int(i)), []).append(tuple(z[f'A{v}'][t][j] for t in range(3)))
            for j, k in enumerate(z['B_keys']):
                per.setdefault(('B', str(k)), []).append(tuple(z['B'][t][j] for t in range(3)))
    for key, parts in per.items():
        assert len(parts) == 3, key
        d = ln_mix(parts)
        assert abs(d[1].sum() - 1) < 1e-6
        if key[0] == 'A':
            gsA[key[1]][key[2]] = d
        else:
            gsB[key[1]] = d
    return gsA, gsB


def load_roost(variant):
    per = {}
    for s in SEEDS:
        for f in range(1, 6):
            p = CV / 'roost_runs' / f'pred_{variant}_f{f}_s{s}.csv'
            for r in pd.read_csv(p).itertuples():
                per.setdefault(int(r.id), []).append((r.mu, r.ale / math.sqrt(2)))
    rA, rB = {}, {}
    for i, lst in per.items():
        assert len(lst) == 3, i
        d = ('lap', np.array([x[0] for x in lst]), np.array([x[1] for x in lst]))
        if i < 200000:
            rA[i - 100000] = d
        else:
            rB[Bkeys[i - 200000]] = d
    return {v: rA for v in range(4)}, rB


def evaluate(variant, name, gsA, gsB, view=3, rows_pred=None):
    """returns per-A-record dict list, per-B-comp dict list, window units."""
    ua = []
    for i, r in enumerate(A):
        dist = gsA[view][i]
        e = unit_eval(dist, [float(r['gap_ev'])])
        ua.append(dict(model=name, tier='A', idx=i, key=r['composition_key'], w=wA[i], crps=e['crps'][0], nll=e['nll'][0], mae=e['mae'][0], cov80=e['cov80'][0],
                       width80=e['width80'], median=e['median'], pit=e['pit'][0], y=float(r['gap_ev'])))
    ub, pits = [], []
    for k in Bkeys:
        e = unit_eval(gsB[k], Bvals[k])
        ub.append(dict(model=name, tier='B', key=k, w=1.0, crps=e['crps'].mean(), nll=e['nll'].mean(), mae=e['mae'].mean(), cov80=e['cov80'].mean(),
                       width80=e['width80'], median=e['median'], n_y=len(Bvals[k])))
        pits += [(k, float(p)) for p in e['pit']]
    return pd.DataFrame(ua), pd.DataFrame(ub), pits


def comp_level_A(ua):
    """tier-A record metrics -> per-composition (hierarchical weights normalised within composition)."""
    out = []
    for k, g in ua.groupby('key'):
        w = g.w.values / g.w.sum()
        out.append(dict(key=k, **{m: float((g[m].values * w).sum()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
    return pd.DataFrame(out)


def window_units(name, d_comp):
    """d_comp: key -> formula-only distribution; ys = union of tier-A reports and tier-B distinct values."""
    rows = []
    for k in sorted(d_comp):
        ys = np.array(sorted(set(Avals.get(k, [])) | set(Bvals.get(k, []))))
        dist = d_comp[k]
        e = unit_eval(dist, ys[:1])
        med = e['median']
        r = dict(model=name, key=k, n_y=len(ys), median=med)
        for wn, (a, b) in WIN.items():
            p = win_p(dist, a, b)
            o = ((ys >= a) & (ys <= b)).astype(float)
            r[wn + '_p'] = p
            r[wn + '_any'] = bool(o.any())
            r[wn + '_brier'] = float(np.mean((p - o) ** 2))
            r[wn + '_hard'] = float(np.mean((float(a <= med <= b) - o) ** 2))
            r[wn + '_dist'] = -max(a - med, med - b, 0.0)
            r[wn + '_cdist'] = -abs(med - (a + b) / 2)
        rows.append(r)
    return pd.DataFrame(rows)


def auc(lab, sc):
    lab = np.asarray(lab, bool)
    sc = np.asarray(sc, float)
    P, N = sc[lab], sc[~lab]
    if len(P) == 0 or len(N) == 0:
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([P, N]))
    return float((r[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))


def window_summary(wu):
    out = {}
    for wn in WIN:
        out[wn + '_brier_prob'] = wu[wn + '_brier'].mean()
        out[wn + '_brier_hard'] = wu[wn + '_hard'].mean()
        out[wn + '_auc_prob'] = auc(wu[wn + '_any'], wu[wn + '_p'])
        out[wn + '_auc_dist'] = auc(wu[wn + '_any'], wu[wn + '_dist'])
        out[wn + '_n_any'] = int(wu[wn + '_any'].sum())
        pr = wu.sort_values(wn + '_p', ascending=False, kind='mergesort')
        ds = wu.assign(_k=-wu[wn + '_dist']).sort_values([ '_k'] , kind='mergesort')
        ds = wu.assign(_a=-wu[wn + '_dist'], _b=-wu[wn + '_cdist']).sort_values(['_a', '_b'], kind='mergesort')
        for k in (10, 20, 50):
            out[f'{wn}_top{k}_prob'] = int(pr[wn + '_any'].values[:k].sum())
            out[f'{wn}_top{k}_dist'] = int(ds[wn + '_any'].values[:k].sum())
    return out


def omega2(vals_by_phase):
    groups = [np.array(v, float) for v in vals_by_phase.values()]
    allv = np.concatenate(groups)
    N, k = len(allv), len(groups)
    gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    msw = ssw / (N - k)
    return float(max(0.0, (ssb - (k - 1) * msw) / (sst + msw))) if sst > 0 else float('nan')


def phase_check(gsA):
    byc = {}
    for i, r in enumerate(A):
        if r['phase_label']:
            byc.setdefault(r['composition_key'], []).append(i)
    rows = []
    for k, ids in byc.items():
        ph = {}
        for i in ids:
            ph.setdefault(A[i]['phase_label'], []).append(float(A[i]['gap_ev']))
        if len(ph) < 2 or max(len(v) for v in ph.values()) < 2:
            continue
        nar = []
        for i in ids:
            w1 = unit_eval(gsA[1][i], [float(A[i]['gap_ev'])])['width80']
            w0 = unit_eval(gsA[0][i], [float(A[i]['gap_ev'])])['width80']
            nar.append(1 - w1 / w0)
        rows.append(dict(composition_key=k, formula=A[ids[0]]['formula'], n_phases=len(ph), n_reports=len(ids), omega2=omega2(ph), narrowing=float(np.mean(nar))))
    return pd.DataFrame(rows)


def boot_diff(d_a, d_b, comp_index, ncomp, rng, nb=4000):
    """d: per-unit values; comp_index: comp id of each unit. Paired composition bootstrap of mean(a)-mean(b) (units pooled)."""
    diff = np.asarray(d_a) - np.asarray(d_b)
    s = np.bincount(comp_index, weights=diff, minlength=ncomp)
    c = np.bincount(comp_index, minlength=ncomp).astype(float)
    pt = s.sum() / c.sum()
    present = np.where(c > 0)[0]
    n = len(present)
    draws = rng.integers(0, n, size=(nb, n))
    sp, cp = s[present], c[present]
    est = np.array([sp[d].sum() / cp[d].sum() for d in draws])
    return pt, np.percentile(est, [2.5, 97.5])


def main():
    allrows, met, pitrows, winrows, boots = [], [], [], [], []
    res = {}
    for variant in ('comp', 'chem'):
        models = ['GS', 'GS_MEAN', 'GS_noHSE', 'GS_noB'] if variant == 'comp' else ['GS']
        sets = {}
        for mname in models:
            if not (CV / 'gs_out' / f'{variant}_{mname}_seed41_fold5.npz').exists():
                continue
            sets[mname] = load_gs(variant, mname)
        if (CV / 'roost_runs' / f'pred_{variant}_f5_s41.csv').exists() and all((CV / 'roost_runs' / f'pred_{variant}_f{f}_s{s}.csv').exists() for f in range(1, 6) for s in SEEDS):
            sets['ROOST'] = load_roost(variant)
        for mname, (gA, gB) in sets.items():
            ua, ub, pits = evaluate(variant, mname, gA, gB)
            ca = comp_level_A(ua)
            res[(variant, mname)] = (ua, ub, ca)
            for k, g in ((mname, None),):
                pass
            for tier, df in (('A', ca), ('B', ub)):
                met.append(dict(variant=variant, model=mname, tier=tier, view='all_available' if mname != 'ROOST' else 'composition', n_comp=len(df),
                                **{m: float(df[m].mean()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            both = pd.concat([ca, ub])
            met.append(dict(variant=variant, model=mname, tier='pooled', view='all_available' if mname != 'ROOST' else 'composition', n_comp=len(both),
                            **{m: float(both[m].mean()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            # record-level tier-A (hierarchical weights over all records, equal composition mass)
            met.append(dict(variant=variant, model=mname, tier='A_records_hier', view='all_available' if mname != 'ROOST' else 'composition', n_comp=len(ua),
                            **{m: float((ua[m] * ua.w).sum() / ua.w.sum()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
            # PIT
            pa = np.histogram(ua.pit.values, bins=10, range=(0, 1), weights=None)[0]
            pb = np.histogram([p for _, p in pits], bins=10, range=(0, 1))[0]
            for tier, h in (('A', pa), ('B', pb)):
                pitrows.append(dict(variant=variant, model=mname, tier=tier, **{f'bin{i + 1}': int(h[i]) for i in range(10)}))
            # predictions
            allrows.append(ua.assign(variant=variant))
            allrows.append(ub.assign(variant=variant))
            # windows (formula-only distribution per composition)
            dcomp = {}
            for k in Bkeys:
                dcomp[k] = gB[k]
            first = {}
            for i, k in enumerate(Akey):
                first.setdefault(k, i)
            for k, i in first.items():
                if k not in dcomp:
                    dcomp[k] = gA[0][i]
            wu = window_units(mname, dcomp)
            wu['variant'] = variant
            winrows.append(wu)
            if mname == 'GS':
                res[(variant, 'gsA')] = gA
        # ablation views tier A
        if 'GS' in sets:
            gA, gB = sets['GS']
            for v, vn in enumerate(('formula_only', 'formula_sg', 'formula_structure', 'full')):
                uav = []
                for i, r in enumerate(A):
                    e = unit_eval(gA[v][i], [float(r['gap_ev'])])
                    uav.append(dict(key=r['composition_key'], w=wA[i], crps=e['crps'][0], nll=e['nll'][0], mae=e['mae'][0], cov80=e['cov80'][0], width80=e['width80']))
                cav = comp_level_A(pd.DataFrame(uav))
                met.append(dict(variant=variant, model='GS', tier='A', view=f'ablation_{vn}', n_comp=len(cav), **{m: float(cav[m].mean()) for m in ('crps', 'nll', 'mae', 'cov80', 'width80')}))
        # bootstrap
        if 'GS' in sets and 'ROOST' in sets:
            pass
        rng = np.random.default_rng(20261113)
        cid = {k: j for j, k in enumerate(sorted(set(Akey) | set(Bkeys)))}
        comps = [m_ for m_ in sets if m_ != 'GS']
        for other in comps:
            ga, ca_, ua_ = res[(variant, 'GS')][2], None, None
            for tier in ('A', 'B', 'pooled'):
                def units(mn):
                    ua, ub, ca = res[(variant, mn)]
                    if tier == 'A':
                        d = ca
                    elif tier == 'B':
                        d = ub
                    else:
                        d = pd.concat([ca, ub])
                    return d
                da, db = units('GS'), units(other)
                if tier == 'A':
                    key_a, key_b = da['key'].values, db['key'].values
                elif tier == 'B':
                    key_a, key_b = da['key'].values, db['key'].values
                else:
                    key_a, key_b = da['key'].values, db['key'].values
                assert (key_a == key_b).all()
                ci = np.array([cid[k] for k in key_a])
                pt, (lo, hi) = boot_diff(da.crps.values, db.crps.values, ci, len(cid), rng)
                boots.append(dict(variant=variant, comparison=f'GS_vs_{other}', tier=tier, n_comp=len(set(key_a)), crps_diff_GS_minus_other=pt, ci95_lo=lo, ci95_hi=hi,
                                  GS_crps=float(da.crps.mean()), other_crps=float(db.crps.mean())))
        # phase check
        if variant == 'comp' and 'GS' in sets:
            pc = phase_check(sets['GS'][0])
            pc.to_csv(CV / 'phase_check.csv', index=False)
            rho, pv = spearmanr(pc.omega2, pc.narrowing)
            res['phase'] = dict(n_comp=len(pc), spearman_rho=float(rho), p=float(pv), mean_omega2=float(pc.omega2.mean()), mean_narrowing=float(pc.narrowing.mean()))
    pd.DataFrame(met).to_csv(CV / 'metrics.csv', index=False)
    pd.DataFrame(boots).to_csv(CV / 'bootstrap.csv', index=False)
    pd.DataFrame(pitrows).to_csv(CV / 'pit.csv', index=False)
    wins = pd.concat(winrows)
    wins.to_csv(CV / 'windows_units.csv', index=False)
    srows = []
    for (variant, mname), g in wins.groupby(['variant', 'model']):
        srows.append(dict(variant=variant, model=mname, n_comp=len(g), **window_summary(g.reset_index(drop=True))))
    pd.DataFrame(srows).to_csv(CV / 'windows.csv', index=False)
    pd.concat(allrows).to_csv(CV / 'predictions_summary.csv', index=False)
    json.dump(res.get('phase', {}), open(CV / 'phase_summary.json', 'w'))
    print('done', res.get('phase'))


if __name__ == '__main__':
    main()
