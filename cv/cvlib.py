"""Shared pieces for the grouped-CV GapScope arms (GS, GS_MEAN, GS_noHSE, GS_noB). Reuses the H_AUX recipe code unchanged."""
import sys, json, csv, time, hashlib, copy
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
from hse_common import *
import train as native
import distribution_head_compare as dc
import training_data as td
sys.path.insert(0, str(STAGE2))
import runner as s2
import pandas as pd
CV = Path('/data/czx/GapScope_Revision_20261113/cv')
dev = None
m = None


def init():
    global dev, m
    dev = rt.configure()
    m = rt.modules()
    return dev, m


def restore(state):
    return dc._restore_mixture(m, state, dev)


def load_fixed():
    A = torch.load(CV / 'data/tierA.pt', weights_only=False)
    B = torch.load(CV / 'data/tierB_features.pt', weights_only=False)
    rowsB = pd.read_csv(CV / 'data/tierB_rows.csv', dtype={'target_ev': str})
    return A['records'], A['features'], B, rowsB


def pool_all():
    """aggregate pool (2753 labels incl. symmetry completion) exactly as training_data.build_fold loads it"""
    root = Path(SRC)
    agg = root / 'data/composition'
    f = td._load_features(agg / 'features.pt')
    md = td._require_rows(td._read_json(agg / 'metadata.json'), agg / 'metadata.json')
    p = agg / 'symmetry_completion.json'
    if p.is_file():
        from candidate_symmetry_completion import apply_completion
        f, md, _ = apply_completion(f, md, rt.read_json(p))
    return f, md


def build_fold_cv(variant, fold, fixed, pool, folds_df, inner_val):
    """fold: 0..4. Returns d (same keys as training_data.build_fold), tier-B aux, tier-B test inputs, counts."""
    recs, feat, B, rowsB = fixed
    pf, pmd = pool
    fd = folds_df.set_index('composition_key').fold.to_dict()
    test_keys = {k for k, v in fd.items() if v == fold + 1}
    ivk = set(inner_val[str(fold + 1)])
    rk = [r['composition_key'] for r in recs]
    oi = [i for i, k in enumerate(rk) if k in test_keys]
    vi = [i for i, k in enumerate(rk) if k in ivk]
    ti = [i for i, k in enumerate(rk) if k not in test_keys and k not in ivk]
    assert not (test_keys & ivk)
    drop = test_keys | ivk
    keep = [i for i, r in enumerate(pmd) if r['composition_key'] not in drop]
    counts = dict(pool_total=len(pmd), pool_dropped_test=sum(r['composition_key'] in test_keys for r in pmd),
                  pool_dropped_innerval=sum(r['composition_key'] in ivk for r in pmd))
    af = td._feature_subset(pf, keep, len(pmd))
    amd = [pmd[i] for i in keep]
    ay = af['targets'].reshape(-1).to(dtype=torch.float64)
    train_keys = {rk[i] for i in ti}
    avi = [i for i, r in enumerate(amd) if r['composition_key'] not in train_keys and td._hash_validation_key(r['composition_key'])]
    avs = set(avi)
    afi = [i for i in range(len(amd)) if i not in avs]
    symtab = m['loader43'].symmetry_table()
    ah = m['t35'].transform(af, 'hybrid', symtab, set())
    ridx = {r['measurement_id']: i for i, r in enumerate(recs)}
    cm, cs, sm, ss, st = m['t10'].fold_scalers(ah, amd, afi, feat, recs, [recs[i]['measurement_id'] for i in ti], recs, ridx, dev)
    views = m['t10'].four_views(feat, cm, cs, sm, ss, dev)
    av = m['t10'].four_views(ah, cm, cs, sm, ss, dev)
    T = lambda x: torch.tensor(x, dtype=torch.long, device=dev)
    tt = torch.tensor([float(r['gap_ev']) for r in recs], dtype=torch.float64, device=dev)
    d = dict(views=views, av=av, targets=tt, ay=ay.to(dev), ti=T(ti), vi=T(vi), oi=T(oi), fi=T(afi), ai=T(avi),
             tw=td.hierarchical_weights(recs, ti, dev), vw=td.hierarchical_weights(recs, vi, dev), ow=td.hierarchical_weights(recs, oi, dev),
             records=recs, scalers=dict(composition_mean=cm, composition_std=cs, structure_mean=sm, structure_std=ss, **st))
    counts.update(A_train=len(ti), A_val=len(vi), A_test=len(oi), agg_fit=len(afi), agg_val=len(avi), agg_retained=len(amd))
    bk = list(B['keys'])
    bf = B['composition']
    trB = [i for i, k in enumerate(bk) if fd[k] != fold + 1]
    teB = [i for i, k in enumerate(bk) if fd[k] == fold + 1]
    trk = [bk[i] for i in trB]
    kidx = {k: j for j, k in enumerate(trk)}
    sub = rowsB[rowsB.composition_key.isin(set(trk))]
    sc = d['scalers']
    aux = dict(views=s2.formula_view(bf[trB], sc, dev, torch),
               row_indices=torch.tensor([kidx[k] for k in sub.composition_key], dtype=torch.long, device=dev),
               targets=torch.tensor(sub.target_ev.astype(float).values, dtype=torch.float64, device=dev),
               weights=torch.tensor(sub.within_composition_weight.values, dtype=torch.float64, device=dev), N=len(trk))
    mean = sub.assign(t=sub.target_ev.astype(float)).groupby('composition_key').t.mean()
    aux['means'] = torch.tensor([mean[k] for k in trk], dtype=torch.float64, device=dev)
    assert abs(float(aux['weights'].sum()) - len(trk)) < 1e-6
    test_b = dict(keys=[bk[i] for i in teB], views=s2.formula_view(bf[teB], sc, dev, torch))
    counts.update(B_train_comps=len(trk), B_train_values=len(sub), B_test_comps=len(teB))
    return d, aux, test_b, counts


def apply_hse(model, cp, scalers):
    st = {k: v for k, v in cp['state'].items() if not k.startswith('report_head')}
    mh, sh = cp['comp_mean'].double(), cp['comp_std'].double()
    mf, sf = scalers['composition_mean'].cpu().double(), scalers['composition_std'].cpu().double()
    W, b = st['composition_projection.weight'].double(), st['composition_projection.bias'].double()
    st['composition_projection.weight'] = (W * (sf / sh)[None]).float()
    st['composition_projection.bias'] = (b + W @ ((mf - mh) / sh)).float()
    res = model.load_state_dict(st, strict=False)
    assert not res.unexpected_keys and all(k.startswith('report_head') for k in res.missing_keys), res


def aggregate_prior_h(d, seed, fold, hse):
    """hse/arm.py aggregate_prior_h verbatim (hse=None -> cold)."""
    torch.manual_seed(seed * 100 + fold)
    fi, ai, av, ay = (d[k] for k in ('fi', 'ai', 'av', 'ay'))
    positive = fi[ay[fi] > 0]
    x = av[0, positive, :183].cpu().double()
    beta = torch.linalg.lstsq(torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), 1), ay[positive].log().cpu(), rcond=1e-6, driver='gelsd').solution
    raw = av[0, fi, :183].double() @ beta[:183].to(dev) + beta[183].to(dev)
    model = m['interaction']('cross_attention', beta, float(raw.mean()), float(raw.std(unbiased=False).clamp_min(1e-6)), mode='F').to(dev)
    if hse is not None:
        apply_hse(model, hse, d['scalers'])
    mean = ay[fi].log1p().mean()
    std = ay[fi].log1p().std(unbiased=False).clamp_min(1e-6)
    standardized = (ay.log1p() - mean) / std
    loss = lambda ix: (model.aggregate(av[:, ix]).double() - standardized[ix]).square().mean()
    opt = torch.optim.Adam(model.parameters(), lr=native.RECIPE['aggregate_lr'])
    fit = native.optimize(model, lambda: loss(fi), lambda: loss(ai), opt, native.RECIPE['aggregate_epochs'])
    m['mixture_model'].initialize_report_head(model, 'mixture_lognormal', ay[positive], torch.full((len(positive),), 1 / len(positive), dtype=torch.float64, device=dev))
    with torch.no_grad():
        model.report_head.alpha.zero_()
    return model, fit, dict(log1p_mean=float(mean), log1p_std=float(std))


def mean_targets(d):
    target = d['targets'].clone()
    groups = {}
    for j, i in enumerate(d['ti'].tolist()):
        groups.setdefault(d['records'][i]['composition_key'], []).append((i, j))
    for k, mem in groups.items():
        ii, jj = zip(*mem)
        w = d['tw'][list(jj)]
        target[list(ii)] = (w * d['targets'][list(ii)]).sum() / w.sum()
    return target


def fit_head(prior_state, ts, d, aux, seed, fold, mode):
    """mode: GS (core + DIST aux/N), GS_noB (core only), GS_MEAN (mean targets, MEAN_AUX).
    Head-only, shared net frozen (326 params), selection on tier-A inner-validation NLL."""
    model = restore(prior_state)
    model.mode = 'M'
    model.requires_grad_(False)
    model.report_head.requires_grad_(True)
    replay = s2.replay_constant(model, d, ts)
    params = [q for q in model.parameters() if q.requires_grad]
    assert sum(q.numel() for q in params) == 326
    opt = torch.optim.Adam(params, lr=s2.HEAD_LR)
    targets = mean_targets(d) if mode == 'GS_MEAN' else d['targets']
    dm = m['mixture_model']
    counter = {'e': 0}
    rs = 100000 + seed * 1000 + fold * 400

    def objective():
        e = counter['e']
        counter['e'] += 1
        s2.set_rng(torch, rs + e)
        core_nll = model.per_record_nll(d['views'][:, d['ti']], targets[d['ti']])
        core = (core_nll.mean(dim=0) * d['tw']).sum()
        total = core + s2.REPLAY_COEFFICIENT * replay
        if mode == 'GS_noB':
            return total
        raw = model.raw_report(aux['views'])
        if mode == 'GS_MEAN':
            auxl = dm.nll_per_record(raw, aux['means'], 'mixture_lognormal').mean()
        else:
            auxl = (dm.nll_per_record(raw[aux['row_indices']], aux['targets'], 'mixture_lognormal') * aux['weights']).sum() / float(aux['N'])
        return total + auxl
    fit = native.optimize(model, objective, lambda: s2.core_valid_nll(model, d), opt, s2.MAX_EPOCHS)
    return model, fit


def arr(parts):
    return np.stack([p.detach().cpu().numpy().astype(np.float64) for p in parts])


def predict_cv(model, d, test_b):
    out = {}
    model.eval()
    with torch.no_grad():
        for v in range(4):
            out[f'A{v}'] = arr(model.components(d['views'][v, d['oi']]))
        out['A_idx'] = d['oi'].cpu().numpy()
        out['B'] = arr(model.components(test_b['views']))
    return out


def pretrain_hse(seed, excluded):
    """hse/pretrain.py verbatim, on the HSE pool minus `excluded` composition keys."""
    pool = torch.load(H / 'hse_pool_features.pt', map_location='cpu', weights_only=False)
    y = pool.pop('targets')
    keys = pool.pop('keys')
    keep = torch.tensor([k not in excluded for k in keys])
    info = dict(rows_total=len(keys), rows_dropped=int((~keep).sum()),
                keys_dropped=len({k for k, kk in zip(keys, keep.tolist()) if not kk}))
    pool = {k: v[keep] for k, v in pool.items()}
    y = y[keep]
    keys = [k for k, kk in zip(keys, keep.tolist()) if kk]
    feats = m['t35'].transform(pool, 'hybrid', m['loader43'].symmetry_table(), set())
    first = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    idx = torch.tensor(sorted(first.values()))
    cm, cs = m['t10'].standardize(feats['composition'][idx].to(dev))
    views = m['t10'].four_views(feats, cm, cs, torch.zeros(160, device=dev), torch.ones(160, device=dev), dev)[[0, 1]]
    t = y.to(dev).log1p()
    lm, ls = float(t.mean()), float(t.std(unbiased=False))
    z = ((t - lm) / ls).float()
    N = len(z)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    nv = int(round(.05 * N))
    vi, ti = perm[:nv].to(dev), perm[nv:].to(dev)
    torch.manual_seed(seed)
    model = m['interaction']('cross_attention', torch.zeros(184, dtype=torch.float64), 0., 1., mode='F').to(dev)
    params = [q for n, q in model.named_parameters() if not n.startswith('report_head')]
    opt = torch.optim.Adam(params, lr=1e-3)

    def loss(ix):
        return (model.aggregate(views[:, ix]) - z[ix][None]).square().mean()

    def val():
        model.eval()
        with torch.no_grad():
            return float(sum(loss(vi[s:s + 8192]) * len(vi[s:s + 8192]) for s in range(0, len(vi), 8192)) / len(vi))
    best = val()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_ep = wait = 0
    for ep in range(1, 101):
        model.train()
        order = ti[torch.randperm(len(ti), generator=g).to(dev)]
        for s in range(0, len(order), 2048):
            ix = order[s:s + 2048]
            opt.zero_grad(set_to_none=True)
            l = loss(ix)
            l.backward()
            opt.step()
        v = val()
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state = {k: q.detach().cpu().clone() for k, q in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break
    info.update(selected_epoch=best_ep, best_val=best, epochs_run=ep)
    return dict(state=best_state, comp_mean=cm.cpu(), comp_std=cs.cpu(), log1p_mean=lm, log1p_std=ls), info
