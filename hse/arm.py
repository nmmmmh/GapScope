"""Arms: H (HSE-warm aggregate prior, FROZEN recipe), H_MEAN, H_AUX, FID (fidelity, no HSE), ARCH / ARCH_S2 (archived states replay)."""
import sys, time, json, argparse, copy
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
from hse_common import *
import train as native
import distribution_head_compare as dc
from training_data import build_fold
p = argparse.ArgumentParser()
p.add_argument('--arm', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--folds', default='0,1,2,3,4')
p.add_argument('--out', default=str(H / 'out'))
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
dev = rt.configure()
m = rt.modules()
seed = a.seed
ext = external_inputs(dev)
folds = [int(x) for x in a.folds.split(',')]
tag = a.arm


def load_hse():
    return torch.load(H / f'hse_pretrain_seed{seed}.pt', map_location='cpu', weights_only=False)


def apply_hse(model, cp, scalers):
    """Load the pretrained shared network; re-express the composition projection exactly in this fold's input scaling."""
    st = {k: v for k, v in cp['state'].items() if not k.startswith('report_head')}
    mh, sh = cp['comp_mean'].double(), cp['comp_std'].double()
    mf, sf = scalers['composition_mean'].cpu().double(), scalers['composition_std'].cpu().double()
    W, b = st['composition_projection.weight'].double(), st['composition_projection.bias'].double()
    st['composition_projection.weight'] = (W * (sf / sh)[None]).float()
    st['composition_projection.bias'] = (b + W @ ((mf - mh) / sh)).float()
    res = model.load_state_dict(st, strict=False)
    assert not res.unexpected_keys and all(k.startswith('report_head') for k in res.missing_keys), res


def aggregate_prior_h(d, fold, hse):
    """train.aggregate_prior verbatim, plus optional HSE initialization of the shared network before the fit."""
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
    return model, standardized, fit, dict(log1p_mean=float(mean), log1p_std=float(std))


def mean_targets(d):
    target = d['targets'].clone()
    groups = {}
    for j, i in enumerate(d['ti'].tolist()):
        groups.setdefault(d['records'][i]['composition_key'], []).append((i, j))
    for key, mem in groups.items():
        ii, jj = zip(*mem)
        w = d['tw'][list(jj)]
        target[list(ii)] = (w * d['targets'][list(ii)]).sum() / w.sum()
    assert torch.equal(target[d['vi']], d['targets'][d['vi']]) and torch.equal(target[d['oi']], d['targets'][d['oi']])
    return target


def restore(state):
    return dc._restore_mixture(m, state, dev)


if a.arm == 'H_AUX':
    sys.path.insert(0, str(STAGE2))
    import runner as s2
    external = s2.load_external_inputs(INPUTS, STAGE1, SRC, torch, rt)

for fold in folds:
    t0 = time.time()
    fdir = out / f'{tag}_seed{seed}_fold{fold+1}'
    d = build_fold(SRC, fold, dev)
    info = dict(seed=seed, fold=fold, arm=tag)
    if a.arm in ('H', 'FID'):
        hse = load_hse() if a.arm == 'H' else None
        prior, std, fit, ts = aggregate_prior_h(d, fold, hse)
        info.update(aggregate_selected_epoch=fit['selected_epoch'], aggregate_epochs_run=fit['epochs_run'],
                    aggregate_initial_val=fit['initial_validation'], aggregate_selected_val=fit['selected_validation'])
        model, fit2 = native.adapt(prior, d, std, 'FROZEN')
        info.update(frozen_selected_epoch=fit2['selected_epoch'], frozen_epochs_run=fit2['epochs_run'])
        torch.save(dict(prior=native.state(prior), frozen=native.state(model), target_scalers=ts, info=info), str(fdir) + '_ckpt.pt')
    elif a.arm == 'H_MEAN':
        cp = torch.load(out / f'H_seed{seed}_fold{fold+1}_ckpt.pt', map_location='cpu', weights_only=False)
        prior = restore(cp['prior'])
        std = (d['ay'].log1p() - cp['target_scalers']['log1p_mean']) / cp['target_scalers']['log1p_std']
        d2 = dict(d)
        d2['targets'] = mean_targets(d)
        model, fit2 = native.adapt(prior, d2, std, 'FROZEN')
        info.update(frozen_selected_epoch=fit2['selected_epoch'], frozen_epochs_run=fit2['epochs_run'])
    elif a.arm == 'H_AUX':
        cp = torch.load(out / f'H_seed{seed}_fold{fold+1}_ckpt.pt', map_location='cpu', weights_only=False)
        d, aux, _ = s2.fold_context(SRC, fold, dev, torch, build_fold, external)
        model = restore(cp['prior'])
        model.mode = 'M'
        model.requires_grad_(False)
        model.report_head.requires_grad_(True)
        replay = s2.replay_constant(model, d, cp['target_scalers'])
        params = [q for q in model.parameters() if q.requires_grad]
        assert sum(q.numel() for q in params) == 326
        opt = torch.optim.Adam(params, lr=s2.HEAD_LR)
        counter = {'e': 0}
        rs = 100000 + seed * 1000 + fold * 400

        def objective():
            e = counter['e']
            counter['e'] += 1
            s2.set_rng(torch, rs + e)
            return s2.loss_terms(model, d, aux, m['mixture_model'], 'DIST', replay)['total']
        fit2 = native.optimize(model, objective, lambda: s2.core_valid_nll(model, d), opt, s2.MAX_EPOCHS)
        info.update(frozen_selected_epoch=fit2['selected_epoch'], frozen_epochs_run=fit2['epochs_run'])
    elif a.arm == 'ARCH':
        cp = torch.load(SRC / f'models/checkpoints/seed_{seed}_fold_{fold+1:02d}/models.pt', map_location='cpu', weights_only=False)
        model = restore(cp['models']['FROZEN'])
    elif a.arm == 'ARCH_S2':
        c = torch.load(STAGE2 / f'runs/fold_{fold+1:02d}/seed_{seed}/DIST/checkpoint.pt', map_location='cpu', weights_only=False)
        model = restore(c['selected_state'])
    if a.arm == 'FID':
        cp = torch.load(SRC / f'models/checkpoints/seed_{seed}_fold_{fold+1:02d}/models.pt', map_location='cpu', weights_only=False)
        ref = cp['frozen_aggregate_prior']
        pr = native.state(prior)
        info['prior_vs_archived_max_abs_diff'] = max(float((pr[k].double() - ref[k].double()).abs().max()) for k in ref if ref[k].is_floating_point())
        fr = native.state(model)
        info['frozen_vs_archived_max_abs_diff'] = max(float((fr[k].double() - cp['models']['FROZEN'][k].double()).abs().max()) for k in fr if fr[k].is_floating_point())
    save_pred(str(fdir) + '_pred.npz', predict_all(model, d, ext, m, dev))
    json.dump(info, open(str(fdir) + '_info.json', 'w'))
    print(tag, seed, fold + 1, {k: v for k, v in info.items() if k not in ('seed', 'fold', 'arm')}, f'{time.time()-t0:.0f}s', flush=True)
