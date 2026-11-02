"""One cold-start, composition-grouped training run for the current report pool."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import runtime as rt
import anchored_transfer as anchor
from training_data import build_fold

torch = rt.torch
RECIPE = dict(seeds=[17, 29, 41], folds=5, architecture='cross_attention',
              token_width=32, heads=4, fusion_width=64, components=2,
              aggregate_epochs=300, report_epochs=300, patience=30,
              aggregate_lr=.001, shared_lr=.0001, head_lr=.001,
              replay_coefficient=.1, report_selection='mean four training views VALID NLL; epoch0 eligible',
              initialization='cold aggregate FIT only; positive-FIT log moments initialize report head; no historical report weights',
              budget='15 aggregate fits + 60 matched report fits + up to 15 additional low-rank transfer fits; no parameter search',
              arms=['F', 'J', 'J_SG', 'FROZEN', 'ANCHORED'],
              default_model='FROZEN; shared prior frozen, four-view report head optimized',
              claim='incremental report-stage structure contribution conditional on common multimodal aggregate prior')


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def optimize(model, objective, validation, optimizer, epochs):
    model.eval()
    with torch.no_grad(): best = float(validation())
    initial = best; selected = state(model)
    best_epoch = wait = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training objective')
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): current = float(validation())
        if not np.isfinite(current): raise ValueError('Nonfinite validation objective')
        history.append(dict(epoch=epoch, training_loss=float(loss.detach()), validation_loss=current))
        if current < best - 1e-6:
            best, best_epoch, selected, wait = current, epoch, state(model), 0
        else: wait += 1
        if wait >= RECIPE['patience']: break
    model.load_state_dict(selected); model.eval()
    return dict(initial_validation=initial, selected_validation=best,
                selected_epoch=best_epoch, epochs_run=epoch, history=history)


def aggregate_prior(m, d, seed, fold, device):
    torch.manual_seed(seed * 100 + fold)
    fi, ai, av, ay = (d[k] for k in ('fi', 'ai', 'av', 'ay'))
    positive = fi[ay[fi] > 0]
    x = av[0, positive, :183].cpu().double()
    beta = torch.linalg.lstsq(torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), 1),
                             ay[positive].log().cpu(), rcond=1e-6, driver='gelsd').solution
    raw = av[0, fi, :183].double() @ beta[:183].to(device) + beta[183].to(device)
    # Keep initialization entirely aggregate-derived, as in the report-supervision control.
    model = m['interaction']('cross_attention', beta, float(raw.mean()),
                            float(raw.std(unbiased=False).clamp_min(1e-6)), mode='F').to(device)
    mean = ay[fi].log1p().mean(); std = ay[fi].log1p().std(unbiased=False).clamp_min(1e-6)
    standardized = (ay.log1p() - mean) / std
    loss = lambda indices: (model.aggregate(av[:, indices]).double() - standardized[indices]).square().mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['aggregate_lr'])
    fit = optimize(model, lambda: loss(fi), lambda: loss(ai), optimizer, RECIPE['aggregate_epochs'])
    m['mixture_model'].initialize_report_head(model, 'mixture_lognormal', ay[positive],
        torch.full((len(positive),), 1 / len(positive), dtype=torch.float64, device=device))
    with torch.no_grad(): model.report_head.alpha.zero_()
    return model, standardized, fit, dict(log1p_mean=float(mean), log1p_std=float(std))


def adapt(prior, d, standardized, arm):
    model = copy.deepcopy(prior); model.mode = 'F' if arm == 'F' else 'M'
    views = d['views'][[0, 1, 0, 1]] if arm == 'J_SG' else d['views']
    av = d['av'][[0, 1, 0, 1]] if arm == 'J_SG' else d['av']
    def nll(indices, weights):
        inputs = views[0, indices] if arm == 'F' else views[:, indices]
        loss = model.per_record_nll(inputs, d['targets'][indices])
        if loss.ndim == 2: loss = loss.mean(0)
        return (loss * weights).sum()
    def objective():
        replay = (model.aggregate(av[:, d['fi']]).double() - standardized[d['fi']]).square().mean()
        return nll(d['ti'], d['tw']) + RECIPE['replay_coefficient'] * replay
    head = list(model.report_head.parameters()); head_ids = {id(p) for p in head}
    if arm == 'FROZEN':
        model.requires_grad_(False)
        for parameter in head: parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(head, lr=RECIPE['head_lr'])
    else:
        optimizer = torch.optim.Adam([
            dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=RECIPE['shared_lr']),
            dict(params=head, lr=RECIPE['head_lr'])])
    fit = optimize(model, objective, lambda: nll(d['vi'], d['vw']), optimizer, RECIPE['report_epochs'])
    return model, fit


@torch.no_grad()
def base_parts(model, formula):
    w, mu, sd = model.components(formula)
    return dict(weights=w, mus=mu, sigmas=sd, gate=model.raw_report(formula)[..., 0],
                raw_scale=torch.logit((sd - .05) / 2.45), hidden=model.encode(formula).double())


def transfer(baseline, teacher, d, device):
    baseline.requires_grad_(False); teacher.requires_grad_(False)
    b = base_parts(baseline, d['views'][0]); rows = []
    with torch.no_grad():
        for view, label in [(1, 'SG_KD'), (3, 'FULL_KD')]:
            parts = teacher.components(d['views'][view, d['ti']])
            for j, i in enumerate(d['ti'].tolist()):
                r = d['records'][i]
                rows.append(dict(arm=label, measurement_id=r['measurement_id'], formula=r['formula'],
                    composition_key=r['composition_key'], train_weight=float(d['tw'][j]),
                    component_weights=json.dumps(parts[0][j].tolist()),
                    log_mus=json.dumps(parts[1][j].tolist()), sigmas=json.dumps(parts[2][j].tolist())))
    ids = [d['records'][i]['measurement_id'] for i in d['ti'].tolist()]
    groups, ti, grid, target, mass, zero = anchor.teacher_targets(pd.DataFrame(rows), b, ids, d['records'], device)
    hidden = b['hidden'][ti]; center = hidden.mean(0)
    _, s, vh = torch.linalg.svd(hidden - center, full_matrices=False)
    rank = min(4, int((s > torch.finfo(s.dtype).eps * s[0] * max(hidden.shape)).sum()))
    basis = vh[:rank].T; scale = s[:rank] / np.sqrt(len(hidden) - 1)
    features = (b['hidden'] - center) @ basis / scale
    head = torch.nn.Linear(rank, 5, bias=False, dtype=torch.float64, device=device)
    torch.nn.init.zeros_(head.weight)
    tb = anchor.subset_base(b, ti)
    original = anchor.cdf((tb['weights'], tb['mus'], tb['sigmas']), grid)
    energy = (mass * torch.trapezoid(target.square(), grid, dim=-1)).sum()
    cdf_history = []
    def cdf_diagnostics():
        increment = anchor.cdf(anchor.corrected(tb, head(features[ti])), grid) - original
        residual = (mass * torch.trapezoid((increment - target).square(), grid, dim=-1)).sum()
        return increment, dict(normalized_cdf_residual=float(residual / energy) if float(energy) > 0 else None,
            increment_energy=float((mass * torch.trapezoid(increment.square(), grid, dim=-1)).sum()),
            head_norm=float(head.weight.norm()))
    def objective():
        increment = anchor.cdf(anchor.corrected(tb, head(features[ti])), grid) - original
        return ((mass * torch.trapezoid((increment - target).square(), grid, dim=-1)).sum() / energy
                + .001 * head.weight.square().sum())
    def validation():
        with torch.no_grad():
            _, diagnostic = cdf_diagnostics()
            cdf_history.append(dict(epoch=len(cdf_history), **diagnostic))
        parts = anchor.corrected(anchor.subset_base(b, d['vi']), head(features[d['vi']]))
        return (d['vw'] * anchor.nll(parts, d['targets'][d['vi']])).sum()
    if zero or rank == 0:
        initial = float(validation())
        fit = dict(initial_validation=initial, selected_validation=initial, selected_epoch=0, epochs_run=0, history=[])
    else:
        fit = optimize(head, objective, validation, torch.optim.Adam(head.parameters(), lr=.001), 300)
    with torch.no_grad():
        parts = anchor.corrected(b, head(features))
        selected_increment, selected_diagnostic = cdf_diagnostics()
    fit.update(teacher_energy=float(energy), pca_rank=rank, zero_teacher=zero,
               nonzero_teacher_formulas=sum(g['teacher_delta_energy'] > 0 for g in groups),
               selected_normalized_cdf_residual=selected_diagnostic['normalized_cdf_residual'],
               final_normalized_cdf_residual=cdf_history[-1]['normalized_cdf_residual'],
               minimum_normalized_cdf_residual=min((r['normalized_cdf_residual'] for r in cdf_history
                   if r['normalized_cdf_residual'] is not None), default=None),
               selected_increment_energy=selected_diagnostic['increment_energy'], cdf_history=cdf_history)
    cp = dict(head_state=state(head), center=center.cpu(), basis=basis.cpu(), scale=scale.cpu())
    diagnostics = [{k: v for k, v in g.items() if k not in ('grid', 'target', 'report_index')} for g in groups]
    for j, diagnostic in enumerate(diagnostics):
        fitted = tuple(p[ti[j]] for p in parts)
        base = tuple(tb[k][j] for k in ('weights', 'mus', 'sigmas'))
        mean = lambda p: (p[0] * torch.exp(p[1] + .5 * p[2].square())).sum()
        diagnostic.update(learned_mean_increment_ev=float(mean(fitted) - mean(base)),
            selected_increment_log_w1=float(torch.trapezoid(selected_increment[j].abs(), grid[j])),
            selected_residual_energy=float(torch.trapezoid((selected_increment[j] - target[j]).square(), grid[j])))
    return parts, fit, cp, diagnostics


def prediction_rows(parts, d, seed, fold, arm, view, selected_indices=None):
    indices = d['oi'].tolist() if selected_indices is None else selected_indices
    result = []
    for j, i in enumerate(indices):
        r = d['records'][i]
        result.append(dict(model=arm, view=view, seed=seed, fold_id=fold,
            measurement_id=r['measurement_id'], formula=r['formula'], composition_key=r['composition_key'],
            phase_label=r.get('phase_label', ''),
            sg_supported=int(d['views'][3, i, -2].item()), structure_supported=int(d['views'][3, i, -1].item()),
            component_weights=json.dumps(parts[0][j].tolist()), log_mus=json.dumps(parts[1][j].tolist()),
            sigmas=json.dumps(parts[2][j].tolist())))
    return result


def run_pair(root, work, seed, fold, device, m):
    out = work / f'seed_{seed}_fold_{fold+1:02d}'; out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists(): return
    started = time.time(); d = build_fold(root, fold, device)
    rt.write_json(out/'partition.json', d['provenance'])
    prior, standardized, fit, target_scalers = aggregate_prior(m, d, seed, fold, device)
    fits = dict(aggregate=fit); models = {}
    prior_hash = hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes()
                                       for k,v in sorted(state(prior).items()))).hexdigest()
    print(f'seed={seed} fold={fold+1} aggregate selected={fit["selected_epoch"]}', flush=True)
    for arm in ('F', 'J', 'J_SG', 'FROZEN'):
        models[arm], fits[arm] = adapt(prior, d, standardized, arm)
        fits[arm]['initial_state_hash'] = prior_hash
        fits[arm]['parameters'] = sum(p.numel() for p in models[arm].parameters())
        print(f'seed={seed} fold={fold+1} {arm} selected={fits[arm]["selected_epoch"]}', flush=True)
    parts, fits['ANCHORED'], anchor_cp, diagnostics = transfer(models['F'], models['J'], d, device)
    predictions = prediction_rows(tuple(p[d['oi']] for p in parts), d, seed, fold, 'ANCHORED', 'formula_only')
    with torch.no_grad():
        for arm, model in models.items():
            for vi, view in enumerate(rt.VIEWS):
                if arm == 'F' and vi: continue
                used = (0 if vi in (0, 2) else 1) if arm == 'J_SG' else vi
                parts = model.components(d['views'][used, d['oi']])
                predictions += prediction_rows(parts, d, seed, fold, arm, view)
    rt.write_csv(out/'predictions.csv', predictions)
    rt.write_csv(out/'teacher_signal.csv', diagnostics)
    rt.write_json(out/'fits.json', fits)
    torch.save(dict(schema='current_training_v1', models={k: state(v) for k, v in models.items()},
                    aggregate_prior=state(prior), anchor=anchor_cp, scalers=d['scalers'], target_scalers=target_scalers,
                    recipe=RECIPE, partition=d['provenance']), out/'models.pt')
    rt.write_json(out/'complete.json', dict(seed=seed, fold=fold, seconds=time.time()-started,
                  predictions=len(predictions), report_counts={k: len(d[k]) for k in ('ti', 'vi', 'oi')},
                  aggregate_counts={k: len(d[k]) for k in ('fi', 'ai')}))
    print(f'COMPLETE seed={seed} fold={fold+1} seconds={time.time()-started:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Dataset root containing data/reports and data/composition.')
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args(); args.work_dir.mkdir(parents=True, exist_ok=True)
    device = rt.configure(); m = rt.modules()
    package_root = Path(__file__).resolve().parent
    sources = [Path(__file__), Path(__file__).with_name('training_data.py'),
               Path(__file__).with_name('runtime.py'), Path(__file__).with_name('anchored_transfer.py')]
    sources += sorted((Path(__file__).parent/'modules').glob('*.py'))
    sources += [Path(__file__).parent/'modules/symmetry_table.json']
    data_files = [
        args.root/'data/reports/features.pt', args.root/'data/reports/records.jsonl',
        args.root/'data/reports/metadata.json', args.root/'data/composition/features.pt',
        args.root/'data/composition/metadata.json']
    binding = {str(p.relative_to(args.root)): rt.sha(p) for p in data_files}
    binding.update({f"code/{p.relative_to(package_root).as_posix()}": rt.sha(p) for p in sources})
    protocol = RECIPE | dict(input_binding=binding, device=str(device), gpu=torch.cuda.get_device_name(0))
    path = args.work_dir/'protocol.json'
    if path.exists() and rt.read_json(path) != protocol: raise ValueError('Input or recipe changed; existing run cannot be resumed')
    rt.write_json(path, protocol)
    for seed in RECIPE['seeds']:
        for fold in range(5): run_pair(args.root, args.work_dir, seed, fold, device, m)


if __name__ == '__main__': main()
