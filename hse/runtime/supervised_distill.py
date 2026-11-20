"""Fixed 30-fit, formula-level absolute-CDF distillation controls."""
import argparse
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import runtime as rt
import train
import anchored_transfer as anchor
import transfer_stage as stage
from training_data import build_fold

torch = rt.torch
SEEDS = (17, 29, 41)
FOLDS = 5
ARMS = ("SG_DISTILL", "FULL_DISTILL")
RECIPE = dict(
    seeds=list(SEEDS), folds=FOLDS, report_fits=dict(SG_DISTILL=15, FULL_DISTILL=15, total=30),
    aggregate_fits=0, parameter_grid=[], parameters_per_head=325, hidden_width=64,
    head="zero-start Linear(64,5,bias=True); frozen FROZEN baseline; train-formula standardized hidden",
    objective="hierarchical true-report TRAIN NLL + .1 absolute teacher CDF MSE/common_E + .001 sum(head weight^2,bias^2)",
    teacher="same fixed FROZEN checkpoint absolute TRAIN formula-marginal SG or FULL CDF",
    common_energy=".5 * sum_formula mass * integral[(CDF_FULL-CDF_base)^2+(CDF_SG-CDF_base)^2] dlogEg",
    grid_points=1025, grid_sigma=9, learning_rate=.001, max_epochs=300, patience=30,
    selection="inner VALID hierarchical true-report NLL; epoch0 eligible",
    outer_targets_read=False, sealed_targets_read=False)


def _cdf(weights, mus, sigmas, grid):
    z = (grid[:, None] - mus[None, :]) / sigmas[None, :]
    return (weights[None, :] * .5 * (1 + torch.erf(z / math.sqrt(2)))).sum(-1)


def _correct(base, raw):
    first = torch.sigmoid(base['gate'] + raw[:, 0])
    weights = torch.stack((first, 1 - first), -1)
    mus = base['mus'] + raw[:, 1:3]
    scale0 = .05 + 2.45 * torch.sigmoid(base['raw_scale'])
    sigmas = base['sigmas'] + (.05 + 2.45 * torch.sigmoid(base['raw_scale'] + raw[:, 3:5]) - scale0)
    return weights, mus, sigmas


def _subset(base, indices):
    return {key: value[indices] for key, value in base.items()}


def _teacher_cdfs(rows, groups, ti, base, grid, mass, device):
    targets = {}
    base_cdf = []
    for k, group in enumerate(groups):
        formula_grid = grid[k]
        i = int(ti[k])
        base_cdf.append(_cdf(base['weights'][i], base['mus'][i], base['sigmas'][i], formula_grid))
        subset = rows[rows.composition_key == group['composition_key']]
        formula_mass = float(subset[subset.arm == 'SG_KD'].train_weight.sum())
        for arm, label in (("SG_DISTILL", "SG_KD"), ("FULL_DISTILL", "FULL_KD")):
            w_all, mu_all, sd_all = [], [], []
            for row in subset[subset.arm == label].itertuples(index=False):
                fraction = float(row.train_weight) / formula_mass
                w = rt.json.loads(row.component_weights)
                mu = rt.json.loads(row.log_mus)
                sd = rt.json.loads(row.sigmas)
                w_all.extend(fraction * value for value in w)
                mu_all.extend(mu)
                sd_all.extend(sd)
            targets.setdefault(arm, []).append(_cdf(
                torch.tensor(w_all, dtype=torch.float64, device=device),
                torch.tensor(mu_all, dtype=torch.float64, device=device),
                torch.tensor(sd_all, dtype=torch.float64, device=device), formula_grid))
    base_cdf = torch.stack(base_cdf)
    targets = {key: torch.stack(value) for key, value in targets.items()}
    energy = .5 * (mass * torch.trapezoid(
        (targets['FULL_DISTILL'] - base_cdf).square() +
        (targets['SG_DISTILL'] - base_cdf).square(), grid, dim=-1)).sum()
    return base_cdf, targets, energy


def _fit_arm(arm, teacher, energy, baseline_parts, features, data, ti_formula, grid, mass):
    head = torch.nn.Linear(64, 5, bias=True, dtype=torch.float64, device=features.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    ti, vi = data['ti'], data['vi']
    tw, vw = data['tw'], data['vw']
    y = data['targets']
    train_base, formula_base, valid_base = (_subset(baseline_parts, idx) for idx in (ti, ti_formula, vi))
    positive_energy = float(energy) > 0

    def terms():
        parts = _correct(train_base, head(features[ti]))
        nll = (anchor.nll(parts, y[ti]) * tw).sum()
        predicted = anchor.cdf(_correct(formula_base, head(features[ti_formula])), grid)
        numerator = (mass * torch.trapezoid((predicted - teacher).square(), grid, dim=-1)).sum()
        kd = numerator / energy if positive_energy else numerator * 0
        return nll, kd

    def objective():
        nll, kd = terms()
        return nll + .1 * kd + .001 * (head.weight.square().sum() + head.bias.square().sum())

    trace = []

    def validation():
        with torch.no_grad():
            nll, kd = terms()
            trace.append(dict(epoch=len(trace), train_nll=float(nll), kd_normalized=float(kd)))
            return (anchor.nll(_correct(valid_base, head(features[vi])), y[vi]) * vw).sum()

    fit = train.optimize(head, objective, validation, torch.optim.Adam(head.parameters(), lr=.001), 300)
    selected, final = trace[fit['selected_epoch']], trace[-1]
    fit.update(model_key=arm, baseline_key='FROZEN', baseline_fixed=True,
               teacher_target='absolute_formula_marginal_CDF', target_loss_not_phase_assignment=True,
               trainable_parameters=325, common_teacher_energy=float(energy),
               zero_common_energy=not positive_energy,
               selected_train_nll=selected['train_nll'], final_train_nll=final['train_nll'],
               minimum_train_nll=min(row['train_nll'] for row in trace),
               selected_kd_normalized=selected['kd_normalized'],
               final_kd_normalized=final['kd_normalized'],
               minimum_kd_normalized=min(row['kd_normalized'] for row in trace),
               validation_trace_rows=len(trace))
    return head, fit


def _run_fold(root, work, seed, fold, device, modules, protocol_sha):
    out = work / f'seed_{seed}_fold_{fold + 1:02d}'
    out.mkdir(parents=True, exist_ok=True)
    complete = out / 'complete.json'
    if complete.exists():
        receipt = rt.read_json(complete)
        for name in ('predictions.csv', 'fits.json', 'models.pt'):
            key = name.split('.')[0] + '_sha256'
            if receipt.get(key) != rt.sha(out / name):
                raise ValueError(f'completed_output_hash_mismatch:{out.name}/{name}')
        if receipt.get('protocol_sha256') != protocol_sha:
            raise ValueError(f'completed_protocol_mismatch:{out.name}')
        return

    started = time.time()
    data = build_fold(root, fold, device)
    source = root / 'models' / 'checkpoints' / f'seed_{seed}_fold_{fold + 1:02d}' / 'models.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    stage._check_checkpoint(checkpoint, data)
    baseline_state = checkpoint['models']['FROZEN']
    baseline = stage._restore(modules, baseline_state, device).requires_grad_(False).eval()
    baseline_parts = train.base_parts(baseline, data['views'][0])
    teacher_rows = stage._teacher_rows(baseline, data, device)
    ids = [data['records'][i]['measurement_id'] for i in data['ti'].tolist()]
    groups, ti_formula, grid, _delta, mass, _zero = anchor.teacher_targets(
        teacher_rows, baseline_parts, ids, data['records'], device)
    base_cdf, teachers, energy = _teacher_cdfs(
        teacher_rows, groups, ti_formula, baseline_parts, grid, mass, device)
    del base_cdf

    hidden = baseline_parts['hidden'][ti_formula]
    if hidden.shape[1] != 64:
        raise ValueError(f'expected_64_hidden_features:{hidden.shape[1]}')
    center = hidden.mean(0)
    scale = hidden.std(0, unbiased=False)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale.clamp_min(1e-6))
    basis = torch.eye(64, dtype=torch.float64, device=device)
    features = (baseline_parts['hidden'] - center) / scale

    torch.manual_seed(seed * 100 + fold)
    fits, heads, anchors = {}, {}, {}
    for arm in ARMS:
        head, fit = _fit_arm(arm, teachers[arm], energy, baseline_parts,
                             features, data, ti_formula, grid, mass)
        fit['baseline_state_sha256'] = stage._state_hash(baseline_state)
        fits[arm] = fit
        heads[arm] = head
        anchors[arm] = dict(head_state=train.state(head), center=center.detach().cpu(),
                            basis=basis.detach().cpu(), scale=scale.detach().cpu())

    predictions = train.prediction_rows(
        tuple(value[data['oi']] for value in (baseline_parts['weights'], baseline_parts['mus'], baseline_parts['sigmas'])),
        data, seed, fold, 'FROZEN', 'formula_only')
    for arm in ARMS:
        outer = _subset(baseline_parts, data['oi'])
        parts = _correct(outer, heads[arm](features[data['oi']]))
        predictions += train.prediction_rows(parts, data, seed, fold, arm, 'formula_only')
    expected = 3 * len(data['oi'])
    if len(predictions) != expected:
        raise ValueError(f'prediction_count_mismatch:{len(predictions)}:{expected}')
    rt.write_csv(out / 'predictions.csv', predictions)
    rt.write_json(out / 'fits.json', fits)
    models_path = out / 'models.pt'
    torch.save(dict(schema='supervised_distill_v1', seed=seed, fold=fold,
                    models={'FROZEN': baseline_state}, baseline_key='FROZEN', anchors=anchors,
                    scalers=checkpoint['scalers'], target_scalers=checkpoint['target_scalers'],
                    partition=checkpoint['partition'], frozen_recipe=checkpoint['frozen_recipe']), models_path)
    rt.write_json(complete, dict(status='complete', seed=seed, fold=fold, report_fits=2,
                  total_fits=2, predictions=len(predictions), protocol_sha256=protocol_sha,
                  predictions_sha256=rt.sha(out / 'predictions.csv'), fits_sha256=rt.sha(out / 'fits.json'),
                  models_sha256=rt.sha(models_path), seconds=time.time() - started))
    print(f'COMPLETE seed={seed} fold={fold + 1} fits=2', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work_dir.resolve()
    try:
        work.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError('work_dir_must_be_outside_formal_root')
    work.mkdir(parents=True, exist_ok=True)
    device, modules = rt.configure(), rt.modules()
    inputs = [root / 'data' / 'reports' / name for name in
              ('features.pt', 'records.jsonl', 'metadata.json')]
    inputs += [root / 'data' / 'composition' / name for name in ('features.pt', 'metadata.json')]
    checkpoints = [root / 'models' / 'checkpoints' / f'seed_{seed}_fold_{fold:02d}' / 'models.pt'
                   for seed in SEEDS for fold in range(1, FOLDS + 1)]
    code = [Path(__file__).resolve(), Path(train.__file__).resolve(), Path(anchor.__file__).resolve(),
            Path(stage.__file__).resolve(), Path(build_fold.__code__.co_filename).resolve(),
            Path(rt.__file__).resolve(), root / 'runtime' / 'inference.py',
            root / 'runtime' / 'frozen_fusion_compare.py']
    code += sorted((root / 'runtime' / 'modules').glob('*.py'))
    code += [root / 'runtime' / 'modules' / 'symmetry_table.json']
    protocol = RECIPE | dict(device=str(device), gpu=torch.cuda.get_device_name(0),
        input_binding={str(path.relative_to(root)): rt.sha(path) for path in inputs},
        checkpoint_binding={str(path.relative_to(root)): rt.sha(path) for path in checkpoints},
        code_binding={str(path.relative_to(root)): rt.sha(path) for path in code})
    protocol_path = work / 'protocol.json'
    if protocol_path.exists() and rt.read_json(protocol_path) != protocol:
        raise ValueError('input_or_recipe_changed_existing_run_cannot_be_resumed')
    rt.write_json(protocol_path, protocol)
    protocol_sha = rt.sha(protocol_path)
    for seed in SEEDS:
        for fold in range(FOLDS):
            _run_fold(root, work, seed, fold, device, modules, protocol_sha)


if __name__ == '__main__':
    main()
