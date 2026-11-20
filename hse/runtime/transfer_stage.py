"""Fixed 30-fit FROZEN_SG and formula-anchored structure-transfer stage."""
import argparse
import copy
import hashlib
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import runtime as rt
import train
import anchored_transfer as anchor
from training_data import build_fold

torch = rt.torch
SEEDS = (17, 29, 41)
FOLDS = 5
SG_VIEWS = (0, 1, 0, 1)
RECIPE = dict(
    seeds=list(SEEDS), folds=FOLDS, aggregate_fits=0,
    report_fits=dict(FROZEN_SG=15, STRUCTURE_TRANSFER=15, total=30),
    frozen_sg=dict(views=list(SG_VIEWS), train_loss='hierarchical four-view LogNormal NLL',
                   validation='hierarchical four-view VALID NLL; epoch0 eligible',
                   frozen='all parameters except report_head including alpha',
                   learning_rate=.001, epochs=300, patience=30, replay_coefficient=.1),
    structure_transfer=dict(rank_max=4, outputs=5, bias=False, epochs=300,
                            patience=30, learning_rate=.001, l2_coefficient=.001,
                            grid_points=1025, grid_sigma=9,
                            selection='inner VALID weighted true-report NLL including epoch0',
                            train_objective='formula-weighted integrated squared CDF increment residual / teacher increment energy + fixed L2',
                            teacher='fixed FROZEN TRAIN FULL marginal CDF minus SG marginal CDF',
                            baseline='fixed current FROZEN checkpoint; formula-only input'),
    parameter_grid=[],
)


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _restore(modules, state, device):
    model = modules['interaction']('cross_attention', state['report_head.raw_beta'],
                                   float(state['report_head.raw_feature_mean']),
                                   float(state['report_head.raw_feature_std']), mode='M')
    model.load_state_dict(state)
    return model.to(device).eval()


def _fit_frozen_sg(prior, d, standardized):
    model = copy.deepcopy(prior)
    model.mode = 'M'
    views, av = d['views'][list(SG_VIEWS)], d['av'][list(SG_VIEWS)]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = list(model.report_head.parameters())
    for parameter in head:
        parameter.requires_grad_(True)

    def nll(indices, weights):
        losses = model.per_record_nll(views[:, indices], d['targets'][indices])
        if losses.ndim == 2:
            losses = losses.mean(0)
        return (losses * weights).sum()

    def objective():
        replay = (model.aggregate(av[:, d['fi']]).double() - standardized[d['fi']]).square().mean()
        return nll(d['ti'], d['tw']) + .1 * replay

    fit = train.optimize(model, objective, lambda: nll(d['vi'], d['vw']),
                         torch.optim.Adam(head, lr=.001), 300)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(not name.startswith('report_head.') for name in names):
        raise ValueError('frozen_sg_unexpected_trainable_parameter')
    fit.update(selected_arm='FROZEN_SG', trainable_parameter_names=names,
               parameters=sum(p.numel() for p in model.parameters()))
    return model, fit


def _teacher_rows(teacher, d, device):
    rows = []
    with torch.no_grad():
        for view, label in ((1, 'SG_KD'), (3, 'FULL_KD')):
            parts = teacher.components(d['views'][view, d['ti']])
            for j, i in enumerate(d['ti'].tolist()):
                record = d['records'][i]
                rows.append(dict(arm=label, measurement_id=record['measurement_id'],
                                 formula=record['formula'], composition_key=record['composition_key'],
                                 train_weight=float(d['tw'][j]),
                                 component_weights=rt.json.dumps(parts[0][j].tolist()),
                                 log_mus=rt.json.dumps(parts[1][j].tolist()),
                                 sigmas=rt.json.dumps(parts[2][j].tolist())))
    return train.pd.DataFrame(rows)


def _transfer_diagnostics(baseline, teacher, d, device, cp, fit):
    base = train.base_parts(baseline, d['views'][0])
    ids = [d['records'][i]['measurement_id'] for i in d['ti'].tolist()]
    groups, ti, grid, target, mass, zero = anchor.teacher_targets(
        _teacher_rows(teacher, d, device), base, ids, d['records'], device)
    center = cp['center'].to(device=device, dtype=torch.float64)
    basis = cp['basis'].to(device=device, dtype=torch.float64)
    scale = cp['scale'].to(device=device, dtype=torch.float64)
    weight = cp['head_state']['weight'].to(device=device, dtype=torch.float64)
    features = (base['hidden'] - center) @ basis / scale
    delta = torch.nn.functional.linear(features, weight)
    train_base = anchor.subset_base(base, ti)
    original = anchor.cdf((train_base['weights'], train_base['mus'], train_base['sigmas']), grid)
    increment = anchor.cdf(anchor.corrected(train_base, delta[ti]), grid) - original
    energy = (mass * torch.trapezoid(target.square(), grid, dim=-1)).sum()
    residual = None if float(energy) <= 0 else float(
        (mass * torch.trapezoid((increment - target).square(), grid, dim=-1)).sum() / energy)
    if bool(zero) != bool(fit['zero_teacher']) or basis.shape[1] != int(fit['pca_rank']):
        raise ValueError('transfer_diagnostic_reconstruction_mismatch')

    outer = d['oi']
    outer_delta = delta[outer]
    ob = anchor.subset_base(base, outer)
    fitted = anchor.corrected(ob, outer_delta)
    def mean_ev(parts):
        return (parts[0] * torch.exp(parts[1] + .5 * parts[2].square())).sum(-1)
    mean_shift = mean_ev(fitted) - mean_ev(tuple(ob[k] for k in ('weights', 'mus', 'sigmas')))
    fit.update(selected_normalized_cdf_residual=residual,
               selected_head_norm=float(torch.linalg.vector_norm(weight)),
               outer_maxraw_delta=float(outer_delta.abs().max()) if outer_delta.numel() else 0.0,
               outer_mean_density_increment_ev=float((d['ow'] * mean_shift).sum()),
               outer_mean_abs_density_increment_ev=float((d['ow'] * mean_shift.abs()).sum()))
    return groups


def _check_checkpoint(checkpoint, data):
    required = ('models', 'frozen_aggregate_prior', 'frozen_recipe', 'scalers',
                'target_scalers', 'partition')
    if checkpoint.get('schema') != 'current_training_v1' or any(k not in checkpoint for k in required):
        raise ValueError('incomplete_current_frozen_checkpoint')
    if 'FROZEN' not in checkpoint['models']:
        raise ValueError('missing_fixed_FROZEN_baseline')
    if not _same(checkpoint['partition'], data['provenance']):
        raise ValueError('checkpoint_partition_mismatch')
    if not _same(checkpoint['scalers'], data['scalers']):
        raise ValueError('checkpoint_scalers_mismatch')
    recipe = checkpoint['frozen_recipe']
    target = checkpoint['target_scalers']
    if not isinstance(recipe, dict) or not isinstance(target, dict):
        raise ValueError('invalid_frozen_recipe_or_target_scalers')
    declared = recipe.get('target_scalers')
    if declared is None and all(k in recipe for k in ('log1p_mean', 'log1p_std')):
        declared = {k: recipe[k] for k in ('log1p_mean', 'log1p_std')}
    if declared is not None and not _same(declared, target):
        raise ValueError('frozen_recipe_target_scalers_mismatch')
    mean, std = float(target['log1p_mean']), float(target['log1p_std'])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError('invalid_saved_target_scalers')
    return mean, std


def _run_fold(root, work, seed, fold, device, modules, protocol_sha):
    out = work / f'seed_{seed}_fold_{fold + 1:02d}'
    out.mkdir(parents=True, exist_ok=True)
    complete = out / 'complete.json'
    if complete.exists():
        receipt = rt.read_json(complete)
        for name in ('predictions.csv', 'fits.json'):
            if receipt.get(name.replace('.csv', '').replace('.json', '') + '_sha256') != rt.sha(out / name):
                raise ValueError(f'completed_output_hash_mismatch:{out.name}/{name}')
        if receipt.get('protocol_sha256') != protocol_sha:
            raise ValueError(f'completed_protocol_mismatch:{out.name}')
        return

    started = time.time()
    data = build_fold(root, fold, device)
    source = root / 'models' / 'checkpoints' / f'seed_{seed}_fold_{fold + 1:02d}' / 'models.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    mean, std = _check_checkpoint(checkpoint, data)
    target_scalers = checkpoint['target_scalers']
    standardized = (data['ay'].log1p() - mean) / std
    prior_state = checkpoint['frozen_aggregate_prior']
    prior = _restore(modules, prior_state, device)
    baseline_state = checkpoint['models']['FROZEN']
    baseline = _restore(modules, baseline_state, device).requires_grad_(False)
    prior_hash, baseline_hash = _state_hash(prior_state), _state_hash(baseline_state)

    sg_model, sg_fit = _fit_frozen_sg(prior, data, standardized)
    sg_fit['initial_state_hash'] = prior_hash
    parts, transfer_fit, anchor_cp, diagnostics = train.transfer(baseline, baseline, data, device)
    transfer_fit.update(selected_arm='STRUCTURE_TRANSFER', baseline_key='FROZEN',
                        baseline_state_hash=baseline_hash, teacher_key='FROZEN')
    groups = _transfer_diagnostics(baseline, baseline, data, device, anchor_cp, transfer_fit)
    predictions = []
    with torch.no_grad():
        for model_name, model in (('FROZEN', baseline), ('FROZEN_SG', sg_model)):
            for view_index, view_name in enumerate(rt.VIEWS):
                current = model.components(data['views'][view_index, data['oi']])
                predictions += train.prediction_rows(current, data, seed, fold, model_name, view_name)
    predictions += train.prediction_rows(tuple(p[data['oi']] for p in parts), data, seed,
                                         fold, 'STRUCTURE_TRANSFER', 'formula_only')
    expected = 9 * len(data['oi'])
    if len(predictions) != expected:
        raise ValueError(f'prediction_count_mismatch:{len(predictions)}:{expected}')
    rt.write_csv(out / 'predictions.csv', predictions)
    rt.write_csv(out / 'teacher_signal.csv', diagnostics)
    if not (out / 'teacher_signal.csv').exists():
        (out / 'teacher_signal.csv').write_text(
            'composition_key,formula,mass,teacher_delta_energy,teacher_log_w1,teacher_mean_increment_ev\n',
            encoding='utf-8')
    rt.write_json(out / 'fits.json', dict(FROZEN_SG=sg_fit, STRUCTURE_TRANSFER=transfer_fit))
    models_path = out / 'models.pt'
    torch.save(dict(schema='structure_diagnosis_transfer_v1', seed=seed, fold=fold,
                    models={'FROZEN': baseline_state, 'FROZEN_SG': train.state(sg_model)},
                    baseline_key='FROZEN', transfer_anchor=anchor_cp,
                    scalers=checkpoint['scalers'], target_scalers=target_scalers,
                    partition=checkpoint['partition'], frozen_recipe=checkpoint['frozen_recipe']), models_path)
    rt.write_json(complete, dict(status='complete', seed=seed, fold=fold, aggregate_fits=0,
                  report_fits=2, total_fits=2, predictions=len(predictions), teacher_signal_rows=len(groups),
                  protocol_sha256=protocol_sha, predictions_sha256=rt.sha(out / 'predictions.csv'),
                  fits_sha256=rt.sha(out / 'fits.json'), teacher_signal_sha256=rt.sha(out / 'teacher_signal.csv'),
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
    device = rt.configure()
    modules = rt.modules()

    inputs = [root / 'data' / 'reports' / name for name in
              ('features.pt', 'records.jsonl', 'metadata.json')]
    inputs += [root / 'data' / 'composition' / name for name in ('features.pt', 'metadata.json')]
    checkpoint_paths = [root / 'models' / 'checkpoints' / f'seed_{s}_fold_{f:02d}' / 'models.pt'
                        for s in SEEDS for f in range(1, FOLDS + 1)]
    code = [Path(__file__).resolve(), Path(train.__file__).resolve(),
            Path(build_fold.__code__.co_filename).resolve(), Path(rt.__file__).resolve(),
            Path(anchor.__file__).resolve(), root / 'runtime' / 'inference.py',
            root / 'runtime' / 'frozen_fusion_compare.py']
    code += sorted((root / 'runtime' / 'modules').glob('*.py'))
    code += [root / 'runtime' / 'modules' / 'symmetry_table.json']
    protocol = RECIPE | dict(device=str(device), gpu=torch.cuda.get_device_name(0),
        input_binding={str(p.relative_to(root)): rt.sha(p) for p in inputs},
        checkpoint_binding={str(p.relative_to(root)): rt.sha(p) for p in checkpoint_paths},
        code_binding={str(p.relative_to(root)): rt.sha(p) for p in code})
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
