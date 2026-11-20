"""Read-only TRAIN diagnostics for fixed FROZEN teachers and zero-head gradients."""
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
import supervised_distill as distill
from training_data import build_fold

torch = rt.torch
SEEDS = (17, 29, 41)
FOLDS = 5
ARMS = ("SG_DISTILL", "FULL_DISTILL")
RECIPE = dict(
    seeds=list(SEEDS), folds=FOLDS, diagnostic_pairs=15, output_rows=30,
    fits=0, optimization=0, parameter_grid=[],
    scope="TRAIN only; no validation or outer-target metric calculation",
    teacher="fixed FROZEN checkpoint; shared distillation-stage formula groups/grid/mass",
    gradient_point="zero-start Linear(64,5,bias=True); shared unique-TRAIN-formula standardization",
    phase_label_count="exact raw phase_label strings; aliases remain separate; not a true phase count",
    gradient_interpretation="local zero-head geometry only; not optimization or generalization evidence")


def _weighted_nll(parts, targets, weights):
    return float((anchor.nll(parts, targets) * weights).sum())


def _marginal_parts(rows, groups, device):
    result = {arm: {} for arm in ARMS}
    for group in groups:
        subset = rows[rows.composition_key == group['composition_key']]
        formula_mass = float(subset[subset.arm == 'SG_KD'].train_weight.sum())
        for arm, label in (("SG_DISTILL", "SG_KD"), ("FULL_DISTILL", "FULL_KD")):
            ws, mus, sigmas = [], [], []
            for row in subset[subset.arm == label].itertuples(index=False):
                fraction = float(row.train_weight) / formula_mass
                ws.extend(fraction * value for value in rt.json.loads(row.component_weights))
                mus.extend(rt.json.loads(row.log_mus))
                sigmas.extend(rt.json.loads(row.sigmas))
            result[arm][group['composition_key']] = tuple(torch.tensor(values, dtype=torch.float64, device=device)
                                                            for values in (ws, mus, sigmas))
    return result


def _marginal_nll(parts_by_formula, data):
    losses = []
    for index in data['ti'].tolist():
        key = data['records'][index]['composition_key']
        parts = tuple(value[None, :] for value in parts_by_formula[key])
        losses.append(anchor.nll(parts, data['targets'][index:index + 1])[0])
    return float((torch.stack(losses) * data['tw']).sum())


@torch.enable_grad()
def _gradient_diagnostics(arm, teacher, energy, base, features, data, ti_formula, grid, mass):
    head = torch.nn.Linear(64, 5, bias=True, dtype=torch.float64, device=features.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    ti = data['ti']
    train_parts = distill._correct({key: value[ti] for key, value in base.items()}, head(features[ti]))
    train_nll = (anchor.nll(train_parts, data['targets'][ti]) * data['tw']).sum()
    formula_parts = distill._correct({key: value[ti_formula] for key, value in base.items()},
                                     head(features[ti_formula]))
    predicted = anchor.cdf(formula_parts, grid)
    numerator = (mass * torch.trapezoid((predicted - teacher).square(), grid, dim=-1)).sum()
    normalized_kd = numerator / energy if float(energy) > 0 else numerator * 0
    scaled_kd = .1 * normalized_kd
    parameters = (head.weight, head.bias)
    nll_grads = torch.autograd.grad(train_nll, parameters, retain_graph=True)
    kd_grads = torch.autograd.grad(scaled_kd, parameters)
    grad_nll = torch.cat([value.reshape(-1) for value in nll_grads])
    grad_kd = torch.cat([value.reshape(-1) for value in kd_grads])
    nll_norm, kd_norm = torch.linalg.vector_norm(grad_nll), torch.linalg.vector_norm(grad_kd)
    dot = torch.dot(grad_nll, grad_kd)
    cosine = float(dot / (nll_norm * kd_norm)) if float(nll_norm * kd_norm) > 0 else None
    return dict(zero_head_train_nll=float(train_nll), normalized_kd=float(normalized_kd),
                scaled_kd=float(scaled_kd), train_nll_gradient_norm=float(nll_norm),
                scaled_kd_gradient_norm=float(kd_norm), gradient_dot=float(dot),
                gradient_cosine=cosine, common_teacher_energy=float(energy),
                zero_common_energy=bool(float(energy) == 0))


def _run_pair(root, seed, fold, device, modules):
    data = build_fold(root, fold, device)
    source = root / 'models' / 'checkpoints' / f'seed_{seed}_fold_{fold + 1:02d}' / 'models.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    stage._check_checkpoint(checkpoint, data)
    baseline = stage._restore(modules, checkpoint['models']['FROZEN'], device).requires_grad_(False).eval()
    with torch.no_grad():
        train_indices = data['ti']
        train_base = train.base_parts(baseline, data['views'][0, train_indices])
        base = {}
        for key, value in train_base.items():
            sparse = torch.zeros((len(data['records']), *value.shape[1:]),
                                 dtype=value.dtype, device=device)
            sparse[train_indices] = value
            base[key] = sparse
        ids = [data['records'][i]['measurement_id'] for i in data['ti'].tolist()]
        teacher_rows = stage._teacher_rows(baseline, data, device)
        groups, ti_formula, grid, _delta, mass, _zero = anchor.teacher_targets(
            teacher_rows, base, ids, data['records'], device)
        _base_cdf, teachers, energy = distill._teacher_cdfs(
            teacher_rows, groups, ti_formula, base, grid, mass, device)
        marginal_parts = _marginal_parts(teacher_rows, groups, device)
        formula_nll = _weighted_nll(
            tuple(base[key][data['ti']] for key in ('weights', 'mus', 'sigmas')),
            data['targets'][data['ti']], data['tw'])
        sg_conditional = baseline.components(data['views'][1, data['ti']])
        full_conditional = baseline.components(data['views'][3, data['ti']])
        sg_conditional_nll = _weighted_nll(sg_conditional, data['targets'][data['ti']], data['tw'])
        full_conditional_nll = _weighted_nll(full_conditional, data['targets'][data['ti']], data['tw'])
        sg_marginal_nll = _marginal_nll(marginal_parts['SG_DISTILL'], data)
        full_marginal_nll = _marginal_nll(marginal_parts['FULL_DISTILL'], data)

        hidden = base['hidden'][ti_formula]
        if hidden.shape[1] != 64:
            raise ValueError(f'expected_64_hidden_features:{hidden.shape[1]}')
        center = hidden.mean(0)
        scale = hidden.std(0, unbiased=False)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale.clamp_min(1e-6))
        features = (base['hidden'] - center) / scale
        train_indices = data['ti'].tolist()
        formula_keys = [group['composition_key'] for group in groups]
        records_by_key = {key: [data['records'][i] for i in train_indices
                                if data['records'][i]['composition_key'] == key]
                          for key in formula_keys}
        labels = {}
        for key, rows in records_by_key.items():
            labels[key] = {row.get('phase_label') for row in rows
                           if isinstance(row.get('phase_label'), str) and row.get('phase_label') != ''}
        sg_supported = {key: sum(int(data['views'][3, i, -2].item() > .5) for i in train_indices
                                 if data['records'][i]['composition_key'] == key) for key in formula_keys}
        structure_supported = {key: sum(int(data['views'][3, i, -1].item() > .5) for i in train_indices
                                        if data['records'][i]['composition_key'] == key) for key in formula_keys}
        counts = dict(train_report_count=len(train_indices), train_formula_count=len(groups),
            train_sg_supported_report_count=sum(sg_supported.values()),
            train_structure_supported_report_count=sum(structure_supported.values()),
            train_formula_with_sg_count=sum(value > 0 for value in sg_supported.values()),
            train_formula_with_structure_count=sum(value > 0 for value in structure_supported.values()),
            train_formula_with_ge2_distinct_phase_label_strings=sum(len(value) >= 2 for value in labels.values()),
            train_formula_with_ge2_structure_supported_rows=sum(value >= 2 for value in structure_supported.values()))
        common = dict(seed=seed, fold_id=fold + 1, **counts,
            frozen_formula_train_nll=formula_nll,
            frozen_sg_conditional_train_nll=sg_conditional_nll,
            frozen_full_conditional_train_nll=full_conditional_nll,
            sg_formula_marginal_train_nll=sg_marginal_nll,
            full_formula_marginal_train_nll=full_marginal_nll,
            phase_label_count_semantics='exact raw strings; aliases not merged; not true phase count')
        rows = []
        for arm in ARMS:
            rows.append(common | dict(arm=arm) | _gradient_diagnostics(
                arm, teachers[arm], energy, base, features, data, ti_formula, grid, mass))
    return rows


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
    code = [Path(__file__).resolve(), Path(distill.__file__).resolve(), Path(stage.__file__).resolve(),
            Path(anchor.__file__).resolve(), Path(train.__file__).resolve(),
            Path(build_fold.__code__.co_filename).resolve(), Path(rt.__file__).resolve(),
            root / 'runtime' / 'inference.py', root / 'runtime' / 'frozen_fusion_compare.py']
    code += sorted((root / 'runtime' / 'modules').glob('*.py'))
    code += [root / 'runtime' / 'modules' / 'symmetry_table.json']
    protocol = RECIPE | dict(device=str(device), gpu=torch.cuda.get_device_name(0),
        input_binding={str(path.relative_to(root)): rt.sha(path) for path in inputs},
        checkpoint_binding={str(path.relative_to(root)): rt.sha(path) for path in checkpoints},
        code_binding={str(path.relative_to(root)): rt.sha(path) for path in code})
    protocol_path = work / 'protocol.json'
    if protocol_path.exists() and rt.read_json(protocol_path) != protocol:
        raise ValueError('input_or_recipe_changed_existing_diagnostic_cannot_be_resumed')
    rt.write_json(protocol_path, protocol)
    rows = []
    started = time.time()
    for seed in SEEDS:
        for fold in range(FOLDS):
            rows.extend(_run_pair(root, seed, fold, device, modules))
    if len(rows) != 30:
        raise ValueError(f'diagnostic_row_count_mismatch:{len(rows)}')
    csv_path = work / 'diagnostics.csv'
    rt.write_csv(csv_path, rows)
    rt.write_json(work / 'complete.json', dict(status='complete', diagnostic_pairs=15,
        rows=len(rows), fits=0, optimization=0, protocol_sha256=rt.sha(protocol_path),
        diagnostics_sha256=rt.sha(csv_path), seconds=time.time() - started))


if __name__ == '__main__':
    main()
