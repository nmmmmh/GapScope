"""Train the fixed conditional-adapter teacher and gated marginal student."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import runtime as rt
import train
from conditional_adapter import (
    ConditionalAdapter,
    FrozenHeadStudent,
    adapter_state,
    original_state,
    restore_adapter,
    state_matches,
)
from inference import _restore_model
from train_decoupled_distribution import context as report_context
from training_data import hierarchical_weights


torch = rt.torch
SEEDS = (17, 29, 41)
FOLDS = range(1, 4)
FIT_FILES = ("models.pt", "predictions.csv", "fits.json", "partition.json")
RECIPE = dict(
    seeds=list(SEEDS), folds=3, teacher_fits=9, student_fits=9,
    aggregate_fits=0, maximum_fits=18,
    evaluated_views={
        "TEACHER_BASE": list(rt.VIEWS), "ADAPTER_TEACHER": list(rt.VIEWS),
        "STUDENT_BASE": ["formula_only"], "MARGINAL_STUDENT": ["formula_only"],
    },
    adapter_rank=4, adapter_trainable_parameters=2704,
    output_equivalence=dict(atol=1e-6, rtol=1e-6, frozen_state='exact'),
    adapter_paths={
        "sg_projection": "290-to-32, no-bias rank-four residual",
        "structure_projection": "160-to-32, no-bias rank-four residual",
        "fusion": "98-to-64, no-bias rank-four residual gated by either final mask",
    },
    adapter_initialization="random down projections; zero up projections; deterministic seed/fold fork",
    teacher_objective="supported TRAIN reports; per-report mean NLL across available SG, structure, and FULL views; supported-report hierarchical weights",
    teacher_selection="supported VALID report NLL across available SG, structure, and FULL views; epoch zero eligible",
    student_initialization="selected parent FROZEN_FORMULA state; only its 326-parameter report head trainable",
    student_objective="all TRAIN reports retain hierarchical-weighted real NLL plus one fixed gradient-matched teacher-CDF term",
    student_selection="formula-only VALID report hierarchical NLL; epoch zero eligible",
    student_kd="supported FULL teacher reports aggregated within formula by original TRAIN weights normalized over support; formula mass is original supported TRAIN mass normalized across supported formulas; no unsupported-report fill-in",
    student_kd_grid="257 points uniformly spaced in log Eg from TRAIN log-target minimum minus 2 to maximum plus 2; integrate squared CDF difference by trapezoid over Eg in eV",
    student_kd_strength="one coefficient from initial TRAIN report-head gradient-norm ratio ||g_NLL||/||g_KD||; zero if KD gradient norm is exactly zero; no validation or outer tuning",
    optimizer="Adam learning rate 0.001; maximum 300 epochs; patience 30",
    teacher_gate="same structure-supported grouped OOF FULL density-ensemble CRPS improves, at least two of three seed CRPS improve, and ensemble NLL is nonworse; development gate only",
    selected_views_and_targets="205 report records from the fixed parent inputs; no aggregate fitting, new CIF targets, or confirmation targets",
    default_model_changed=False,
)


def _state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parts_equal(left, right):
    return all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(left, right))


def _component_max_difference(left, right):
    differences = [float((a - b).abs().max().item()) for a, b in zip(left, right)]
    return max(differences, default=0.0)


def _source_files(root):
    runtime_dir = root / "runtime"
    paths = [
        runtime_dir / name for name in (
            "conditional_adapter.py", "train_conditional_adapter.py",
            "analyze_conditional_adapter.py", "train_decoupled_distribution.py",
            "decoupled_distribution.py", "grouped_adaptation_compare.py",
            "train.py", "runtime.py",
            "training_data.py", "inference.py", "density_score.py",
        )
    ]
    paths += sorted((runtime_dir / "modules").glob("*.py"))
    paths.append(runtime_dir / "modules" / "symmetry_table.json")
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError("missing_protocol_source:" + ",".join(missing))
    return paths


def _protocol(root, parent, work, device):
    parent_checkpoints = {}
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"seed_{seed}_fold_{fold:02d}"
            path = root / "models" / "report_adaptation" / name / "models.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing_parent_checkpoint:{path}")
            parent_checkpoints[path.relative_to(root).as_posix()] = rt.sha(path)
    input_files = ("assembly.json", "features.pt", "metadata.json", "records.jsonl")
    input_binding = {}
    for name in input_files:
        path = parent / "inputs" / name
        if not path.is_file():
            raise FileNotFoundError(f"missing_parent_input:{path}")
        input_binding[name] = rt.sha(path)
    sources = _source_files(root)
    result = RECIPE | dict(
        source_binding={path.relative_to(root).as_posix(): rt.sha(path) for path in sources},
        parent_checkpoint_binding=parent_checkpoints,
        input_binding=input_binding,
        device=str(device), gpu=torch.cuda.get_device_name(0),
    )
    path = work / "training_protocol.json"
    if path.exists():
        if rt.read_json(path) != result:
            raise ValueError("conditional_adapter_protocol_or_inputs_changed")
    else:
        rt.write_json(path, result)
    return result, rt.sha(path)


def _load_parent(root, parent, name, fold, modules, device):
    checkpoint_path = root / "models" / "report_adaptation" / name / "models.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "grouped_adaptation_v1":
        raise ValueError(f"unexpected_parent_checkpoint_schema:{checkpoint_path}")
    partition = checkpoint.get("partition", {})
    if partition.get("fold") != fold - 1:
        raise ValueError(f"parent_checkpoint_fold_mismatch:{checkpoint_path}")
    data = report_context(root, parent, checkpoint, modules, device)
    if sum(len(data[key]) for key in ("ti", "vi", "oi")) != 205:
        raise ValueError(f"parent_report_partition_mismatch:{name}")
    return checkpoint, data, checkpoint_path


def _supported_indices(data, split):
    indices = data[f"{split}i"]
    supported = data["views"][3, indices, 633:635].bool().any(-1)
    return indices[supported]


def _conditional_nll(model, data, indices, weights):
    flags = data["views"][3, indices, 633:635].bool()
    view_mask = torch.stack((flags[:, 0], flags[:, 1], flags.all(-1))).double()
    if torch.any(view_mask.sum(0) == 0):
        raise ValueError("unsupported_report_in_conditional_objective")
    losses = torch.stack([
        model.per_record_nll(data["views"][view, indices], data["targets"][indices])
        for view in (1, 2, 3)
    ])
    per_report = (losses * view_mask).sum(0) / view_mask.sum(0)
    return (per_report * weights).sum()


def _weighted_formula_nll(model, data, indices, weights):
    losses = model.per_record_nll(data["views"][0, indices], data["targets"][indices])
    return (losses * weights).sum()


def _component_audit(left, right, views, indices):
    differences = {}
    exact = True
    with torch.no_grad():
        for view_index, view in enumerate(rt.VIEWS):
            a = left.components(views[view_index, indices])
            b = right.components(views[view_index, indices])
            equal = _parts_equal(a, b)
            exact = exact and equal
            differences[view] = _component_max_difference(a, b)
    return exact, differences


def _fit_teacher(root, parent, work, protocol_sha, name, seed, fold,
                 modules, device, checkpoint, data, checkpoint_path, out):
    initial_state = checkpoint["models"]["FROZEN_MULTIVIEW"]
    baseline = _restore_model(modules, initial_state, "M", device)
    baseline_state = train.state(baseline)
    base = copy.deepcopy(baseline)
    seed_value = (seed * 100 + fold) % (2**31 - 1)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed_value)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed_value)
        model = ConditionalAdapter(base).to(device)
    initial_exact, initial_differences = _component_audit(
        baseline, model, data["views"], torch.arange(205, device=device))
    if not initial_exact:
        raise ValueError(f"zero_initialized_adapter_changed_parent:{name}")
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 2704:
        raise ValueError("conditional_adapter_trainable_parameter_count_mismatch")

    train_indices = _supported_indices(data, "t")
    valid_indices = _supported_indices(data, "v")
    if not len(train_indices) or not len(valid_indices):
        raise ValueError(f"empty_supported_teacher_split:{name}")
    train_weights = hierarchical_weights(data["records"], train_indices.tolist(), device)
    valid_weights = hierarchical_weights(data["records"], valid_indices.tolist(), device)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001)
    attempt = _begin_attempt(work, out, "teacher", seed, fold,
                             protocol_sha, checkpoint_path)
    started = time.time()
    print(f"START phase=teacher seed={seed} fold={fold} attempt={attempt}", flush=True)
    result = train.optimize(
        model,
        lambda: _conditional_nll(model, data, train_indices, train_weights),
        lambda: _conditional_nll(model, data, valid_indices, valid_weights),
        optimizer, 300,
    )
    with torch.no_grad():
        base_formula = baseline.components(data["views"][0])
        adapted_formula = model.components(data["views"][0])
        formula_equal = _parts_equal(base_formula, adapted_formula)
        formula_differences = {"formula_only": _component_max_difference(
            base_formula, adapted_formula)}
    frozen_equal = state_matches(original_state(model), baseline_state)
    if not formula_equal:
        raise ValueError(f"adapter_changed_formula_components:{name}")
    if not frozen_equal:
        raise ValueError(f"adapter_changed_frozen_parent_state:{name}")
    if model.base.training:
        raise ValueError(f"frozen_teacher_base_left_eval_mode:{name}")

    result.update(
        trainable_parameters=2704,
        supported_train_reports=len(train_indices),
        supported_valid_reports=len(valid_indices),
        train_support_hierarchical_weight_sum=float(train_weights.sum().item()),
        valid_support_hierarchical_weight_sum=float(valid_weights.sum().item()),
        available_train_view_reports={
            "SG": int(data["views"][3, train_indices, 633].bool().sum().item()),
            "structure": int(data["views"][3, train_indices, 634].bool().sum().item()),
            "FULL": len(train_indices),
        },
        initial_all_four_view_components_numerically_equal=initial_exact,
        output_equivalence_tolerance=dict(atol=1e-6, rtol=1e-6),
        initial_all_four_view_max_abs_difference=initial_differences,
        formula_components_equal_parent_across_205=formula_equal,
        formula_components_max_abs_difference=formula_differences,
        original_parameters_and_buffers_unchanged=frozen_equal,
        frozen_base_eval_after_fit=not model.base.training,
        selected_adapter_state_sha256=_state_hash(adapter_state(model)),
        attempt=attempt,
        seconds=time.time() - started,
    )

    predictions = []
    with torch.no_grad():
        for arm, selected_model in (("TEACHER_BASE", baseline), ("ADAPTER_TEACHER", model)):
            for view_index, view in enumerate(rt.VIEWS):
                parts = selected_model.components(data["views"][view_index, data["oi"]])
                predictions += train.prediction_rows(
                    parts, data, seed, fold, arm, view)
    if len(predictions) != len(data["oi"]) * len(rt.VIEWS) * 2:
        raise ValueError(f"teacher_outer_prediction_count_mismatch:{name}")

    rt.write_csv(out / "predictions.csv", predictions)
    rt.write_json(out / "partition.json", checkpoint["partition"])
    rt.write_json(out / "fits.json", result)
    torch.save(dict(
        schema="conditional_adapter_teacher_v1", seed=seed, fold=fold - 1,
        base_state=baseline_state, adapter_state=adapter_state(model),
        scalers=checkpoint["scalers"], partition=checkpoint["partition"],
    ), out / "models.pt")
    _complete(out, protocol_sha, seed, fold, "teacher", checkpoint_path,
              attempt=attempt, seconds=time.time() - started)


def _cdf(parts, grid_y):
    weights, mus, sigmas = parts
    log_y = grid_y.log()
    z = (log_y[None, :, None] - mus[:, None, :]) / sigmas[:, None, :]
    return (weights[:, None, :] * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).sum(-1)


def _formula_feature_audit(data):
    first = {}
    maximum = 0.0
    for index, row in enumerate(data["records"]):
        key = row["composition_key"]
        features = data["views"][0, index, :183]
        if key not in first:
            first[key] = (index, features)
            continue
        reference_index, reference = first[key]
        difference = float((features - reference).abs().max().item())
        maximum = max(maximum, difference)
        if not torch.allclose(features, reference, atol=1e-5, rtol=1e-5):
            raise ValueError(f"formula_feature_mismatch_before_kd_grouping:{key}:{reference_index}:{index}")
    return dict(formulas=len(first), maximum_abs_difference=maximum, tolerance=1e-5)


def _prepare_kd(teacher, data, device):
    audit = _formula_feature_audit(data)
    ti = data["ti"]
    log_targets = data["targets"][ti].log()
    grid_log = torch.linspace(float(log_targets.min() - 2.0), float(log_targets.max() + 2.0),
                              257, dtype=torch.float64, device=device)
    grid_y = grid_log.exp()
    train_ids = ti.detach().cpu().tolist()
    original_weights = data["tw"].detach().double()
    weight_by_index = {index: original_weights[position]
                       for position, index in enumerate(train_ids)}
    ids_by_formula = {}
    for index in train_ids:
        ids_by_formula.setdefault(data["records"][index]["composition_key"], []).append(index)

    formula_keys, representatives, targets, masses, report_details, rows = [], [], [], [], [], []
    supported_mass_total = 0.0
    with torch.no_grad():
        for key, all_ids in sorted(ids_by_formula.items()):
            supported_ids = [index for index in all_ids if bool(
                data["views"][3, index, 633:635].bool().any().item())]
            total_mass = torch.stack([weight_by_index[index] for index in all_ids]).sum()
            if not supported_ids:
                rows.append(dict(composition_key=key, train_reports=len(all_ids),
                    supported_reports=0, original_train_mass=float(total_mass.item()),
                    original_supported_mass=0.0, support_coverage_fraction=0.0,
                    kd_formula_mass=0.0, supported_report_weights="[]",
                    target_cdf_sha256="", teacher_cdf_lower=None, teacher_cdf_upper=None,
                    teacher_left_tail_mass=None, teacher_right_tail_mass=None))
                continue
            report_mass = torch.stack([weight_by_index[index] for index in supported_ids])
            supported_mass = report_mass.sum()
            normalized_report_weights = report_mass / supported_mass
            parts = teacher.components(data["views"][3, supported_ids])
            per_report_cdf = _cdf(parts, grid_y)
            target_cdf = (normalized_report_weights[:, None] * per_report_cdf).sum(0)
            formula_keys.append(key)
            representatives.append(all_ids[0])
            targets.append(target_cdf)
            masses.append(supported_mass)
            supported_mass_total += float(supported_mass.item())
            detail = [dict(measurement_id=data["records"][index]["measurement_id"],
                           original_train_weight=float(weight_by_index[index].item()),
                           normalized_supported_report_weight=float(weight.item()))
                      for index, weight in zip(supported_ids, normalized_report_weights)]
            report_details.append(detail)
            target_hash = _tensor_hash(target_cdf)
            rows.append(dict(composition_key=key, train_reports=len(all_ids),
                supported_reports=len(supported_ids), original_train_mass=float(total_mass.item()),
                original_supported_mass=float(supported_mass.item()),
                support_coverage_fraction=float((supported_mass / total_mass).item()),
                kd_formula_mass=0.0, supported_report_weights=json.dumps(detail, ensure_ascii=False),
                target_cdf_sha256=target_hash,
                teacher_cdf_lower=float(target_cdf[0].item()),
                teacher_cdf_upper=float(target_cdf[-1].item()),
                teacher_left_tail_mass=float(target_cdf[0].item()),
                teacher_right_tail_mass=float((1.0 - target_cdf[-1]).item())))
    if not formula_keys or supported_mass_total <= 0:
        raise ValueError("empty_supported_formula_kd_target")
    target_tensor = torch.stack(targets)
    mass_tensor = torch.stack(masses).double()
    mass_tensor /= mass_tensor.sum()
    mass_by_key = {key: float(mass.item()) for key, mass in zip(formula_keys, mass_tensor)}
    for row in rows:
        row["kd_formula_mass"] = mass_by_key.get(row["composition_key"], 0.0)
    return dict(
        grid_y=grid_y, formula_keys=formula_keys,
        representative_indices=torch.tensor(representatives, dtype=torch.long, device=device),
        target_cdf=target_tensor, formula_mass=mass_tensor,
        report_details=report_details, diagnostics=rows, formula_feature_audit=audit,
        grid_y_sha256=_tensor_hash(grid_y), target_cdf_sha256=_tensor_hash(target_tensor),
        target_report_count=sum(row["supported_reports"] for row in rows),
        target_formula_count=len(formula_keys),
        supported_train_mass=supported_mass_total,
    )


def _kd_loss(model, data, kd):
    parts = model.components(data["views"][0, kd["representative_indices"]])
    student_cdf = _cdf(parts, kd["grid_y"])
    per_formula = torch.trapezoid((student_cdf - kd["target_cdf"]).square(),
                                  x=kd["grid_y"], dim=-1)
    return (kd["formula_mass"] * per_formula).sum()


def _gradient_norm(gradients):
    terms = [gradient.detach().double().square().sum()
             for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(terms).sum()).item()) if terms else 0.0


def _initial_kd_coefficient(model, data, kd):
    parameters = [parameter for parameter in model.base.report_head.parameters()
                  if parameter.requires_grad]
    nll = _weighted_formula_nll(model, data, data["ti"], data["tw"])
    kd_value = _kd_loss(model, data, kd)
    nll_gradients = torch.autograd.grad(nll, parameters, allow_unused=True)
    kd_gradients = torch.autograd.grad(kd_value, parameters, allow_unused=True)
    nll_norm = _gradient_norm(nll_gradients)
    kd_norm = _gradient_norm(kd_gradients)
    products = [a.detach().double().mul(b.detach().double()).sum()
                for a, b in zip(nll_gradients, kd_gradients)
                if a is not None and b is not None]
    dot = float(torch.stack(products).sum().item()) if products else 0.0
    cosine = dot / (nll_norm * kd_norm) if nll_norm > 0 and kd_norm > 0 else None
    coefficient = 0.0 if kd_norm == 0.0 else nll_norm / kd_norm
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("nonfinite_initial_gradient_matched_kd_coefficient")
    return dict(
        coefficient=coefficient, kd_active=kd_norm != 0.0,
        initial_train_nll_gradient_norm=nll_norm,
        initial_train_kd_gradient_norm=kd_norm,
        initial_train_gradient_cosine=cosine,
        initial_train_gradient_dot=dot,
        rule="||g_NLL||/||g_KD|| on initial TRAIN report-head gradients; zero coefficient for exactly zero KD gradient",
    )


def _formula_route_audit(model, views):
    with torch.no_grad():
        reference = model.components(views[0])
        differences = {
            view: _component_max_difference(reference, model.components(views[index]))
            for index, view in enumerate(rt.VIEWS)
        }
        exact = all(_parts_equal(reference, model.components(views[index])) for index in range(4))
    return exact, differences


def _fit_student(root, parent, work, protocol_sha, name, seed, fold,
                 modules, device, checkpoint, data, checkpoint_path,
                 teacher_checkpoint_path, out):
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
    if teacher_checkpoint.get("schema") != "conditional_adapter_teacher_v1":
        raise ValueError(f"unexpected_teacher_checkpoint_schema:{name}")
    teacher_state = checkpoint["models"]["FROZEN_MULTIVIEW"]
    if not state_matches(teacher_checkpoint["base_state"], teacher_state):
        raise ValueError(f"teacher_parent_state_mismatch:{name}")
    teacher = restore_adapter(modules, teacher_state, teacher_checkpoint["adapter_state"], device)
    kd = _prepare_kd(teacher, data, device)

    student_base = _restore_model(modules, checkpoint["models"]["FROZEN_FORMULA"], "F", device)
    initial_base_state = train.state(student_base)
    model = FrozenHeadStudent(copy.deepcopy(student_base)).to(device)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 326:
        raise ValueError("marginal_student_trainable_parameter_count_mismatch")
    initial_exact, initial_differences = _component_audit(
        student_base, model, data["views"], torch.arange(205, device=device))
    if not initial_exact:
        raise ValueError(f"student_wrapper_changed_parent_formula_state:{name}")
    coefficient = _initial_kd_coefficient(model, data, kd)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001)
    attempt = _begin_attempt(work, out, "student", seed, fold, protocol_sha,
                             checkpoint_path, teacher_checkpoint_path)
    started = time.time()
    print(f"START phase=student seed={seed} fold={fold} attempt={attempt}", flush=True)

    def objective():
        nll = _weighted_formula_nll(model, data, data["ti"], data["tw"])
        if coefficient["kd_active"]:
            return nll + coefficient["coefficient"] * _kd_loss(model, data, kd)
        return nll

    result = train.optimize(
        model, objective,
        lambda: _weighted_formula_nll(model, data, data["vi"], data["vw"]),
        optimizer, 300,
    )
    trainable_head_keys = {
        f"report_head.{key}" for key, parameter in model.base.report_head.named_parameters()
        if parameter.requires_grad
    }
    frozen_parent = {key: value for key, value in initial_base_state.items()
                     if key not in trainable_head_keys}
    selected_base = train.state(model.base)
    selected_nonhead = {key: value for key, value in selected_base.items()
                        if key not in trainable_head_keys}
    frozen_equal = state_matches(selected_nonhead, frozen_parent)
    formula_route_equal, formula_route_differences = _formula_route_audit(model, data["views"])
    if not frozen_equal:
        raise ValueError(f"student_changed_frozen_parent_state:{name}")
    if not formula_route_equal:
        raise ValueError(f"student_formula_route_depends_on_optional_view:{name}")
    if model.base.training:
        raise ValueError(f"frozen_student_base_left_eval_mode:{name}")

    with torch.no_grad():
        student_parts = model.components(data["views"][0, kd["representative_indices"]])
        student_cdf = _cdf(student_parts, kd["grid_y"])
        diagnostics = []
        supported_rows = {key: index for index, key in enumerate(kd["formula_keys"])}
        for row in kd["diagnostics"]:
            if row["supported_reports"]:
                index = supported_rows[row["composition_key"]]
                row = dict(row)
                row.update(
                    student_cdf_sha256=_tensor_hash(student_cdf[index]),
                    student_cdf_lower=float(student_cdf[index, 0].item()),
                    student_cdf_upper=float(student_cdf[index, -1].item()),
                    student_left_tail_mass=float(student_cdf[index, 0].item()),
                    student_right_tail_mass=float((1.0 - student_cdf[index, -1]).item()),
                )
            else:
                row = dict(row, student_cdf_sha256="", student_cdf_lower=None,
                           student_cdf_upper=None, student_left_tail_mass=None,
                           student_right_tail_mass=None)
            diagnostics.append(row)
    with torch.no_grad():
        kd_distance = float(_kd_loss(model, data, kd).item())
    result.update(
        trainable_parameters=326,
        full_original_state_unchanged=False,
        original_nontrainable_parameters_and_buffers_unchanged=frozen_equal,
        report_head_is_the_only_trainable_state=True,
        frozen_base_eval_after_fit=not model.base.training,
        initial_wrapper_all_four_view_components_numerically_equal=initial_exact,
        output_equivalence_tolerance=dict(atol=1e-6, rtol=1e-6),
        initial_wrapper_max_abs_difference=initial_differences,
        formula_route_numerically_equal_across_205_reports_and_four_views=formula_route_equal,
        formula_route_max_abs_difference=formula_route_differences,
        kd_coefficient=coefficient["coefficient"],
        kd_active=coefficient["kd_active"],
        initial_train_nll_gradient_norm=coefficient["initial_train_nll_gradient_norm"],
        initial_train_kd_gradient_norm=coefficient["initial_train_kd_gradient_norm"],
        initial_train_gradient_cosine=coefficient["initial_train_gradient_cosine"],
        initial_train_gradient_dot=coefficient["initial_train_gradient_dot"],
        kd_distance_name="training-grid truncated eV-CDF squared distance; not full CRPS",
        selected_train_kd_distance_ev=float(kd_distance),
        kd_formula_count=kd["target_formula_count"],
        kd_supported_report_count=kd["target_report_count"],
        kd_supported_train_mass=float(kd["supported_train_mass"]),
        kd_grid_y_sha256=kd["grid_y_sha256"],
        kd_teacher_cdf_sha256=kd["target_cdf_sha256"],
        kd_formula_feature_audit=kd["formula_feature_audit"],
        kd_diagnostics=diagnostics,
        selected_student_state_sha256=_state_hash(selected_base),
        attempt=attempt,
        seconds=time.time() - started,
    )

    predictions = []
    with torch.no_grad():
        for arm, selected_model in (("STUDENT_BASE", student_base), ("MARGINAL_STUDENT", model)):
            parts = selected_model.components(data["views"][0, data["oi"]])
            predictions += train.prediction_rows(
                parts, data, seed, fold, arm, "formula_only")
    if len(predictions) != len(data["oi"]) * 2:
        raise ValueError(f"student_outer_prediction_count_mismatch:{name}")

    kd_cache = dict(
        schema="conditional_adapter_kd_targets_v1", grid_y=kd["grid_y"].detach().cpu(),
        formula_keys=kd["formula_keys"],
        representative_indices=kd["representative_indices"].detach().cpu(),
        target_cdf=kd["target_cdf"].detach().cpu(),
        formula_mass=kd["formula_mass"].detach().cpu(),
        report_details=kd["report_details"],
        grid_y_sha256=kd["grid_y_sha256"], target_cdf_sha256=kd["target_cdf_sha256"],
    )
    rt.write_csv(out / "predictions.csv", predictions)
    rt.write_csv(out / "kd_diagnostics.csv", diagnostics)
    rt.write_json(out / "partition.json", checkpoint["partition"])
    rt.write_json(out / "fits.json", result)
    torch.save(dict(
        schema="conditional_adapter_student_v1", seed=seed, fold=fold - 1,
        models={"STUDENT_BASE": initial_base_state, "MARGINAL_STUDENT": selected_base},
        scalers=checkpoint["scalers"], partition=checkpoint["partition"],
        kd_targets=kd_cache,
    ), out / "models.pt")
    torch.save(kd_cache, out / "kd_targets.pt")
    _complete(out, protocol_sha, seed, fold, "student", checkpoint_path,
              teacher_checkpoint_path=teacher_checkpoint_path,
              attempt=attempt,
              seconds=time.time() - started)


def _begin_attempt(work, out, phase, seed, fold, protocol_sha, checkpoint_path,
                   teacher_checkpoint_path=None):
    path = out / "attempts.json"
    attempts = rt.read_json(path).get("attempts", []) if path.exists() else []
    global_count = 0
    for phase_dir_name in ("teachers", "students"):
        phase_dir = work / phase_dir_name
        if not phase_dir.exists():
            continue
        for attempt_path in phase_dir.glob("*/attempts.json"):
            global_count += len(rt.read_json(attempt_path).get("attempts", []))
    if global_count >= 18:
        raise ValueError("conditional_adapter_18_fit_attempt_budget_exhausted")
    attempt_id = global_count + 1
    attempts.append(dict(attempt_id=attempt_id, attempt_in_seed_fold=len(attempts) + 1,
        phase=phase, seed=seed, fold=fold,
        protocol_sha256=protocol_sha,
        parent_checkpoint_sha256=rt.sha(checkpoint_path),
        teacher_checkpoint_sha256=(rt.sha(teacher_checkpoint_path)
                                   if teacher_checkpoint_path is not None else None),
        started_unix_seconds=time.time()))
    rt.write_json(path, dict(attempts=attempts))
    return attempt_id


def _complete(out, protocol_sha, seed, fold, phase, checkpoint_path,
              teacher_checkpoint_path=None, attempt=None, seconds=None):
    receipt = dict(phase=phase, seed=seed, fold=fold, seconds=seconds,
        attempt=attempt, protocol_sha256=protocol_sha,
        parent_checkpoint_sha256=rt.sha(checkpoint_path))
    if teacher_checkpoint_path is not None:
        receipt["teacher_checkpoint_sha256"] = rt.sha(teacher_checkpoint_path)
    for filename in FIT_FILES:
        receipt[filename.replace(".", "_") + "_sha256"] = rt.sha(out / filename)
    if phase == "student":
        receipt["kd_diagnostics_csv_sha256"] = rt.sha(out / "kd_diagnostics.csv")
        receipt["kd_targets_pt_sha256"] = rt.sha(out / "kd_targets.pt")
    rt.write_json(out / "complete.json", receipt)


def _validate_complete(out, protocol_sha):
    receipt_path = out / "complete.json"
    if not receipt_path.exists():
        return False
    receipt = rt.read_json(receipt_path)
    if receipt.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"completed_fit_protocol_mismatch:{out}")
    for filename in FIT_FILES:
        key = filename.replace(".", "_") + "_sha256"
        if receipt.get(key) != rt.sha(out / filename):
            raise ValueError(f"completed_fit_artifact_changed:{out / filename}")
    for filename, key in (("kd_diagnostics.csv", "kd_diagnostics_csv_sha256"),
                           ("kd_targets.pt", "kd_targets_pt_sha256")):
        if key in receipt and receipt[key] != rt.sha(out / filename):
            raise ValueError(f"completed_fit_artifact_changed:{out / filename}")
    return True


def _validate_teacher_gate(work, protocol_sha):
    gate_path = work / "teacher_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError("student_phase_requires_root_teacher_gate")
    gate = rt.read_json(gate_path)
    if gate.get("passed") is not True:
        raise ValueError("root_teacher_gate_did_not_pass")
    if gate.get("protocol_sha256") != protocol_sha:
        raise ValueError("teacher_gate_protocol_mismatch")
    teacher_root = work / "teachers"
    expected_complete, expected_outputs = {}, {}
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"seed_{seed}_fold_{fold:02d}"
            out = teacher_root / name
            if not _validate_complete(out, protocol_sha):
                raise ValueError(f"teacher_gate_missing_completed_fit:{name}")
            expected_complete[name] = rt.sha(out / "complete.json")
            for filename in FIT_FILES:
                relative = (Path("teachers") / name / filename).as_posix()
                expected_outputs[relative] = rt.sha(out / filename)
    if gate.get("teacher_bindings") != expected_complete:
        raise ValueError("teacher_gate_output_receipts_mismatch")
    if gate.get("teacher_output_binding") != expected_outputs:
        raise ValueError("teacher_gate_flat_output_binding_mismatch")
    return gate


def run(root, parent, work, phase):
    root, parent, work = Path(root).resolve(), Path(parent).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    device = rt.configure()
    modules = rt.modules()
    protocol, protocol_sha = _protocol(root, parent, work, device)
    if phase == "student":
        _validate_teacher_gate(work, protocol_sha)

    phase_dir = work / ("teachers" if phase == "teacher" else "students")
    phase_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"seed_{seed}_fold_{fold:02d}"
            out = phase_dir / name
            out.mkdir(parents=True, exist_ok=True)
            if _validate_complete(out, protocol_sha):
                completed += 1
                continue
            checkpoint, data, checkpoint_path = _load_parent(
                root, parent, name, fold, modules, device)
            if phase == "teacher":
                _fit_teacher(root, parent, work, protocol_sha, name, seed, fold,
                             modules, device, checkpoint, data, checkpoint_path, out)
            else:
                teacher_path = work / "teachers" / name / "models.pt"
                _fit_student(root, parent, work, protocol_sha, name, seed, fold,
                             modules, device, checkpoint, data, checkpoint_path,
                             teacher_path, out)
            completed += 1
            print(f"COMPLETE phase={phase} seed={seed} fold={fold} output={out}", flush=True)
    if completed != 9:
        raise ValueError(f"unexpected_phase_fit_count:{phase}:{completed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--phase", choices=("teacher", "student"), required=True)
    args = parser.parse_args()
    run(args.root, args.parent, args.work, args.phase)


if __name__ == "__main__":
    main()
