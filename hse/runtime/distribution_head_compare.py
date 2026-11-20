"""Paired single- versus two-component report heads on the frozen 205 backbone."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import runtime as rt
import train
import frozen_fusion_compare
from training_data import build_fold

torch = rt.torch
SEEDS = tuple(train.RECIPE["seeds"])
FOLDS = int(train.RECIPE["folds"])
SCALE_BOUNDS = (0.05, 2.5)
RECIPE = {
    "seeds": list(SEEDS),
    "folds": FOLDS,
    "aggregate_fits": 0,
    "single_report_fits": 15,
    "paired_frozen_mixture2_fits": 0,
    "paired_frozen_mixture2_source": "the matching formal current_training_v1 checkpoint",
    "backbone": "cross_attention from frozen_aggregate_prior; every non-report-head tensor fixed",
    "single_density": "single_lognormal, two linear outputs [log_mu, raw_log_sigma]",
    "mixture_density": "existing FROZEN two-component lognormal checkpoint",
    "scale_bounds": list(SCALE_BOUNDS),
    "single_trainable_parameters": 131,
    "train": "hierarchical weighted positive-report NLL averaged over the same four TRAIN views",
    "validation": "hierarchical weighted VALID NLL averaged over the same four views; epoch 0 eligible",
    "nll": "true lognormal density NLL including +log(target) from the Jacobian",
    "initialization": "positive aggregate FIT target log moments from each saved frozen_aggregate_prior fold",
    "report_optimizer": "Adam over report_head weight, bias, and alpha only",
    "head_lr": 0.001,
    "report_epochs": 300,
    "patience": 30,
    "replay_coefficient": 0.1,
    "aggregate_replay": "same aggregate FIT reconstruction loss; constant because the backbone is frozen",
    "parameter_search": [],
}


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_same(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, (list, tuple)) and len(left) == len(right)
                and all(_same(a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _tensor_hash(*tensors):
    digest = hashlib.sha256()
    for value in tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _single_class(modules):
    base = modules["interaction"]
    raw_head = modules["model_raw"].RawBypassReportHead
    density = modules["mixture_model"]

    class SingleDensityFusion(base):
        def __init__(self, architecture, beta, raw_mean, raw_std, mode="M"):
            super().__init__(architecture, beta, raw_mean, raw_std, mode=mode)
            original = self.report_head
            replacement = raw_head(original.in_features, 2, beta, raw_mean, raw_std)
            replacement = replacement.to(device=original.weight.device)
            with torch.no_grad():
                replacement.alpha.copy_(original.alpha)
            self.report_head = replacement

        def raw_report(self, inputs):
            raw = self.report_head(self.encode_report(inputs)).double()
            shift = (self.report_head.alpha.double()
                     * self.report_head.normalized_raw_score(inputs[..., :183]))
            return torch.stack((raw[..., 0] + shift, raw[..., 1]), dim=-1)

        def components(self, inputs, bounds=SCALE_BOUNDS):
            return density.component_parameters(self.raw_report(inputs),
                                                "single_lognormal", bounds)

        def per_record_nll(self, inputs, targets, bounds=SCALE_BOUNDS):
            return density.nll_per_record(self.raw_report(inputs), targets,
                                          "single_lognormal", bounds)

    return SingleDensityFusion


def restore_single(modules, state, device):
    """Restore a saved SINGLE state for downstream prediction or replay."""
    required = ("report_head.raw_beta", "report_head.raw_feature_mean",
                "report_head.raw_feature_std", "report_head.weight", "report_head.bias")
    if any(key not in state for key in required):
        raise ValueError("incomplete_single_density_state")
    if tuple(state["report_head.weight"].shape) != (2, 64):
        raise ValueError("unexpected_single_density_head_shape")
    if tuple(state["report_head.bias"].shape) != (2,):
        raise ValueError("unexpected_single_density_bias_shape")
    model_class = _single_class(modules)
    model = model_class(
        "cross_attention", state["report_head.raw_beta"],
        float(state["report_head.raw_feature_mean"]),
        float(state["report_head.raw_feature_std"]), mode="M")
    model.load_state_dict(state)
    return model.to(device).eval()


def _restore_mixture(modules, state, device):
    if tuple(state["report_head.weight"].shape) != (5, 64):
        raise ValueError("unexpected_frozen_mixture_head_shape")
    model = modules["interaction"](
        "cross_attention", state["report_head.raw_beta"],
        float(state["report_head.raw_feature_mean"]),
        float(state["report_head.raw_feature_std"]), mode="M")
    model.load_state_dict(state)
    return model.to(device).eval()


def _check_recipe():
    expected = {
        "report_epochs": 300,
        "patience": 30,
        "head_lr": 0.001,
        "replay_coefficient": 0.1,
    }
    for name, value in expected.items():
        actual = train.RECIPE[name]
        if actual != value:
            raise ValueError(f"current_train_recipe_mismatch:{name}:{actual}:{value}")
    if train.RECIPE["architecture"] != "cross_attention":
        raise ValueError("current_backbone_architecture_mismatch")
    if len(SEEDS) != 3 or FOLDS != 5:
        raise ValueError("current_cv_recipe_mismatch")


def _check_checkpoint(checkpoint, data, seed, fold):
    required = ("models", "frozen_aggregate_prior", "frozen_recipe", "scalers",
                "target_scalers", "partition")
    if checkpoint.get("schema") != "current_training_v1" or any(
            key not in checkpoint for key in required):
        raise ValueError("incomplete_current_frozen_checkpoint")
    if "FROZEN" not in checkpoint["models"]:
        raise ValueError("missing_fixed_FROZEN_baseline")
    if checkpoint["partition"].get("fold") != fold:
        raise ValueError(f"checkpoint_fold_mismatch:{seed}:{fold}")
    if not _same(checkpoint["partition"], data["provenance"]):
        raise ValueError(f"checkpoint_partition_mismatch:{seed}:{fold}")
    if not _same(checkpoint["scalers"], data["scalers"]):
        raise ValueError(f"checkpoint_scalers_mismatch:{seed}:{fold}")

    target = checkpoint["target_scalers"]
    recipe = checkpoint["frozen_recipe"]
    if not isinstance(target, dict) or not isinstance(recipe, dict):
        raise ValueError("invalid_frozen_recipe_or_target_scalers")
    declared = recipe.get("target_scalers")
    if declared is None and all(key in recipe for key in ("log1p_mean", "log1p_std")):
        declared = {key: recipe[key] for key in ("log1p_mean", "log1p_std")}
    if declared is not None and not _same(declared, target):
        raise ValueError("frozen_recipe_target_scalers_mismatch")

    mean, std = float(target["log1p_mean"]), float(target["log1p_std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("invalid_saved_target_scalers")
    fit_targets = data["ay"][data["fi"]].log1p()
    fit_mean = float(fit_targets.mean())
    fit_std = float(fit_targets.std(unbiased=False).clamp_min(1e-6))
    if not math.isclose(mean, fit_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("saved_aggregate_log1p_mean_mismatch")
    if not math.isclose(std, fit_std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("saved_aggregate_log1p_std_mismatch")

    prior = checkpoint["frozen_aggregate_prior"]
    baseline = checkpoint["models"]["FROZEN"]
    if tuple(prior["report_head.weight"].shape) != (5, 64):
        raise ValueError("unexpected_frozen_aggregate_prior_head_shape")
    if tuple(baseline["report_head.weight"].shape) != (5, 64):
        raise ValueError("unexpected_fixed_FROZEN_head_shape")
    if any(key not in baseline or not _same(value, baseline[key])
           for key, value in prior.items() if not key.startswith("report_head.")):
        raise ValueError("fixed_FROZEN_backbone_differs_from_saved_prior")
    if float(prior["report_head.alpha"]) != 0.0:
        raise ValueError("frozen_aggregate_prior_alpha_not_zero")
    return mean, std


def _mixture_log_moments(modules, prior_state):
    raw = prior_state["report_head.bias"].detach().cpu().double()
    weights, mus, sigmas = modules["mixture_model"].component_parameters(
        raw, "mixture_lognormal", SCALE_BOUNDS)
    mean = torch.sum(weights * mus)
    variance = torch.sum(weights * (sigmas.square() + (mus - mean).square()))
    return float(mean), float(torch.sqrt(variance.clamp_min(0.0)))


def _initialize_single(modules, prior_state, data, device):
    model_class = _single_class(modules)
    model = model_class(
        "cross_attention", prior_state["report_head.raw_beta"],
        float(prior_state["report_head.raw_feature_mean"]),
        float(prior_state["report_head.raw_feature_std"]), mode="M")
    partial_state = {
        key: value for key, value in prior_state.items()
        if key not in ("report_head.weight", "report_head.bias")
    }
    loaded = model.load_state_dict(partial_state, strict=False)
    if set(loaded.missing_keys) != {"report_head.weight", "report_head.bias"}:
        raise ValueError(f"unexpected_single_prior_missing_keys:{loaded.missing_keys}")
    if loaded.unexpected_keys:
        raise ValueError(f"unexpected_single_prior_keys:{loaded.unexpected_keys}")
    model = model.to(device)

    positive = data["fi"][data["ay"][data["fi"]] > 0]
    targets = data["ay"][positive]
    if targets.numel() == 0:
        raise ValueError("empty_positive_aggregate_fit")
    weights = torch.full((targets.numel(),), 1.0 / targets.numel(),
                         dtype=torch.float64, device=device)
    with torch.no_grad():
        model.report_head.alpha.zero_()
    initialization = modules["mixture_model"].initialize_report_head(
        model, "single_lognormal", targets, weights, SCALE_BOUNDS)

    if float(model.report_head.alpha.detach()) != 0.0:
        raise ValueError("single_initial_alpha_not_zero")
    if torch.count_nonzero(model.report_head.weight).item() != 0:
        raise ValueError("single_initial_head_weights_not_zero")

    model_state = train.state(model)
    for key, value in prior_state.items():
        if key in ("report_head.weight", "report_head.bias"):
            continue
        if key not in model_state or not _same(value, model_state[key]):
            raise ValueError(f"single_prior_state_mismatch:{key}")

    target_mean = initialization["weighted_log_gap_mean"]
    target_std = initialization["weighted_log_gap_std"]
    mixture_mean, mixture_std = _mixture_log_moments(modules, prior_state)
    single_parts = modules["mixture_model"].component_parameters(
        model.report_head.bias.detach().double(), "single_lognormal", SCALE_BOUNDS)
    single_mean = float(single_parts[1][0])
    single_std = float(single_parts[2][0])
    if not (math.isclose(mixture_mean, target_mean, rel_tol=0.0, abs_tol=1e-5)
            and math.isclose(mixture_std, target_std, rel_tol=0.0, abs_tol=1e-5)
            and math.isclose(single_mean, target_mean, rel_tol=0.0, abs_tol=1e-5)
            and math.isclose(single_std, target_std, rel_tol=0.0, abs_tol=1e-5)):
        raise ValueError("single_and_mixture_initial_log_moments_mismatch")

    initialization.update(
        positive_aggregate_fit_count=int(targets.numel()),
        initial_alpha=float(model.report_head.alpha.detach()),
        mixture_prior_log_mean=mixture_mean,
        mixture_prior_log_std=mixture_std,
        single_prior_log_mean=single_mean,
        single_prior_log_std=single_std,
        prior_state_sha256=_state_hash(prior_state),
        single_initial_state_sha256=_state_hash(model_state),
    )
    return model, initialization


def _run_fold(root, work, seed, fold, device, modules, protocol_sha):
    out = work / f"seed_{seed}_fold_{fold + 1:02d}"
    out.mkdir(parents=True, exist_ok=True)
    complete_path = out / "complete.json"
    if complete_path.exists():
        receipt = rt.read_json(complete_path)
        for filename, key in (("predictions.csv", "predictions_sha256"),
                              ("fits.json", "fits_sha256"),
                              ("models.pt", "models_sha256")):
            if receipt.get(key) != rt.sha(out / filename):
                raise ValueError(f"completed_output_hash_mismatch:{out.name}/{filename}")
        if receipt.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"completed_protocol_mismatch:{out.name}")
        return

    started = time.time()
    data = build_fold(root, fold, device)
    source = root / "models" / "checkpoints" / f"seed_{seed}_fold_{fold + 1:02d}" / "models.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    _check_checkpoint(checkpoint, data, seed, fold)

    prior_state = checkpoint["frozen_aggregate_prior"]
    frozen_state = checkpoint["models"]["FROZEN"]
    single, initialization = _initialize_single(modules, prior_state, data, device)
    frozen = _restore_mixture(modules, frozen_state, device).requires_grad_(False)
    prior_hash = _state_hash(prior_state)
    frozen_hash = _state_hash(frozen_state)

    target_scalers = checkpoint["target_scalers"]
    mean, std = float(target_scalers["log1p_mean"]), float(target_scalers["log1p_std"])
    standardized = (data["ay"].log1p() - mean) / std
    single, single_fit = frozen_fusion_compare.fit_frozen(single, data, standardized)
    trainable_names = [name for name, parameter in single.named_parameters()
                       if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in single.parameters()
                          if parameter.requires_grad)
    if any(not name.startswith("report_head.") for name in trainable_names):
        raise ValueError("single_fit_unexpected_trainable_parameter")
    if trainable_count != 131:
        raise ValueError(f"single_trainable_parameter_count_mismatch:{trainable_count}")
    single_fit.update(
        selected_arm="SINGLE",
        initial_state_hash=initialization["single_initial_state_sha256"],
        aggregate_prior_state_hash=prior_hash,
        trainable_parameter_names=trainable_names,
        trainable_head_parameters=trainable_count,
        model_parameters=sum(parameter.numel() for parameter in single.parameters()),
        initialization=initialization,
    )

    frozen_head_parameters = sum(parameter.numel() for parameter in frozen.report_head.parameters())
    frozen_fit = {
        "fit_performed": False,
        "source": "existing current_training_v1 FROZEN checkpoint",
        "source_checkpoint": str(source.relative_to(root)),
        "source_checkpoint_sha256": rt.sha(source),
        "aggregate_prior_state_hash": prior_hash,
        "frozen_state_hash": frozen_hash,
        "model_parameters": sum(parameter.numel() for parameter in frozen.parameters()),
        "head_parameters": frozen_head_parameters,
        "head_family": "two_component_lognormal",
    }

    predictions = []
    with torch.no_grad():
        for name, model in (("SINGLE", single), ("FROZEN", frozen)):
            for view_index, view_name in enumerate(rt.VIEWS):
                parts = model.components(data["views"][view_index, data["oi"]], SCALE_BOUNDS)
                predictions.extend(train.prediction_rows(
                    parts, data, seed, fold, name, view_name))
    expected = 8 * len(data["oi"])
    if len(predictions) != expected:
        raise ValueError(f"prediction_count_mismatch:{len(predictions)}:{expected}")
    rt.write_csv(out / "predictions.csv", predictions)
    rt.write_json(out / "fits.json", {
        "SINGLE": single_fit,
        "FROZEN": frozen_fit,
        "fit_count": 1,
        "paired_mixture_fit_count": 0,
        "outer_count": len(data["oi"]),
        "four_view_tensor_sha256": _tensor_hash(data["views"]),
        "aggregate_four_view_tensor_sha256": _tensor_hash(data["av"]),
    })
    models_path = out / "models.pt"
    torch.save({
        "schema": "distribution_head_compare_v1",
        "seed": seed,
        "fold": fold,
        "models": {"SINGLE": train.state(single), "FROZEN": frozen_state},
        "frozen_aggregate_prior": prior_state,
        "scalers": checkpoint["scalers"],
        "target_scalers": target_scalers,
        "partition": checkpoint["partition"],
        "recipe": RECIPE,
        "source_checkpoint_sha256": rt.sha(source),
    }, models_path)
    rt.write_json(complete_path, {
        "status": "complete",
        "seed": seed,
        "fold": fold,
        "aggregate_fits": 0,
        "single_report_fits": 1,
        "paired_frozen_mixture2_fits": 0,
        "predictions": len(predictions),
        "protocol_sha256": protocol_sha,
        "predictions_sha256": rt.sha(out / "predictions.csv"),
        "fits_sha256": rt.sha(out / "fits.json"),
        "models_sha256": rt.sha(models_path),
        "seconds": time.time() - started,
    })
    print(f"COMPLETE seed={seed} fold={fold + 1} fits=1", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work_dir.resolve()
    try:
        work.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("work_dir_must_be_outside_formal_root")

    _check_recipe()
    work.mkdir(parents=True, exist_ok=True)
    device = rt.configure()
    modules = rt.modules()
    inputs = [root / "data" / "reports" / name
              for name in ("features.pt", "records.jsonl", "metadata.json")]
    inputs += [root / "data" / "composition" / name
               for name in ("features.pt", "metadata.json")]
    checkpoint_paths = [root / "models" / "checkpoints"
                         / f"seed_{seed}_fold_{fold:02d}" / "models.pt"
                         for seed in SEEDS for fold in range(1, FOLDS + 1)]
    code = [Path(__file__).resolve(), Path(train.__file__).resolve(),
            Path(frozen_fusion_compare.__file__).resolve(),
            Path(build_fold.__code__.co_filename).resolve(), Path(rt.__file__).resolve(),
            root / "runtime" / "inference.py"]
    code += sorted((root / "runtime" / "modules").glob("*.py"))
    code += [root / "runtime" / "modules" / "symmetry_table.json"]
    protocol = RECIPE | {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "input_binding": {str(path.relative_to(root)): rt.sha(path) for path in inputs},
        "checkpoint_binding": {str(path.relative_to(root)): rt.sha(path)
                                for path in checkpoint_paths},
        "code_binding": {str(path.relative_to(root)): rt.sha(path) for path in code},
    }
    protocol_path = work / "protocol.json"
    if protocol_path.exists() and rt.read_json(protocol_path) != protocol:
        raise ValueError("input_or_recipe_changed_existing_run_cannot_be_resumed")
    rt.write_json(protocol_path, protocol)
    protocol_sha = rt.sha(protocol_path)

    for seed in SEEDS:
        for fold in range(FOLDS):
            _run_fold(root, work, seed, fold, device, modules, protocol_sha)


if __name__ == "__main__":
    main()
