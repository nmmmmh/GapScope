"""Inference for the current frozen-head model and anchored-transfer comparison."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import runtime as rt


SEEDS = (17, 29, 41)
FOLDS = range(1, 6)
MODEL_ARMS = {"formula": "FROZEN", "multimodal": "FROZEN", "formula_alternative": "SG_DISTILL"}
SCALE_BOUNDS = (0.05, 2.5)
_MODULES = None


def _modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = rt.modules()
    return _MODULES


def validate_features(data):
    if not isinstance(data, dict):
        raise ValueError("features must be a JSON object")
    composition = data.get("composition")
    if (not isinstance(composition, list) or len(composition) != 183
            or not all(math.isfinite(value) for value in composition)):
        raise ValueError("composition must contain 183 finite raw encoder features")
    space_group = data.get("space_group")
    if space_group is not None and (type(space_group) is not int or not 1 <= space_group <= 230):
        raise ValueError("space_group must be an integer from 1 to 230")
    structure = data.get("structure")
    if (structure is not None and
            (not isinstance(structure, list) or len(structure) != 160
             or not all(math.isfinite(value) for value in structure))):
        raise ValueError("structure must contain 160 finite raw encoder features")


def _input_features(data):
    space_group = data.get("space_group")
    structure = data.get("structure")
    sg_features = torch.zeros(1, 237, dtype=torch.float32)
    if space_group is not None:
        sg_features[0, space_group - 1] = 1
        system = (0 if space_group <= 2 else 1 if space_group <= 15 else
                  2 if space_group <= 74 else 3 if space_group <= 142 else
                  4 if space_group <= 167 else 5 if space_group <= 194 else 6)
        sg_features[0, 230 + system] = 1
    features = {
        "composition": torch.tensor([data["composition"]], dtype=torch.float32),
        "sg_features": sg_features,
        "structure": torch.tensor([structure if structure is not None else [0.0] * 160],
                                  dtype=torch.float32),
        "sg_mask": torch.tensor([[float(space_group is not None)]], dtype=torch.float32),
        "structure_mask": torch.tensor([[float(structure is not None)]], dtype=torch.float32),
    }
    modules = _modules()
    return modules["t35"].transform(
        features, "hybrid", modules["loader43"].symmetry_table(), set())


def _restore_model(modules, state, mode, device):
    if state["report_head.raw_beta"].shape != (184,):
        raise ValueError("unexpected_current_training_raw_beta")
    model = modules["interaction"](
        "cross_attention", state["report_head.raw_beta"],
        float(state["report_head.raw_feature_mean"]),
        float(state["report_head.raw_feature_std"]), mode=mode)
    model.load_state_dict(state)
    return model.to(device).eval()


def _base_parts(model, formula_view):
    weights, mus, sigmas = model.components(formula_view, SCALE_BOUNDS)
    raw = model.raw_report(formula_view)
    return {
        "weights": weights, "mus": mus, "sigmas": sigmas,
        "gate": raw[..., 0],
        "raw_scale": torch.logit((sigmas - SCALE_BOUNDS[0]) /
                                 (SCALE_BOUNDS[1] - SCALE_BOUNDS[0])),
        "hidden": model.encode(formula_view).double(),
    }


def _anchored_parts(model, formula_view, anchor, device):
    base = _base_parts(model, formula_view)
    center = anchor["center"].to(device=device, dtype=torch.float64)
    basis = anchor["basis"].to(device=device, dtype=torch.float64)
    scale = anchor["scale"].to(device=device, dtype=torch.float64)
    head_weight = anchor["head_state"]["weight"].to(device=device, dtype=torch.float64)
    head_bias = anchor['head_state'].get('bias')
    if head_bias is not None:
        head_bias = head_bias.to(device=device, dtype=torch.float64)
    coordinates = (base["hidden"] - center) @ basis / scale
    delta = torch.nn.functional.linear(coordinates, head_weight, head_bias)

    first_weight = torch.sigmoid(base["gate"] + delta[:, 0])
    weights = torch.stack((first_weight, 1.0 - first_weight), dim=-1)
    mus = base["mus"] + delta[:, 1:3]
    scale0 = SCALE_BOUNDS[0] + (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]) * torch.sigmoid(base["raw_scale"])
    sigmas = base["sigmas"] + (
        SCALE_BOUNDS[0] + (SCALE_BOUNDS[1] - SCALE_BOUNDS[0])
        * torch.sigmoid(base["raw_scale"] + delta[:, 3:5]) - scale0)
    return weights, mus, sigmas


def _checkpoint_paths(checkpoint_dir):
    return [(seed, fold, Path(checkpoint_dir) / f"seed_{seed}_fold_{fold:02d}" / "models.pt")
            for seed in SEEDS for fold in FOLDS]


def predict_features(data, model_name, device="cpu", checkpoint_dir=None):
    """Return an equal-fold mixture of the requested frozen model predictions."""
    validate_features(data)
    if model_name not in MODEL_ARMS:
        raise ValueError(f"unknown_model:{model_name}")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required; trained weights are distributed separately")
    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device(device)
    modules = _modules()
    features = _input_features(data)
    space_group = data.get("space_group")
    structure = data.get("structure")
    view_index = int(space_group is not None) + 2 * int(structure is not None)
    arm = MODEL_ARMS[model_name]
    paths = _checkpoint_paths(checkpoint_dir)
    if any(not path.is_file() for _, _, path in paths):
        missing = [str(path) for _, _, path in paths if not path.is_file()]
        raise FileNotFoundError("missing_current_training_checkpoints:" + ",".join(missing))

    weights, log_mus, log_sigmas, assets = [], [], [], []
    for seed, fold, path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "current_training_v1":
            raise ValueError(f"unexpected_checkpoint_schema:{path}")
        partition = checkpoint.get("partition", {})
        if partition.get("fold") != fold - 1:
            raise ValueError(f"checkpoint_fold_mismatch:{path}")
        state_key = "FROZEN" if arm in ("SG_DISTILL", "FULL_DISTILL") else ("F" if arm == "ANCHORED" else arm)
        state = checkpoint["models"][state_key]
        mode = "F" if state_key == "F" else "J"
        model = _restore_model(modules, state, mode, device)
        scalers = checkpoint["scalers"]
        views = modules["t10"].four_views(
            features,
            *(scalers[key].to(device=device, dtype=torch.float32) for key in
              ("composition_mean", "composition_std", "structure_mean", "structure_std")),
            device)
        with torch.inference_mode():
            if arm == "ANCHORED":
                parts = _anchored_parts(model, views[0], checkpoint["anchor"], device)
            elif arm in ("SG_DISTILL", "FULL_DISTILL"):
                parts = _anchored_parts(model, views[0], checkpoint["supervised_distillation"][arm], device)
            else:
                parts = model.components(views[0 if model_name == 'formula' else view_index], SCALE_BOUNDS)
        fold_weights, fold_mus, fold_sigmas = (
            value[0].detach().cpu().double().tolist() for value in parts)
        weights.extend(float(value) / len(paths) for value in fold_weights)
        log_mus.extend(float(value) for value in fold_mus)
        log_sigmas.extend(float(value) for value in fold_sigmas)
        assets.append({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed, "fold_id": fold,
        })

    if (len(assets) != 15 or len(weights) != 30 or
            not all(math.isfinite(value) for value in (*weights, *log_mus, *log_sigmas))):
        raise ValueError("invalid_15_fold_density_ensemble")
    prediction_model = "ANCHORED" if arm == "ANCHORED" else arm
    result = {
        "schema": "experimental_multiphase_density_v1",
        "model_choice": model_name,
        "prediction_model": prediction_model,
        "trained_report_count": 205,
        "target_scope": "positive experimental report bandgap distribution in eV",
        "weights": weights, "log_mu": log_mus, "log_sigma": log_sigmas,
        "mean_ev": sum(weight * math.exp(mu + 0.5 * sigma * sigma)
                       for weight, mu, sigma in zip(weights, log_mus, log_sigmas)),
        "input_metadata": {key: data[key] for key in
                           ("formula", "encoder_checkpoint_sha256", "cif_sha256", "structure_source")
                           if key in data},
        "available": {"space_group": space_group is not None, "structure": structure is not None},
        "optional_modalities_used": {
            "space_group": model_name == "multimodal" and space_group is not None,
            "structure": model_name == "multimodal" and structure is not None,
        },
        "ensemble_use": "15-fold equal-weight density mixture; no all-report refit",
        "components_are_phase_labels": False,
        "checkpoints": assets,
    }
    if arm == "ANCHORED":
        result["interpretation"] = (
            "Experimental anchored transfer: an F baseline with a low-rank, "
            "structure-informed density correction; not a physical phase assignment.")
    elif arm in ("SG_DISTILL", "FULL_DISTILL"):
        result["interpretation"] = (
            "Experimental formula-only affine correction of the fixed FROZEN baseline, "
            "trained with real reports and a formula-marginal teacher CDF; not a phase assignment.")
    return result
