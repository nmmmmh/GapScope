#!/usr/bin/env python
"""Fold-safe T10 shared-fusion dual-task experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from dual_task_model import DualTaskFusion


STUDY_DIR = "analysis/multiphase_research_20261004/t10_multimodal_dual_task_20261007"
VIEWS = ("formula_only", "formula_sg", "formula_structure", "full")
SCORE_NAMES = ("nll", "crps_ev", "median_mae_ev", "coverage_50", "width_50_ev",
               "coverage_80", "width_80_ev", "coverage_90", "width_90_ev")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot_load_module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_contract(protocol: dict) -> None:
    if protocol.get("arms") != ["frozen_fusion", "joint_replay"]:
        raise ValueError("unexpected_t10_arms")
    if tuple(protocol.get("views", [])) != VIEWS:
        raise ValueError("unexpected_t10_views")
    if [int(value) for value in protocol.get("seeds", [])] != [17, 29, 41]:
        raise ValueError("unexpected_t10_seeds")
    if int(protocol.get("folds", 0)) != 5:
        raise ValueError("unexpected_t10_fold_count")
    model = protocol["model"]
    if (int(model["composition_dim"]) != 183 or int(model["sg_dim"]) != 237
            or int(model["structure_dim"]) != 160 or int(model["input_dim"]) != 582
            or int(model["width"]) != 64 or int(model["depth"]) != 1
            or int(model["report_components"]) != 1
            or tuple(float(x) for x in model["scale_bounds"]) != (0.05, 2.5)):
        raise ValueError("unexpected_t10_model_contract")
    if int(protocol["budget"]["aggregate_base_fits"]) != 15 or int(protocol["budget"]["report_fits"]) != 30:
        raise ValueError("unexpected_t10_fit_budget")


def receipt_path(root: Path, hashes: dict, fragment: str) -> Path:
    matches = [resolve(root, name) for name in hashes
               if fragment.replace("\\", "/") in name.replace("\\", "/")]
    if len(matches) != 1:
        raise ValueError(f"receipt_input_not_unique:{fragment}:{len(matches)}")
    return matches[0]


def check_hashes(root: Path, hashes: dict, label: str) -> None:
    for name, expected in hashes.items():
        path = resolve(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{label}_hash_mismatch:{name}")


def verify_data(root: Path, protocol: dict, protocol_path: Path):
    base_path = resolve(root, protocol["base_report_protocol"])
    base_protocol = read_json(base_path)
    t4_path = resolve(root, base_protocol["runner"]["path"]) if "runner" in base_protocol else (
        root / "analysis/multiphase_research_20261004/t4_single_lognormal_20261007/run_formula_cv.py")
    t4 = load_module("t10_t4_runner", t4_path)
    paths, report_records, target_values, base_features, fold_by_id, fold_ids, fold_weights, oof_weights, t4_hashes = (
        t4.verify_inputs(root, base_protocol))

    data_dir = resolve(root, protocol["data_dir"])
    receipt = read_json(data_dir / "receipt.json")
    if receipt.get("status") != "complete":
        raise ValueError("t10_data_receipt_incomplete")
    check_hashes(root, receipt.get("input_sha256", {}), "t10_receipt_input")
    check_hashes(data_dir, receipt.get("output_sha256", {}), "t10_receipt_output")
    aggregate = torch.load(data_dir / "aggregate_features.pt", map_location="cpu", weights_only=True)
    reports = torch.load(data_dir / "report_features.pt", map_location="cpu", weights_only=True)
    aggregate_meta = read_json(data_dir / "aggregate_metadata.json")
    report_meta = read_json(data_dir / "report_metadata.json")
    if not isinstance(aggregate, dict) or not isinstance(reports, dict):
        raise ValueError("invalid_t10_tensor_payload")
    required = {"composition", "sg_features", "structure", "sg_mask", "structure_mask"}
    if required - set(aggregate) or required - set(reports) or "targets" not in aggregate:
        raise ValueError("t10_tensor_keys_missing")

    aggregate_count, report_count = len(aggregate_meta), len(report_records)
    shapes = {"composition": 183, "sg_features": 237, "structure": 160,
              "sg_mask": 1, "structure_mask": 1}
    for name, width in shapes.items():
        if tuple(aggregate[name].shape) != (aggregate_count, width):
            raise ValueError(f"aggregate_shape_mismatch:{name}")
        if tuple(reports[name].shape) != (report_count, width):
            raise ValueError(f"report_shape_mismatch:{name}")
    targets = aggregate["targets"].reshape(-1).to(dtype=torch.float64)
    if tuple(targets.shape) != (aggregate_count,) or not torch.isfinite(targets).all() or torch.any(targets < 0):
        raise ValueError("invalid_aggregate_targets")
    for name in required:
        if not torch.isfinite(aggregate[name]).all() or not torch.isfinite(reports[name]).all():
            raise ValueError(f"nonfinite_features:{name}")
    for name in ("sg_mask", "structure_mask"):
        for payload in (aggregate, reports):
            mask = payload[name]
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"nonbinary_mask:{name}")
    if len({row["composition_key"] for row in aggregate_meta}) != aggregate_count:
        raise ValueError("aggregate_rows_not_unique_by_composition")
    if any(str(row.get("split", "")).lower() != "train" for row in aggregate_meta):
        raise ValueError("nontrain_aggregate_row")
    if len(report_meta) != report_count:
        raise ValueError("report_metadata_count_mismatch")
    for index, row in enumerate(report_records):
        meta = report_meta[index]
        if (meta.get("measurement_id", meta.get("record_id")) != row["measurement_id"]
                or meta.get("composition_key") != row["composition_key"]
                or meta.get("formula") != row["formula"]):
            raise ValueError(f"report_feature_order_mismatch:{row['measurement_id']}")
        if not torch.allclose(reports["composition"][index], base_features["composition"][index],
                              atol=1e-6, rtol=1e-6):
            raise ValueError(f"report_formula_feature_mismatch:{row['measurement_id']}")
    if int(reports["sg_mask"].sum().item()) != int(receipt["report_sg_encoded"]):
        raise ValueError("report_sg_receipt_mismatch")
    if int(reports["structure_mask"].sum().item()) != int(receipt["report_structure_encoded"]):
        raise ValueError("report_structure_receipt_mismatch")
    if int(aggregate["sg_mask"].sum().item()) != int(receipt["aggregate_sg_encoded"]):
        raise ValueError("aggregate_sg_receipt_mismatch")
    if int(aggregate["structure_mask"].sum().item()) != int(receipt["aggregate_structure_encoded"]):
        raise ValueError("aggregate_structure_receipt_mismatch")

    protected_path = receipt_path(root, receipt["input_sha256"], "stage123_20261004/protocol/split_registry.csv")
    protected_rows = t4.read_csv(protected_path)
    protected = {row["composition_key"] for row in protected_rows
                 if row.get("partition", "").lower() != "train"}
    aggregate_keys = {row["composition_key"] for row in aggregate_meta}
    if aggregate_keys & protected:
        raise ValueError("protected_composition_in_aggregate_input")
    for label, path in receipt["output_sha256"].items():
        if sha256(data_dir / label) != receipt["output_sha256"][label]:
            raise ValueError(f"t10_output_changed_after_load:{label}")
    check_hashes(root, receipt.get("input_sha256", {}), "t10_receipt_input_after_load")
    for label, path in paths.items():
        if sha256(path) != t4_hashes[label]:
            raise ValueError(f"t4_input_changed_after_load:{label}")
    return {
        "t4": t4, "base_protocol": base_protocol, "report_records": report_records,
        "report_targets": torch.tensor(target_values, dtype=torch.float64),
        "fold_by_id": fold_by_id, "fold_ids": fold_ids, "fold_weights": fold_weights,
        "oof_weights": oof_weights, "aggregate": aggregate, "reports": reports,
        "aggregate_meta": aggregate_meta, "report_meta": report_meta,
        "aggregate_targets": targets, "protected": protected, "receipt": receipt,
        "t4_hashes": t4_hashes, "data_hashes": {
            name: sha256(data_dir / name) for name in receipt["output_sha256"]
        },
    }


def standardize(values: torch.Tensor):
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def fold_scalers(aggregate, aggregate_meta, aggregate_fit, report, report_meta, report_train_ids,
                 report_records, report_index, device):
    formula_by_key, structure_by_key = {}, {}
    for index in aggregate_fit:
        key = aggregate_meta[index]["composition_key"]
        formula_by_key.setdefault(key, aggregate["composition"][index])
        if float(aggregate["structure_mask"][index].item()) > 0:
            structure_by_key.setdefault(key, aggregate["structure"][index])
    for record_id in report_train_ids:
        index = report_index[record_id]
        key = report_records[index]["composition_key"]
        vector = report["composition"][index]
        if key in formula_by_key and not torch.allclose(formula_by_key[key], vector, atol=1e-5, rtol=1e-5):
            raise ValueError(f"overlap_formula_features_differ:{key}")
        formula_by_key.setdefault(key, vector)
        if (float(report["structure_mask"][index].item()) > 0
                and key not in structure_by_key):
            structure_by_key[key] = report["structure"][index]
    if not formula_by_key:
        raise ValueError("empty_input_normalization_support")
    comp = torch.stack([formula_by_key[key] for key in sorted(formula_by_key)]).to(device=device)
    comp_mean, comp_std = standardize(comp)
    if structure_by_key:
        structure = torch.stack([structure_by_key[key] for key in sorted(structure_by_key)]).to(device=device)
        structure_mean, structure_std = standardize(structure)
    else:
        structure_mean = torch.zeros(160, device=device)
        structure_std = torch.ones(160, device=device)
    return comp_mean, comp_std, structure_mean, structure_std, {
        "normalization_compositions": len(formula_by_key),
        "normalization_structure_compositions": len(structure_by_key),
    }


def four_views(features, comp_mean, comp_std, structure_mean, structure_std, device):
    comp = (features["composition"].to(device=device, dtype=torch.float32) - comp_mean) / comp_std
    sg = features["sg_features"].to(device=device, dtype=torch.float32)
    sg_mask = features["sg_mask"].to(device=device, dtype=torch.float32)
    structure_mask = features["structure_mask"].to(device=device, dtype=torch.float32)
    structure = (features["structure"].to(device=device, dtype=torch.float32) - structure_mean) / structure_std
    sg = sg * sg_mask
    structure = structure * structure_mask
    none = torch.zeros_like(sg_mask)
    empty_sg = torch.zeros_like(sg)
    empty_structure = torch.zeros_like(structure)
    parts = []
    for include_sg, include_structure in ((False, False), (True, False), (False, True), (True, True)):
        current_sg_mask = sg_mask if include_sg else none
        current_structure_mask = structure_mask if include_structure else none
        current_sg = sg if include_sg else empty_sg
        current_structure = structure if include_structure else empty_structure
        parts.append(torch.cat((comp, current_sg, current_structure,
                                current_sg_mask, current_structure_mask), dim=1))
    return torch.stack(parts)


def aggregate_loss(model, views, targets_standardized, indices=None):
    if indices is not None:
        views = views[:, indices]
        targets_standardized = targets_standardized[indices]
    count = targets_standardized.numel()
    predictions = model.aggregate(views.reshape(-1, views.shape[-1])).reshape(4, count)
    return ((predictions - targets_standardized.unsqueeze(0)) ** 2).mean(dim=0).mean()


def lognormal_nll(log_y, log_mu, sigma):
    log_y = log_y.to(dtype=log_mu.dtype)
    return (0.5 * ((log_y - log_mu) / sigma) ** 2 + log_y
            + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi))


def report_loss(model, views, targets, weights, bounds):
    count = targets.numel()
    log_mu, sigma = model.report(views.reshape(-1, views.shape[-1]), bounds)
    log_mu = log_mu.reshape(4, count)
    sigma = sigma.reshape(4, count)
    per_record = lognormal_nll(torch.log(targets).unsqueeze(0), log_mu, sigma).mean(dim=0)
    return torch.sum(weights * per_record)


def report_validation_loss(model, formula_view, targets, weights, bounds):
    log_mu, sigma = model.report(formula_view, bounds)
    return float(torch.sum(weights * lognormal_nll(torch.log(targets), log_mu, sigma)).item())


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().view(torch.uint8).reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(bytes(value.tolist()))
    return digest.hexdigest()


def init_report_head(model, targets, weights, bounds):
    log_y = torch.log(targets).to(dtype=torch.float64)
    mean = torch.sum(weights * log_y)
    scale = torch.sqrt(torch.sum(weights * (log_y - mean) ** 2)).clamp(
        min=float(bounds[0]) + 1e-6, max=float(bounds[1]) - 1e-6)
    fraction = ((scale - float(bounds[0])) / (float(bounds[1]) - float(bounds[0]))).clamp(1e-6, 1 - 1e-6)
    scale_bias = torch.log(fraction / (1 - fraction))
    with torch.no_grad():
        model.report_head.weight.zero_()
        model.report_head.bias[0] = mean.to(model.report_head.bias.dtype)
        model.report_head.bias[1] = scale_bias.to(model.report_head.bias.dtype)
    return {"weighted_log_gap_mean": float(mean.item()), "weighted_log_gap_std": float(scale.item()),
            "raw_scale_bias": float(scale_bias.item())}


def nonzero_gradient(grads) -> bool:
    return any(value is not None and torch.isfinite(value).all() and float(value.abs().sum().item()) > 0
               for value in grads)


def assert_aggregate_gradients(model, views, targets, label):
    loss = aggregate_loss(model, views, targets)
    params = list(model.shared.parameters()) + list(model.aggregate_head.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    split = len(list(model.shared.parameters()))
    if not nonzero_gradient(grads[:split]) or not nonzero_gradient(grads[split:]):
        raise ValueError(f"aggregate_task_gradient_missing:{label}")
    first_weight_grad = grads[0]
    composition_norm = float(first_weight_grad[:, :183].abs().sum().item())
    sg_norm = float(first_weight_grad[:, 183:420].abs().sum().item())
    structure_norm = float(first_weight_grad[:, 420:580].abs().sum().item())
    if composition_norm == 0 or sg_norm == 0 or structure_norm == 0:
        raise ValueError(f"aggregate_input_branch_gradient_missing:{label}:{composition_norm}:{sg_norm}:{structure_norm}")
    return {"composition_input_gradient_l1": composition_norm,
            "sg_input_gradient_l1": sg_norm, "structure_input_gradient_l1": structure_norm}


def assert_report_gradients(model, views, targets, weights, bounds, arm, label):
    loss = report_loss(model, views, targets, weights, bounds)
    report_params = list(model.report_head.parameters())
    shared_params = list(model.shared.parameters()) if arm == "joint_replay" else []
    grads = torch.autograd.grad(loss, shared_params + report_params, allow_unused=True)
    shared_grads, report_grads = grads[:len(shared_params)], grads[len(shared_params):]
    if not nonzero_gradient(report_grads):
        raise ValueError(f"report_head_gradient_missing:{label}")
    shared_norm = sum(float(grad.abs().sum().item()) for grad in shared_grads if grad is not None)
    return {"report_head_gradient_l1": sum(float(grad.abs().sum().item()) for grad in report_grads if grad is not None),
            "report_shared_gradient_l1": shared_norm}


def train_aggregate(torch, model, aggregate_views, target_values, fit_indices, valid_indices,
                    config, seed, fold_id, epoch_rows):
    log_targets = torch.log1p(target_values)
    target_mean = log_targets[fit_indices].mean()
    target_std = log_targets[fit_indices].std(unbiased=False)
    if float(target_std.item()) <= 0:
        target_std = torch.ones_like(target_std)
    standardized = (log_targets - target_mean) / target_std
    optimizer = torch.optim.Adam(
        list(model.shared.parameters()) + list(model.aggregate_head.parameters()),
        lr=float(config["learning_rate"]), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    best_loss = float(aggregate_loss(model, aggregate_views, standardized, valid_indices).item())
    best_epoch, bad_epochs, best_state = 0, 0, clone_state(model)
    epoch_rows.append({"stage": "aggregate_base", "seed": seed, "fold_id": fold_id,
                       "epoch": 0, "train_mse": "", "validation_mse": best_loss})
    gradient_info = None
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = aggregate_loss(model, aggregate_views, standardized, fit_indices)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite_aggregate_loss:{seed}:{fold_id}:{epoch}")
        loss.backward()
        if epoch == 1:
            first_weight_grad = model.shared.weight.grad
            composition_norm = float(first_weight_grad[:, :183].abs().sum().item())
            sg_norm = float(first_weight_grad[:, 183:420].abs().sum().item())
            structure_norm = float(first_weight_grad[:, 420:580].abs().sum().item())
            if composition_norm == 0 or sg_norm == 0 or structure_norm == 0:
                raise ValueError(f"aggregate_input_branch_gradient_missing:{seed}:{fold_id}:{composition_norm}:{sg_norm}:{structure_norm}")
            gradient_info = {"composition_input_gradient_l1": composition_norm,
                             "sg_input_gradient_l1": sg_norm,
                             "structure_input_gradient_l1": structure_norm}
        old_shared = model.shared.weight.detach().clone() if epoch == 1 else None
        old_head = model.aggregate_head.weight.detach().clone() if epoch == 1 else None
        optimizer.step()
        if epoch == 1 and (torch.equal(old_shared, model.shared.weight) or torch.equal(old_head, model.aggregate_head.weight)):
            raise ValueError(f"aggregate_task_parameter_update_missing:{seed}:{fold_id}")
        model.eval()
        with torch.no_grad():
            valid_loss = float(aggregate_loss(model, aggregate_views, standardized, valid_indices).item())
        epoch_rows.append({"stage": "aggregate_base", "seed": seed, "fold_id": fold_id,
                           "epoch": epoch, "train_mse": float(loss.item()), "validation_mse": valid_loss})
        if valid_loss < best_loss - float(config["min_delta"]):
            best_loss, best_epoch, bad_epochs, best_state = valid_loss, epoch, 0, clone_state(model)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    return model, standardized, target_mean, target_std, best_epoch, best_loss, gradient_info


def report_init_state(base_state, bounds, report_targets, report_weights, seed, fold_id):
    model = DualTaskFusion().to(report_targets.device)
    model.load_state_dict(base_state)
    info = init_report_head(model, report_targets, report_weights, bounds)
    return model, info


def adapt_report(torch, model, arm, report_views, report_targets, report_train_idx,
                 report_valid_idx, train_weights, valid_weights, aggregate_views,
                 aggregate_standardized, aggregate_fit_idx, report_config, bounds,
                 seed, fold_id, epoch_rows):
    if arm == "frozen_fusion":
        for parameter in model.shared.parameters():
            parameter.requires_grad_(False)
        for parameter in model.aggregate_head.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": model.report_head.parameters(),
                   "lr": float(report_config["output_learning_rate"])}]
    else:
        groups = [
            {"params": model.shared.parameters(), "lr": float(report_config["fusion_learning_rate"])},
            {"params": model.aggregate_head.parameters(), "lr": float(report_config["output_learning_rate"])},
            {"params": model.report_head.parameters(), "lr": float(report_config["output_learning_rate"])},
        ]
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train_y, valid_y = report_targets[report_train_idx], report_targets[report_valid_idx]
    valid_formula = report_views[0, report_valid_idx]
    with torch.no_grad():
        initial_val = report_validation_loss(model, valid_formula, valid_y, valid_weights, bounds)
    best_loss, best_epoch, bad_epochs, best_state = initial_val, 0, 0, clone_state(model)
    epoch_rows.append({"stage": "report_adaptation", "arm": arm, "seed": seed,
                       "fold_id": fold_id, "epoch": 0, "train_report_nll": "",
                       "train_replay_mse": "", "validation_formula_nll": initial_val})
    gradient_rows = []
    changed_report = False
    changed_shared = False
    changed_aggregate = False
    for epoch in range(1, int(report_config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_x = report_views[:, report_train_idx]
        train_loss = report_loss(model, train_x, train_y, train_weights, bounds)
        replay = None
        if arm == "joint_replay":
            replay = aggregate_loss(model, aggregate_views, aggregate_standardized, aggregate_fit_idx)
            loss = train_loss + float(report_config["replay_lambda"]) * replay
        else:
            loss = train_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite_report_loss:{arm}:{seed}:{fold_id}:{epoch}")
        if epoch == 1:
            gradients = assert_report_gradients(
                model, train_x, train_y, train_weights, bounds, arm, f"{arm}:{seed}:{fold_id}:epoch1")
            gradient_rows.append({"stage": "report_gradient", "arm": arm, "seed": seed,
                                  "fold_id": fold_id, **gradients})
            if replay is not None:
                replay_gradients = assert_aggregate_gradients(
                    model, aggregate_views[:, aggregate_fit_idx],
                    aggregate_standardized[aggregate_fit_idx], f"joint_replay:{seed}:{fold_id}")
                gradient_rows.append({"stage": "joint_replay_gradient", "arm": arm,
                                      "seed": seed, "fold_id": fold_id, **replay_gradients})
            old_report = [value.detach().clone() for value in model.report_head.parameters()]
            old_shared = [value.detach().clone() for value in model.shared.parameters()]
            old_aggregate = [value.detach().clone() for value in model.aggregate_head.parameters()]
        loss.backward()
        optimizer.step()
        if epoch == 1:
            changed_report = any(not torch.equal(old, new) for old, new in zip(old_report, model.report_head.parameters()))
            changed_shared = any(not torch.equal(old, new) for old, new in zip(old_shared, model.shared.parameters()))
            changed_aggregate = any(not torch.equal(old, new) for old, new in zip(old_aggregate, model.aggregate_head.parameters()))
            if not changed_report or (arm == "joint_replay" and (not changed_shared or not changed_aggregate)):
                raise ValueError(f"report_adaptation_parameter_update_missing:{arm}:{seed}:{fold_id}")
            if arm == "frozen_fusion" and (changed_shared or changed_aggregate):
                raise ValueError(f"frozen_parameter_changed:{seed}:{fold_id}")
        model.eval()
        with torch.no_grad():
            val_loss = report_validation_loss(model, valid_formula, valid_y, valid_weights, bounds)
        epoch_rows.append({"stage": "report_adaptation", "arm": arm, "seed": seed,
                           "fold_id": fold_id, "epoch": epoch,
                           "train_report_nll": float(train_loss.item()),
                           "train_replay_mse": "" if replay is None else float(replay.item()),
                           "validation_formula_nll": val_loss})
        if val_loss < best_loss - float(report_config["min_delta"]):
            best_loss, best_epoch, bad_epochs, best_state = val_loss, epoch, 0, clone_state(model)
        else:
            bad_epochs += 1
            if bad_epochs >= int(report_config["patience"]):
                break
        if arm == "joint_replay" and epoch == 1:
            # The zero-initialized report output has no first-step gradient into the trunk.
            # Confirm that the warmed report head connects report loss to the shared layer.
            warmed = assert_report_gradients(
                model, train_x, train_y, train_weights, bounds, arm,
                f"{arm}:{seed}:{fold_id}:warmed_report_head")
            gradient_rows.append({"stage": "warmed_report_gradient", "arm": arm,
                                  "seed": seed, "fold_id": fold_id, **warmed})
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, epoch, gradient_rows, {
        "report_head_updated": changed_report, "shared_updated": changed_shared,
        "aggregate_head_updated": changed_aggregate,
    }


def aggregate_retention(torch, t4, model, aggregate_views, target_values, standardized,
                        target_mean, target_std, indices, seed, fold_id, arm, state):
    rows = []
    targets = target_values[indices].to(device=aggregate_views.device, dtype=torch.float64)
    actual_log = torch.log1p(targets)
    with torch.no_grad():
        for view_index in (0, 3):
            pred_standardized = model.aggregate(aggregate_views[view_index, indices])
            pred_log = pred_standardized * target_std + target_mean
            mse = torch.mean((pred_log.to(torch.float64) - actual_log) ** 2)
            estimate = torch.clamp_min(torch.expm1(pred_log.to(torch.float64)), 0.0)
            mae = torch.mean(torch.abs(estimate - targets))
            rows.append({"arm": arm, "seed": seed, "fold_id": fold_id, "state": state,
                         "view": VIEWS[view_index], "validation_records": int(indices.numel()),
                         "log1p_mse": float(mse.item()), "inverse_point_mae_ev": float(mae.item())})
    return rows


def make_prediction_rows(torch, t4, model, report_views, targets, valid_indices, valid_ids,
                         valid_weights_map, oof_weights, rows_by_id, bounds, arm, seed, fold_id):
    output_rows = []
    y = targets[valid_indices].to(device=report_views.device, dtype=torch.float64)
    fold_weights = [float(valid_weights_map[value]) for value in valid_ids]
    position_by_id = {record_id: index for index, record_id in enumerate(valid_ids)}
    with torch.no_grad():
        for view_index, view_name in enumerate(VIEWS):
            log_mu, sigma = model.report(report_views[view_index, valid_indices], bounds)
            mus, sigmas = log_mu.unsqueeze(1), sigma.unsqueeze(1)
            mixture_weights = torch.ones_like(mus, dtype=torch.float64)
            metrics = t4.distribution_scores(torch, y, mus, sigmas, mixture_weights)
            scored = t4.tensor_rows(metrics, y, valid_ids, fold_weights, mus, sigmas,
                                    mixture_weights, "dual_task_mlp", seed, fold_id)
            for row in scored:
                source = rows_by_id[row["measurement_id"]]
                position = position_by_id[row["measurement_id"]]
                row_log_mu = float(log_mu[position].item())
                row_sigma = float(sigma[position].item())
                row.update({"arm": arm, "view": view_name, "formula": source["formula"],
                            "composition_key": source["composition_key"],
                            "oof_weight": float(oof_weights[row["measurement_id"]]),
                            "log_mu_value": row_log_mu, "sigma_value": row_sigma,
                            "median_ev": math.exp(row_log_mu),
                            "sg_supported": int(source["_sg_supported"]),
                            "structure_supported": int(source["_structure_supported"])})
            output_rows.extend(scored)
    return output_rows


def normalized_weights(rows, weight_key):
    values = [float(row[weight_key]) for row in rows]
    total = sum(values)
    if total <= 0:
        raise ValueError("empty_score_weight_support")
    return [value / total for value in values]


def summarize(t4, prediction_rows):
    pooled, formula_scores, paired, fold_scores = [], [], [], []
    fold_groups = {}
    for row in prediction_rows:
        fold_groups.setdefault((row["arm"], int(row["seed"]), row["fold_id"], row["view"]), []).append(row)
    for key, rows in fold_groups.items():
        fold_scores.append({"arm": key[0], "seed": key[1], "fold_id": key[2], "view": key[3],
                            "records": len(rows), **t4.weighted_summary(
                                rows, [float(row["fold_validation_weight"]) for row in rows])})
    grouped = {}
    for row in prediction_rows:
        grouped.setdefault((row["arm"], int(row["seed"]), row["view"]), []).append(row)
    by_id = {}
    for key, rows in grouped.items():
        if len(rows) != 117 or len({row["measurement_id"] for row in rows}) != 117:
            raise ValueError(f"oof_view_support_mismatch:{key}:{len(rows)}")
        weights = normalized_weights(rows, "oof_weight")
        pooled.append({"arm": key[0], "seed": key[1], "view": key[2], "scope": "all117",
                       "records": len(rows), **t4.weighted_summary(rows, weights)})
        by_id[key] = {row["measurement_id"]: row for row in rows}
        masks = {
            "sg_supported": lambda row: int(row["sg_supported"]) == 1,
            "structure_supported": lambda row: int(row["structure_supported"]) == 1,
            "any_optional_supported": lambda row: int(row["sg_supported"]) == 1 or int(row["structure_supported"]) == 1,
            "both_optional_supported": lambda row: int(row["sg_supported"]) == 1 and int(row["structure_supported"]) == 1,
        }
        for scope, predicate in masks.items():
            subset = [row for row in rows if predicate(row)]
            if subset:
                pooled.append({"arm": key[0], "seed": key[1], "view": key[2], "scope": scope,
                               "records": len(subset), **t4.weighted_summary(
                                   subset, normalized_weights(subset, "oof_weight"))})
        by_formula = {}
        for row in rows:
            by_formula.setdefault(row["formula"], []).append(row)
        for formula, subset in by_formula.items():
            formula_scores.append({"arm": key[0], "seed": key[1], "view": key[2],
                                   "formula": formula, "records": len(subset),
                                   **t4.weighted_summary(subset, normalized_weights(subset, "oof_weight"))})

    comparison_views = (("formula_sg", "formula_only", "sg_supported"),
                        ("formula_structure", "formula_only", "structure_supported"),
                        ("full", "formula_only", "any_optional_supported"))
    for arm in ("frozen_fusion", "joint_replay"):
        for seed in (17, 29, 41):
            for candidate, baseline, scope in comparison_views:
                left, right = by_id[(arm, seed, candidate)], by_id[(arm, seed, baseline)]
                ids = sorted(record_id for record_id in left
                             if (int(left[record_id]["sg_supported"]) == 1 if scope == "sg_supported" else
                                 int(left[record_id]["structure_supported"]) == 1 if scope == "structure_supported" else
                                 (int(left[record_id]["sg_supported"]) == 1 or int(left[record_id]["structure_supported"]) == 1)))
                if not ids:
                    continue
                weights = [float(left[record_id]["oof_weight"]) for record_id in ids]
                weights = [weight / sum(weights) for weight in weights]
                for metric in ("nll", "crps_ev", "median_mae_ev"):
                    paired.append({"arm": arm, "seed": seed, "comparison": f"{candidate}_minus_{baseline}",
                                   "scope": scope, "records": len(ids), "metric": metric,
                                   "weighted_delta": sum(weight * (float(left[record_id][metric]) -
                                                                     float(right[record_id][metric]))
                                                         for record_id, weight in zip(ids, weights))})
                for metric, out_name in (("log_mu_value", "log_mu"), ("sigma_value", "sigma"),
                                         ("median_ev", "median_ev")):
                    def row_value(row):
                        return float(row[metric])
                    paired.append({"arm": arm, "seed": seed, "comparison": f"{candidate}_minus_{baseline}",
                                   "scope": scope, "records": len(ids), "metric": f"delta_{out_name}",
                                   "weighted_delta": sum(weight * (row_value(left[record_id]) -
                                                                     row_value(right[record_id]))
                                                         for record_id, weight in zip(ids, weights))})

    for arm in ("frozen_fusion", "joint_replay"):
        for view in VIEWS:
            for scope in ("all117", "sg_supported", "structure_supported", "any_optional_supported", "both_optional_supported"):
                seed_rows = [row for row in pooled if row["arm"] == arm and row["view"] == view and row["scope"] == scope]
                if len(seed_rows) == 3:
                    pooled.append({"arm": arm, "seed": "three_seed_mean", "view": view, "scope": scope,
                                   "records": seed_rows[0]["records"],
                                   **{metric: sum(float(row[metric]) for row in seed_rows) / 3
                                      for metric in SCORE_NAMES}})
    return pooled, formula_scores, paired, fold_scores


def run(args):
    root = Path(args.root).resolve()
    protocol_path = resolve(root, args.protocol)
    protocol = read_json(protocol_path)
    verify_contract(protocol)
    data = verify_data(root, protocol, protocol_path)
    summary = {
        "status": "preflight_complete", "protocol_sha256": sha256(protocol_path),
        "t10_data_hashes": data["data_hashes"], "t4_input_hashes": data["t4_hashes"],
        "aggregate_records": len(data["aggregate_meta"]),
        "aggregate_positive": int((data["aggregate_targets"] > 0).sum().item()),
        "aggregate_zero": int((data["aggregate_targets"] == 0).sum().item()),
        "report_records": len(data["report_records"]), "report_folds": data["fold_ids"],
        "report_sg_supported": int(data["reports"]["sg_mask"].sum().item()),
        "report_structure_supported": int(data["reports"]["structure_mask"].sum().item()),
        "protected_compositions": len(data["protected"]),
        "training_started": False,
    }
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return

    runtime = protocol["runtime"]
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_by_t10_protocol")
    torch.set_num_threads(int(runtime["num_threads"]))
    torch.use_deterministic_algorithms(bool(runtime["deterministic"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
    device = torch.device(runtime["device"])
    t4 = data["t4"]
    report_records, report_meta = data["report_records"], data["report_meta"]
    report_ids = [row["measurement_id"] for row in report_records]
    rows_by_id = {row["measurement_id"]: row for row in report_records}
    report_index = {record_id: index for index, record_id in enumerate(report_ids)}
    report_targets = data["report_targets"].to(device=device)
    aggregate_targets = data["aggregate_targets"].to(device=device, dtype=torch.float32)
    aggregate_meta, aggregate_features = data["aggregate_meta"], data["aggregate"]
    report_features = data["reports"]
    for index, row in enumerate(report_records):
        row["_sg_supported"] = int(float(report_features["sg_mask"][index].item()) > 0)
        row["_structure_supported"] = int(float(report_features["structure_mask"][index].item()) > 0)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    prediction_rows, fit_rows, epoch_rows, retention_rows, exclusion_rows = [], [], [], [], []
    gradient_rows, checkpoint_rows = [], []
    completed_base, completed_report = 0, 0
    aggregate_config, report_config = protocol["aggregate"], protocol["report"]
    bounds = tuple(float(value) for value in protocol["model"]["scale_bounds"])
    aggregate_keys = {row["composition_key"] for row in aggregate_meta}

    for seed in protocol["seeds"]:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        for fold_id in data["fold_ids"]:
            valid_ids = [record_id for record_id in report_ids if data["fold_by_id"][record_id] == fold_id]
            train_ids = [record_id for record_id in report_ids if data["fold_by_id"][record_id] != fold_id]
            valid_compositions = {rows_by_id[record_id]["composition_key"] for record_id in valid_ids}
            train_weights_map = data["fold_weights"][(fold_id, "train")]
            valid_weights_map = data["fold_weights"][(fold_id, "validation")]
            train_indices = torch.tensor([report_index[value] for value in train_ids], device=device)
            valid_indices = torch.tensor([report_index[value] for value in valid_ids], device=device)
            train_weights = torch.tensor([train_weights_map[value] for value in train_ids],
                                         device=device, dtype=torch.float64)
            valid_weights = torch.tensor([valid_weights_map[value] for value in valid_ids],
                                         device=device, dtype=torch.float64)
            aggregate_candidates = [index for index, row in enumerate(aggregate_meta)
                             if row["composition_key"] not in data["protected"]
                             and row["composition_key"] not in valid_compositions]
            if not aggregate_candidates:
                raise ValueError(f"empty_aggregate_fit:{seed}:{fold_id}")
            aggregate_fold_excluded = len(aggregate_meta) - len(aggregate_candidates)
            hash_order = sorted(aggregate_candidates, key=lambda index: hashlib.sha256(
                ("t10_aux_validation|" + aggregate_meta[index]["composition_key"]).encode("utf-8")).hexdigest())
            validation_count = math.ceil(float(aggregate_config["validation_fraction"]) * len(hash_order))
            aggregate_validation = hash_order[:validation_count]
            aggregate_fit = hash_order[validation_count:]
            if not aggregate_fit or not aggregate_validation:
                raise ValueError(f"aggregate_internal_split_empty:{seed}:{fold_id}")
            aggregate_fit_idx = torch.tensor(aggregate_fit, device=device, dtype=torch.long)
            aggregate_validation_idx = torch.tensor(aggregate_validation, device=device, dtype=torch.long)

            scaler = fold_scalers(aggregate_features, aggregate_meta, aggregate_fit,
                                  report_features, report_meta, train_ids, report_records,
                                  report_index, device)
            comp_mean, comp_std, structure_mean, structure_std, scale_info = scaler
            aggregate_views = four_views(aggregate_features, comp_mean, comp_std,
                                         structure_mean, structure_std, device)
            report_views = four_views(report_features, comp_mean, comp_std,
                                      structure_mean, structure_std, device)
            log_aggregate = torch.log1p(aggregate_targets)
            target_mean = log_aggregate[aggregate_fit_idx].mean()
            target_std = log_aggregate[aggregate_fit_idx].std(unbiased=False)
            if float(target_std.item()) <= 0:
                target_std = torch.ones_like(target_std)
            aggregate_standardized = (log_aggregate - target_mean) / target_std

            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            base_model = DualTaskFusion().to(device)
            with torch.no_grad():
                base_model.report_head.weight.zero_()
                base_model.report_head.bias.zero_()
            history_start = len(epoch_rows)
            base_model, aggregate_standardized, target_mean, target_std, base_epoch, base_best, base_gradients = train_aggregate(
                torch, base_model, aggregate_views, aggregate_targets, aggregate_fit_idx,
                aggregate_validation_idx, aggregate_config, int(seed), fold_id, epoch_rows)
            gradient_rows.append({"stage": "aggregate_gradient", "seed": seed,
                                  "fold_id": fold_id, **(base_gradients or {})})
            aggregate_fit_ids = [aggregate_meta[index]["record_id"] for index in aggregate_fit]
            aggregate_validation_ids = [aggregate_meta[index]["record_id"] for index in aggregate_validation]
            base_name = f"aggregate_base_seed_{seed}_fold_{fold_id}.pt"
            base_path = checkpoint_dir / base_name
            base_checkpoint = {
                "state_dict": clone_state(base_model), "model_config": protocol["model"],
                "seed": seed, "fold_id": fold_id, "best_epoch": base_epoch,
                "best_validation_standardized_mse": base_best,
                "aggregate_fit_ids": aggregate_fit_ids,
                "aggregate_internal_validation_ids": aggregate_validation_ids,
                "report_fold_validation_ids": valid_ids,
                "composition_mean": comp_mean.detach().cpu(), "composition_std": comp_std.detach().cpu(),
                "structure_mean": structure_mean.detach().cpu(), "structure_std": structure_std.detach().cpu(),
                "aggregate_log1p_mean": float(target_mean.item()),
                "aggregate_log1p_std": float(target_std.item()),
                "normalization": scale_info,
                "gradient_checks": base_gradients,
                "trainable_parameter_count": sum(p.numel() for p in
                                                  list(base_model.shared.parameters())
                                                  + list(base_model.aggregate_head.parameters())),
                "torch_version": str(torch.__version__), "device": str(device),
            }
            torch.save(base_checkpoint, base_path)
            replay_base = torch.load(base_path, map_location=device, weights_only=True)
            replay_model = DualTaskFusion().to(device)
            replay_model.load_state_dict(replay_base["state_dict"])
            replay_model.eval()
            base_replay_difference = 0.0
            with torch.no_grad():
                original_agg = base_model.aggregate(aggregate_views[:, aggregate_validation_idx].reshape(-1, 582))
                replay_agg = replay_model.aggregate(aggregate_views[:, aggregate_validation_idx].reshape(-1, 582))
                base_replay_difference = float((original_agg - replay_agg).abs().max().item())
            if base_replay_difference > 1e-7:
                raise ValueError(f"aggregate_checkpoint_replay_mismatch:{seed}:{fold_id}")
            checkpoint_rows.append({"stage": "aggregate_base", "seed": seed, "fold_id": fold_id,
                                    "path": base_path.relative_to(output_dir).as_posix(),
                                    "sha256": sha256(base_path),
                                    "replay_max_abs_difference": base_replay_difference})
            completed_base += 1
            fit_rows.append({
                "stage": "aggregate_base", "seed": seed, "fold_id": fold_id,
                "aggregate_fit_records": len(aggregate_fit),
                "aggregate_internal_validation_records": len(aggregate_validation),
                "aggregate_fold_excluded_records": aggregate_fold_excluded,
                "aggregate_fit_compositions": len({aggregate_meta[index]["composition_key"] for index in aggregate_fit}),
                **scale_info, "best_epoch": base_epoch,
                "best_validation_standardized_mse": base_best,
                "trainable_parameter_count": base_checkpoint["trainable_parameter_count"],
                "total_parameter_count": sum(p.numel() for p in base_model.parameters()),
                "composition_input_gradient_l1": base_gradients["composition_input_gradient_l1"],
                "sg_input_gradient_l1": base_gradients["sg_input_gradient_l1"],
                "structure_input_gradient_l1": base_gradients["structure_input_gradient_l1"],
                "history_rows": len(epoch_rows) - history_start,
                "checkpoint": base_path.relative_to(output_dir).as_posix(),
            })
            retention_rows.extend(aggregate_retention(
                torch, t4, base_model, aggregate_views, aggregate_targets,
                aggregate_standardized, target_mean, target_std, aggregate_validation_idx,
                seed, fold_id, "shared_base", "before_report_adaptation"))
            report_init, init_info = report_init_state(
                clone_state(base_model), bounds, report_targets[train_indices], train_weights, seed, fold_id)
            initial_report_state = clone_state(report_init)
            arm_initial_hash = state_hash(initial_report_state)

            for arm in protocol["arms"]:
                model = DualTaskFusion().to(device)
                model.load_state_dict(initial_report_state)
                for name, value in model.state_dict().items():
                    if not torch.equal(value.detach().cpu(), initial_report_state[name]):
                        raise ValueError(f"arm_initialization_mismatch:{arm}:{seed}:{fold_id}:{name}")
                arm_epoch_start = len(epoch_rows)
                model, best_epoch, best_nll, epochs_run, arm_gradients, update_info = adapt_report(
                    torch, model, arm, report_views, report_targets, train_indices, valid_indices,
                    train_weights, valid_weights, aggregate_views, aggregate_standardized,
                    aggregate_fit_idx, report_config, bounds, int(seed), fold_id, epoch_rows)
                gradient_rows.extend(arm_gradients)
                if arm == "frozen_fusion":
                    for key in ("shared.weight", "shared.bias", "aggregate_head.weight", "aggregate_head.bias"):
                        if not torch.equal(model.state_dict()[key].detach().cpu(),
                                           initial_report_state[key]):
                            raise ValueError(f"frozen_shared_changed:{seed}:{fold_id}:{key}")

                checkpoint_name = f"{arm}_seed_{seed}_fold_{fold_id}.pt"
                checkpoint_path = checkpoint_dir / checkpoint_name
                checkpoint = {
                    "state_dict": clone_state(model), "model_config": protocol["model"],
                    "arm": arm, "seed": seed, "fold_id": fold_id,
                    "best_epoch": best_epoch, "best_validation_formula_nll": best_nll,
                    "report_train_ids": train_ids, "report_validation_ids": valid_ids,
                    "aggregate_fit_ids": aggregate_fit_ids,
                    "aggregate_internal_validation_ids": aggregate_validation_ids,
                    "composition_mean": comp_mean.detach().cpu(), "composition_std": comp_std.detach().cpu(),
                    "structure_mean": structure_mean.detach().cpu(), "structure_std": structure_std.detach().cpu(),
                    "aggregate_log1p_mean": float(target_mean.item()),
                    "aggregate_log1p_std": float(target_std.item()),
                    "report_initialization": init_info,
                    "initial_state_sha256": arm_initial_hash,
                    "normalization": scale_info,
                    "frozen_fusion": arm == "frozen_fusion",
                    "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "total_parameter_count": sum(p.numel() for p in model.parameters()),
                    "epochs_run": epochs_run,
                    "parameter_updates": update_info,
                    "torch_version": str(torch.__version__), "device": str(device),
                }
                torch.save(checkpoint, checkpoint_path)
                replay = torch.load(checkpoint_path, map_location=device, weights_only=True)
                replay_model = DualTaskFusion().to(device)
                replay_model.load_state_dict(replay["state_dict"])
                replay_model.eval()
                with torch.no_grad():
                    original = model.report(report_views[:, valid_indices].reshape(-1, 582), bounds)
                    reloaded = replay_model.report(report_views[:, valid_indices].reshape(-1, 582), bounds)
                    differences = [float((left - right).abs().max().item()) for left, right in zip(original, reloaded)]
                if max(differences) > 1e-7:
                    raise ValueError(f"report_checkpoint_replay_mismatch:{arm}:{seed}:{fold_id}")
                checkpoint_rows.append({"stage": "report_adaptation", "arm": arm, "seed": seed,
                                        "fold_id": fold_id,
                                        "path": checkpoint_path.relative_to(output_dir).as_posix(),
                                        "sha256": sha256(checkpoint_path),
                                        "replay_max_abs_output_difference": max(differences)})
                retention_rows.extend(aggregate_retention(
                    torch, t4, replay_model, aggregate_views, aggregate_targets,
                    aggregate_standardized, target_mean, target_std, aggregate_validation_idx,
                    seed, fold_id, arm, "after_report_adaptation"))
                prediction_rows.extend(make_prediction_rows(
                    torch, t4, replay_model, report_views, report_targets, valid_indices, valid_ids,
                    valid_weights_map, data["oof_weights"], rows_by_id, bounds, arm, int(seed), fold_id))
                fit_rows.append({
                    "stage": "report_adaptation", "arm": arm, "seed": seed, "fold_id": fold_id,
                    "report_train_records": len(train_ids), "report_validation_records": len(valid_ids),
                    "aggregate_fit_records": len(aggregate_fit),
                    "aggregate_internal_validation_records": len(aggregate_validation),
                    "aggregate_fit_compositions": len({aggregate_meta[index]["composition_key"] for index in aggregate_fit}),
                    **scale_info, **init_info,
                    "aggregate_base_best_epoch": base_epoch,
                    "aggregate_base_best_validation_standardized_mse": base_best,
                    "best_report_epoch": best_epoch, "report_epochs_run": epochs_run,
                    "best_validation_formula_nll": best_nll,
                    "trainable_parameter_count": checkpoint["trainable_parameter_count"],
                    "total_parameter_count": checkpoint["total_parameter_count"],
                    "frozen_fusion": checkpoint["frozen_fusion"],
                    "checkpoint": checkpoint_path.relative_to(output_dir).as_posix(),
                    "history_rows": len(epoch_rows) - arm_epoch_start,
                })
                completed_report += 1
                print(json.dumps({"completed_base_fits": completed_base, "completed_report_fits": completed_report,
                                  "arm": arm, "seed": seed, "fold": fold_id,
                                  "best_epoch": best_epoch, "best_validation_formula_nll": best_nll},
                                 ensure_ascii=False), flush=True)

            for index, row in enumerate(aggregate_meta):
                reasons = []
                if row["composition_key"] in valid_compositions:
                    reasons.append("current_report_fold_validation_composition")
                if row["composition_key"] in data["protected"]:
                    reasons.append("frozen_report_nontrain_composition")
                if row["composition_key"] not in aggregate_keys:
                    reasons.append("not_in_aggregate_train_export")
                if reasons:
                    exclusion_rows.append({"seed": seed, "fold_id": fold_id,
                                           "record_id": row["record_id"],
                                           "composition_key": row["composition_key"],
                                           "reasons": ";".join(reasons)})

    if completed_base != int(protocol["budget"]["aggregate_base_fits"]):
        raise ValueError(f"aggregate_fit_count_mismatch:{completed_base}")
    if completed_report != int(protocol["budget"]["report_fits"]):
        raise ValueError(f"report_fit_count_mismatch:{completed_report}")
    expected_predictions = 117 * len(protocol["seeds"]) * len(protocol["arms"]) * len(VIEWS)
    if len(prediction_rows) != expected_predictions:
        raise ValueError(f"oof_view_row_count_mismatch:{len(prediction_rows)}:{expected_predictions}")
    pooled, formula_scores, paired, fold_scores = summarize(t4, prediction_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "fold_scores.csv", fold_scores)
    write_csv(output_dir / "pooled_scores.csv", pooled)
    write_csv(output_dir / "per_formula_scores.csv", formula_scores)
    write_csv(output_dir / "paired_view_effects.csv", paired)
    write_csv(output_dir / "aggregate_retention.csv", retention_rows)
    write_csv(output_dir / "fit_manifest.csv", fit_rows)
    write_csv(output_dir / "epoch_history.csv", epoch_rows)
    write_csv(output_dir / "gradient_checks.csv", gradient_rows)
    write_csv(output_dir / "aggregate_exclusions.csv", exclusion_rows)
    write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    final_summary = {
        **summary, "status": "complete", "training_started": True,
        "completed_aggregate_base_fits": completed_base,
        "completed_report_fits": completed_report,
        "oof_prediction_rows": len(prediction_rows),
        "checkpoint_count": len(checkpoint_rows),
        "interpretation": protocol["evaluation"]["interpretation"],
        "outputs": ["predictions.csv", "pooled_scores.csv", "per_formula_scores.csv",
                    "fold_scores.csv", "paired_view_effects.csv", "aggregate_retention.csv", "fit_manifest.csv",
                    "epoch_history.csv", "gradient_checks.csv", "aggregate_exclusions.csv",
                    "checkpoint_manifest.csv", "checkpoints/"],
    }
    write_json(output_dir / "summary.json", final_summary)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                      "aggregate_base_fits": completed_base, "report_fits": completed_report,
                      "oof_prediction_rows": len(prediction_rows)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="verify registered inputs without training")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
