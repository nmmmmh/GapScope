"""Leak-free five-fold inputs for the 205-report and composition tasks.

``build_fold(root, fold, device)`` returns a dictionary with:

* ``views`` / ``av``: report / retained-aggregate four-view tensors, shaped
  ``[4, N, 635]`` in formula-only, formula+SG, formula+structure, full order.
* ``targets`` / ``ay``: positive report ``gap_ev`` targets and aggregate targets
  from ``features.pt`` (including zeros), as float64 tensors on ``device``.
* ``ti``, ``vi``, ``oi``: report train, validation, outer indices; ``fi`` and
  ``ai``: aggregate fit and validation indices. All are device-local long
  tensors. Aggregate indices are local to the physically retained pool.
* ``tw``, ``vw``, ``ow``: per-report hierarchical weights, normalized within
  each split. ``records`` merges feature metadata with source records, with
  ``records.jsonl`` taking precedence for targets and provenance.
* ``aggregate_metadata``: metadata for rows retained in ``av`` / ``ay``;
  ``scalers``: formula and structure means/stds plus scaler support counts;
  ``provenance``: fold assignment, aggregate filtering, and hash-split details.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import runtime as rt


FOLD_SEED = 20261011
_AUXILIARY_TARGET_FIELDS = {
    "additional_optical_edges", "fundamental_gap_eV", "secondary_same_sample_gap_ev",
    "source_conflict_alternative_gap_ev",
}
_MODULES = None


def _modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = rt.modules()
    return _MODULES


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_features(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"features_not_mapping:{path}")
    return value


def _load_records(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            for key in _AUXILIARY_TARGET_FIELDS:
                row.pop(key, None)
            rows.append(row)
    return rows


def _require_rows(value, path: Path):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"metadata_not_row_list:{path}")
    return value


def _check_features(features, metadata, label, has_targets=False):
    required = {"composition", "sg_features", "structure", "sg_mask", "structure_mask"}
    if has_targets:
        required.add("targets")
    missing = required - set(features)
    if missing:
        raise ValueError(f"{label}_feature_keys_missing:{sorted(missing)}")
    count = len(metadata)
    widths = {"composition": 183, "sg_features": 237, "structure": 160,
              "sg_mask": 1, "structure_mask": 1}
    for name, width in widths.items():
        tensor = features[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label}_feature_not_tensor:{name}")
        if tuple(tensor.shape) != (count, width):
            raise ValueError(f"{label}_feature_shape_mismatch:{name}:{tuple(tensor.shape)}:{count}x{width}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label}_nonfinite_feature:{name}")
    for name in ("sg_mask", "structure_mask"):
        tensor = features[name]
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"{label}_nonbinary_mask:{name}")
    if has_targets:
        targets = features["targets"].reshape(-1)
        if tuple(targets.shape) != (count,) or not torch.isfinite(targets).all() or torch.any(targets < 0):
            raise ValueError("invalid_aggregate_targets")


def _max_abs_difference(left, right):
    return float((left - right).abs().max().item())


def _audit_formula_features(aggregate, aggregate_metadata, reports, report_metadata):
    """Check formula embeddings across every same-key row before fitting scalers."""
    seen = {}
    for index, row in enumerate(aggregate_metadata):
        key = row["composition_key"]
        seen.setdefault(key, ("aggregate", index, row, aggregate["composition"][index]))
    max_difference = 0.0
    overlap = set(seen)
    for index, row in enumerate(report_metadata):
        key = row["composition_key"]
        vector = reports["composition"][index]
        previous = seen.get(key)
        if previous is None:
            seen[key] = ("report", index, row, vector)
            continue
        difference = _max_abs_difference(previous[3], vector)
        max_difference = max(max_difference, difference)
        if not torch.allclose(previous[3], vector, atol=1e-5, rtol=1e-5):
            raise ValueError(
                "formula_feature_mismatch:"
                f"key={key}:first_source={previous[0]}:first_id="
                f"{previous[2].get('record_id', previous[2].get('measurement_id'))}:"
                f"report_id={row.get('measurement_id', row.get('record_id'))}:"
                f"first_formula={previous[2].get('formula')}:report_formula={row.get('formula')}:"
                f"max_abs_diff={difference:.9g}"
            )
    return {"overlap_composition_keys": len(overlap & {
                row["composition_key"] for row in report_metadata}),
            "max_abs_difference": max_difference, "tolerance": 1e-5}


def _feature_subset(features, indices, count):
    return {key: value[indices] if isinstance(value, torch.Tensor) and value.ndim > 0
            and len(value) == count else value
            for key, value in features.items()}


def _fold_keys(report_metadata, fold):
    keys = sorted({row["composition_key"] for row in report_metadata})
    if len(keys) != 56:
        raise ValueError(f"unexpected_report_composition_count:{len(keys)}")
    random.Random(FOLD_SEED).shuffle(keys)
    assigned = [set(keys[index::5]) for index in range(5)]
    outer = assigned[fold]
    valid = assigned[(fold + 1) % 5]
    train = set().union(*[assigned[index] for index in range(5)
                          if index not in (fold, (fold + 1) % 5)])
    return assigned, train, valid, outer


def _hash_validation_key(key):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def _selected_indices(metadata, selected_keys):
    return [index for index, row in enumerate(metadata)
            if row["composition_key"] in selected_keys]


def _index_tensor(indices, device):
    return torch.tensor(indices, dtype=torch.long, device=device)


def hierarchical_weights(records, indices, device):
    """Equalize mass formula -> source -> specimen -> report within a split."""
    if isinstance(indices, torch.Tensor):
        selected = indices.detach().cpu().reshape(-1).tolist()
    else:
        selected = [int(index) for index in indices]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate_report_index_in_weight_split")
    if not selected:
        return torch.empty(0, dtype=torch.float64, device=device)

    groups = []
    sources_by_formula = defaultdict(set)
    specimens_by_source = defaultdict(set)
    reports_by_specimen = defaultdict(int)
    for index in selected:
        row = records[index]
        formula = row.get("composition_key")
        source = row.get("source_group_id")
        specimen = row.get("specimen_group_id")
        if not formula or not source or not specimen:
            raise ValueError(f"missing_report_weight_group:{index}:{formula}:{source}:{specimen}")
        group = (formula, source, specimen)
        groups.append(group)
        sources_by_formula[formula].add(source)
        specimens_by_source[(formula, source)].add(specimen)
        reports_by_specimen[group] += 1

    formula_count = len(sources_by_formula)
    weights = [
        1.0 / formula_count
        / len(sources_by_formula[formula])
        / len(specimens_by_source[(formula, source)])
        / reports_by_specimen[(formula, source, specimen)]
        for formula, source, specimen in groups
    ]
    result = torch.tensor(weights, dtype=torch.float64, device=device)
    result /= result.sum()
    return result


def build_fold(root, fold: int, device):
    """Load one fold without exposing held-out report formulas to aggregate fit."""
    if not isinstance(fold, int) or fold < 0 or fold > 4:
        raise ValueError(f"fold_out_of_range:{fold}")
    root = Path(root)
    data = root / "data"
    reports_dir = data / "reports"
    aggregate_dir = data / "composition"
    device = torch.device(device)

    report_features = _load_features(reports_dir / "features.pt")
    aggregate_all_features = _load_features(aggregate_dir / "features.pt")
    report_metadata = _require_rows(_read_json(reports_dir / "metadata.json"),
                                    reports_dir / "metadata.json")
    aggregate_all_metadata = _require_rows(_read_json(aggregate_dir / "metadata.json"),
                                           aggregate_dir / "metadata.json")
    source_records = _load_records(reports_dir / "records.jsonl")
    _check_features(report_features, report_metadata, "report")
    _check_features(aggregate_all_features, aggregate_all_metadata, "aggregate", has_targets=True)

    if len(source_records) != len(report_metadata):
        raise ValueError("report_record_metadata_count_mismatch")
    for index, (source, meta) in enumerate(zip(source_records, report_metadata)):
        if (source.get("measurement_id") != meta.get("measurement_id")
                or source.get("composition_key") != meta.get("composition_key")
                or source.get("formula") != meta.get("formula")):
            raise ValueError(f"report_record_metadata_order_mismatch:{index}")
    records = []
    targets_values = []
    for meta, source in zip(report_metadata, source_records):
        merged = dict(meta)
        merged.update(source)
        target = source.get("gap_ev")
        if not isinstance(target, (int, float)) or not math.isfinite(float(target)) or target <= 0:
            raise ValueError(f"invalid_report_gap_ev:{source.get('measurement_id')}:{target}")
        records.append(merged)
        targets_values.append(float(target))
    if len(records) != 205:
        raise ValueError(f"unexpected_report_count:{len(records)}")
    if len({row["measurement_id"] for row in records}) != len(records):
        raise ValueError("duplicate_report_measurement_id")

    formula_audit = _audit_formula_features(aggregate_all_features, aggregate_all_metadata,
                                           report_features, report_metadata)
    assigned, train_keys, valid_keys, outer_keys = _fold_keys(report_metadata, fold)
    held_out_keys = valid_keys | outer_keys

    # Remove held-out formulas from the aggregate pool itself before making indices.
    retained_original_indices = [index for index, row in enumerate(aggregate_all_metadata)
                                 if row["composition_key"] not in held_out_keys]
    removed_metadata = [row for row in aggregate_all_metadata
                        if row["composition_key"] in held_out_keys]
    aggregate_features = _feature_subset(aggregate_all_features, retained_original_indices,
                                         len(aggregate_all_metadata))
    aggregate_metadata = [aggregate_all_metadata[index] for index in retained_original_indices]
    aggregate_targets = aggregate_features["targets"].reshape(-1).to(dtype=torch.float64)
    if len({row["composition_key"] for row in aggregate_metadata}) != len(aggregate_metadata):
        raise ValueError("aggregate_rows_not_unique_by_composition")

    report_train_indices = _selected_indices(report_metadata, train_keys)
    report_valid_indices = _selected_indices(report_metadata, valid_keys)
    report_outer_indices = _selected_indices(report_metadata, outer_keys)
    train_aggregate_keys = train_keys & {row["composition_key"] for row in aggregate_metadata}
    aggregate_valid_indices = [
        index for index, row in enumerate(aggregate_metadata)
        if row["composition_key"] not in train_aggregate_keys
        and _hash_validation_key(row["composition_key"])
    ]
    aggregate_valid_set = set(aggregate_valid_indices)
    aggregate_fit_indices = [index for index in range(len(aggregate_metadata))
                             if index not in aggregate_valid_set]
    if not aggregate_fit_indices or not aggregate_valid_indices:
        raise ValueError("empty_aggregate_fit_or_validation_split")

    report_index = {row["measurement_id"]: index for index, row in enumerate(records)}
    modules = _modules()
    symmetry_table = modules["loader43"].symmetry_table()
    report_hybrid = modules["t35"].transform(report_features, "hybrid", symmetry_table, set())
    aggregate_hybrid = modules["t35"].transform(aggregate_features, "hybrid", symmetry_table, set())
    (comp_mean, comp_std, structure_mean, structure_std, scaler_stats) = modules["t10"].fold_scalers(
        aggregate_hybrid, aggregate_metadata, aggregate_fit_indices,
        report_hybrid, report_metadata,
        [records[index]["measurement_id"] for index in report_train_indices],
        records, report_index, device)
    views = modules["t10"].four_views(report_hybrid, comp_mean, comp_std,
                                     structure_mean, structure_std, device)
    aggregate_views = modules["t10"].four_views(aggregate_hybrid, comp_mean, comp_std,
                                                structure_mean, structure_std, device)
    if tuple(views.shape) != (4, len(records), 635):
        raise ValueError(f"report_four_view_shape_mismatch:{tuple(views.shape)}")
    if tuple(aggregate_views.shape) != (4, len(aggregate_metadata), 635):
        raise ValueError(f"aggregate_four_view_shape_mismatch:{tuple(aggregate_views.shape)}")

    report_targets = torch.tensor(targets_values, dtype=torch.float64, device=device)
    aggregate_targets = aggregate_targets.to(device=device)
    ti = _index_tensor(report_train_indices, device)
    vi = _index_tensor(report_valid_indices, device)
    oi = _index_tensor(report_outer_indices, device)
    fi = _index_tensor(aggregate_fit_indices, device)
    ai = _index_tensor(aggregate_valid_indices, device)
    tw = hierarchical_weights(records, ti, device)
    vw = hierarchical_weights(records, vi, device)
    ow = hierarchical_weights(records, oi, device)

    aggregate_keys = {row["composition_key"] for row in aggregate_metadata}
    if aggregate_keys & held_out_keys:
        raise ValueError("held_out_report_composition_retained_in_aggregate_pool")
    if {aggregate_metadata[index]["composition_key"] for index in aggregate_valid_indices} & train_keys:
        raise ValueError("report_train_composition_in_aggregate_validation")
    if any(abs(float(weights.sum().item()) - 1.0) > 1e-12 for weights in (tw, vw, ow)):
        raise ValueError("report_split_weights_not_normalized")

    scalers = {
        "composition_mean": comp_mean, "composition_std": comp_std,
        "structure_mean": structure_mean, "structure_std": structure_std,
        **scaler_stats,
    }
    provenance = {
        "schema": "training_data_fold_v1", "fold": fold, "fold_seed": FOLD_SEED,
        "report_fold_rule": "sorted composition_key values shuffled with random.Random(seed), then round-robin across five folds",
        "report_fold_composition_keys": [sorted(keys) for keys in assigned],
        "report_train_composition_keys": sorted(train_keys),
        "report_valid_composition_keys": sorted(valid_keys),
        "report_outer_composition_keys": sorted(outer_keys),
        "aggregate_pool_rule": "physically remove aggregate rows whose composition_key is report valid or outer",
        "aggregate_removed_report_valid_outer_rows": len(removed_metadata),
        "aggregate_removed_report_valid_outer_composition_keys": sorted(held_out_keys),
        "aggregate_validation_rule": "sha256(composition_key UTF-8) first 8 bytes modulo 10 equals zero, excluding report-train formulas",
        "aggregate_forced_fit_report_train_keys": sorted(train_aggregate_keys),
        "report_counts": {"train": len(report_train_indices), "valid": len(report_valid_indices),
                          "outer": len(report_outer_indices)},
        "aggregate_counts": {"retained": len(aggregate_metadata), "fit": len(aggregate_fit_indices),
                             "valid": len(aggregate_valid_indices)},
        "formula_feature_audit": formula_audit,
    }
    return {
        "views": views, "av": aggregate_views,
        "targets": report_targets, "ay": aggregate_targets,
        "ti": ti, "vi": vi, "oi": oi, "fi": fi, "ai": ai,
        "tw": tw, "vw": vw, "ow": ow,
        "records": records, "report_metadata": report_metadata,
        "aggregate_metadata": aggregate_metadata,
        "scalers": scalers, "provenance": provenance,
    }
