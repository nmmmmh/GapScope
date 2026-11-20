"""Inventory same-formula actual-CIF and SG-only pair coverage by fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pack(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_group_weights(group_indices, split_weight_by_index):
    """Normalize build_fold split weights within one formula/CIF group."""
    total = sum(split_weight_by_index[index] for index in group_indices)
    if total <= 0:
        raise ValueError("empty_group_weight")
    return {index: split_weight_by_index[index] / total for index in group_indices}


def pair_groups(groups, prefix):
    pairs = []
    by_formula = defaultdict(list)
    for group_key in groups:
        by_formula[group_key[0]].append(group_key)
    for formula_key, formula_groups in sorted(by_formula.items()):
        for group_a, group_b in itertools.combinations(sorted(formula_groups), 2):
            pair_id = f"{prefix}|{formula_key}|{group_a[1]}|{group_b[1]}"
            pairs.append((pair_id, formula_key, group_a, group_b))
    return pairs


def group_side(records, metadata, resolved, indices, weights, group_value, side):
    normalized = condition_group_weights(indices, weights)
    report_rows = []
    for index in indices:
        record = records[index]
        meta = metadata[index]
        resolution = resolved.get(record["measurement_id"], {})
        report_rows.append({
            "report_index": index,
            "measurement_id": record["measurement_id"],
            "split_weight": weights[index],
            "conditioned_weight": normalized[index],
            "phase_label": record.get("phase_label", ""),
            "source_group_id": record.get("source_group_id", ""),
            "specimen_group_id": record.get("specimen_group_id", ""),
            "space_group_number": meta.get("model_space_group_number", ""),
            "structure_relation": resolution.get("structure_relation", meta.get("model_structure_relation", "")),
            "actual_cif_sha256": resolution.get("actual_cif_sha256", ""),
            "input_support": resolution.get("input_support", "SG-only report"),
        })
    report_rows.sort(key=lambda row: row["measurement_id"])
    ids = [row["measurement_id"] for row in report_rows]
    context = {
        "group_value": str(group_value),
        "reports": len(report_rows),
        "report_ids": ids,
        "report_weights": [{"measurement_id": row["measurement_id"],
                             "weight": row["conditioned_weight"]} for row in report_rows],
        "phase_labels": sorted({row["phase_label"] for row in report_rows if row["phase_label"]}),
        "source_group_ids": sorted({row["source_group_id"] for row in report_rows if row["source_group_id"]}),
        "specimen_group_ids": sorted({row["specimen_group_id"] for row in report_rows if row["specimen_group_id"]}),
        "space_group_numbers": sorted({str(row["space_group_number"]) for row in report_rows if row["space_group_number"]}),
        "structure_relations": sorted({row["structure_relation"] for row in report_rows if row["structure_relation"]}),
        "input_support": sorted({row["input_support"] for row in report_rows if row["input_support"]}),
    }
    detail_rows = [{
        "side": side,
        "group_value": str(group_value),
        **row,
    } for row in report_rows]
    return context, detail_rows, report_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root.parent / "BandgapFusion_PhaseWork" / "inventory").resolve()
    if output.is_relative_to(root):
        raise ValueError("inventory_output_must_be_outside_formal_root")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "runtime"))
    import torch
    from training_data import build_fold

    torch.set_num_threads(2)
    reports_dir = root / "data" / "reports"
    physical_dir = root / "results" / "physical_cases"
    metadata = read_json(reports_dir / "metadata.json")
    records = [json.loads(line) for line in (reports_dir / "records.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    features = torch.load(reports_dir / "features.pt", map_location="cpu", weights_only=True)
    structure_mask = [int(value) for value in features["structure_mask"].reshape(-1).tolist()]
    sg_mask = [int(value) for value in features["sg_mask"].reshape(-1).tolist()]
    resolved_rows = read_csv(physical_dir / "resolved_structure_index.csv")
    resolved = {row["measurement_id"]: row for row in resolved_rows}
    record_index = {row["measurement_id"]: index for index, row in enumerate(records)}
    if len(metadata) != len(records) or len(records) != len(structure_mask) or len(records) != len(sg_mask):
        raise ValueError("report_feature_metadata_count_mismatch")
    for index, (meta, record) in enumerate(zip(metadata, records)):
        if meta["measurement_id"] != record["measurement_id"]:
            raise ValueError(f"report_order_mismatch:{index}")

    structure_ids = {records[index]["measurement_id"] for index, value in enumerate(structure_mask) if value}
    if structure_ids != set(resolved):
        raise ValueError("resolved_structure_ids_do_not_match_structure_feature_mask")
    hash_by_path = {}
    for measurement_id, row in resolved.items():
        index = record_index[measurement_id]
        if row["composition_key"] != records[index]["composition_key"]:
            raise ValueError(f"resolved_formula_mismatch:{measurement_id}")
        cif_path = (root / row["cif_path"]).resolve()
        if not cif_path.is_relative_to(root):
            raise ValueError(f"resolved_cif_outside_root:{measurement_id}")
        actual_hash = row["actual_cif_sha256"].lower()
        if len(actual_hash) != 64:
            raise ValueError(f"invalid_actual_cif_sha256:{measurement_id}")
        if cif_path not in hash_by_path:
            hash_by_path[cif_path] = sha256(cif_path)
        if hash_by_path[cif_path] != actual_hash:
            raise ValueError(f"actual_cif_byte_hash_mismatch:{measurement_id}")
        if metadata[index].get("cif_sha256") and metadata[index]["cif_sha256"].lower() != actual_hash:
            raise ValueError(f"metadata_actual_cif_hash_mismatch:{measurement_id}")

    cif_groups = defaultdict(list)
    sg_groups = defaultdict(list)
    for index, record in enumerate(records):
        if structure_mask[index]:
            row = resolved[record["measurement_id"]]
            cif_groups[(record["composition_key"], row["actual_cif_sha256"].lower())].append(index)
        elif sg_mask[index]:
            sg_number = metadata[index].get("model_space_group_number")
            if sg_number in (None, ""):
                raise ValueError(f"sg_feature_missing_space_group:{record['measurement_id']}")
            sg_groups[(record["composition_key"], int(sg_number))].append(index)
    cif_pairs = pair_groups(cif_groups, "CIF")
    sg_pairs = pair_groups(sg_groups, "SG_ONLY")

    existing_pairs = read_csv(physical_dir / "same_formula_cif_pairs.csv")
    existing_keys = {
        (row["composition_key"], *sorted((row["cif_a"].lower(), row["cif_b"].lower())))
        for row in existing_pairs
    }
    reconstructed_keys = {(key, a[1], b[1]) for _, key, a, b in cif_pairs}
    if reconstructed_keys != existing_keys:
        raise ValueError("resolved_actual_hash_pairs_do_not_match_existing_pair_identities")

    pair_rows = []
    sg_pair_rows = []
    detail_rows = []
    sg_detail_rows = []
    target_rows = []
    formula_rows = []
    count_rows = []
    leakage_rows = []
    global_delta_checks = defaultdict(list)
    report_keys = sorted({row["composition_key"] for row in records})

    for fold in range(5):
        data = build_fold(root, fold, "cpu")
        fold_records = data["records"]
        if [row["measurement_id"] for row in fold_records] != [row["measurement_id"] for row in records]:
            raise ValueError(f"fold_report_order_mismatch:{fold}")
        split_inputs = {
            "TRAIN": (data["ti"], data["tw"]),
            "inner_VALID": (data["vi"], data["vw"]),
            "outer": (data["oi"], data["ow"]),
        }
        split_indices = {}
        split_weights = {}
        formula_role = {}
        for role, (raw_indices, raw_weights) in split_inputs.items():
            indices = [int(index) for index in raw_indices.detach().cpu().tolist()]
            weights = [float(value) for value in raw_weights.detach().cpu().tolist()]
            if len(indices) != len(weights):
                raise ValueError(f"split_weight_length_mismatch:{fold}:{role}")
            split_indices[role] = set(indices)
            split_weights[role] = dict(zip(indices, weights))
            for index in indices:
                key = fold_records[index]["composition_key"]
                if key in formula_role and formula_role[key] != role:
                    raise ValueError(f"formula_crosses_splits:{fold}:{key}")
                formula_role[key] = role
        if set(formula_role) != set(report_keys):
            raise ValueError(f"formula_fold_assignment_incomplete:{fold}")

        fold_pair_counts = defaultdict(lambda: {"pairs": 0, "formulas": set(), "reports": set()})
        fold_formula_counts = defaultdict(lambda: {"CIF": {"pairs": 0, "reports": set()},
                                                  "SG_ONLY": {"pairs": 0, "reports": set()}})
        for pair_type, pairs, groups, destination, detail_destination in (
                ("CIF", cif_pairs, cif_groups, pair_rows, detail_rows),
                ("SG_ONLY", sg_pairs, sg_groups, sg_pair_rows, sg_detail_rows)):
            for pair_id, key, group_a, group_b in pairs:
                role = formula_role[key]
                weight_map = split_weights[role]
                indices_a, indices_b = groups[group_a], groups[group_b]
                if any(index not in split_indices[role] for index in indices_a + indices_b):
                    raise ValueError(f"pair_group_crosses_splits:{fold}:{pair_id}")
                context_a, side_rows_a, report_rows_a = group_side(
                    fold_records, metadata, resolved, indices_a, weight_map, group_a[1], "a")
                context_b, side_rows_b, report_rows_b = group_side(
                    fold_records, metadata, resolved, indices_b, weight_map, group_b[1], "b")
                row = {
                    "fold": fold,
                    "split": role,
                    "pair_type": pair_type,
                    "pair_id": pair_id,
                    "composition_key": key,
                    "formula": fold_records[indices_a[0]]["formula"],
                    "group_a": str(group_a[1]),
                    "group_b": str(group_b[1]),
                    "reports_a": context_a["reports"],
                    "reports_b": context_b["reports"],
                    "group_a_context_json": pack(context_a),
                    "group_b_context_json": pack(context_b),
                }
                destination.append(row)
                detail_destination.extend(
                    {"fold": fold, "split": role, "pair_type": pair_type, "pair_id": pair_id, **detail}
                    for detail in side_rows_a + side_rows_b)
                counts = fold_pair_counts[(role, pair_type)]
                counts["pairs"] += 1
                counts["formulas"].add(key)
                counts["reports"].update(indices_a + indices_b)
                formula_counts = fold_formula_counts[(key, role)][pair_type]
                formula_counts["pairs"] += 1
                formula_counts["reports"].update(indices_a + indices_b)

                if pair_type == "CIF" and role in ("TRAIN", "inner_VALID"):
                    mean_a = sum(report["conditioned_weight"] * float(fold_records[report["report_index"]]["gap_ev"])
                                 for report in report_rows_a)
                    mean_b = sum(report["conditioned_weight"] * float(fold_records[report["report_index"]]["gap_ev"])
                                 for report in report_rows_b)
                    delta = mean_b - mean_a
                    target_rows.append({
                        "fold": fold, "split": role, "fit_target": int(role == "TRAIN"),
                        "pair_id": pair_id, "composition_key": key,
                        "cif_a": group_a[1], "cif_b": group_b[1],
                        "weighted_mean_a_ev": mean_a, "weighted_mean_b_ev": mean_b,
                        "observed_delta_ev": delta,
                        "report_ids_a_json": pack(context_a["report_ids"]),
                        "report_ids_b_json": pack(context_b["report_ids"]),
                    })
                    global_delta_checks[(key, group_a[1], group_b[1])].append(delta)

        for key in report_keys:
            role = formula_role[key]
            cif_hashes = sorted(group[1] for group in cif_groups if group[0] == key)
            sg_numbers = sorted(group[1] for group in sg_groups if group[0] == key)
            formula_rows.append({
                "fold": fold, "split": role, "composition_key": key,
                "formula": next(row["formula"] for row in fold_records if row["composition_key"] == key),
                "report_count": sum(row["composition_key"] == key for row in fold_records),
                "cif_supported_reports": sum(1 for index, row in enumerate(fold_records)
                                              if row["composition_key"] == key and structure_mask[index]),
                "cif_group_count": len(cif_hashes), "cif_pair_count": len(cif_hashes) * (len(cif_hashes) - 1) // 2,
                "cif_pair_ids_json": pack([pair_id for pair_id, pair_key, _, _ in cif_pairs if pair_key == key]),
                "sg_only_report_count": sum(1 for index, row in enumerate(fold_records)
                                             if row["composition_key"] == key and sg_mask[index] and not structure_mask[index]),
                "sg_only_group_count": len(sg_numbers),
                "sg_only_pair_count": len(sg_numbers) * (len(sg_numbers) - 1) // 2,
                "sg_only_pair_ids_json": pack([pair_id for pair_id, pair_key, _, _ in sg_pairs if pair_key == key]),
                "cif_pair_count_in_split": fold_formula_counts[(key, role)]["CIF"]["pairs"],
                "sg_only_pair_count_in_split": fold_formula_counts[(key, role)]["SG_ONLY"]["pairs"],
            })

        for role in ("TRAIN", "inner_VALID", "outer"):
            for pair_type in ("CIF", "SG_ONLY"):
                counts = fold_pair_counts[(role, pair_type)]
                count_rows.append({
                    "fold": fold, "split": role, "pair_type": pair_type,
                    "formula_count": len(counts["formulas"]),
                    "pair_count": counts["pairs"],
                    "participating_report_count": len(counts["reports"]),
                    "pair_ids_json": pack(sorted(row["pair_id"] for row in pair_rows + sg_pair_rows
                                                  if row["fold"] == fold and row["split"] == role
                                                  and row["pair_type"] == pair_type)),
                })

        train_ids = split_indices["TRAIN"]
        valid_ids = split_indices["inner_VALID"]
        outer_ids = split_indices["outer"]
        fold_targets = [row for row in target_rows if row["fold"] == fold]
        fit_targets = [row for row in fold_targets if row["fit_target"]]
        valid_targets = [row for row in fold_targets if not row["fit_target"]]
        if any(row["split"] != "TRAIN" for row in fit_targets):
            raise ValueError(f"fit_target_outside_train:{fold}")
        if any(record_index[mid] not in train_ids for row in fit_targets
               for mid in json.loads(row["report_ids_a_json"]) + json.loads(row["report_ids_b_json"])):
            raise ValueError(f"fit_target_report_outside_train:{fold}")
        if any(row["split"] != "inner_VALID" for row in valid_targets):
            raise ValueError(f"validation_target_not_inner_valid:{fold}")
        if any(record_index[mid] not in valid_ids for row in valid_targets
               for mid in json.loads(row["report_ids_a_json"]) + json.loads(row["report_ids_b_json"])):
            raise ValueError(f"validation_target_report_outside_valid:{fold}")
        if any(record_index[mid] in outer_ids for row in fold_targets
               for mid in json.loads(row["report_ids_a_json"]) + json.loads(row["report_ids_b_json"])):
            raise ValueError(f"outer_report_in_target_rows:{fold}")
        leakage_rows.append({
            "fold": fold,
            "train_pair_fit_target_rows": len(fit_targets),
            "inner_valid_pair_evaluation_rows": len(valid_targets),
            "outer_pair_target_rows": 0,
            "fit_targets_only_train": True,
            "validation_targets_only_inner_valid": True,
            "outer_targets_excluded": True,
            "reserved_targets_read": False,
        })

    reference_deltas = defaultdict(set)
    for row in existing_pairs:
        key, cif_a, cif_b = row["composition_key"], row["cif_a"].lower(), row["cif_b"].lower()
        a, b = sorted((cif_a, cif_b))
        reference_deltas[(key, a, b)].add(float(row["observed_mean_delta_ev"]) * (1 if (cif_a, cif_b) == (a, b) else -1))
    pair_mean_check = []
    for pair_key, values in sorted(global_delta_checks.items()):
        reference = reference_deltas[pair_key]
        if len(reference) != 1:
            raise ValueError(f"inconsistent_existing_pair_means:{pair_key}")
        expected = next(iter(reference))
        difference = max(abs(value - expected) for value in values)
        pair_mean_check.append({"pair_id": f"CIF|{pair_key[0]}|{pair_key[1]}|{pair_key[2]}",
                                "fold_role_estimates": len(values), "existing_delta_ev": expected,
                                "max_abs_difference_ev": difference})
        if difference > 1e-9:
            raise ValueError(f"conditioned_split_weights_do_not_recover_existing_pair_mean:{pair_key}:{difference}")
    if len(pair_mean_check) != len(cif_pairs):
        raise ValueError("existing_pair_mean_check_incomplete")

    for row in leakage_rows:
        if not row["fit_targets_only_train"] or not row["validation_targets_only_inner_valid"] or not row["outer_targets_excluded"]:
            raise ValueError(f"leakage_check_failed:{row['fold']}")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fold_pair_counts.csv", count_rows,
              ["fold", "split", "pair_type", "formula_count", "pair_count", "participating_report_count", "pair_ids_json"])
    write_csv(output / "formula_fold_coverage.csv", formula_rows,
              ["fold", "split", "composition_key", "formula", "report_count", "cif_supported_reports",
               "cif_group_count", "cif_pair_count", "cif_pair_ids_json", "sg_only_report_count",
               "sg_only_group_count", "sg_only_pair_count", "sg_only_pair_ids_json",
               "cif_pair_count_in_split", "sg_only_pair_count_in_split"])
    write_csv(output / "cif_pair_inventory.csv", pair_rows,
              ["fold", "split", "pair_type", "pair_id", "composition_key", "formula", "group_a", "group_b",
               "reports_a", "reports_b", "group_a_context_json", "group_b_context_json"])
    write_csv(output / "cif_pair_reports.csv", detail_rows,
              ["fold", "split", "pair_type", "pair_id", "side", "group_value", "report_index", "measurement_id",
               "split_weight", "conditioned_weight", "phase_label", "source_group_id", "specimen_group_id",
               "space_group_number", "structure_relation", "actual_cif_sha256", "input_support"])
    write_csv(output / "sg_only_pair_inventory.csv", sg_pair_rows,
              ["fold", "split", "pair_type", "pair_id", "composition_key", "formula", "group_a", "group_b",
               "reports_a", "reports_b", "group_a_context_json", "group_b_context_json"])
    write_csv(output / "sg_only_pair_reports.csv", sg_detail_rows,
              ["fold", "split", "pair_type", "pair_id", "side", "group_value", "report_index", "measurement_id",
               "split_weight", "conditioned_weight", "phase_label", "source_group_id", "specimen_group_id",
               "space_group_number", "structure_relation", "actual_cif_sha256", "input_support"])
    write_csv(output / "cif_pair_targets_train_and_valid.csv", target_rows,
              ["fold", "split", "fit_target", "pair_id", "composition_key", "cif_a", "cif_b",
               "weighted_mean_a_ev", "weighted_mean_b_ev", "observed_delta_ev", "report_ids_a_json", "report_ids_b_json"])
    write_csv(output / "pair_mean_weight_check.csv", pair_mean_check,
              ["pair_id", "fold_role_estimates", "existing_delta_ev", "max_abs_difference_ev"])
    write_csv(output / "leakage_checks.csv", leakage_rows,
              ["fold", "train_pair_fit_target_rows", "inner_valid_pair_evaluation_rows", "outer_pair_target_rows",
               "fit_targets_only_train", "validation_targets_only_inner_valid", "outer_targets_excluded", "reserved_targets_read"])

    fit_counts = [row for row in count_rows if row["pair_type"] == "CIF" and row["split"] == "TRAIN"]
    valid_counts = [row for row in count_rows if row["pair_type"] == "CIF" and row["split"] == "inner_VALID"]
    summary = {
        "schema": "phase_pair_inventory_v1",
        "root": str(root),
        "outputs": str(output),
        "report_count": len(records),
        "structure_supported_reports": sum(structure_mask),
        "actual_cif_hashes": len({row["actual_cif_sha256"].lower() for row in resolved_rows}),
        "structure_supported_formulas": len({key for key, _ in cif_groups}),
        "global_cif_pair_formulas": len({key for _, key, _, _ in cif_pairs}),
        "global_cif_pairs": len(cif_pairs),
        "global_sg_only_pair_formulas": len({key for _, key, _, _ in sg_pairs}),
        "global_sg_only_pairs": len(sg_pairs),
        "fits_30_layout": "5 folds x 3 seeds x 2 arms",
        "fit_count_budget_coverage_sufficient": all(row["pair_count"] > 0 for row in fit_counts + valid_counts),
        "min_train_cif_pairs_per_fold": min(row["pair_count"] for row in fit_counts),
        "min_inner_valid_cif_pairs_per_fold": min(row["pair_count"] for row in valid_counts),
        "fit_budget_scope_note": "Coverage supports the bounded fit schedule at the inventory level; it is not a power or performance claim.",
        "pair_mean_max_abs_difference_ev": max(row["max_abs_difference_ev"] for row in pair_mean_check),
        "fit_target_rows_all_train": all(row["fit_targets_only_train"] for row in leakage_rows),
        "outer_pair_target_rows": sum(row["outer_pair_target_rows"] for row in leakage_rows),
        "reserved_targets_read": False,
        "fits_run": 0,
        "input_binding": {
            str(path.relative_to(root)): sha256(path) for path in (
                reports_dir / "metadata.json", reports_dir / "records.jsonl", reports_dir / "features.pt",
                physical_dir / "resolved_structure_index.csv", physical_dir / "same_formula_cif_pairs.csv")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
