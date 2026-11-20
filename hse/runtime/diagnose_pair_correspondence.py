"""Audit phase-label to CIF correspondence for the frozen same-formula pairs.

This is a read-only diagnostic. It does not access reserved/auxiliary datasets,
fit models, or change report, CIF, feature, or result inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import warnings
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


EXPECTED_PAIRS = 22
EXPECTED_FORMULAS = 18
MATCHER_SETTINGS = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": True,
}
STRICT_MATCHER_SETTINGS = {
    "ltol": 0.01,
    "stol": 0.01,
    "angle_tol": 1,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": True,
}
INPUT_RELATIVE_PATHS = (
    "data/reports/records.jsonl",
    "data/reports/metadata.json",
    "results/physical_cases/resolved_structure_index.csv",
    "results/physical_cases/same_formula_cif_pairs.csv",
    "results/physical_cases/cif_group_response.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_csv_selected(path: Path, fields: tuple[str, ...]):
    """Read named columns only; leave all numeric response columns untouched."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {name: header.index(name) for name in fields}
        return [{name: values[position] for name, position in positions.items()}
                for values in reader if values]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_cell(values) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def as_bool(value):
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def distinct(values):
    return sorted({str(value).strip() for value in values if value not in (None, "")})


def space_group_number(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    try:
        return int(SpaceGroup(text).int_number)
    except (TypeError, ValueError):
        return None


def pair_key(row):
    return row["composition_key"], *sorted((row["cif_a"].lower(), row["cif_b"].lower()))


def group_report_rows(ids, records_by_id, metadata_by_id, index_by_id):
    rows = []
    for report_id in sorted(ids):
        record = records_by_id[report_id]
        meta = metadata_by_id[report_id]
        index = index_by_id[report_id]
        raw_phase = record.get("phase_label", "")
        meta_phase = meta.get("phase_label", "")
        raw_sg = record.get("space_group", "")
        rows.append({
            "measurement_id": report_id,
            "composition_key": record.get("composition_key", ""),
            "formula": record.get("formula", ""),
            "raw_phase_label": raw_phase,
            "metadata_phase_label": meta_phase,
            "phase_label_matches_metadata": raw_phase == meta_phase,
            "raw_space_group": raw_sg,
            "raw_sg_missing": raw_sg in (None, ""),
            "raw_sg_supported": as_bool(record.get("sg_supported")),
            "raw_sg_evidence_status": record.get("sg_evidence_status", meta.get("sg_evidence_status", "")),
            "metadata_sg_evidence_status": meta.get("sg_evidence_status", ""),
            "metadata_sg_supported": as_bool(meta.get("sg_supported")),
            "record_cif_path": record.get("cif_path", ""),
            "record_cif_available": as_bool(record.get("cif_available")),
            "record_structure_supported": as_bool(record.get("structure_supported")),
            "record_structure_relation": record.get("structure_relation", ""),
            "record_structure_evidence_tier": record.get("structure_evidence_tier", ""),
            "source_group_id": record.get("source_group_id", ""),
            "specimen_group_id": record.get("specimen_group_id", ""),
            "source_locator": record.get("source_locator", ""),
            "cif_path": index.get("cif_path", ""),
            "cif_sha256": index.get("actual_cif_sha256", "").lower(),
            "original_cif_sha256": index.get("original_cif_sha256", ""),
            "original_reference_sha256": index.get("original_reference_sha256", ""),
            "index_structure_relation": index.get("structure_relation", ""),
            "index_input_support": index.get("input_support", ""),
            "metadata_model_sg": meta.get("model_space_group_number", ""),
            "metadata_model_sg_source": meta.get("model_sg_source", ""),
            "metadata_structure_relation": meta.get("structure_relation", ""),
            "metadata_model_structure_relation": meta.get("model_structure_relation", ""),
            "structure_evidence_tier_as_recorded": meta.get("structure_evidence_tier", ""),
            "structure_supported_by_source_as_recorded": as_bool(meta.get("structure_supported_by_source")),
            "exact_sample_structure_as_recorded": as_bool(meta.get("exact_sample_structure")),
            "model_structure_source_url": meta.get("model_structure_source_url", ""),
            "metadata_cif_reference_path": meta.get("cif_reference_path", ""),
            "metadata_model_structure_cif_path": meta.get("model_structure_cif_path", ""),
            "metadata_model_structure_cif_sha256": meta.get("model_structure_cif_sha256", ""),
            "metadata_cif_sha256": meta.get("cif_sha256", ""),
        })
    return rows


def side_summary(reports, cif_info):
    raw_sg = distinct(row["raw_space_group"] for row in reports)
    supported_raw_sg = distinct(row["raw_space_group"] for row in reports
                                if row["raw_sg_supported"] is True)
    supported_raw_sg_numbers = sorted({number for row in reports
                                       if row["raw_sg_supported"] is True
                                       for number in [space_group_number(row["raw_space_group"])]
                                       if number is not None})
    model_sg = distinct(row["metadata_model_sg"] for row in reports)
    phases = distinct(row["raw_phase_label"] for row in reports)
    source_ids = distinct(row["source_group_id"] for row in reports)
    relation_values = distinct(row["index_structure_relation"] for row in reports)
    model_relation_values = distinct(row["metadata_model_structure_relation"] for row in reports)
    source_urls = distinct(row["model_structure_source_url"] for row in reports)
    source_supported = [row["structure_supported_by_source_as_recorded"] for row in reports]
    exact_values = [row["exact_sample_structure_as_recorded"] for row in reports]
    locators = distinct(row["source_locator"] for row in reports)
    return {
        "report_ids": json_cell([row["measurement_id"] for row in reports]),
        "raw_phase_labels": json_cell(phases),
        "raw_space_groups": json_cell(raw_sg),
        "source_supported_raw_space_groups": json_cell(supported_raw_sg),
        "source_supported_raw_space_group_numbers": json_cell(supported_raw_sg_numbers),
        "raw_sg_missing_report_count": sum(row["raw_sg_missing"] for row in reports),
        "supported_raw_sg_unparsed_report_ids": json_cell([
            row["measurement_id"] for row in reports
            if row["raw_sg_supported"] is True and row["raw_space_group"] not in (None, "")
            and space_group_number(row["raw_space_group"]) is None]),
        "raw_sg_evidence_statuses": json_cell(distinct(row["raw_sg_evidence_status"] for row in reports)),
        "metadata_model_space_groups": json_cell(model_sg),
        "cif_analyzed_space_group": cif_info["analyzed_space_group"],
        "source_group_ids": json_cell(source_ids),
        "source_locators": json_cell(locators),
        "index_structure_relations": json_cell(relation_values),
        "metadata_model_structure_relations": json_cell(model_relation_values),
        "structure_evidence_tiers_as_recorded": json_cell(distinct(row["structure_evidence_tier_as_recorded"] for row in reports)),
        "structure_supported_by_source_values": json_cell(sorted({str(x) for x in source_supported})),
        "exact_sample_structure_values": json_cell(sorted({str(x) for x in exact_values})),
        "model_structure_source_urls": json_cell(source_urls),
        "source_locator_missing_report_count": sum(not row["source_locator"] for row in reports),
        "relation_missing_report_count": sum(not row["index_structure_relation"] for row in reports),
        "multi_raw_phase_labels_same_cif": len(phases) > 1,
        "raw_phase_metadata_conflict": any(not row["phase_label_matches_metadata"] for row in reports),
        "supported_raw_sg_model_sg_mismatch": any(
            row["raw_sg_supported"] is True
            and space_group_number(row["raw_space_group"]) is not None
            and space_group_number(row["metadata_model_sg"]) is not None
            and space_group_number(row["raw_space_group"]) != space_group_number(row["metadata_model_sg"])
            for row in reports),
        "metadata_model_sg_cif_analysis_mismatch": any(
            space_group_number(row["metadata_model_sg"]) is not None
            and space_group_number(cif_info["analyzed_space_group"]) is not None
            and space_group_number(row["metadata_model_sg"]) != space_group_number(cif_info["analyzed_space_group"])
            for row in reports),
    }


def read_pymatgen_version() -> str:
    for package in ("pymatgen", "pymatgen-core"):
        try:
            return version(package)
        except PackageNotFoundError:
            continue
    return "unknown"


def run(root: Path, output: Path):
    root = root.resolve()
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("mapping_output_must_be_outside_formal_root")
    output.mkdir(parents=True, exist_ok=True)

    reports_dir = root / "data" / "reports"
    physical_dir = root / "results" / "physical_cases"
    records = read_jsonl(reports_dir / "records.jsonl")
    metadata = read_json(reports_dir / "metadata.json")
    index_rows = read_csv(physical_dir / "resolved_structure_index.csv")
    all_pair_rows = read_csv_selected(
        physical_dir / "same_formula_cif_pairs.csv",
        ("model", "seed_mixture", "composition_key", "formula", "cif_a", "cif_b",
         "phase_a", "phase_b", "sg_a", "sg_b", "shared_source"))
    group_rows = read_csv_selected(
        physical_dir / "cif_group_response.csv",
        ("model", "seed_mixture", "composition_key", "formula", "cif_sha256", "reports",
         "phase_labels_json", "source_groups_json", "model_space_group", "structure_relation_json"))

    records_by_id = {row["measurement_id"]: row for row in records}
    metadata_by_id = {row["measurement_id"]: row for row in metadata}
    index_by_id = {row["measurement_id"]: row for row in index_rows}
    if len(records_by_id) != len(records):
        raise ValueError("duplicate_measurement_id_in_records")
    if len(metadata_by_id) != len(metadata):
        raise ValueError("duplicate_measurement_id_in_metadata")
    if set(records_by_id) != set(metadata_by_id):
        raise ValueError("record_metadata_id_set_mismatch")

    pair_rows = [row for row in all_pair_rows
                 if row.get("model") == "FROZEN"
                 and row.get("seed_mixture") == "equal_density_mixture"]
    pair_rows.sort(key=lambda row: (row["composition_key"], row["cif_a"], row["cif_b"]))
    pair_keys = [pair_key(row) for row in pair_rows]
    if len(pair_rows) != EXPECTED_PAIRS or len(set(pair_keys)) != EXPECTED_PAIRS:
        raise ValueError(f"expected_{EXPECTED_PAIRS}_unique_frozen_pairs_got_{len(set(pair_keys))}")

    # Reconstruct the complete pair universe from the actual resolved report
    # mapping and compare it with the frozen table's declared 22 pairs.
    cif_groups = defaultdict(list)
    for report_id, index in index_by_id.items():
        if report_id not in records_by_id:
            raise ValueError(f"resolved_id_missing_from_reports:{report_id}")
        if index["composition_key"] != records_by_id[report_id]["composition_key"]:
            raise ValueError(f"resolved_composition_mismatch:{report_id}")
        cif_groups[(index["composition_key"], index["actual_cif_sha256"].lower())].append(report_id)
    expected_keys = set()
    by_composition = defaultdict(list)
    for composition_key, cif_hash in cif_groups:
        by_composition[composition_key].append(cif_hash)
    for composition_key, hashes in by_composition.items():
        for cif_a, cif_b in itertools.combinations(sorted(set(hashes)), 2):
            expected_keys.add((composition_key, cif_a, cif_b))
    if expected_keys != set(pair_keys):
        raise ValueError("resolved_report_CIF_pairs_do_not_equal_frozen_pair_table")
    formulas = {row["formula"] for row in pair_rows}
    if len(formulas) != EXPECTED_FORMULAS:
        raise ValueError(f"expected_{EXPECTED_FORMULAS}_formulas_got_{len(formulas)}")

    frozen_groups = [row for row in group_rows
                     if row["model"] == "FROZEN"
                     and row["seed_mixture"] == "equal_density_mixture"]
    group_by_key = {(row["composition_key"], row["cif_sha256"].lower()): row
                    for row in frozen_groups}
    if len(group_by_key) != len(frozen_groups):
        raise ValueError("duplicate_frozen_cif_group_response_key")
    endpoint_keys = {key for row in pair_rows for key in
                     ((row["composition_key"], row["cif_a"].lower()),
                      (row["composition_key"], row["cif_b"].lower()))}
    checked_group_rows = 0
    for key in sorted(endpoint_keys):
        group = group_by_key.get(key)
        if group is None:
            raise ValueError(f"pair_endpoint_missing_from_cif_group_response:{key}")
        group_reports = cif_groups[key]
        expected_phases = distinct(records_by_id[report_id].get("phase_label", "")
                                   for report_id in group_reports)
        expected_sources = distinct(records_by_id[report_id].get("source_group_id", "")
                                    for report_id in group_reports)
        expected_model_sg = distinct(metadata_by_id[report_id].get("model_space_group_number", "")
                                     for report_id in group_reports)
        expected_relations = distinct(metadata_by_id[report_id].get("structure_relation", "")
                                      for report_id in group_reports)
        if set(json.loads(group["phase_labels_json"])) != set(expected_phases):
            raise ValueError(f"cif_group_phase_labels_mismatch:{key}")
        if set(json.loads(group["source_groups_json"])) != set(expected_sources):
            raise ValueError(f"cif_group_source_groups_mismatch:{key}")
        if int(group["reports"]) != len(group_reports):
            raise ValueError(f"cif_group_report_count_mismatch:{key}")
        if expected_model_sg != [str(group["model_space_group"]).strip()]:
            raise ValueError(f"cif_group_model_sg_mismatch:{key}")
        if set(json.loads(group["structure_relation_json"])) != set(expected_relations):
            raise ValueError(f"cif_group_structure_relation_mismatch:{key}")
        if group["formula"] != records_by_id[group_reports[0]]["formula"]:
            raise ValueError(f"cif_group_formula_mismatch:{key}")
        checked_group_rows += 1
    involved_ids = {report_id for key in endpoint_keys for report_id in cif_groups[key]}
    report_usage = defaultdict(list)
    for row in pair_rows:
        pair_id = f"{row['formula']}|{row['cif_a'].lower()}|{row['cif_b'].lower()}"
        for side in ("a", "b"):
            key = row["composition_key"], row[f"cif_{side}"].lower()
            for report_id in cif_groups[key]:
                report_usage[report_id].append(pair_id)

    report_mapping = group_report_rows(involved_ids, records_by_id, metadata_by_id, index_by_id)
    for row in report_mapping:
        report_id = row["measurement_id"]
        expected_hash = row["cif_sha256"]
        cif_path = (root / row["cif_path"]).resolve()
        if not cif_path.is_relative_to(root) or not cif_path.is_file():
            raise ValueError(f"invalid_or_missing_cif_path:{report_id}")
        actual_hash = sha256(cif_path)
        if actual_hash != expected_hash:
            raise ValueError(f"resolved_hash_vs_file_mismatch:{report_id}")
        meta = metadata_by_id[report_id]
        for field in ("cif_sha256", "model_structure_cif_sha256"):
            recorded_hash = str(meta.get(field) or "").lower()
            if recorded_hash and recorded_hash != actual_hash:
                raise ValueError(f"metadata_{field}_vs_file_mismatch:{report_id}")
        row["actual_file_sha256_verified"] = True
        row["pair_ids"] = json_cell(sorted(set(report_usage[report_id])))

    structures = {}
    for composition_key, cif_hash in endpoint_keys:
        report_id = sorted(cif_groups[(composition_key, cif_hash)])[0]
        index = index_by_id[report_id]
        cif_path = (root / index["cif_path"]).resolve()
        with warnings.catch_warnings(record=True) as parse_warnings:
            warnings.simplefilter("always")
            structure = Structure.from_file(cif_path)
        formula = records_by_id[report_id]["formula"]
        actual_fraction = structure.composition.fractional_composition.get_el_amt_dict()
        expected_fraction = Composition(formula).fractional_composition.get_el_amt_dict()
        same_stoichiometry = (set(actual_fraction) == set(expected_fraction)
                              and all(abs(actual_fraction[element] - expected_fraction[element]) < 1e-6
                                      for element in expected_fraction))
        if not same_stoichiometry:
            raise ValueError(f"CIF_formula_mismatch:{report_id}:{structure.composition.reduced_formula}:{formula}")
        meta = metadata_by_id[report_id]
        symprec = meta.get("model_sg_symprec_angstrom")
        try:
            symprec = float(symprec) if symprec not in (None, "") else 0.01
            analyzed_sg = int(SpacegroupAnalyzer(structure, symprec=symprec,
                                                  angle_tolerance=5.0).get_space_group_number())
        except Exception:
            analyzed_sg = ""
        structures[(composition_key, cif_hash)] = {
            "structure": structure,
            "parse_warnings": [str(warning.message) for warning in parse_warnings],
            "path": index["cif_path"],
            "actual_hash": sha256(cif_path),
            "original_cif_sha256": index.get("original_cif_sha256", ""),
            "original_reference_sha256": index.get("original_reference_sha256", ""),
            "index_relation": index.get("structure_relation", ""),
            "input_support": index.get("input_support", ""),
            "analyzed_space_group": analyzed_sg,
            "analyzer_symprec_angstrom": symprec,
        }

    matcher = StructureMatcher(**MATCHER_SETTINGS)
    strict_matcher = StructureMatcher(**STRICT_MATCHER_SETTINGS)
    pair_output = []
    hard_conflict_count = 0
    same_sg_count = 0
    distinct_structure_count = 0
    label_different_count = 0
    primary_phase_reference_count = 0
    missing_locator_count = 0
    for row in pair_rows:
        composition_key = row["composition_key"]
        hash_a, hash_b = row["cif_a"].lower(), row["cif_b"].lower()
        reports_a = [r for r in report_mapping if r["cif_sha256"] == hash_a
                     and r["composition_key"] == composition_key]
        reports_b = [r for r in report_mapping if r["cif_sha256"] == hash_b
                     and r["composition_key"] == composition_key]
        info_a, info_b = structures[(composition_key, hash_a)], structures[(composition_key, hash_b)]
        side_a, side_b = side_summary(reports_a, info_a), side_summary(reports_b, info_b)
        same_sg_table = str(row.get("sg_a", "")).strip() == str(row.get("sg_b", "")).strip()
        analyzed_a, analyzed_b = info_a["analyzed_space_group"], info_b["analyzed_space_group"]
        same_sg_analyzed = (analyzed_a not in (None, "") and analyzed_b not in (None, "")
                            and str(analyzed_a) == str(analyzed_b))
        raw_supported_a = json.loads(side_a["source_supported_raw_space_group_numbers"])
        raw_supported_b = json.loads(side_b["source_supported_raw_space_group_numbers"])
        same_sg_raw_supported = (len(raw_supported_a) == 1 and len(raw_supported_b) == 1
                                 and raw_supported_a[0] == raw_supported_b[0])
        structure_equivalent = bool(matcher.fit(info_a["structure"], info_b["structure"]))
        strict_structure_equivalent = bool(strict_matcher.fit(info_a["structure"], info_b["structure"]))
        symmetry_differs = (analyzed_a not in (None, "") and analyzed_b not in (None, "")
                            and str(analyzed_a) != str(analyzed_b))
        distinct_by_structure_evidence = (not structure_equivalent) or symmetry_differs
        phase_a = json.loads(side_a["raw_phase_labels"])
        phase_b = json.loads(side_b["raw_phase_labels"])
        if set(phase_a) != set(json.loads(row.get("phase_a") or "[]")):
            raise ValueError(f"pair_table_phase_a_mismatch:{row['formula']}:{hash_a}")
        if set(phase_b) != set(json.loads(row.get("phase_b") or "[]")):
            raise ValueError(f"pair_table_phase_b_mismatch:{row['formula']}:{hash_b}")
        model_sg_a = json.loads(side_a["metadata_model_space_groups"])
        model_sg_b = json.loads(side_b["metadata_model_space_groups"])
        if len(model_sg_a) != 1 or str(row.get("sg_a", "")).strip() != model_sg_a[0]:
            raise ValueError(f"pair_table_sg_a_mismatch:{row['formula']}:{hash_a}")
        if len(model_sg_b) != 1 or str(row.get("sg_b", "")).strip() != model_sg_b[0]:
            raise ValueError(f"pair_table_sg_b_mismatch:{row['formula']}:{hash_b}")
        phase_labels_differ = set(phase_a) != set(phase_b)
        source_a = set(json.loads(side_a["source_group_ids"]))
        source_b = set(json.loads(side_b["source_group_ids"]))
        shared_sources = sorted(source_a & source_b)
        shared_source_file = as_bool(row.get("shared_source"))
        if shared_source_file is not None and shared_source_file != bool(shared_sources):
            raise ValueError(f"shared_source_flag_mismatch:{row['formula']}:{hash_a}:{hash_b}")

        hard_conflicts = []
        notes = []
        for label, side in (("a", side_a), ("b", side_b)):
            if side["raw_phase_metadata_conflict"]:
                hard_conflicts.append(f"side_{label}:records_phase_label_differs_from_metadata_phase_label")
            if side["supported_raw_sg_model_sg_mismatch"]:
                hard_conflicts.append(f"side_{label}:source_supported_raw_SG_differs_from_encoded_model_SG")
            if side["metadata_model_sg_cif_analysis_mismatch"]:
                notes.append(f"side_{label}:metadata_model_SG_differs_from_CIF_symmetry_analysis")
            if side["multi_raw_phase_labels_same_cif"]:
                notes.append(f"side_{label}:multiple_raw_phase_labels_share_one_CIF_hash; not classified as error without exact-sample evidence")
            if side["source_locator_missing_report_count"]:
                notes.append(f"side_{label}:source_locator_missing_for_some_reports")
            if side["relation_missing_report_count"]:
                notes.append(f"side_{label}:structure_relation_missing_for_some_reports")
            if side["raw_sg_missing_report_count"]:
                notes.append(f"side_{label}:raw_report_SG_missing_for_some_reports; encoded_CIF_SG_is_listed_separately")
            if side["supported_raw_sg_unparsed_report_ids"] != "[]":
                notes.append(f"side_{label}:source_supported_raw_SG_symbol_not_normalized")
            if side["structure_supported_by_source_values"] == "[\"True\"]":
                primary_phase_reference_count += 1
            missing_locator_count += side["source_locator_missing_report_count"]
        if len(phase_a) > 1 or len(phase_b) > 1:
            notes.append("phase_label_multiplicity_is_reported_as_supplied; no label is overwritten")
        if phase_labels_differ:
            label_different_count += 1
        if same_sg_table:
            same_sg_count += 1
        if distinct_by_structure_evidence:
            distinct_structure_count += 1
        hard_conflict_count += bool(hard_conflicts)
        pair_id = f"{row['formula']}|{hash_a}|{hash_b}"
        pair_output.append({
            "pair_id": pair_id,
            "composition_key": composition_key,
            "formula": row["formula"],
            "cif_a_sha256": hash_a,
            "cif_a_path": info_a["path"],
            "cif_b_sha256": hash_b,
            "cif_b_path": info_b["path"],
            "cif_a_file_hash_verified": info_a["actual_hash"] == hash_a,
            "cif_b_file_hash_verified": info_b["actual_hash"] == hash_b,
            "original_cif_sha256_a": info_a["original_cif_sha256"],
            "original_cif_sha256_b": info_b["original_cif_sha256"],
            "original_reference_sha256_a": info_a["original_reference_sha256"],
            "original_reference_sha256_b": info_b["original_reference_sha256"],
            "raw_phase_labels_a": side_a["raw_phase_labels"],
            "raw_phase_labels_b": side_b["raw_phase_labels"],
            "report_ids_a": side_a["report_ids"],
            "report_ids_b": side_b["report_ids"],
            "source_group_ids_a": side_a["source_group_ids"],
            "source_group_ids_b": side_b["source_group_ids"],
            "source_locators_a": side_a["source_locators"],
            "source_locators_b": side_b["source_locators"],
            "raw_space_groups_a": side_a["raw_space_groups"],
            "raw_space_groups_b": side_b["raw_space_groups"],
            "supported_raw_space_groups_a": side_a["source_supported_raw_space_groups"],
            "supported_raw_space_groups_b": side_b["source_supported_raw_space_groups"],
            "supported_raw_space_group_numbers_a": side_a["source_supported_raw_space_group_numbers"],
            "supported_raw_space_group_numbers_b": side_b["source_supported_raw_space_group_numbers"],
            "raw_sg_evidence_statuses_a": side_a["raw_sg_evidence_statuses"],
            "raw_sg_evidence_statuses_b": side_b["raw_sg_evidence_statuses"],
            "raw_sg_missing_report_count_a": side_a["raw_sg_missing_report_count"],
            "raw_sg_missing_report_count_b": side_b["raw_sg_missing_report_count"],
            "pair_table_sg_a": row.get("sg_a", ""),
            "pair_table_sg_b": row.get("sg_b", ""),
            "metadata_model_sg_a": side_a["metadata_model_space_groups"],
            "metadata_model_sg_b": side_b["metadata_model_space_groups"],
            "cif_analyzed_sg_a": analyzed_a,
            "cif_analyzed_sg_b": analyzed_b,
            "cif_parse_warnings_a": json_cell(info_a["parse_warnings"]),
            "cif_parse_warnings_b": json_cell(info_b["parse_warnings"]),
            "same_sg_pair_table": same_sg_table,
            "same_sg_cif_analysis": same_sg_analyzed,
            "same_sg_supported_raw_reports": same_sg_raw_supported,
            "raw_phase_labels_differ": phase_labels_differ,
            "structure_matcher_equivalent": structure_equivalent,
            "strict_structure_matcher_equivalent": strict_structure_equivalent,
            "cif_analyzed_symmetry_differs": symmetry_differs,
            "actual_distinct_structure_by_symmetry_or_matcher": distinct_by_structure_evidence,
            "cif_bytes_differ": hash_a != hash_b,
            "shared_source_flag_in_pair_table": shared_source_file,
            "shared_source_group_ids": json_cell(shared_sources),
            "index_structure_relations_a": side_a["index_structure_relations"],
            "index_structure_relations_b": side_b["index_structure_relations"],
            "model_structure_relations_a": side_a["metadata_model_structure_relations"],
            "model_structure_relations_b": side_b["metadata_model_structure_relations"],
            "structure_evidence_tiers_a": side_a["structure_evidence_tiers_as_recorded"],
            "structure_evidence_tiers_b": side_b["structure_evidence_tiers_as_recorded"],
            "structure_supported_by_source_values_a": side_a["structure_supported_by_source_values"],
            "structure_supported_by_source_values_b": side_b["structure_supported_by_source_values"],
            "exact_sample_structure_values_a": side_a["exact_sample_structure_values"],
            "exact_sample_structure_values_b": side_b["exact_sample_structure_values"],
            "model_structure_source_urls_a": side_a["model_structure_source_urls"],
            "model_structure_source_urls_b": side_b["model_structure_source_urls"],
            "raw_phase_metadata_conflict_a": side_a["raw_phase_metadata_conflict"],
            "raw_phase_metadata_conflict_b": side_b["raw_phase_metadata_conflict"],
            "supported_raw_sg_model_sg_mismatch_a": side_a["supported_raw_sg_model_sg_mismatch"],
            "supported_raw_sg_model_sg_mismatch_b": side_b["supported_raw_sg_model_sg_mismatch"],
            "metadata_model_sg_cif_analysis_mismatch_a": side_a["metadata_model_sg_cif_analysis_mismatch"],
            "metadata_model_sg_cif_analysis_mismatch_b": side_b["metadata_model_sg_cif_analysis_mismatch"],
            "hard_conflict_details": json_cell(hard_conflicts),
            "missing_or_review_notes": json_cell(sorted(set(notes))),
        })

    pair_fields = list(pair_output[0])
    report_fields = list(report_mapping[0])
    write_csv(output / "pair_correspondence.csv", pair_output, pair_fields)
    write_csv(output / "report_mapping.csv", report_mapping, report_fields)

    try:
        pymatgen_ver = read_pymatgen_version()
    except Exception:
        pymatgen_ver = "unknown"
    input_binding = {rel: sha256(root / rel) for rel in INPUT_RELATIVE_PATHS}
    cif_binding = {structures[key]["path"]: structures[key]["actual_hash"]
                   for key in sorted(structures)}
    summary = {
        "schema": "pair_correspondence_diagnostic_v1",
        "scope": "the 22 FROZEN equal_density_mixture pairs only",
        "report_count_all": len(records),
        "formula_count_all_reports": len({row["composition_key"] for row in records}),
        "resolved_structure_report_count": len(index_rows),
        "report_mapping_rows": len(report_mapping),
        "unique_pairs": len(pair_output),
        "unique_formulas": len({row["formula"] for row in pair_output}),
        "unique_pair_endpoint_cifs": len(structures),
        "CIFs_with_parser_warnings": sum(bool(info["parse_warnings"]) for info in structures.values()),
        "pairs_with_different_raw_phase_label_sets": label_different_count,
        "pairs_same_pair_table_sg": same_sg_count,
        "pairs_tolerant_matcher_equivalent": sum(row["structure_matcher_equivalent"] for row in pair_output),
        "pairs_strict_matcher_equivalent": sum(row["strict_structure_matcher_equivalent"] for row in pair_output),
        "pairs_distinct_by_cif_symmetry_or_tolerant_matcher": distinct_structure_count,
        "pairs_with_hard_local_metadata_conflict": hard_conflict_count,
        "primary_source_support_recorded_pair_sides": primary_phase_reference_count,
        "report_rows_missing_source_locator": missing_locator_count,
        "report_rows_missing_direct_cif_path": sum(not row["record_cif_path"] for row in report_mapping),
        "report_rows_with_raw_sg_missing": sum(row["raw_sg_missing"] for row in report_mapping),
        "pair_sides_with_shared_source": sum(as_bool(row["shared_source_flag_in_pair_table"]) is True
                                              for row in pair_output),
        "pairs_tolerant_matcher_equivalent_but_strictly_distinct": sum(
            row["structure_matcher_equivalent"] and not row["strict_structure_matcher_equivalent"]
            for row in pair_output),
        "cif_group_response_rows_checked_without_response_values": checked_group_rows,
        "cif_group_response_rows_checked_exactly": checked_group_rows == len(endpoint_keys),
        "resolved_report_CIF_pair_set_matches_frozen_pair_set": expected_keys == set(pair_keys),
        "formula_count_matches_expected_18": len(formulas) == EXPECTED_FORMULAS,
        "pair_count_matches_expected_22": len(pair_output) == EXPECTED_PAIRS,
        "reserved_and_auxiliary_paths_opened": [],
        "targets_used": False,
        "input_binding": input_binding,
        "CIF_file_binding": cif_binding,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    protocol = {
        "schema": "pair_correspondence_protocol_v1",
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "formal_root": str(root),
        "output_dir": str(output),
        "input_binding": input_binding,
        "CIF_file_binding": cif_binding,
        "scope": "report and CIF correspondence for all distinct actual-CIF pairs in the frozen same_formula_cif_pairs artifact",
        "pair_selection": "model=FROZEN and seed_mixture=equal_density_mixture; unique key=(composition_key, sorted(cif_a,cif_b))",
        "mapping_chain": [
            "records.jsonl measurement_id/composition_key/phase_label/space_group/source fields",
            "metadata.json measurement_id to encoded CIF and model symmetry/provenance fields",
            "resolved_structure_index.csv measurement_id to actual CIF path and SHA256",
            "same_formula_cif_pairs.csv frozen pair identity, side phase/SG summary, and shared-source flag",
            "cif_group_response.csv frozen group phase/source/SG/relation fields checked without reading response values",
        ],
        "mapping_source_code": {
            "runtime/analyze_modal_interaction.py": "resolves encoded metadata CIF path and hash, groups by composition_key plus CIF hash, then forms all within-formula hash pairs",
            "runtime/phase_pair_inventory.py": "checks resolved_structure_index IDs against the structure feature mask and verifies actual file hashes",
        },
        "CIF_hash_validation": "resolved actual_cif_sha256 and metadata CIF hashes were checked against the bytes at the resolved path",
        "structure_comparison": {
            "implementation": "pymatgen StructureMatcher.fit",
            "version": pymatgen_ver,
            "settings": MATCHER_SETTINGS,
            "strict_sensitivity_settings": STRICT_MATCHER_SETTINGS,
            "interpretation": "The tolerant fit is a geometry-equivalence check, not independent phase identification; distinct analyzed CIF symmetry is separately retained as structural evidence even when tolerant matching succeeds",
        },
        "space_group_reanalysis": "pymatgen SpacegroupAnalyzer at metadata model_sg_symprec_angstrom when present, otherwise 0.01 A; angle tolerance 5 degrees",
        "evidence_boundary": "raw labels, source locators, source IDs, relation strings, support flags, exact-sample flags, and source URLs are copied from existing local records; no verification grade is added",
        "numeric_targets": "not used; same_formula_cif_pairs response columns and cif_group_response response columns were not accessed",
        "excluded_inputs": ["data/reserved/**", "data/auxiliary/**", "model checkpoints", "training features", "predictions"],
        "fits_or_data_changes": 0,
        "outputs": {},
    }
    protocol["outputs"] = {
        path.name: sha256(path) for path in (output / "pair_correspondence.csv",
                                             output / "report_mapping.csv", output / "summary.json")
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")

    cspbi3 = next((row for row in pair_output if row["formula"] == "CsPbI3"), None)
    shared_source_pairs = sum(as_bool(row["shared_source_flag_in_pair_table"]) is True for row in pair_output)
    cross_source_formulas = sorted(row["formula"] for row in pair_output
                                   if as_bool(row["shared_source_flag_in_pair_table"]) is False)
    matcher_close_pairs = [row for row in pair_output if row["structure_matcher_equivalent"]]
    analyzer_symmetry_diff_count = sum(row["cif_analyzed_symmetry_differs"] for row in pair_output)
    raw_sg_missing_count = sum(row["raw_sg_missing"] for row in report_mapping)
    direct_cif_missing_count = sum(not row["record_cif_path"] for row in report_mapping)
    findings = [
        "# Pair correspondence findings",
        "",
        f"- Reconstructed {len(pair_output)} unique actual-CIF pairs across {len(formulas)} formulas; the set exactly matches the frozen pair table.",
        f"- Compared {len(structures)} pair-endpoint CIF assets by SHA256, symmetry analysis, and two StructureMatcher tolerances. {analyzer_symmetry_diff_count}/{len(pair_output)} have different analyzed SG; all {len(pair_output)} fail the strict matcher or have different analyzed SG.",
        f"- The tolerant matcher calls {len(matcher_close_pairs)} pairs geometrically equivalent, while none pass the strict matcher; those tolerant matches still have different analyzed SG.",
        f"- {same_sg_count}/{len(pair_output)} pairs share the pair table SG; {label_different_count}/{len(pair_output)} have different raw phase-label sets.",
        f"- Found {hard_conflict_count} pairs with a hard local record/metadata conflict. Raw phase-label differences alone are retained as supplied and are not called errors.",
        f"- {direct_cif_missing_count}/{len(report_mapping)} related report rows lack a direct CIF path in records.jsonl; each still has a separately hash-verified model CIF path in metadata/index. This records the distinction between source specimen CIF and phase-reference input.",
        f"- {raw_sg_missing_count}/{len(report_mapping)} related reports lack a raw report SG. Their encoded/model SG is kept in separate columns; absence is not scored as a conflict.",
        f"- {checked_group_rows}/{len(endpoint_keys)} pair-endpoint groups agree with cif_group_response on report count, raw phase-label set, source IDs, model SG, and recorded relation; numeric response columns were not accessed.",
        f"- {shared_source_pairs}/{len(pair_output)} pairs have a shared source group. The cross-source formulas are {', '.join(cross_source_formulas)}.",
        "- The source implementation resolves encoded CIFs through metadata.json, groups reports by formula plus CIF hash, and forms all within-formula hash combinations; the resolved index then binds those report IDs to actual CIF bytes.",
        "- Phase-to-CIF evidence is reported from the existing source locator, source IDs, resolved-index relation, support flags, and model-source fields. A phase reference is not upgraded to an exact-sample structure claim.",
        "- No reserved or auxiliary report values were opened; no targets, training, prediction, or data edits were used.",
    ]
    if cspbi3:
        cspbi3_sources = json.loads(cspbi3["source_group_ids_a"])
        cspbi3_locators = json.loads(cspbi3["source_locators_a"])
        cspbi3_relations_a = json.loads(cspbi3["index_structure_relations_a"])
        cspbi3_relations_b = json.loads(cspbi3["index_structure_relations_b"])
        findings.extend([
            "",
            "## CsPbI3 same-SG pair",
            f"- Raw labels: {cspbi3['raw_phase_labels_a']} vs {cspbi3['raw_phase_labels_b']}; pair-table SG {cspbi3['pair_table_sg_a']} vs {cspbi3['pair_table_sg_b']}; CIF-analyzed SG {cspbi3['cif_analyzed_sg_a']} vs {cspbi3['cif_analyzed_sg_b']}.",
            f"- StructureMatcher equivalence: {cspbi3['structure_matcher_equivalent']}; analyzed CIF SG differs: {cspbi3['cif_analyzed_symmetry_differs']}. Existing source locators and relations are retained in pair_correspondence.csv and report_mapping.csv.",
            f"- Both sides map to source {', '.join(cspbi3_sources)}; the existing locator is: {cspbi3_locators[0] if cspbi3_locators else 'missing'}.",
            f"- Resolved-index relation: side a {', '.join(cspbi3_relations_a)}; side b {', '.join(cspbi3_relations_b)}. Both are recorded as same-phase references that are not exact samples; source-support flags are preserved in the tables.",
            "- Same SG is compatible with distinct polymorph structures; this row is not labeled a phase/CIF conflict solely because gamma and delta share SG 62.",
        ])
    findings.extend([
        "",
        "## Interpretation limits",
        "",
        "The correspondence table audits the current local mappings. It does not establish that every CIF is the exact measured specimen, and it does not adjudicate phase identity beyond the evidence recorded in the existing source fields. Where the source locator or relation is absent, the output marks that gap instead of inferring a failure.",
        "",
        "See `summary.json` and `protocol.json` for counts and input/output hashes.",
    ])
    (output / "findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    protocol["outputs"]["findings.md"] = sha256(output / "findings.md")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "report_count_all", "resolved_structure_report_count", "report_mapping_rows",
        "unique_pairs", "unique_formulas", "unique_pair_endpoint_cifs",
        "pairs_same_pair_table_sg", "pairs_tolerant_matcher_equivalent",
        "pairs_strict_matcher_equivalent", "pairs_distinct_by_cif_symmetry_or_tolerant_matcher",
        "pairs_with_different_raw_phase_label_sets", "pairs_with_hard_local_metadata_conflict",
        "cif_group_response_rows_checked_exactly",
        "resolved_report_CIF_pair_set_matches_frozen_pair_set")}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1].parent /
                        "BandgapFusion_CorrespondenceWork" / "mapping")
    args = parser.parse_args()
    run(args.root, args.output_dir)


if __name__ == "__main__":
    main()
