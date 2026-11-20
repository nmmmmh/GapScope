"""Recompute missing SG numbers for the fixed candidate CIF set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


EXPECTED_ROWS = 45
SYMPREC_ANGSTROM = 0.01
ANGLE_TOLERANCE_DEGREES = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def complete(input_path: Path, structures_dir: Path, output_path: Path) -> bool:
    rows = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("input_must_be_a_json_list")

    root = structures_dir.resolve()
    global_errors = []
    if len(rows) != EXPECTED_ROWS:
        global_errors.append(f"expected_{EXPECTED_ROWS}_rows_got_{len(rows)}")

    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        versions = {
            "pymatgen": importlib.metadata.version("pymatgen"),
            "spglib": importlib.metadata.version("spglib"),
        }
    except Exception as exc:
        Structure = SpacegroupAnalyzer = None
        versions = {"pymatgen": None, "spglib": None}
        global_errors.append(f"symmetry_dependencies_unavailable:{type(exc).__name__}:{exc}")

    completed = []
    record_ids = []
    composition_indices = []
    for position, source in enumerate(rows):
        result = {
            "input_position": position,
            "record_id": source.get("record_id") if isinstance(source, dict) else None,
            "composition_index": source.get("composition_index") if isinstance(source, dict) else None,
            "cif_file": source.get("cif_file") if isinstance(source, dict) else None,
            "cif_sha256": source.get("cif_sha256") if isinstance(source, dict) else None,
            "previous_numbers": source.get("previous_numbers") if isinstance(source, dict) else None,
        }
        try:
            if not isinstance(source, dict):
                raise ValueError("row_must_be_object")
            for key in ("record_id", "composition_index", "cif_file", "cif_sha256", "previous_numbers"):
                if key not in source:
                    raise ValueError(f"missing_{key}")
            if not isinstance(source["record_id"], str) or not source["record_id"]:
                raise ValueError("invalid_record_id")
            if type(source["composition_index"]) is not int or source["composition_index"] < 0:
                raise ValueError("invalid_composition_index")
            if not isinstance(source["previous_numbers"], list):
                raise ValueError("previous_numbers_must_be_list")
            cif_path = (root / source["cif_file"]).resolve()
            if not _inside(cif_path, root) or not cif_path.is_file():
                raise ValueError("cif_missing_or_outside_structures_directory")
            actual_hash = sha256(cif_path)
            if actual_hash.lower() != str(source["cif_sha256"]).lower():
                raise ValueError("cif_sha256_mismatch")
            if Structure is None or SpacegroupAnalyzer is None:
                raise RuntimeError("symmetry_dependencies_unavailable")
            structure = Structure.from_file(str(cif_path))
            number = int(SpacegroupAnalyzer(
                structure,
                symprec=SYMPREC_ANGSTROM,
                angle_tolerance=ANGLE_TOLERANCE_DEGREES,
            ).get_space_group_number())
            if not 1 <= number <= 230:
                raise ValueError(f"computed_space_group_out_of_range:{number}")
            result.update(
                status="computed",
                space_group_number=number,
                verified_cif_sha256=actual_hash,
            )
            record_ids.append(source["record_id"])
            composition_indices.append(source["composition_index"])
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}:{exc}")
        completed.append(result)

    if len(set(record_ids)) != len(record_ids):
        global_errors.append("duplicate_record_id")
    if len(set(composition_indices)) != len(composition_indices):
        global_errors.append("duplicate_composition_index")
    all_computed = len(completed) == EXPECTED_ROWS and all(row["status"] == "computed" for row in completed)
    result = {
        "schema": "candidate_symmetry_completion_v1",
        "method": {
            "analyzer": "pymatgen.symmetry.analyzer.SpacegroupAnalyzer.get_space_group_number",
            "symprec_angstrom": SYMPREC_ANGSTROM,
            "angle_tolerance_degrees": ANGLE_TOLERANCE_DEGREES,
        },
        "versions": versions,
        "expected_rows": EXPECTED_ROWS,
        "row_count": len(completed),
        "complete": bool(all_computed and not global_errors),
        "global_errors": global_errors,
        "rows": completed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result["complete"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if complete(args.input, args.structures, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
