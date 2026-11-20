"""Apply an explicit, hash-checked SG overlay to candidate composition features."""

from __future__ import annotations

import copy
import hashlib
import json

import torch


SYMPREC_ANGSTROM = 0.01
ANGLE_TOLERANCE_DEGREES = 5.0
EXPECTED_ROWS = 45


def _payload_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _crystal_system_index(space_group_number: int) -> int:
    return (0 if space_group_number <= 2 else 1 if space_group_number <= 15 else
            2 if space_group_number <= 74 else 3 if space_group_number <= 142 else
            4 if space_group_number <= 167 else 5 if space_group_number <= 194 else 6)


def apply_completion(features, metadata, completion):
    """Return patched features, metadata, and an audit; inputs remain unchanged.

    Only ``sg_features`` and ``sg_mask`` are copied and edited. The positional
    index, record ID, structure SHA, supported-structure mask, and missing-SG
    state must all agree before an overlay is accepted.
    """
    if completion.get("schema") != "candidate_symmetry_completion_v1" or not completion.get("complete"):
        raise ValueError("candidate_symmetry_completion_not_complete")
    method = completion.get("method", {})
    if (method.get("symprec_angstrom") != SYMPREC_ANGSTROM or
            method.get("angle_tolerance_degrees") != ANGLE_TOLERANCE_DEGREES or
            not str(method.get("analyzer", "")).startswith("pymatgen.symmetry.analyzer.SpacegroupAnalyzer.")):
        raise ValueError("candidate_symmetry_completion_method_mismatch")
    rows = completion.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("candidate_symmetry_completion_row_count")
    if not isinstance(metadata, list):
        raise ValueError("metadata_must_be_row_list")

    count = len(metadata)
    required = {"sg_features", "sg_mask", "structure_mask"}
    if not required.issubset(features):
        raise ValueError(f"feature_keys_missing:{sorted(required - set(features))}")
    sg = features["sg_features"]
    sg_mask = features["sg_mask"]
    structure_mask = features["structure_mask"]
    if (not isinstance(sg, torch.Tensor) or tuple(sg.shape) != (count, 237) or
            not isinstance(sg_mask, torch.Tensor) or tuple(sg_mask.shape) != (count, 1) or
            not isinstance(structure_mask, torch.Tensor) or tuple(structure_mask.shape) != (count, 1)):
        raise ValueError("candidate_feature_shape_mismatch")

    patched_features = dict(features)
    patched_features["sg_features"] = sg.clone()
    patched_features["sg_mask"] = sg_mask.clone()
    patched_metadata = copy.deepcopy(metadata)
    payload_hash = _payload_sha256(completion)
    seen_ids, seen_indices = set(), set()
    corrected = []

    for row in rows:
        if row.get("status") != "computed":
            raise ValueError(f"candidate_sg_not_computed:{row.get('record_id')}")
        index = row.get("composition_index")
        number = row.get("space_group_number")
        record_id = row.get("record_id")
        cif_hash = row.get("verified_cif_sha256")
        if type(index) is not int or not 0 <= index < count:
            raise ValueError(f"composition_index_out_of_range:{index}")
        if type(number) is not int or not 1 <= number <= 230:
            raise ValueError(f"invalid_computed_space_group:{record_id}")
        if index in seen_indices or record_id in seen_ids:
            raise ValueError("duplicate_candidate_identity")
        seen_indices.add(index)
        seen_ids.add(record_id)

        meta = metadata[index]
        if meta.get("record_id") != record_id:
            raise ValueError(f"record_id_index_mismatch:{index}")
        if str(meta.get("structure_sha256", "")).lower() != str(cif_hash).lower():
            raise ValueError(f"structure_sha256_mismatch:{record_id}")
        if float(structure_mask[index, 0].item()) != 1.0:
            raise ValueError(f"structure_not_supported:{record_id}")
        if float(sg_mask[index, 0].item()) != 0.0:
            raise ValueError(f"space_group_already_supported:{record_id}")
        if torch.count_nonzero(sg[index]).item() != 0:
            raise ValueError(f"masked_space_group_features_not_zero:{record_id}")

        patched_features["sg_features"][index].zero_()
        patched_features["sg_features"][index, number - 1] = 1.0
        patched_features["sg_features"][index, 230 + _crystal_system_index(number)] = 1.0
        patched_features["sg_mask"][index, 0] = 1.0
        patched_metadata[index].update({
            "model_space_group_number": number,
            "model_sg_symprec_angstrom": SYMPREC_ANGSTROM,
            "model_sg_angle_tolerance_degrees": ANGLE_TOLERANCE_DEGREES,
            "model_space_group_origin": "candidate_CIF_symmetry_completion",
            "model_sg_candidate_structure_sha256": cif_hash,
            "model_sg_previous_tolerance_numbers": row.get("previous_numbers", []),
            "model_sg_completion_payload_sha256": payload_hash,
        })
        corrected.append({"record_id": record_id, "composition_index": index,
                          "space_group_number": number, "structure_sha256": cif_hash})

    return patched_features, patched_metadata, {
        "schema": "candidate_symmetry_overlay_v1",
        "completion_payload_sha256": payload_hash,
        "rows_patched": len(corrected),
        "patched_rows": corrected,
        "feature_fields_changed": ["sg_features", "sg_mask"],
    }
