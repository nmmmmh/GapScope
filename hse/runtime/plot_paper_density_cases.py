"""Make one bounded, provenance-recorded FROZEN OOF density figure for three cases."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "BandgapFusion_CorrespondenceWork" / "density_cases"
REPORT_SCORES = ROOT / "results/formula_and_modalities/report_scores.csv"
OOF_DENSITIES = ROOT / "results/formula_and_modalities/oof_densities.csv"
STRUCTURE_INDEX = ROOT / "results/physical_cases/resolved_structure_index.csv"
PAIR_INDEX = ROOT / "results/physical_cases/same_formula_cif_pairs.csv"
REPORT_RECORDS = ROOT / "data/reports/records.jsonl"
SEEDS = ("17", "29", "41")
X_MIN = 0.001
DISPLAY_CAP = 8.0
STEP = 0.005
TOL = 1e-8
PHASE_LABELS = {
    "07e6db13": "δ (SG 62)", "dd937dd6": "γ (SG 62)",
    "3ebd5475": "tetragonal zircon (SG 141)",
    "dc01a29b": "tetragonal scheelite (SG 88)",
    "f7b43c09": "monoclinic scheelite (SG 15)",
    "0eaf5155": "rutile (SG 136)", "39111d3e": "anatase (SG 141)",
    "fab07256": "brookite (SG 61)",
}

CASES = (
    {
        "name": "CsPbI3",
        "composition_key": "Cs:0.200000000000|I:0.600000000000|Pb:0.200000000000",
        "role": "unique same-SG pair (SG 62)",
    },
    {
        "name": "BiVO4",
        "composition_key": "Bi:0.166666666667|O:0.666666666667|V:0.166666666667",
        "role": "multiple CIF structures",
    },
    {
        "name": "TiO2",
        "composition_key": "O:0.666666666667|Ti:0.333333333333",
        "role": "phase-labeled report series",
    },
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_records(path: Path, wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            mid = row.get("measurement_id")
            if mid in wanted:
                if mid in found:
                    raise ValueError(f"duplicate_report_metadata:{mid}")
                found[mid] = row
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"report_metadata_missing:{sorted(missing)}")
    return found


def require_one(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    if len(rows) != 1:
        raise ValueError(f"expected_one_row:{label}:{len(rows)}")
    return rows[0]


def parse_components(row: dict[str, str]) -> dict[str, list[float]]:
    parsed = {
        field: json.loads(row[field])
        for field in ("component_weights", "log_mus", "sigmas")
    }
    if any(len(values) != 2 for values in parsed.values()):
        raise ValueError(f"expected_two_density_components:{row['measurement_id']}:{row['seed']}")
    if not np.isfinite(np.asarray(sum(parsed.values(), []), dtype=float)).all():
        raise ValueError(f"nonfinite_density_parameters:{row['measurement_id']}:{row['seed']}")
    if any(value <= 0 for value in parsed["sigmas"]):
        raise ValueError(f"nonpositive_sigma:{row['measurement_id']}:{row['seed']}")
    if not math.isclose(sum(parsed["component_weights"]), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"component_weights_do_not_sum_to_one:{row['measurement_id']}:{row['seed']}")
    return parsed


def check_same_prediction(rows: list[dict[str, str]], label: str) -> None:
    if len(rows) < 2:
        return
    parsed = [parse_components(row) for row in rows]
    for field in ("component_weights", "log_mus", "sigmas"):
        baseline = np.asarray(parsed[0][field], dtype=float)
        for item in parsed[1:]:
            delta = np.max(np.abs(np.asarray(item[field], dtype=float) - baseline))
            if delta > TOL:
                details = ",".join(row["measurement_id"] for row in rows)
                raise ValueError(f"inconsistent_predictions:{label}:{field}:max_delta={delta}:{details}")


def density(x: np.ndarray, components: list[dict[str, float]]) -> np.ndarray:
    """Evaluate the original positive-support lognormal mixture, without display renormalization."""
    log_x = np.log(x)
    total = np.zeros_like(x, dtype=float)
    for component in components:
        weight = component["weight"]
        if weight <= 0:
            continue
        sigma = component["sigma"]
        z = (log_x - component["log_mu"]) / sigma
        total += (weight / (x * sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * z * z)
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports_all = read_csv(REPORT_SCORES)
    reports = [
        row for row in reports_all
        if row["model"] == "FROZEN"
        and row["view"] == "full"
        and row["seed_mixture"] == "equal_density_mixture"
    ]
    if len(reports) != 205 or len({row["measurement_id"] for row in reports}) != 205:
        raise ValueError(f"expected_205_unique_frozen_full_mixture_reports:{len(reports)}")

    case_by_key = {case["composition_key"]: case for case in CASES}
    report_by_case: dict[str, list[dict[str, str]]] = {
        case["name"]: sorted(
            [row for row in reports if row["composition_key"] == case["composition_key"]],
            key=lambda row: row["measurement_id"],
        )
        for case in CASES
    }
    if any(not report_by_case[case["name"]] for case in CASES):
        raise ValueError("one_or_more_fixed_case_keys_not_found")
    wanted_ids = {row["measurement_id"] for rows in report_by_case.values() for row in rows}
    metadata = read_records(REPORT_RECORDS, wanted_ids)

    index_rows = read_csv(STRUCTURE_INDEX)
    index_by_id: dict[str, dict[str, str]] = {}
    for row in index_rows:
        if row["input_support"] != "current encoded OOF structure mask":
            continue
        if row["measurement_id"] in wanted_ids:
            if row["measurement_id"] in index_by_id:
                raise ValueError(f"duplicate_current_structure_index:{row['measurement_id']}")
            index_by_id[row["measurement_id"]] = row

    pair_rows = read_csv(PAIR_INDEX)
    sg_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        for suffix in ("a", "b"):
            cif_hash, sg = row[f"cif_{suffix}"], row[f"sg_{suffix}"]
            if cif_hash and sg:
                sg_by_hash[cif_hash].add(sg)

    selected_hashes: set[str] = set()
    for case in CASES:
        for row in report_by_case[case["name"]]:
            index_row = index_by_id.get(row["measurement_id"])
            has_cif = index_row is not None
            if has_cif != (row["structure_supported"] == "1"):
                raise ValueError(f"structure_mask_index_mismatch:{row['measurement_id']}")
            if index_row:
                cif_hash = index_row["actual_cif_sha256"]
                selected_hashes.add(cif_hash)
                sgs = sg_by_hash.get(cif_hash, set())
                if len(sgs) != 1:
                    raise ValueError(f"missing_or_conflicting_cif_sg:{cif_hash}:{sorted(sgs)}")
                cif_path = ROOT / index_row["cif_path"]
                if not cif_path.is_file() or sha256(cif_path) != cif_hash:
                    raise ValueError(f"cif_file_hash_mismatch:{cif_path}:{cif_hash}")

    oof_rows = read_csv(OOF_DENSITIES)
    oof_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in oof_rows:
        if row["model"] == "FROZEN" and row["view"] in ("formula_only", "full"):
            oof_by_key[(row["view"], row["seed"], row["measurement_id"])].append(row)

    representatives: dict[tuple[str, str], str] = {}
    curve_specs: list[dict] = []
    consistency_audit: list[dict] = []
    for case in CASES:
        case_name = case["name"]
        case_reports = report_by_case[case_name]
        formula_rep = case_reports[0]["measurement_id"]
        representatives[(case_name, "formula_only")] = formula_rep
        check_rows: list[dict[str, str]] = []
        for report in case_reports:
            for seed in SEEDS:
                check_rows.append(require_one(
                    oof_by_key.get(("formula_only", seed, report["measurement_id"]), []),
                    f"formula_only:{case_name}:{seed}:{report['measurement_id']}",
                ))
        for seed in SEEDS:
            seed_rows = [row for row in check_rows if row["seed"] == seed]
            check_same_prediction(seed_rows, f"formula_only:{case_name}:{seed}")
        curve_specs.append({
            "case": case,
            "kind": "formula_only",
            "hash": "",
            "sg": "",
            "representative": formula_rep,
        })

        by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
        for report in case_reports:
            index_row = index_by_id.get(report["measurement_id"])
            if index_row:
                by_hash[index_row["actual_cif_sha256"]].append(report)
        case_sgs = {cif_hash: next(iter(sg_by_hash[cif_hash])) for cif_hash in by_hash}
        if case_name == "CsPbI3" and sorted(case_sgs.values()) != ["62", "62"]:
            raise ValueError(f"CsPbI3_expected_unique_same_sg62_pair:{case_sgs}")
        for cif_hash, members in sorted(by_hash.items()):
            rep = sorted(members, key=lambda row: row["measurement_id"])[0]["measurement_id"]
            representatives[(case_name, f"full:{cif_hash}")] = rep
            for seed in SEEDS:
                group_rows = [require_one(
                    oof_by_key.get(("full", seed, row["measurement_id"]), []),
                    f"full:{case_name}:{cif_hash}:{seed}:{row['measurement_id']}",
                ) for row in members]
                check_same_prediction(group_rows, f"full:{case_name}:{cif_hash}:{seed}")
                consistency_audit.append({
                    "case": case_name,
                    "actual_cif_sha256": cif_hash,
                    "seed": seed,
                    "report_count": len(group_rows),
                    "max_parameter_delta": 0.0 if len(group_rows) < 2 else max(
                        float(np.max(np.abs(np.asarray(json.loads(group_rows[i][field]), dtype=float)
                                                   - np.asarray(json.loads(group_rows[0][field]), dtype=float))))
                        for i in range(1, len(group_rows))
                        for field in ("component_weights", "log_mus", "sigmas")
                    ),
                    "consistent_within_tolerance": True,
                    "tolerance": TOL,
                })
            curve_specs.append({
                "case": case,
                "kind": "full_actual_cif",
                "hash": cif_hash,
                "sg": case_sgs[cif_hash],
                "representative": rep,
            })

    curve_components: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    q99_max = X_MIN
    for spec in curve_specs:
        case_name = spec["case"]["name"]
        view = "formula_only" if spec["kind"] == "formula_only" else "full"
        components: list[dict[str, float]] = []
        for seed in SEEDS:
            row = require_one(oof_by_key.get((view, seed, spec["representative"]), []),
                              f"curve:{case_name}:{view}:{seed}:{spec['representative']}")
            parsed = parse_components(row)
            for weight, mu, sigma in zip(parsed["component_weights"], parsed["log_mus"], parsed["sigmas"]):
                components.append({"weight": weight / len(SEEDS), "log_mu": mu, "sigma": sigma})
                log_q99 = mu + 2.3263478740408408 * sigma
                q99_max = max(q99_max, DISPLAY_CAP if log_q99 >= math.log(DISPLAY_CAP)
                              else math.exp(log_q99))
        curve_components[(case_name, spec["kind"], spec["hash"])] = components

    all_report_targets = [float(row["target_ev"]) for rows in report_by_case.values() for row in rows]
    x_max_candidate = min(DISPLAY_CAP, max(max(all_report_targets) + 1.0, q99_max))
    n_steps = int(np.floor((x_max_candidate - X_MIN) / STEP + 1e-12))
    x_values = X_MIN + np.arange(n_steps + 1, dtype=float) * STEP
    x_max = float(x_values[-1])
    if not len(x_values) or x_values[0] != X_MIN:
        raise ValueError("invalid_display_grid")

    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
    dash_styles = ((0, (5, 2)), (0, (3, 1.3, 1, 1.3)), (0, (1.5, 1.2)))
    colors_by_case: dict[str, dict[str, str]] = {}
    specs_by_case: dict[str, list[dict]] = defaultdict(list)
    for spec in curve_specs:
        specs_by_case[spec["case"]["name"]].append(spec)
    for case_name, specs in specs_by_case.items():
        hashes = sorted(spec["hash"] for spec in specs if spec["hash"])
        colors_by_case[case_name] = {h: palette[i % len(palette)] for i, h in enumerate(hashes)}

    OUT.mkdir(parents=True, exist_ok=True)
    curves_path = OUT / "density_curves.csv"
    curve_fields = ["case", "case_role", "composition_key", "formula", "curve_kind", "view",
                    "seed_mixture", "actual_cif_sha256", "actual_cif_short", "cif_sg",
                    "representative_measurement_id", "mixture_component_count", "x_ev", "density_per_ev"]
    curve_rows: list[dict[str, str]] = []
    curve_y: dict[tuple[str, str, str], np.ndarray] = {}
    for spec in curve_specs:
        case = spec["case"]
        key = (case["name"], spec["kind"], spec["hash"])
        y_values = density(x_values, curve_components[key])
        curve_y[key] = y_values
        representative = spec["representative"]
        formula = next(row["formula"] for row in report_by_case[case["name"]]
                       if row["measurement_id"] == representative)
        view = "formula_only" if spec["kind"] == "formula_only" else "full"
        for x, y in zip(x_values, y_values):
            curve_rows.append({
                "case": case["name"], "case_role": case["role"],
                "composition_key": case["composition_key"], "formula": formula,
                "curve_kind": spec["kind"], "view": view, "seed_mixture": "equal_density_mixture",
                "actual_cif_sha256": spec["hash"], "actual_cif_short": spec["hash"][:10] if spec["hash"] else "",
                "cif_sg": spec["sg"], "representative_measurement_id": representative,
                "mixture_component_count": len(SEEDS) * 2, "x_ev": f"{x:.3f}", "density_per_ev": f"{y:.12g}",
            })
    with curves_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    report_csv = OUT / "report_observations.csv"
    report_fields = ["case", "case_role", "measurement_id", "formula", "composition_key", "phase_label",
                     "target_ev", "source_group_id", "source_record_id", "source_title", "source_url",
                     "source_locator", "report_space_group", "sg_supported", "structure_supported",
                     "actual_cif_sha256", "actual_cif_short", "cif_sg", "cif_path", "input_support",
                     "prediction_model", "prediction_view", "prediction_seed_mixture", "rug_group"]
    report_rows: list[dict[str, str]] = []
    for case in CASES:
        for report in report_by_case[case["name"]]:
            mid = report["measurement_id"]
            meta = metadata[mid]
            index_row = index_by_id.get(mid)
            cif_hash = index_row["actual_cif_sha256"] if index_row else ""
            report_rows.append({
                "case": case["name"], "case_role": case["role"], "measurement_id": mid,
                "formula": report["formula"], "composition_key": report["composition_key"],
                "phase_label": report["phase_label"], "target_ev": report["target_ev"],
                "source_group_id": meta.get("source_group_id", ""),
                "source_record_id": meta.get("source_record_id", ""),
                "source_title": meta.get("source_title", ""), "source_url": meta.get("source_url", ""),
                "source_locator": meta.get("source_locator", ""),
                "report_space_group": "" if meta.get("space_group") is None else str(meta.get("space_group")),
                "sg_supported": report["sg_supported"], "structure_supported": report["structure_supported"],
                "actual_cif_sha256": cif_hash, "actual_cif_short": cif_hash[:10] if cif_hash else "",
                "cif_sg": next(iter(sg_by_hash[cif_hash])) if cif_hash else "",
                "cif_path": index_row["cif_path"] if index_row else "",
                "input_support": index_row["input_support"] if index_row else "no indexed CIF",
                "prediction_model": "FROZEN", "prediction_view": "full",
                "prediction_seed_mixture": "equal_density_mixture",
                "rug_group": f"SG{next(iter(sg_by_hash[cif_hash]))} {cif_hash[:10]}" if cif_hash else "no CIF",
            })
    with report_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharex=True, layout="constrained")
    fig.suptitle("FROZEN · composition-held-out development OOF",
                 fontsize=13, fontweight="bold")
    for ax, case in zip(axes, CASES):
        case_name = case["name"]
        specs = specs_by_case[case_name]
        formula_spec = next(spec for spec in specs if spec["kind"] == "formula_only")
        formula_key = (case_name, "formula_only", "")
        ax.plot(x_values, curve_y[formula_key], color="#1A1A1A", linewidth=2.1,
                label="Formula-only density", zorder=4)
        handles = [Line2D([0], [0], color="#1A1A1A", linewidth=2.1, label="Formula-only density")]
        for index, spec in enumerate(spec for spec in specs if spec["kind"] == "full_actual_cif"):
            cif_hash = spec["hash"]
            color = colors_by_case[case_name][cif_hash]
            style = dash_styles[index % len(dash_styles)]
            curve_key = (case_name, "full_actual_cif", cif_hash)
            ax.plot(x_values, curve_y[curve_key], color=color, linewidth=1.8,
                    linestyle=style, label=PHASE_LABELS[cif_hash[:8]], zorder=3)
            handles.append(Line2D([0], [0], color=color, linewidth=1.8, linestyle=style,
                                  label=PHASE_LABELS[cif_hash[:8]]))

        rug_rows = [row for row in report_rows if row["case"] == case_name]
        seen_at_x: dict[float, int] = defaultdict(int)
        for row in rug_rows:
            target = float(row["target_ev"])
            cif_hash = row["actual_cif_sha256"]
            color = colors_by_case[case_name][cif_hash] if cif_hash else "#777777"
            stack = seen_at_x[target]
            seen_at_x[target] += 1
            low = 0.012 + 0.014 * stack
            high = low + 0.025
            ax.plot([target, target], [low, high], transform=ax.get_xaxis_transform(),
                    color=color, linewidth=1.15, alpha=0.9, clip_on=False, zorder=5)
        if any(not row["actual_cif_sha256"] for row in rug_rows):
            handles.append(Line2D([0], [0], color="#777777", linewidth=1.2,
                                  linestyle="None", marker="|", markersize=8,
                                  label="Report without indexed CIF"))
        title_formula = next(row["formula"] for row in report_by_case[case_name])
        ax.set_title(f"{title_formula} — {case['role']}", fontsize=10.5)
        ax.set_xlim(X_MIN, x_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Reported bandgap, $E_g$ (eV)")
        ax.set_ylabel("Probability density (eV$^{-1}$)")
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.55, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")

    caption = (
        f"Each prediction curve is the equal-density average of the three FROZEN seeds "
        f"(two lognormal components per seed); mixture components are statistical density terms, not physical phases. "
        f"Ticks show every report target, colored by indexed actual-CIF hash; grey means no indexed CIF. "
        f"Display range: {X_MIN:.3f}–{x_max:.3f} eV; lognormal density is evaluated at 0.001-eV spacing and is not "
        f"renormalized after display truncation."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=8.1, wrap=True)
    png_path, pdf_path = OUT / "paper_density_cases.png", OUT / "paper_density_cases.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    input_paths = [REPORT_SCORES, OOF_DENSITIES, STRUCTURE_INDEX, PAIR_INDEX, REPORT_RECORDS]
    cif_files = {}
    for cif_hash in sorted(selected_hashes):
        match = next(row for row in index_rows if row["actual_cif_sha256"] == cif_hash
                     and row["input_support"] == "current encoded OOF structure mask")
        path = ROOT / match["cif_path"]
        cif_files[cif_hash] = {"path": str(path), "sha256": sha256(path)}
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fixed diagnostic illustration of FROZEN grouped-development OOF densities; not performance selection",
        "model": "FROZEN",
        "evaluation_scope": "composition-held-out development OOF; not independent external validation",
        "fixed_cases": [{"formula": case["name"], "composition_key": case["composition_key"], "role": case["role"]}
                        for case in CASES],
        "prediction_construction": {
            "formula_curve": "one deterministic representative report per composition_key (lexicographically first measurement_id); formula_only view; all three seeds",
            "full_curves": "one deterministic representative report per actual_cif_sha256; full view; all three seeds",
            "seed_mixture": "equal density across seeds; each seed's two lognormal component weights divided by 3",
            "within_cif_prediction_check": {"fields": ["component_weights", "log_mus", "sigmas"], "absolute_tolerance": TOL,
                                             "result": "all grouped predictions agree within tolerance"},
            "formula_only_group_check": {"fields": ["component_weights", "log_mus", "sigmas"], "absolute_tolerance": TOL,
                                          "result": "reports within each selected formula agree within tolerance"},
        },
        "observations": "all FROZEN equal_density_mixture targets in selected composition_key groups; raw phase_label/source fields retained in report_observations.csv; no CIF is grey",
        "phase_display_labels": PHASE_LABELS,
        "actual_cif_space_group": "from same_formula_cif_pairs.csv; conflicting or absent SG mappings stop generation",
        "same_sg_case": {"formula": "CsPbI3", "space_group": "62", "distinct_actual_cif_hashes": sorted(
            spec["hash"] for spec in curve_specs if spec["case"]["name"] == "CsPbI3" and spec["hash"]
        )},
        "density": {
            "distribution": "finite lognormal mixture evaluated on positive Eg support",
            "x_min_eV": X_MIN, "x_max_eV": float(x_values[-1]), "step_eV": STEP,
            "display_rule": "min(8 eV, max(selected observed maximum + 1 eV, largest component 99th percentile)); grid display only; no truncation renormalization",
            "y_axis": "probability density per eV; independent y limits per panel",
        },
        "figure_caption": caption,
        "representative_measurement_ids": {
            f"{case_name}:{kind}": mid for (case_name, kind), mid in sorted(representatives.items())
        },
        "cif_prediction_consistency_audit": consistency_audit,
        "source_files": {str(path): {"sha256": sha256(path)} for path in input_paths},
        "indexed_cif_files": cif_files,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "output_files": {},
    }
    outputs = [png_path, pdf_path, curves_path, report_csv, Path(__file__).resolve()]
    protocol["output_files"] = {path.name if path.parent == OUT else "plot_paper_density_cases.py":
                                {"path": str(path), "sha256": sha256(path)} for path in outputs}
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(OUT), "reports": len(report_rows), "curve_rows": len(curve_rows),
        "x_range_eV": [X_MIN, float(x_values[-1])], "case_counts": {
            case["name"]: {"reports": len(report_by_case[case["name"]]),
                           "actual_cif_groups": sum(1 for spec in curve_specs
                                                     if spec["case"]["name"] == case["name"] and spec["hash"])}
            for case in CASES
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

