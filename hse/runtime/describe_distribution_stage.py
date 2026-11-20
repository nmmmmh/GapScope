"""Create paper-style figures from frozen distribution-head diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VIEWS = ("formula_only", "full")
SUBSETS = ("all", "structure", "missing_structure")
SEEDS = ("17", "29", "41")
SEED_MARKERS = {"17": "o", "29": "s", "41": "^"}
ENSEMBLE = "equal_density_mixture"
MODEL_COLORS = {"SINGLE": "#0072B2", "FROZEN": "#D55E00"}
FIGURE_DATA_FIELDS = (
    "figure", "data_kind", "model", "view", "subset", "metric", "seed_mixture",
    "value", "paired_delta_single_minus_frozen", "reports", "formulas",
    "composition_key", "formula", "cif_a", "cif_b", "observed_mean_delta_ev",
    "pred_mean_delta_formula_only_ev", "pred_mean_delta_formula_sg_ev",
    "pred_mean_delta_full_ev", "attention_structure", "structure_w1_given_sg_ev",
    "formula_report_weight_sum", "report_count",
)


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_input_csv:{path}")
    return rows


def _float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid_numeric_field:{key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite_numeric_field:{key}")
    return value


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_to(path: Path, base: Path):
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _validate_source_protocol(protocol_path: Path, bound_files):
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_source_protocol:{protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    declared = protocol.get("output_binding", {})
    for path in bound_files:
        expected = declared.get(path.name)
        if expected is not None and expected != _sha(path):
            raise ValueError(f"source_output_hash_mismatch:{path.name}")
    return protocol, _sha(protocol_path)


def _mapped_metrics(metrics, contrasts):
    metric_map = {}
    for row in metrics:
        key = (row["model"], row["view"], row["seed_mixture"], row["subset"])
        if key in metric_map:
            raise ValueError(f"duplicate_metric_row:{key}")
        metric_map[key] = row

    contrast_map = {}
    for row in contrasts:
        if row["contrast"] != "two_vs_one_component":
            continue
        key = (row["view"], row["seed_mixture"], row["subset"], row["metric"])
        if key in contrast_map:
            raise ValueError(f"duplicate_contrast_row:{key}")
        contrast_map[key] = row

    values = []
    for view in VIEWS:
        for subset in SUBSETS:
            for model in ("SINGLE", "FROZEN"):
                for mixture in (ENSEMBLE, *SEEDS):
                    key = (model, view, mixture, subset)
                    if key not in metric_map:
                        raise ValueError(f"missing_metric_row:{key}")
            single = metric_map[("SINGLE", view, ENSEMBLE, subset)]
            frozen = metric_map[("FROZEN", view, ENSEMBLE, subset)]
            if single["reports"] != frozen["reports"]:
                raise ValueError(f"unpaired_metric_report_count:{view}:{subset}")
            for metric, field in (("nll", "nll"), ("crps_ev", "crps_ev")):
                key = (view, ENSEMBLE, subset, metric)
                if key not in contrast_map:
                    raise ValueError(f"missing_ensemble_contrast:{key}")
                delta = _float(single, field) - _float(frozen, field)
                declared_delta = _float(contrast_map[key], "improvement")
                if not math.isclose(delta, declared_delta, rel_tol=0.0, abs_tol=1e-10):
                    raise ValueError(f"ensemble_contrast_mismatch:{key}")

            for model in ("SINGLE", "FROZEN"):
                for mixture in (ENSEMBLE, *SEEDS):
                    row = metric_map[(model, view, mixture, subset)]
                    for metric, field in (("nll", "nll"), ("crps_ev", "crps_ev")):
                        entry = {name: "" for name in FIGURE_DATA_FIELDS}
                        entry.update(
                            figure="head_comparison", data_kind="metric_point", model=model,
                            view=view, subset=subset, metric=metric, seed_mixture=mixture,
                            value=_float(row, field), reports=row["reports"],
                            formulas=row.get("formulas", ""),
                        )
                        if mixture == ENSEMBLE:
                            entry["paired_delta_single_minus_frozen"] = _float(
                                contrast_map[(view, ENSEMBLE, subset, metric)], "improvement")
                        values.append(entry)
    return metric_map, values


def _mode_notes(mode_rows):
    notes = {}
    for model in ("SINGLE", "FROZEN"):
        for view in VIEWS:
            key = (model, view, ENSEMBLE)
            matched = [row for row in mode_rows if
                       (row["model"], row["view"], row["seed_mixture"]) == key]
            if len(matched) != 1:
                raise ValueError(f"missing_or_duplicate_ensemble_mode_row:{key}:{len(matched)}")
            notes[(model, view)] = matched[0]
    return notes


def _pair_rows(raw_pairs):
    rows = [row for row in raw_pairs if row["seed_mixture"] == ENSEMBLE]
    if not rows:
        raise ValueError("missing_equal_density_cif_pair_rows")
    seen = set()
    for row in rows:
        key = (row["composition_key"], row["cif_a"], row["cif_b"])
        if key in seen:
            raise ValueError(f"duplicate_ensemble_cif_pair:{key}")
        seen.add(key)
        for field in ("observed_mean_delta_ev", "pred_mean_delta_formula_only_ev",
                      "pred_mean_delta_formula_sg_ev", "pred_mean_delta_full_ev"):
            _float(row, field)
        if abs(_float(row, "pred_mean_delta_formula_only_ev")) > 1e-8:
            raise ValueError("formula_only_same_formula_delta_not_zero")
    return rows


def _weighted_attention_points(interactions):
    rows = [row for row in interactions if row["seed_mixture"] == ENSEMBLE
            and int(float(row["structure_supported"])) == 1]
    if not rows:
        raise ValueError("missing_structurally_supported_ensemble_interactions")
    grouped = defaultdict(list)
    for row in rows:
        weight = _float(row, "weight")
        if weight < 0:
            raise ValueError("negative_report_interaction_weight")
        grouped[row["composition_key"]].append((row, weight))

    points = []
    for composition_key, members in sorted(grouped.items()):
        total = sum(weight for _, weight in members)
        if total <= 0:
            raise ValueError(f"zero_formula_interaction_weight:{composition_key}")
        attention = sum(weight * _float(row, "attention_structure")
                        for row, weight in members) / total
        response = sum(weight * _float(row, "structure_w1_given_sg_ev")
                       for row, weight in members) / total
        points.append({
            "composition_key": composition_key,
            "formula": members[0][0].get("formula", ""),
            "attention_structure": attention,
            "structure_w1_given_sg_ev": response,
            "formula_report_weight_sum": total,
            "report_count": len(members),
        })
    return points


def _write_figure_data(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIGURE_DATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output_dir: Path, stem: str):
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _head_comparison_figure(metric_map, mode_notes, output_dir):
    labels = [
        "Formula-only\nAll", "Formula-only\nStructure", "Formula-only\nMissing structure",
        "Full\nAll", "Full\nStructure", "Full\nMissing structure",
    ]
    categories = [(view, subset) for view in VIEWS for subset in SUBSETS]
    x = list(range(len(categories)))
    model_offsets = {"SINGLE": -0.18, "FROZEN": 0.18}
    seed_offsets = dict(zip(SEEDS, (-0.055, 0.0, 0.055)))
    metrics = (("nll", "NLL (lower is better)"),
               ("crps_ev", "CRPS (eV; lower is better)"))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))

    for axis, (metric, ylabel) in zip(axes, metrics):
        all_values = []
        for model in ("SINGLE", "FROZEN"):
            color = MODEL_COLORS[model]
            for category_index, (view, subset) in enumerate(categories):
                base_x = x[category_index] + model_offsets[model]
                ensemble = metric_map[(model, view, ENSEMBLE, subset)]
                ensemble_value = _float(ensemble, metric)
                all_values.append(ensemble_value)
                axis.scatter(base_x, ensemble_value, marker="D", s=58, color=color,
                             edgecolor="black", linewidth=0.45, zorder=4)
                for seed in SEEDS:
                    seed_row = metric_map[(model, view, seed, subset)]
                    seed_value = _float(seed_row, metric)
                    all_values.append(seed_value)
                    axis.scatter(base_x + seed_offsets[seed], seed_value,
                                 marker=SEED_MARKERS[seed], s=27,
                                 color=color, alpha=0.82, edgecolor="white",
                                 linewidth=0.35, zorder=3)

        for category_index, (view, subset) in enumerate(categories):
            left = _float(metric_map[("SINGLE", view, ENSEMBLE, subset)], metric)
            right = _float(metric_map[("FROZEN", view, ENSEMBLE, subset)], metric)
            axis.plot([x[category_index] + model_offsets["SINGLE"],
                       x[category_index] + model_offsets["FROZEN"]], [left, right],
                      color="#666666", linewidth=0.9, alpha=0.72, zorder=2)

        low, high = min(all_values), max(all_values)
        padding = max((high - low) * 0.12, abs(high) * 0.025, 1e-3)
        axis.set_ylim(low - padding, high + padding)
        axis.set_xlim(-0.55, len(categories) - 0.45)
        axis.set_xticks(x, labels, fontsize=8.2)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(2.5, color="#A0A0A0", linewidth=0.75, linestyle=":")
        axis.set_title("NLL" if metric == "nll" else "CRPS")

    modes = {
        key: 100.0 * _float(row, "report_weighted_multimode_fraction")
        for key, row in mode_notes.items()
    }
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS["SINGLE"],
               markersize=7, label="SINGLE head"),
        Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS["FROZEN"],
               markersize=7, label="FROZEN two-component head"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="#666666",
               markeredgecolor="black", markersize=7, label="Equal-density ensemble"),
        Line2D([], [], marker=SEED_MARKERS["17"], linestyle="", color="#666666",
               markersize=5.5, label="Seed 17"),
        Line2D([], [], marker=SEED_MARKERS["29"], linestyle="", color="#666666",
               markersize=5.5, label="Seed 29"),
        Line2D([], [], marker=SEED_MARKERS["41"], linestyle="", color="#666666",
               markersize=5.5, label="Seed 41"),
        Line2D([], [], color="#666666", linewidth=0.9,
               label="Paired ensemble scores; Δ = SINGLE − FROZEN"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.105), fontsize=9)
    fig.suptitle("Single- and two-component report-density heads", y=0.98, fontsize=14)
    single_formula = modes[("SINGLE", "formula_only")]
    single_full = modes[("SINGLE", "full")]
    frozen_formula = modes[("FROZEN", "formula_only")]
    frozen_full = modes[("FROZEN", "full")]
    fig.text(0.5, 0.035,
             f"Held-formula OOF; report-weighted multimode SINGLE {single_formula:.1f}%/"
             f"{single_full:.1f}%, FROZEN {frozen_formula:.1f}%/"
             f"{frozen_full:.1f}% (formula-only/full). Numerical modes are not phase labels.",
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.32, wspace=0.22)
    _save(fig, output_dir, "head_comparison")


def _structure_response_figure(rows, output_dir):
    x_values = [_float(row, "observed_mean_delta_ev") for row in rows]
    series = (
        ("formula_sg", "Prediction with SG"),
        ("full", "Prediction with SG + structure"),
    )
    y_values = [_float(row, f"pred_mean_delta_{view}_ev")
                for row in rows for view, _ in series]
    limit = max(max(abs(value) for value in x_values + y_values) * 1.08, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharex=True, sharey=True)
    for axis, (view, title) in zip(axes, series):
        axis.scatter(x_values,
                     [_float(row, f"pred_mean_delta_{view}_ev") for row in rows],
                     s=42, color="#0072B2", alpha=0.78, edgecolor="white",
                     linewidth=0.45, zorder=3, label="Same-formula CIF pair")
        axis.plot([-limit, limit], [-limit, limit], color="#666666", linewidth=0.9,
                  linestyle="--", label="Identity line")
        axis.axhline(0.0, color="#D55E00", linewidth=1.0, linestyle=":",
                    label="Formula-only reference: predicted Δ = 0")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E1E1E1", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(title)
        axis.set_xlabel("Observed paired mean-gap difference (eV)")
    axes[0].set_ylabel("Predicted paired mean-gap difference (eV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.075), fontsize=8.5)
    formula_count = len({row["composition_key"] for row in rows})
    fig.suptitle("FROZEN same-formula CIF-reference pair responses", y=0.98, fontsize=14)
    fig.text(0.5, 0.018,
             f"All {len(rows)} encoded CIF pairs across {formula_count} formulas; "
             "reference CIFs are not asserted to be the measured specimens.",
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.27, wspace=0.12)
    _save(fig, output_dir, "structure_response")


def _attention_figure(points, output_dir):
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.scatter([point["attention_structure"] for point in points],
                 [point["structure_w1_given_sg_ev"] for point in points],
                 s=42, color="#009E73", alpha=0.8, edgecolor="white", linewidth=0.45)
    axis.set_xlabel("Full-view structure-token attention\n(within-formula weighted mean)")
    axis.set_ylabel("Structure-conditioned W1 after SG (eV)\n(within-formula weighted mean)")
    axis.set_title("FROZEN attention and structure-conditioned distribution response")
    axis.grid(color="#E1E1E1", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.025,
             f"{len(points)} formulas; descriptive association only. "
             "Attention is not a causal or importance measure.",
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.18)
    _save(fig, output_dir, "attention_structure_w1")


def _is_within(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Directory containing metrics.csv, contrasts.csv, mode_summary.csv, protocol.json")
    parser.add_argument("--interaction-dir", type=Path, required=True,
                        help="Directory containing same_formula_cif_pairs.csv, report_interactions.csv, protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Figure and figure-data directory outside the formal project root")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    interaction_dir = args.interaction_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if _is_within(output_dir, project_root):
        raise ValueError("output_dir_must_be_outside_formal_project_root")
    if any(_is_within(output_dir, source) or _is_within(source, output_dir)
           for source in (analysis_dir, interaction_dir)):
        raise ValueError("output_dir_must_not_overlap_input_directories")

    analysis_files = [analysis_dir / name for name in
                      ("metrics.csv", "contrasts.csv", "mode_summary.csv")]
    interaction_files = [interaction_dir / name for name in
                         ("same_formula_cif_pairs.csv", "report_interactions.csv")]
    protocol_paths = (analysis_dir / "protocol.json", interaction_dir / "protocol.json")
    required_paths = [*analysis_files, *interaction_files, *protocol_paths]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing_figure_input:" + ",".join(missing))

    analysis_protocol, analysis_protocol_sha = _validate_source_protocol(
        protocol_paths[0], analysis_files)
    interaction_protocol, interaction_protocol_sha = _validate_source_protocol(
        protocol_paths[1], interaction_files)
    metric_map, figure_rows = _mapped_metrics(
        _read_csv(analysis_files[0]), _read_csv(analysis_files[1]))
    mode_notes = _mode_notes(_read_csv(analysis_files[2]))
    pair_rows = _pair_rows(_read_csv(interaction_files[0]))
    attention_points = _weighted_attention_points(_read_csv(interaction_files[1]))

    for (model, view), mode_note in mode_notes.items():
        mode_data = {name: "" for name in FIGURE_DATA_FIELDS}
        mode_data.update(
            figure="head_comparison", data_kind="ensemble_mode_summary", model=model,
            view=view, seed_mixture=ENSEMBLE,
            value=_float(mode_note, "report_weighted_multimode_fraction"),
            reports=mode_note.get("reports", ""),
            formulas=mode_note.get("formulas_with_any_multimode_prediction", ""),
        )
        figure_rows.append(mode_data)

    for row in pair_rows:
        entry = {name: "" for name in FIGURE_DATA_FIELDS}
        entry.update(
            figure="structure_response", data_kind="ensemble_cif_pair",
            seed_mixture=ENSEMBLE, composition_key=row["composition_key"],
            formula=row["formula"], cif_a=row["cif_a"], cif_b=row["cif_b"],
            observed_mean_delta_ev=_float(row, "observed_mean_delta_ev"),
            pred_mean_delta_formula_only_ev=_float(row, "pred_mean_delta_formula_only_ev"),
            pred_mean_delta_formula_sg_ev=_float(row, "pred_mean_delta_formula_sg_ev"),
            pred_mean_delta_full_ev=_float(row, "pred_mean_delta_full_ev"),
        )
        figure_rows.append(entry)

    for point in attention_points:
        entry = {name: "" for name in FIGURE_DATA_FIELDS}
        entry.update(
            figure="attention_structure_w1", data_kind="formula_weighted_attention_response",
            seed_mixture=ENSEMBLE, composition_key=point["composition_key"],
            formula=point["formula"], attention_structure=point["attention_structure"],
            structure_w1_given_sg_ev=point["structure_w1_given_sg_ev"],
            formula_report_weight_sum=point["formula_report_weight_sum"],
            report_count=point["report_count"],
        )
        figure_rows.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    _head_comparison_figure(metric_map, mode_notes, output_dir)
    _structure_response_figure(pair_rows, output_dir)
    _attention_figure(attention_points, output_dir)
    figure_data_path = output_dir / "figure_data.csv"
    _write_figure_data(figure_data_path, figure_rows)

    input_binding = {str(path): _sha(path) for path in
                     (*analysis_files, *interaction_files, *protocol_paths)}
    output_binding = {path.name: _sha(path) for path in sorted(output_dir.iterdir())
                      if path.is_file() and path.name != "protocol.json"
                      and path.name != "protocol.sha256"}
    protocol = {
        "schema": "distribution_stage_figures_v1",
        "analysis_protocol_sha256": analysis_protocol_sha,
        "interaction_protocol_sha256": interaction_protocol_sha,
        "input_binding": input_binding,
        "analysis_source_protocol": analysis_protocol,
        "interaction_source_protocol": interaction_protocol,
        "selection": {
            "head_metrics": ["nll", "crps_ev"],
            "views": list(VIEWS),
            "subsets": list(SUBSETS),
            "models": ["SINGLE", "FROZEN"],
            "seed_mixtures": [ENSEMBLE, *SEEDS],
            "seed_markers": dict(SEED_MARKERS),
            "contrast_sign": "SINGLE minus FROZEN; checked against metrics.csv",
            "cif_pair_rows": "all equal_density_mixture rows in same_formula_cif_pairs.csv",
            "formula_reference": "formula-only paired mean delta must be zero within 1e-8 eV",
            "attention_aggregation": "equal-density-mixture reports; structure_supported only; normalize source hierarchical report weights within each formula",
        },
        "counts": {
            "cif_pairs": len(pair_rows),
            "cif_pair_formulas": len({row["composition_key"] for row in pair_rows}),
            "attention_formulas": len(attention_points),
            "figure_data_rows": len(figure_rows),
        },
        "output_binding": output_binding,
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
    protocol_sha = _sha(protocol_path)
    (output_dir / "protocol.sha256").write_text(protocol_sha + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
