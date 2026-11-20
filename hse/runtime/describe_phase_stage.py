"""Render the phase-conditioned residual stage metrics and all CIF-pair responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


MODELS = ("FROZEN", "SINGLE", "PHASE_NLL", "PHASE_PAIR")
PHASE_MODELS = ("PHASE_NLL", "PHASE_PAIR")
SEED_MIXTURES = ("17", "29", "41", "equal_density_mixture")
REPORT_COUNT = {"all": 205, "structure": 96}
COLORS = {
    "17": "#0072B2",
    "29": "#D55E00",
    "41": "#009E73",
    "equal_density_mixture": "#222222",
}
MARKERS = {"17": "o", "29": "s", "41": "^", "equal_density_mixture": "D"}
MIXTURE_LABELS = {
    "17": "Seed 17",
    "29": "Seed 29",
    "41": "Seed 41",
    "equal_density_mixture": "3-seed equal-density mixture",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_fields(rows, fields, label):
    if not rows:
        raise ValueError(f"empty_input:{label}")
    missing = set(fields) - set(rows[0])
    if missing:
        raise ValueError(f"input_fields_missing:{label}:{sorted(missing)}")


def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite_value:{label}:{value}")
    return number


def selected_metrics(rows):
    lookup = {}
    for row in rows:
        key = (row["model"], row["view"], row["seed_mixture"], row["subset"])
        if key in lookup:
            raise ValueError(f"duplicate_metric_row:{key}")
        lookup[key] = row
    selections = (
        ("full", "all", "All reports · full view (n=205)", 205),
        ("full", "structure", "Structure-supported reports · full view (n=96)", 96),
    )
    chosen = []
    for mixture in SEED_MIXTURES:
        frozen = lookup.get(("FROZEN", "formula_only", mixture, "all"))
        if frozen is None:
            raise ValueError(f"formula_only_frozen_row_missing:{mixture}")
        for model in PHASE_MODELS:
            phase_row = lookup.get((model, "formula_only", mixture, "all"))
            if phase_row is None:
                raise ValueError(f"formula_only_phase_row_missing:{model}:{mixture}")
            for metric in ("nll", "crps_ev"):
                difference = abs(finite(phase_row[metric], metric) - finite(frozen[metric], metric))
                if difference > 1e-7:
                    raise ValueError(f"formula_only_not_frozen_equivalent:{model}:{mixture}:{metric}:{difference}")
    for model in MODELS:
        for mixture in SEED_MIXTURES:
            for view, subset, panel, report_count in selections:
                row = lookup.get((model, view, mixture, subset))
                if row is None:
                    raise ValueError(f"metric_row_missing:{model}:{view}:{mixture}:{subset}")
                if int(row["reports"]) != report_count:
                    raise ValueError(f"metric_report_count_mismatch:{model}:{view}:{mixture}:{subset}:{row['reports']}")
                chosen.append({
                    "model": model,
                    "view": view,
                    "seed_mixture": mixture,
                    "subset": subset,
                    "panel": panel,
                    "reports": report_count,
                    "nll": finite(row["nll"], f"nll:{model}:{mixture}:{subset}"),
                    "crps_ev": finite(row["crps_ev"], f"crps:{model}:{mixture}:{subset}"),
                })
    if len(chosen) != len(MODELS) * len(SEED_MIXTURES) * len(selections):
        raise ValueError("selected_metric_row_count_mismatch")
    return chosen


def selected_pairs(pair_rows, pair_summaries):
    summary_lookup = {}
    for row in pair_summaries:
        key = (row["model"], row["seed_mixture"])
        if key in summary_lookup:
            raise ValueError(f"duplicate_pair_summary:{key}")
        summary_lookup[key] = row

    selected = []
    for model in PHASE_MODELS:
        for mixture in SEED_MIXTURES:
            summary = summary_lookup.get((model, mixture))
            if summary is None:
                raise ValueError(f"pair_summary_missing:{model}:{mixture}")
            if int(summary["formulas"]) != 18 or int(summary["pairs"]) != 22:
                raise ValueError(f"pair_summary_coverage_mismatch:{model}:{mixture}")
            if summary["weighting"] != "equal formula then equal within-formula CIF pair":
                raise ValueError(f"pair_summary_weighting_mismatch:{model}:{mixture}")
            rows = [row for row in pair_rows
                    if row["model"] == model and row["seed_mixture"] == mixture]
            identities = {(row["composition_key"], row["cif_a"], row["cif_b"]) for row in rows}
            formulas = {row["composition_key"] for row in rows}
            if len(rows) != 22 or len(identities) != 22 or len(formulas) != 18:
                raise ValueError(f"pair_rows_coverage_mismatch:{model}:{mixture}:{len(rows)}:{len(identities)}:{len(formulas)}")
            weights = [finite(row["weight"], f"pair_weight:{model}:{mixture}") for row in rows]
            if abs(sum(weights) - 1.0) > 1e-10:
                raise ValueError(f"pair_weights_not_normalized:{model}:{mixture}")
            weights_by_formula = defaultdict(float)
            for row in rows:
                weights_by_formula[row["composition_key"]] += finite(row["weight"], "pair_weight")
            if len(weights_by_formula) != 18 or any(abs(weight - 1 / 18) > 1e-10
                                                     for weight in weights_by_formula.values()):
                raise ValueError(f"pair_weights_not_equal_formula:{model}:{mixture}")
            weighted_error = 0.0
            for row in rows:
                observed = finite(row["observed_mean_delta_ev"], f"observed_pair:{model}:{mixture}")
                predicted = finite(row["pred_mean_delta_full_ev"], f"predicted_pair:{model}:{mixture}")
                weighted_error += finite(row["weight"], "pair_weight") * abs(predicted - observed)
                formula_delta = finite(row["pred_mean_delta_formula_only_ev"], "formula_pair_delta")
                if abs(formula_delta) > 1e-9:
                    raise ValueError(f"formula_pair_delta_not_zero:{model}:{mixture}:{formula_delta}")
                selected.append({
                    "model": model,
                    "seed_mixture": mixture,
                    "composition_key": row["composition_key"],
                    "formula": row["formula"],
                    "cif_a": row["cif_a"],
                    "cif_b": row["cif_b"],
                    "phase_a": row.get("phase_a", ""),
                    "phase_b": row.get("phase_b", ""),
                    "observed_mean_delta_ev": observed,
                    "pred_mean_delta_full_ev": predicted,
                    "formula_only_delta_ev": formula_delta,
                    "weight": finite(row["weight"], "pair_weight"),
                    "shared_source": row.get("shared_source", ""),
                })
            expected_error = finite(summary["mean_delta_error_full_ev"], f"pair_summary_error:{model}:{mixture}")
            if abs(weighted_error - expected_error) > 1e-9:
                raise ValueError(f"pair_summary_error_mismatch:{model}:{mixture}:{weighted_error}:{expected_error}")
    if len(selected) != len(PHASE_MODELS) * len(SEED_MIXTURES) * 22:
        raise ValueError("selected_pair_row_count_mismatch")
    return selected


def fit_selection_deltas(work_dir, analysis_protocol):
    run_binding = analysis_protocol.get("run_binding", {})
    rows = []
    files = []
    for seed in (17, 29, 41):
        for fold in range(1, 6):
            directory = f"seed_{seed}_fold_{fold:02d}"
            path = work_dir / "run" / directory / "fits.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = sha256(path)
            if run_binding.get(directory + "/fits.json") != digest:
                raise ValueError(f"fit_json_binding_mismatch:{directory}")
            fits = read_json(path)
            for model in PHASE_MODELS:
                info = fits.get(model)
                if not isinstance(info, dict) or not isinstance(info.get("initial"), dict) or not isinstance(info.get("selected"), dict):
                    raise ValueError(f"fit_json_schema_mismatch:{directory}:{model}")
                row = {"seed": seed, "fold": fold - 1, "model": model,
                       "selected_epoch": info.get("selected_epoch", ""),
                       "train_pair_count": info.get("train_pair_count", ""),
                       "train_pair_formulas": info.get("train_pair_formulas", ""),
                       "pair_scale_ev": info.get("pair_scale_ev", ""),
                       "initial_pair_gradient_norm": info.get("initial_pair_gradient_norm", ""),
                       "initial_nll_gradient_norm": info.get("initial_nll_gradient_norm", ""),
                       "initial_pair_width_gradient_max": info.get("initial_pair_width_gradient_max", ""),
                       "source_fits_json": directory + "/fits.json",
                       "source_sha256": digest}
                for metric in ("train_nll", "pair_loss", "validation_nll"):
                    initial = finite(info["initial"][metric], f"{directory}:{model}:initial:{metric}")
                    selected = finite(info["selected"][metric], f"{directory}:{model}:selected:{metric}")
                    row[f"initial_{metric}"] = initial
                    row[f"selected_{metric}"] = selected
                    row[f"selected_minus_initial_{metric}"] = selected - initial
                rows.append(row)
            files.append({"path": directory + "/fits.json", "sha256": digest})
    if len(rows) != 30 or len(files) != 15:
        raise ValueError(f"fit_json_count_mismatch:{len(rows)}:{len(files)}")
    return rows, files


def plot_metrics(rows, output):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    metric_specs = (("nll", "NLL", 0), ("crps_ev", "CRPS (eV)", 1))
    panel_specs = (("all", "full", "All reports · full view (n=205)"),
                   ("structure", "full", "Structure-supported reports · full view (n=96)"))
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.4), layout="constrained")
    x_positions = {model: index for index, model in enumerate(MODELS)}
    offsets = {"17": -0.21, "29": -0.07, "41": 0.07, "equal_density_mixture": 0.21}
    ranges = {}
    for metric, label, column in metric_specs:
        values = [row[metric] for row in rows]
        span = max(values) - min(values)
        margin = max(span * 0.08, abs(max(values)) * 0.03, 0.02)
        ranges[metric] = (min(values) - margin, max(values) + margin)
    for row_index, (subset, view, panel_title) in enumerate(panel_specs):
        for column, (metric, metric_label, _) in enumerate(metric_specs):
            ax = axes[row_index, column]
            for model in MODELS:
                for mixture in SEED_MIXTURES:
                    row = next(item for item in rows if item["model"] == model and item["seed_mixture"] == mixture
                               and item["subset"] == subset and item["view"] == view)
                    ax.scatter(x_positions[model] + offsets[mixture], row[metric],
                               s=54 if mixture == "equal_density_mixture" else 38,
                               color=COLORS[mixture], marker=MARKERS[mixture],
                               edgecolor="white" if mixture != "equal_density_mixture" else "black",
                               linewidth=0.6, zorder=3)
            ax.set_title(f"{panel_title}\n{metric_label}", loc="left")
            if column == 0:
                ax.set_ylabel(metric_label)
            ax.set_ylim(*ranges[metric])
            ax.set_xticks(list(x_positions.values()), ["FROZEN", "SINGLE", "PHASE NLL", "PHASE PAIR"])
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.98, 0.96, "lower is better", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#555555")
    handles = [Line2D([], [], color=COLORS[mix], marker=MARKERS[mix], linestyle="none",
                      markeredgecolor="white" if mix != "equal_density_mixture" else "black",
                      label=MIXTURE_LABELS[mix]) for mix in SEED_MIXTURES]
    fig.suptitle("Phase-conditioned residual: out-of-fold NLL and CRPS\n"
                 "Formula-only results match FROZEN for both phase arms (not plotted)", fontsize=12)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    png = output / "phase_metrics.png"
    pdf = output / "phase_metrics.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def pair_error_summary(rows):
    mixture = "equal_density_mixture"
    model_rows = {model: [row for row in rows if row["model"] == model and row["seed_mixture"] == mixture]
                  for model in PHASE_MODELS}
    mae = {model: sum(row["weight"] * abs(row["pred_mean_delta_full_ev"] - row["observed_mean_delta_ev"])
                      for row in selected)
           for model, selected in model_rows.items()}
    zero_delta_mae = sum(row["weight"] * abs(row["observed_mean_delta_ev"])
                         for row in model_rows[PHASE_MODELS[0]])
    return {"seed_mixture": mixture, "weighting": "equal formula then equal within-formula CIF pair",
            "mae_full_ev": mae, "formula_only_zero_delta_mae_ev": zero_delta_mae}


def plot_pairs(rows, pair_errors, output):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    bound = max(max(abs(row["observed_mean_delta_ev"]), abs(row["pred_mean_delta_full_ev"])) for row in rows)
    bound = max(0.1, bound * 1.08)
    lims = (-bound, bound)
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), layout="constrained", sharex=True, sharey=True)
    for ax, model in zip(axes, PHASE_MODELS):
        subset = [row for row in rows if row["model"] == model]
        for mixture in SEED_MIXTURES:
            group = [row for row in subset if row["seed_mixture"] == mixture]
            ax.scatter([row["observed_mean_delta_ev"] for row in group],
                       [row["pred_mean_delta_full_ev"] for row in group],
                       s=44 if mixture == "equal_density_mixture" else 31,
                       color=COLORS[mixture], marker=MARKERS[mixture], alpha=0.85,
                       edgecolor="white" if mixture != "equal_density_mixture" else "black",
                       linewidth=0.5, zorder=3)
        ax.plot(lims, lims, color="#555555", linewidth=1.0, zorder=1, label="Ideal agreement")
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(lims[0] + 0.03 * (lims[1] - lims[0]), 0.025 * (lims[1] - lims[0]),
                "formula-only Δ=0", color="#555555", fontsize=8, ha="left", va="bottom")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{model.replace('_', ' ')} · 22 pairs, 18 formulas")
        ax.set_xlabel("Observed pair Δ mean Eg (eV)")
        ax.grid(color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted pair Δ mean Eg, full view (eV)")
    handles = [Line2D([], [], color=COLORS[mix], marker=MARKERS[mix], linestyle="none",
                      markeredgecolor="white" if mix != "equal_density_mixture" else "black",
                      label=MIXTURE_LABELS[mix]) for mix in SEED_MIXTURES]
    handles += [Line2D([], [], color="#555555", linewidth=1.0, label="Ideal agreement (y=x)"),
                Line2D([], [], color="#777777", linestyle="--", linewidth=1.0, label="Formula-only Δ=0")]
    fig.suptitle("Same-formula actual-CIF mean-gap contrasts · all 22 pairs\n"
                 f"Equal-formula MAE (3-seed mixture): NLL={pair_errors['mae_full_ev']['PHASE_NLL']:.3f}, "
                 f"PAIR={pair_errors['mae_full_ev']['PHASE_PAIR']:.3f}, "
                 f"formula-only Δ=0={pair_errors['formula_only_zero_delta_mae_ev']:.3f} eV", fontsize=12)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    png = output / "phase_pair_scatter.png"
    pdf = output / "phase_pair_scatter.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def run(root, work_dir, analysis_dir, output_dir, force=False):
    root, work_dir, analysis_dir, output_dir = (path.resolve() for path in
                                                 (root, work_dir, analysis_dir, output_dir))
    if output_dir.is_relative_to(root):
        raise ValueError("figure_output_must_be_outside_formal_root")
    required_outputs = (
        "phase_metrics.png", "phase_metrics.pdf", "phase_pair_scatter.png", "phase_pair_scatter.pdf",
        "figure1_metrics_source.csv", "figure2_pairs_source.csv", "fit_selection_deltas.csv", "figure_protocol.json")
    existing = [output_dir / name for name in required_outputs if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"figure_outputs_exist; pass --force to replace: {existing}")

    analysis_protocol_path = analysis_dir / "protocol.json"
    training_protocol_path = work_dir / "protocol.json"
    metrics_path = analysis_dir / "metrics.csv"
    pairs_path = analysis_dir / "same_formula_cif_pairs.csv"
    pair_summary_path = analysis_dir / "same_formula_pair_summary.csv"
    for path in (analysis_protocol_path, metrics_path, pairs_path, pair_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis_protocol = read_json(analysis_protocol_path)
    training_protocol = analysis_protocol.get("training")
    if not isinstance(training_protocol, dict):
        raise ValueError("analysis_training_protocol_missing")
    if training_protocol_path.is_file() and read_json(training_protocol_path) != training_protocol:
        raise ValueError("analysis_training_protocol_mismatch")
    output_binding = analysis_protocol.get("output_binding", {})
    for path in (metrics_path, pairs_path, pair_summary_path):
        if output_binding.get(path.name) != sha256(path):
            raise ValueError(f"analysis_output_binding_mismatch:{path.name}")

    metrics = read_csv(metrics_path)
    pair_rows = read_csv(pairs_path)
    pair_summaries = read_csv(pair_summary_path)
    require_fields(metrics, ("model", "view", "seed_mixture", "subset", "reports", "nll", "crps_ev"), "metrics.csv")
    require_fields(pair_rows, ("model", "seed_mixture", "composition_key", "formula", "cif_a", "cif_b",
                               "observed_mean_delta_ev", "pred_mean_delta_full_ev", "pred_mean_delta_formula_only_ev", "weight"),
                   "same_formula_cif_pairs.csv")
    require_fields(pair_summaries, ("model", "seed_mixture", "formulas", "pairs", "mean_delta_error_full_ev", "weighting"),
                   "same_formula_pair_summary.csv")
    metric_rows = selected_metrics(metrics)
    pair_plot_rows = selected_pairs(pair_rows, pair_summaries)
    pair_errors = pair_error_summary(pair_plot_rows)
    fit_rows, fit_files = fit_selection_deltas(work_dir, analysis_protocol)

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = ["model", "view", "seed_mixture", "subset", "panel", "reports", "nll", "crps_ev"]
    pair_fields = ["model", "seed_mixture", "composition_key", "formula", "cif_a", "cif_b", "phase_a", "phase_b",
                   "observed_mean_delta_ev", "pred_mean_delta_full_ev", "formula_only_delta_ev", "weight", "shared_source"]
    fit_fields = ["seed", "fold", "model", "selected_epoch", "train_pair_count", "train_pair_formulas", "pair_scale_ev",
                  "initial_train_nll", "selected_train_nll", "selected_minus_initial_train_nll",
                  "initial_pair_loss", "selected_pair_loss", "selected_minus_initial_pair_loss",
                  "initial_validation_nll", "selected_validation_nll", "selected_minus_initial_validation_nll",
                  "initial_pair_gradient_norm", "initial_nll_gradient_norm", "initial_pair_width_gradient_max",
                  "source_fits_json", "source_sha256"]
    write_csv(output_dir / "figure1_metrics_source.csv", metric_rows, metric_fields)
    write_csv(output_dir / "figure2_pairs_source.csv", pair_plot_rows, pair_fields)
    write_csv(output_dir / "fit_selection_deltas.csv", fit_rows, fit_fields)
    figures = plot_metrics(metric_rows, output_dir) + plot_pairs(pair_plot_rows, pair_errors, output_dir)

    input_paths = [analysis_protocol_path, metrics_path, pairs_path, pair_summary_path]
    if training_protocol_path.is_file():
        input_paths.append(training_protocol_path)
    input_paths += [work_dir / "run" / item["path"] for item in fit_files]
    code_paths = [Path(__file__).resolve(), root / "runtime" / "analyze_phase_residual.py",
                  root / "runtime" / "phase_conditioned_residual.py"]
    protocol = {
        "schema": "phase_stage_figures_v1",
        "status": "stage visualization; no fits run",
        "input_binding": {str(path.relative_to(work_dir) if path.is_relative_to(work_dir) else path.relative_to(root)): sha256(path)
                          for path in input_paths},
        "code_binding": {str(path.relative_to(root)) if path.is_relative_to(root) else path.name: sha256(path)
                         for path in code_paths},
        "figure1": {
            "source": "metrics.csv; values plotted without reaggregation",
            "arms": list(MODELS),
            "seed_mixtures": list(SEED_MIXTURES),
            "panels": [
                {"view": "full", "subset": "all", "reports": 205},
                {"view": "full", "subset": "structure", "reports": 96},
            ],
            "metrics": ["nll", "crps_ev"],
            "note": "Equal-density mixture is the three-seed predictive mixture; points are seed-specific OOF estimates, not confidence intervals. Full-view metrics include all reports; phase residual support is limited to reports with SG and structure.",
            "formula_only_note": "PHASE_NLL and PHASE_PAIR formula-only metrics match FROZEN within 1e-7 for each seed and the three-seed mixture; omitted from chart to avoid overlapping points.",
        },
        "figure2": {
            "source": "same_formula_cif_pairs.csv; all 18 formulas and 22 actual-CIF pairs retained",
            "arms": list(PHASE_MODELS),
            "seed_mixtures": list(SEED_MIXTURES),
            "x": "observed_mean_delta_ev",
            "y": "pred_mean_delta_full_ev",
            "references": {"ideal": "y=x", "formula_only": "y=0"},
            "weighting_check": "per-pair weights sum to one and reproduce same_formula_pair_summary.csv weighted mean absolute error",
            "equal_density_mixture_mae_ev": pair_errors["mae_full_ev"],
            "formula_only_zero_delta_mae_ev": pair_errors["formula_only_zero_delta_mae_ev"],
        },
        "fit_selection_deltas": {
            "source": "15 run-specific fits.json files",
            "rows": len(fit_rows),
            "fields": ["initial", "selected", "selected_minus_initial"],
            "note": "TRAIN/inner-VALID fit diagnostics only; not an outer-fold gain estimate.",
        },
        "formula_improvement_claim": False,
        "outputs": {path.name: sha256(path) for path in
                    [*figures, output_dir / "figure1_metrics_source.csv", output_dir / "figure2_pairs_source.csv",
                     output_dir / "fit_selection_deltas.csv"]},
        "software": {"matplotlib": __import__("matplotlib").__version__},
    }
    (output_dir / "figure_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pairs": 22, "pair_formulas": 18,
                      "fit_delta_rows": len(fit_rows), "figures": [path.name for path in figures]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parents[2] / "BandgapFusion_PhaseWork")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    analysis_dir = (args.analysis_dir or work_dir / "analysis").resolve()
    output_dir = (args.output_dir or work_dir / "figures").resolve()
    run(args.root, work_dir, analysis_dir, output_dir, args.force)


if __name__ == "__main__":
    main()
