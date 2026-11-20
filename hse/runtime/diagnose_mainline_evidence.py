"""Describe existing experimental evidence; never load models or fit parameters."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def weights(records):
    sources, specimens, counts = defaultdict(set), defaultdict(set), Counter()
    for row in records:
        f, s, p = (row[k] for k in ('composition_key', 'source_group_id', 'specimen_group_id'))
        sources[f].add(s)
        specimens[f, s].add(p)
        counts[f, s, p] += 1
    return {row['measurement_id']: 1 / len(sources)
            / len(sources[row['composition_key']])
            / len(specimens[row['composition_key'], row['source_group_id']])
            / counts[row['composition_key'], row['source_group_id'], row['specimen_group_id']]
            for row in records}


def run(root):
    data = root / 'data/reports'
    analysis = root / 'results/conditional_adaptation/analysis'
    metadata_path, records_path = data / 'metadata.json', data / 'records.jsonl'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    original = [json.loads(line) for line in records_path.read_text(encoding='utf-8-sig').splitlines()]
    if len(metadata) != len(original):
        raise ValueError('metadata/record length mismatch')
    records = [dict(m, **r) for m, r in zip(metadata, original)]
    raw_path, scores_path = analysis / 'predictions.csv', analysis / 'report_scores.csv'
    predictions, scores = read_csv(raw_path), read_csv(scores_path)
    all_weights = weights(records)
    by_formula = defaultdict(list)
    for row in records:
        by_formula[row['composition_key']].append(row)
    mean = sum(all_weights[r['measurement_id']] * r['gap_ev'] for r in records)
    within, between = 0.0, 0.0
    for rows in by_formula.values():
        mass = sum(all_weights[r['measurement_id']] for r in rows)
        center = sum(all_weights[r['measurement_id']] * r['gap_ev'] for r in rows) / mass
        within += sum(all_weights[r['measurement_id']] * (r['gap_ev'] - center) ** 2 for r in rows)
        between += mass * (center - mean) ** 2
    ids = {r['measurement_id'] for r in predictions if r['structure_supported'] == '1'}
    availability = {r['measurement_id']: (int(r['sg_supported']), int(r['structure_supported']))
                    for r in predictions}
    formula_coefficients = defaultdict(set)
    direct_formula_mass = 0.0
    for row in records:
        coefficient = 1 / (2 ** sum(availability[row['measurement_id']]))
        formula_coefficients[row['composition_key']].add(coefficient)
        direct_formula_mass += all_weights[row['measurement_id']] * coefficient
    subset = [r for r in records if r['measurement_id'] in ids]
    subset_weights = weights(subset)
    index = {(r['model'], r['view'], r['seed_mixture'], r['measurement_id']): r for r in scores}
    metrics = ('nll', 'crps_ev', 'median_mae_ev', 'width_80_ev', 'coverage_80')
    contrasts, contributions = [], []
    for view in ('formula_structure', 'full'):
        for seed in ('equal_density_mixture', '17', '29', '41'):
            groups = defaultdict(lambda: defaultdict(float))
            for mid, mass in subset_weights.items():
                base = index['TEACHER_BASE', view, seed, mid]
                adapted = index['ADAPTER_TEACHER', view, seed, mid]
                for metric in metrics:
                    groups[base['formula']][metric] += mass * (float(adapted[metric]) - float(base[metric]))
            contrasts.append(dict(view=view, seed=seed,
                adapter_minus_base={k: sum(g[k] for g in groups.values()) for k in metrics},
                formula_crps_improvements=sum(g['crps_ev'] < 0 for g in groups.values()),
                formula_nll_improvements=sum(g['nll'] < 0 for g in groups.values())))
            contributions.extend(dict(view=view, seed=seed, formula=f, **g) for f, g in sorted(groups.items()))
    bindings = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (metadata_path, records_path, raw_path, scores_path)}
    summary = dict(scope='Post-hoc description of existing OOF outputs; no model fitting, selection or independent confirmation',
        source_bindings=bindings, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reports=len(records), formulas=len(by_formula), structure_reports=len(subset),
        structure_formulas=len({r['composition_key'] for r in subset}),
        median_reports_per_formula=statistics.median(len(v) for v in by_formula.values()),
        median_sources_per_formula=statistics.median(len({r['source_group_id'] for r in v}) for v in by_formula.values()),
        four_view_formula_weighting=dict(direct_formula_mass=direct_formula_mass,
            formula_keys_with_variable_coefficients=sum(len(v) > 1 for v in formula_coefficients.values()),
            coefficients={'no_conditional_modalities': 1.0, 'one_available_modality': 0.5, 'both_available': 0.25},
            interpretation='Algebraic formula-view terms in the old four-view average; conditional views also update shared parameters'),
        empirical_variance=dict(mean_ev=mean, total_ev2=within+between, within_formula_ev2=within,
            between_formula_ev2=between, within_formula_fraction=within/(within+between),
            interpretation='Observed report-pool variance, not a phase-only variance or a CRPS decomposition'),
        contrasts=contrasts)
    output = root / 'results/mainline_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'formula_contributions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contributions[0]))
        writer.writeheader()
        writer.writerows(contributions)
    print(json.dumps(dict(output=str(output), reports=len(records), structure_reports=len(subset),
        within_formula_fraction=summary['empirical_variance']['within_formula_fraction']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
