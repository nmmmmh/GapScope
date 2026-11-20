"""Score paired JOINT/FROZEN out-of-fold densities from a completed run."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import runtime as rt
from density_score import score
from training_data import _load_records, hierarchical_weights


METRICS = ['nll', 'crps_ev', 'median_mae_ev', 'coverage_50', 'width_50_ev',
           'coverage_80', 'width_80_ev', 'coverage_90', 'width_90_ev']
MODELS = ('JOINT', 'FROZEN')


def _check_outside(path, root, label):
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise ValueError(f'{label}_must_be_outside_formal_root')


def _read_targets(root):
    reports = root / 'data' / 'reports'
    metadata = rt.read_json(reports / 'metadata.json')
    sources = _load_records(reports / 'records.jsonl')
    if not isinstance(metadata, list) or len(metadata) != 205 or len(sources) != 205:
        raise ValueError('unexpected_report_record_count')
    records = []
    for index, (meta, source) in enumerate(zip(metadata, sources)):
        if any(meta.get(key) != source.get(key)
               for key in ('measurement_id', 'composition_key', 'formula')):
            raise ValueError(f'report_metadata_order_mismatch:{index}')
        record = dict(meta)
        record.update(source)
        target = record.get('gap_ev')
        if not isinstance(target, (int, float)) or not math.isfinite(float(target)) or target <= 0:
            raise ValueError(f'invalid_report_target:{record.get("measurement_id")}')
        records.append(record)
    ids = [record.get('measurement_id') for record in records]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('report_measurement_ids_invalid')
    return records


def _parts(rows):
    arrays = []
    for key in ('component_weights', 'log_mus', 'sigmas'):
        chunks = []
        for row in rows:
            value = np.asarray(json.loads(row[key]), dtype=np.float64)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError(f'invalid_prediction_components:{key}')
            chunks.append(value)
        arrays.append(np.concatenate(chunks))
    weights, log_mus, sigmas = arrays
    weights = weights / len(rows)
    if len(weights) != len(log_mus) or len(weights) != len(sigmas):
        raise ValueError('prediction_component_count_mismatch')
    return weights, log_mus, sigmas


def _fit_rows(path, seed, fold):
    fits = rt.read_json(path)
    if not isinstance(fits, dict):
        raise ValueError(f'fits_not_mapping:{path}')
    fit_count = fits.get('fit_count')
    prior_hash = fits.get('initial_prior_state_hash')
    if type(fit_count) is not int or fit_count != 3:
        raise ValueError(f'fit_count_invalid:{path}')
    if not isinstance(prior_hash, str) or not prior_hash:
        raise ValueError(f'initial_prior_state_hash_invalid:{path}')
    rows = []
    for arm in ('aggregate', *MODELS):
        fit = fits.get(arm)
        if not isinstance(fit, dict):
            raise ValueError(f'fit_summary_missing:{arm}:{path}')
        rows.append(dict(seed=seed, fold=fold, model=arm, fit_count=fit_count,
                         initial_prior_state_hash=prior_hash,
                         **{key: value for key, value in fit.items() if key != 'history'}))
    return rows


def _validate_predictions(raw, records, seeds):
    expected_ids = {record['measurement_id'] for record in records}
    groups = defaultdict(list)
    partition_ids = {}
    for row in raw:
        try:
            seed = int(row['seed'])
            fold = int(row['fold_id'])
            model = row['model']
            view = row['view']
            rid = row['measurement_id']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('prediction_identity_invalid') from exc
        if seed not in seeds or fold not in range(5) or model not in MODELS or view not in rt.VIEWS:
            raise ValueError('prediction_identity_out_of_protocol')
        if rid not in expected_ids:
            raise ValueError(f'prediction_report_unknown:{rid}')
        row['seed'] = seed
        row['fold_id'] = fold
        groups[(model, view, rid)].append(row)
        partition_ids.setdefault((seed, fold, model, view), set()).add(rid)

    expected_group_count = len(MODELS) * len(rt.VIEWS) * len(records)
    if len(groups) != expected_group_count:
        raise ValueError('prediction_report_view_coverage_mismatch')
    for key, rows in groups.items():
        row_seeds = [row['seed'] for row in rows]
        if sorted(row_seeds) != sorted(seeds):
            raise ValueError(f'prediction_seed_count_mismatch:{key[0]}:{key[1]}:{key[2]}')
        if len({row['fold_id'] for row in rows}) != 1:
            raise ValueError(f'prediction_fold_mismatch_across_seeds:{key[2]}')

    for seed in seeds:
        for fold in range(5):
            for model in MODELS:
                for view in rt.VIEWS:
                    key = (seed, fold, model, view)
                    ids = partition_ids.get(key)
                    if ids is None:
                        raise ValueError(f'prediction_partition_missing:{key}')
                    if ids != partition_ids[(seeds[0], fold, MODELS[0], rt.VIEWS[0])]:
                        raise ValueError(f'prediction_partition_membership_mismatch:{key}')
        if any(sum(rid in partition_ids[(seed, fold, MODELS[0], rt.VIEWS[0])] for fold in range(5)) != 1
               for rid in expected_ids):
            raise ValueError(f'prediction_oof_uniqueness_mismatch:{seed}')


def _score_rows(raw, records, seeds):
    by_id = {record['measurement_id']: record for record in records}
    weight_values = hierarchical_weights(records, list(range(len(records))), 'cpu').tolist()
    weight_by_id = {record['measurement_id']: float(weight)
                    for record, weight in zip(records, weight_values)}
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row['model'], row['view'], row['measurement_id'])].append(row)

    scores = []
    score_lookup = {}
    for (model, view, rid), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row['seed'])
        record = by_id[rid]
        for mixture, selected in [('equal_density_mixture', rows)] + [
                (str(row['seed']), [row]) for row in rows]:
            result = score(float(record['gap_ev']), _parts(selected))
            scored = dict(
                model=model, view=view, seed_mixture=mixture,
                measurement_id=rid, formula=record['formula'],
                composition_key=record['composition_key'],
                phase_label=record.get('phase_label', ''),
                sg_supported=int(rows[0]['sg_supported']),
                structure_supported=int(rows[0]['structure_supported']),
                target_ev=float(record['gap_ev']), weight=weight_by_id[rid], **result)
            scores.append(scored)
            score_lookup[(model, view, mixture, rid)] = scored
    return scores, score_lookup


def _metrics(scores, seeds):
    groups = defaultdict(list)
    for row in scores:
        groups[(row['model'], row['view'], row['seed_mixture'])].append(row)
    result = []
    for (model, view, mixture), rows in sorted(groups.items()):
        if len(rows) != 205 or len({row['measurement_id'] for row in rows}) != 205:
            raise ValueError(f'report_score_coverage_incomplete:{model}:{view}:{mixture}')
        subsets = [
            ('all', rows),
            ('structure', [row for row in rows if row['structure_supported']]),
            ('missing_structure', [row for row in rows if not row['structure_supported']]),
        ]
        for subset, selected in subsets:
            if not selected:
                continue
            weights = np.asarray([row['weight'] for row in selected], dtype=np.float64)
            weights /= weights.sum()
            result.append(dict(
                model=model, view=view, seed_mixture=mixture, subset=subset,
                reports=len(selected),
                formulas=len({row['composition_key'] for row in selected}),
                **{metric: float(weights @ np.asarray([row[metric] for row in selected]))
                   for metric in METRICS}))
    return result


def _contrasts(score_lookup, seeds):
    pairs = []
    for view in rt.VIEWS:
        pairs.append(('frozen_vs_joint', 'FROZEN', view, 'JOINT', view))
    for model in MODELS:
        pairs.append(('structure_increment_full_minus_formula_sg', model, 'full', model, 'formula_sg'))
        pairs.append(('total_modality_increment_full_minus_formula_only', model, 'full', model, 'formula_only'))

    output = []
    mixtures = ['equal_density_mixture'] + [str(seed) for seed in seeds]
    for contrast, left_model, left_view, right_model, right_view in pairs:
        label = left_model if left_model == right_model else 'FROZEN_vs_JOINT'
        view_label = left_view if left_view == right_view else f'{left_view}_minus_{right_view}'
        for mixture in mixtures:
            left_ids = {key[3] for key in score_lookup
                        if key[:3] == (left_model, left_view, mixture)}
            right_ids = {key[3] for key in score_lookup
                         if key[:3] == (right_model, right_view, mixture)}
            if left_ids != right_ids or len(left_ids) != 205:
                raise ValueError(f'contrast_pair_coverage_mismatch:{contrast}:{mixture}')
            left = {rid: score_lookup[(left_model, left_view, mixture, rid)] for rid in left_ids}
            right = {rid: score_lookup[(right_model, right_view, mixture, rid)] for rid in right_ids}
            for subset in ('all', 'structure'):
                ids = [rid for rid in sorted(left_ids)
                       if subset == 'all' or left[rid]['structure_supported']]
                if not ids:
                    continue
                weights = np.asarray([left[rid]['weight'] for rid in ids], dtype=np.float64)
                weights /= weights.sum()
                for metric in METRICS:
                    delta = np.asarray([left[rid][metric] - right[rid][metric] for rid in ids])
                    output.append(dict(
                        contrast=contrast, model=label, view=view_label,
                        left_model=left_model, left_view=left_view,
                        right_model=right_model, right_view=right_view,
                        seed_mixture=mixture, subset=subset, metric=metric,
                        delta=float(weights @ delta), reports=len(ids),
                        delta_definition='left_minus_right'))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir.resolve()
    out = args.output_dir.resolve()
    _check_outside(out, root, 'output_dir')

    work_protocol_path = work / 'protocol.json'
    work_protocol = rt.read_json(work_protocol_path)
    seeds = work_protocol.get('seeds')
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('expected_three_distinct_seeds')
    if work_protocol.get('folds') != 5:
        raise ValueError('expected_five_folds')
    if work_protocol.get('arms') != list(MODELS):
        raise ValueError('unexpected_model_arms')
    seeds = [int(seed) for seed in seeds]

    records = _read_targets(root)
    raw = []
    fits = []
    input_binding = {
        'work:protocol.json': rt.sha(work_protocol_path),
        'root:data/reports/records.jsonl': rt.sha(root / 'data' / 'reports' / 'records.jsonl'),
        'root:data/reports/metadata.json': rt.sha(root / 'data' / 'reports' / 'metadata.json'),
        'root:runtime/analyze_frozen.py': rt.sha(Path(__file__).resolve()),
        'root:runtime/density_score.py': rt.sha(root / 'runtime' / 'density_score.py'),
        'root:runtime/training_data.py': rt.sha(root / 'runtime' / 'training_data.py'),
        'root:runtime/runtime.py': rt.sha(root / 'runtime' / 'runtime.py'),
    }
    protocol_sha = rt.sha(work_protocol_path)
    for seed in seeds:
        for fold in range(5):
            path = work / f'seed_{seed}_fold_{fold + 1:02d}'
            complete_path = path / 'complete.json'
            predictions_path = path / 'predictions.csv'
            fits_path = path / 'fits.json'
            for required in (complete_path, predictions_path, fits_path):
                if not required.is_file():
                    raise ValueError(f'incomplete_run_file:{required.name}:{path.name}')
            complete = rt.read_json(complete_path)
            if (complete.get('status') != 'complete' or complete.get('seed') != seed
                    or complete.get('fold') != fold or complete.get('protocol_sha256') != protocol_sha):
                raise ValueError(f'completion_binding_mismatch:{path.name}')
            if complete.get('predictions_sha256') != rt.sha(predictions_path):
                raise ValueError(f'predictions_hash_mismatch:{path.name}')
            if complete.get('fits_sha256') != rt.sha(fits_path):
                raise ValueError(f'fits_hash_mismatch:{path.name}')
            rows = rt.read_csv(predictions_path)
            raw.extend(rows)
            fits.extend(_fit_rows(fits_path, seed, fold))
            for label, file_path in (('predictions', predictions_path),
                                     ('fits', fits_path), ('complete', complete_path)):
                input_binding[f'work:{path.name}/{file_path.name}'] = rt.sha(file_path)

    _validate_predictions(raw, records, seeds)
    scores, score_lookup = _score_rows(raw, records, seeds)
    metrics = _metrics(scores, seeds)
    contrasts = _contrasts(score_lookup, seeds)

    analysis_protocol = dict(
        schema='analyze_frozen_v1',
        work_protocol=work_protocol,
        models=list(MODELS),
        views=list(rt.VIEWS),
        seeds=seeds,
        folds=5,
        reports=205,
        score_metrics=METRICS,
        score_recipe='density_score.score on each report; equal_density_mixture concatenates all three seed component densities with equal density weight',
        report_weight_recipe='training_data.hierarchical_weights over all 205 reports',
        subsets=['all', 'structure', 'missing_structure'],
        contrasts={
            'frozen_vs_joint': 'FROZEN minus JOINT within each same view',
            'structure_increment_full_minus_formula_sg': 'full minus formula_sg per model',
            'total_modality_increment_full_minus_formula_only': 'full minus formula_only per model',
            'contrast_subsets': ['all', 'structure'],
            'delta_definition': 'left minus right; descriptive signed differences',
        },
        fit_summary='Only aggregate, JOINT, and FROZEN fit dictionaries are summarized; fit_count and initial_prior_state_hash are validated as metadata.',
        input_binding=input_binding,
    )
    out.mkdir(parents=True, exist_ok=True)
    rt.write_csv(out / 'oof_densities.csv', raw)
    rt.write_csv(out / 'report_scores.csv', scores)
    rt.write_csv(out / 'metrics.csv', metrics)
    rt.write_csv(out / 'contrasts.csv', contrasts)
    rt.write_csv(out / 'fit_summary.csv', fits)
    analysis_protocol['output_binding'] = {
        name: rt.sha(out / name) for name in
        ('oof_densities.csv', 'report_scores.csv', 'metrics.csv', 'contrasts.csv', 'fit_summary.csv')
    }
    rt.write_json(out / 'protocol.json', analysis_protocol)
    print(json.dumps(dict(reports=len(records), prediction_rows=len(raw),
                          scored_rows=len(scores), metric_rows=len(metrics),
                          contrast_rows=len(contrasts), fit_rows=len(fits)), ensure_ascii=False))


if __name__ == '__main__':
    main()
