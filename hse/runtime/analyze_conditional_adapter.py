"""Matched grouped OOF evidence for conditional adaptation and marginal distillation."""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import runtime as rt
from density_score import score
from training_data import hierarchical_weights

METRICS = ('crps_ev', 'nll', 'median_mae_ev', 'coverage_80', 'width_80_ev')
MIX = 'equal_density_mixture'


def run(root, parent, work, make_gate=False):
    protocol = rt.read_json(work / 'training_protocol.json')
    metadata = rt.read_json(parent / 'inputs/metadata.json')
    records = [dict(m, **json.loads(line)) for m, line in zip(
        metadata, (parent / 'inputs/records.jsonl').read_text(encoding='utf-8').splitlines())]
    features = rt.torch.load(parent / 'inputs/features.pt', map_location='cpu', weights_only=True)
    masks = [(bool(s), bool(t)) for s, t in zip(features['sg_mask'], features['structure_mask'])]
    raw, fit_rows, bindings = [], [], {}
    for phase in ('teachers', 'students'):
        folders = work / phase
        if not folders.exists():
            continue
        completed = [p for p in folders.iterdir() if p.is_dir() and (p / 'complete.json').exists()]
        if not completed:
            continue
        if len(completed) != 9:
            raise ValueError(f'Incomplete {phase}: {len(completed)}/9')
        for folder in sorted(completed):
            receipt = rt.read_json(folder / 'complete.json')
            for name in ('predictions.csv', 'models.pt', 'fits.json', 'partition.json'):
                digest = rt.sha(folder / name)
                if digest != receipt[name.replace('.', '_') + '_sha256']:
                    raise ValueError(f'Changed output: {folder / name}')
                bindings[(folder / name).relative_to(work).as_posix()] = digest
            if receipt['protocol_sha256'] != rt.sha(work / 'training_protocol.json'):
                raise ValueError('Protocol changed after fitting')
            raw.extend(rt.read_csv(folder / 'predictions.csv'))
            fitted = rt.read_json(folder / 'fits.json')
            if 'selected_epoch' in fitted:
                fitted = {'ADAPTER_TEACHER' if phase == 'teachers' else 'MARGINAL_STUDENT': fitted}
            for model, fit in fitted.items():
                fit_rows.append(dict(phase=phase, fit=folder.name, model=model,
                    **{k: fit[k] for k in ('selected_epoch', 'epochs_run',
                       'initial_validation', 'selected_validation', 'trainable_parameters') if k in fit}))
    grouped = defaultdict(list)
    for row in raw:
        grouped[row['model'], row['view'], row['measurement_id']].append(row)
    arms = sorted({r['model'] for r in raw})
    views = {arm: [v for v in rt.VIEWS if any(r['model'] == arm and r['view'] == v for r in raw)] for arm in arms}
    mass = hierarchical_weights(records, list(range(len(records))), 'cpu').tolist()
    scores = []
    for arm in arms:
        for view in views[arm]:
            for mixture in (MIX, *map(str, protocol['seeds'])):
                for record, weight in zip(records, mass):
                    rows = grouped[arm, view, record['measurement_id']]
                    if len(rows) != 3 or {int(r['seed']) for r in rows} != set(protocol['seeds']):
                        raise ValueError('Missing or duplicated seed OOF prediction')
                    selected = rows if mixture == MIX else [r for r in rows if r['seed'] == mixture]
                    parts = tuple(np.concatenate([np.asarray(json.loads(r[k]), dtype=float) for r in selected])
                                  for k in ('component_weights', 'log_mus', 'sigmas'))
                    parts = (parts[0] / len(selected), parts[1], parts[2])
                    scores.append(dict(model=arm, view=view, seed_mixture=mixture,
                        measurement_id=record['measurement_id'], formula=record['formula'],
                        composition_key=record['composition_key'], weight=weight,
                        target_ev=record['gap_ev'], mean_ev=float(parts[0] @ np.exp(parts[1] + .5 * parts[2] ** 2)),
                        **score(record['gap_ev'], parts)))
    subsets = dict(all=list(range(len(records))),
        structure=[i for i, (_, t) in enumerate(masks) if t],
        both=[i for i, (s, t) in enumerate(masks) if s and t],
        conditional=[i for i, (s, t) in enumerate(masks) if s or t],
        natural_sg_only=[i for i, (s, t) in enumerate(masks) if s and not t],
        natural_neither=[i for i, (s, t) in enumerate(masks) if not s and not t])
    metrics = []
    for subset, ids in subsets.items():
        if not ids:
            continue
        weights = hierarchical_weights(records, ids, 'cpu').tolist()
        wm = {records[i]['measurement_id']: w for i, w in zip(ids, weights)}
        for arm in arms:
            for view in views[arm]:
                for mixture in (MIX, *map(str, protocol['seeds'])):
                    rows = [r for r in scores if r['model'] == arm and r['view'] == view and
                            r['seed_mixture'] == mixture and r['measurement_id'] in wm]
                    metrics.append(dict(subset=subset, model=arm, view=view, seed_mixture=mixture,
                        reports=len(rows), formulas=len({r['composition_key'] for r in rows}),
                        **{k: sum(wm[r['measurement_id']] * r[k] for r in rows) for k in METRICS}))
    mi = {(r['subset'], r['model'], r['view'], r['seed_mixture']): r for r in metrics}
    contrasts = []
    for subset in subsets:
        for left, right, view in (('TEACHER_BASE', 'ADAPTER_TEACHER', 'full'),
                                  ('STUDENT_BASE', 'MARGINAL_STUDENT', 'formula_only')):
            if left not in arms or right not in arms:
                continue
            for metric in ('crps_ev', 'nll', 'median_mae_ev'):
                a, b = mi[subset, left, view, MIX][metric], mi[subset, right, view, MIX][metric]
                gains = [mi[subset, left, view, str(s)][metric] - mi[subset, right, view, str(s)][metric]
                         for s in protocol['seeds']]
                contrasts.append(dict(subset=subset, left_model=left, right_model=right, view=view,
                    metric=metric, left_value=a, right_value=b, improvement=a-b,
                    relative_improvement=(a-b)/a if a else None,
                    seed_improvement_count=sum(g > 0 for g in gains), seed_gains=json.dumps(gains)))
    interactions = []
    for arm in ('TEACHER_BASE', 'ADAPTER_TEACHER'):
        for mixture in (MIX, *map(str, protocol['seeds'])):
            f, sg, st, full = [mi['both', arm, v, mixture]['crps_ev'] for v in rt.VIEWS]
            interactions.append(dict(model=arm, seed_mixture=mixture, subset='both',
                structure_given_sg=sg-full, sg_given_structure=st-full,
                sg_shapley=.5*(f-sg+st-full), structure_shapley=.5*(f-st+sg-full),
                scoring_interaction=sg+st-f-full))
    pair_rows, pair_summary = [], []
    compatible_rows, compatible_summary = [], []
    compatible_path = root / 'results/decoupled_distribution/coverage/pairs.json'
    compatible = rt.read_json(compatible_path)['pairs']
    cifs = defaultdict(lambda: defaultdict(list))
    for i, (record, meta) in enumerate(zip(records, metadata)):
        if masks[i][1]:
            cifs[record['composition_key']][meta['actual_cif_sha256']].append(i)
    paired = {key: groups for key, groups in cifs.items() if len(groups) > 1}
    for arm in ('TEACHER_BASE', 'ADAPTER_TEACHER'):
        for mixture in (MIX, *map(str, protocol['seeds'])):
            means = {(r['view'], r['measurement_id']): r['mean_ev'] for r in scores
                     if r['model'] == arm and r['seed_mixture'] == mixture}
            for key, groups in paired.items():
                combinations = list(itertools.combinations(sorted(groups.items()), 2))
                for (ha, ia), (hb, ib) in combinations:
                    wa, wb = np.asarray([mass[i] for i in ia]), np.asarray([mass[i] for i in ib])
                    wa, wb = wa / wa.sum(), wb / wb.sum()
                    observed = float(wb @ np.asarray([records[i]['gap_ev'] for i in ib]) -
                                     wa @ np.asarray([records[i]['gap_ev'] for i in ia]))
                    row = dict(model=arm, seed_mixture=mixture, composition_key=key,
                        formula=records[ia[0]]['formula'], cif_a=ha, cif_b=hb,
                        observed_delta_ev=observed, zero_delta_error_ev=abs(observed),
                        weight=1/len(paired)/len(combinations))
                    for view in rt.VIEWS:
                        delta = float(wb @ np.asarray([means[view, records[i]['measurement_id']] for i in ib]) -
                                      wa @ np.asarray([means[view, records[i]['measurement_id']] for i in ia]))
                        row[view+'_delta_ev'] = delta
                        row[view+'_delta_error_ev'] = abs(delta-observed)
                    pair_rows.append(row)
            rows = [r for r in pair_rows if r['model'] == arm and r['seed_mixture'] == mixture]
            pair_summary.append(dict(model=arm, seed_mixture=mixture, pairs=len(rows), formulas=len(paired),
                zero_delta_mae_ev=sum(r['weight']*r['zero_delta_error_ev'] for r in rows),
                full_delta_mae_ev=sum(r['weight']*r['full_delta_error_ev'] for r in rows),
                mean_absolute_predicted_delta_ev=sum(r['weight']*abs(r['full_delta_ev']) for r in rows),
                direction_agreement=sum(r['weight']*float(r['full_delta_ev']*r['observed_delta_ev'] > 0) for r in rows)))
            for pair in compatible:
                ia, ib = pair['indices_a'], pair['indices_b']
                wa, wb = np.asarray(pair['endpoint_weights_a']), np.asarray(pair['endpoint_weights_b'])
                observed = pair['observed_delta_ev']
                delta = float(wb @ np.asarray([means['full', records[i]['measurement_id']] for i in ib]) -
                              wa @ np.asarray([means['full', records[i]['measurement_id']] for i in ia]))
                compatible_rows.append(dict(model=arm, seed_mixture=mixture, pair_id=pair['pair_id'],
                    composition_key=pair['composition_key'], formula=pair['formula'],
                    source_group_id=pair['source_group_id'], weight=pair['weight'],
                    observed_delta_ev=observed, predicted_delta_ev=delta,
                    delta_error_ev=abs(delta-observed), zero_delta_error_ev=abs(observed)))
            rows = [r for r in compatible_rows if r['model'] == arm and r['seed_mixture'] == mixture]
            compatible_summary.append(dict(model=arm, seed_mixture=mixture, pairs=len(rows),
                formulas=len({r['composition_key'] for r in rows}),
                zero_delta_mae_ev=sum(r['weight']*r['zero_delta_error_ev'] for r in rows),
                full_delta_mae_ev=sum(r['weight']*r['delta_error_ev'] for r in rows),
                mean_absolute_predicted_delta_ev=sum(r['weight']*abs(r['predicted_delta_ev']) for r in rows)))
    out = work / 'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in (('predictions.csv', raw), ('report_scores.csv', scores), ('metrics.csv', metrics),
                       ('contrasts.csv', contrasts), ('fit_summary.csv', fit_rows),
                       ('modal_interaction.csv', interactions), ('pair_response.csv', pair_rows),
                       ('pair_summary.csv', pair_summary), ('compatible_pair_response.csv', compatible_rows),
                       ('compatible_pair_summary.csv', compatible_summary)):
        rt.write_csv(out / name, rows)
    teacher_crps = next(r for r in contrasts if r['subset'] == 'structure' and
                       r['right_model'] == 'ADAPTER_TEACHER' and r['metric'] == 'crps_ev')
    teacher_nll = next(r for r in contrasts if r['subset'] == 'structure' and
                      r['right_model'] == 'ADAPTER_TEACHER' and r['metric'] == 'nll')
    gate = dict(passed=teacher_crps['improvement'] > 0 and teacher_crps['seed_improvement_count'] >= 2
                and teacher_nll['improvement'] >= 0,
                criterion='same structure-supported grouped-development OOF FULL: density-ensemble CRPS improves; >=2/3 seed CRPS improves; ensemble NLL nonworse',
                scope='Development gate for a downstream experiment, not independent confirmation or parameter selection',
                crps=teacher_crps, nll=teacher_nll, protocol_sha256=rt.sha(work/'training_protocol.json'),
                teacher_bindings={p.name:rt.sha(p/'complete.json') for p in sorted((work/'teachers').iterdir()) if p.is_dir()},
                teacher_output_binding={k:v for k,v in bindings.items() if k.startswith('teachers/')})
    if make_gate:
        path = work / 'teacher_gate.json'
        if path.exists() and rt.read_json(path) != gate:
            raise ValueError('Cannot change the predeclared teacher gate')
        rt.write_json(path, gate)
    rt.write_json(out / 'summary.json', dict(reports=len(records), formulas=len({r['composition_key'] for r in records}),
        structure_reports=len(subsets['structure']), conditional_reports=len(subsets['conditional']),
        paired_formulas=len(paired), pairs=sum(len(g)*(len(g)-1)//2 for g in paired.values()),
        scope='Previously developed source/element-grouped OOF; not an independent confirmation',
        source_sha256=rt.sha(Path(__file__)), input_binding={n:rt.sha(parent/'inputs'/n)
            for n in ('features.pt','metadata.json','records.jsonl')}, fit_output_binding=bindings,
        compatible_pairs_sha256=rt.sha(compatible_path),
        actual_teacher_fits=sum(r['phase']=='teachers' for r in fit_rows),
        actual_student_fits=sum(r['phase']=='students' for r in fit_rows), teacher_gate_passed=gate['passed']))
    print(json.dumps(dict(teacher_gate=gate['passed'], teacher_crps_gain=teacher_crps['improvement'],
        teacher_nll_gain=teacher_nll['improvement'], fits=len(fit_rows)), ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','parent','work'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--make-gate', action='store_true')
    args=parser.parse_args()
    run(args.root,args.parent,args.work,args.make_gate)
