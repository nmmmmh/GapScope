"""Score decoupled-distribution OOF predictions and frozen references."""
import argparse, itertools, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from density_score import score
from training_data import hierarchical_weights

METRICS = ('nll', 'crps_ev', 'median_mae_ev', 'coverage_80', 'width_80_ev')
VIEWS = rt.VIEWS
REFERENCES = {'FROZEN_MULTIVIEW': VIEWS, 'FROZEN_FORMULA': ('formula_only',)}
PAIR_MODELS = ('FROZEN_MULTIVIEW', 'DECOUPLED_NLL', 'DECOUPLED_PAIR')


def check_fit(path, complete):
    for name, key in (('predictions.csv', 'predictions_sha256'),
                      ('models.pt', 'models_sha256'), ('fits.json', 'fits_sha256')):
        if rt.sha(path / name) != complete[key]:
            raise ValueError(f'{name} hash mismatch: {path}')


def parts(rows):
    values = tuple(np.concatenate([np.asarray(json.loads(r[k]), dtype=float) for r in rows])
                   for k in ('component_weights', 'log_mus', 'sigmas'))
    return values[0] / len(rows), values[1], values[2]


def run(root, work, parent):
    root, work, parent = map(Path, (root, work, parent))
    protocol_path = work / 'training_protocol.json'
    protocol = rt.read_json(protocol_path)
    seeds, nfolds, arms = protocol['seeds'], protocol['folds'], protocol['arms']
    if set(seeds) != {17, 29, 41} or nfolds != 3 or set(arms) != {'DECOUPLED_NLL', 'DECOUPLED_PAIR'}:
        raise ValueError('expected DECOUPLED_NLL/DECOUPLED_PAIR with seeds 17/29/41 and three folds')
    for arm in arms:
        if set(protocol['evaluated_views'][arm]) != set(VIEWS):
            raise ValueError(f'{arm} must provide all four views')

    inp = parent / 'inputs'
    rp, mp, fp = inp / 'records.jsonl', inp / 'metadata.json', inp / 'features.pt'
    metadata = rt.read_json(mp)
    records = [dict(m, **json.loads(line)) for m, line in
               zip(metadata, rp.read_text(encoding='utf-8').splitlines()) if line.strip()]
    if len(records) != 205 or len(metadata) != 205:
        raise ValueError('parent inputs must be the fixed 205 aligned reports')
    features = rt.torch.load(fp, map_location='cpu', weights_only=True)
    sg, structure = [bool(x) for x in features['sg_mask']], [bool(x) for x in features['structure_mask']]
    ids = {r['measurement_id']: i for i, r in enumerate(records)}
    if len(ids) != 205 or len(sg) != 205 or len(structure) != 205:
        raise ValueError('duplicate IDs or feature masks misaligned with parent reports')

    raw, fit_rows, manifest, outer, counts = [], [], {}, {}, [0, 0]
    for seed in seeds:
        for fold in range(nfolds):
            path = work / 'fits' / f'seed_{seed}_fold_{fold + 1:02d}'
            complete = rt.read_json(path / 'complete.json'); check_fit(path, complete)
            fits = rt.read_json(path / 'fits.json')
            part = rt.read_json(path / 'partition.json')
            if part['fold'] != fold or part['folds'] != nfolds:
                raise ValueError(f'new fold metadata mismatch: {path}')
            outer['new', seed, fold] = set(part['report_outer_composition_keys'])
            job_reports = 0
            for arm in arms:
                fit = fits[arm]
                if fit.get('initial_state_hash') != complete['prior_hash']:
                    raise ValueError(f'prior hash mismatch: {path}/{arm}')
                fit_rows.append({'source': 'new_fit', 'seed': seed, 'fold': fold, 'model': arm,
                                 **{k: v for k, v in fit.items() if isinstance(v, (str, int, float, bool)) or v is None}})
                job_reports += 1
            declared = complete.get('new_report_fits', complete.get('report_fits'))
            counts[1] += int(declared) if declared is not None else job_reports
            counts[0] += int(complete.get('new_aggregate_fits', complete.get('aggregate_fits', 0)))
            if declared is not None and int(declared) != job_reports:
                raise ValueError(f'report fit count mismatch: {path}')
            raw.extend(r for r in rt.read_csv(path / 'predictions.csv') if r['model'] in arms)
            manifest[str(path)] = {n: rt.sha(path / n) for n in
                                   ('complete.json', 'predictions.csv', 'models.pt', 'fits.json')}

    # Parent OOF predictions are references; preserve their original model names and 3-fold identities.
    for seed in seeds:
        for fold in range(nfolds):
            path = parent / 'fits' / f'seed_{seed}_fold_{fold + 1:02d}'
            complete = rt.read_json(path / 'complete.json'); check_fit(path, complete)
            part = rt.read_json(path / 'partition.json')
            if part['fold'] != fold or part['folds'] != nfolds:
                raise ValueError(f'parent fold metadata mismatch: {path}')
            outer['parent', seed, fold] = set(part['report_outer_composition_keys'])
            if outer['parent', seed, fold] != outer['new', seed, fold]:
                raise ValueError(f'parent/new outer split mismatch: {seed}/{fold}')
            raw.extend(r for r in rt.read_csv(path / 'predictions.csv')
                       if r['model'] in REFERENCES and r['view'] in REFERENCES[r['model']])
            manifest[str(path)] = {n: rt.sha(path / n) for n in
                                   ('complete.json', 'predictions.csv', 'models.pt', 'fits.json')}

    views = {**protocol['evaluated_views'], **REFERENCES}
    grouped = defaultdict(list)
    for r in raw:
        model, view, seed, fold = r['model'], r['view'], int(r['seed']), int(r['fold_id'])
        if view not in views[model] or seed not in seeds or fold not in range(nfolds):
            raise ValueError(f'prediction outside protocol: {model}/{view}/{seed}/{fold}')
        i = ids[r['measurement_id']]
        if r['composition_key'] != records[i]['composition_key'] or r['composition_key'] not in outer['new', seed, fold]:
            raise ValueError(f'OOF formula/fold mismatch: {r["measurement_id"]}')
        for key, mask in (('sg_supported', sg), ('structure_supported', structure)):
            if key in r and bool(int(r[key])) != mask[i]:
                raise ValueError(f'{key} differs from parent feature mask: {r["measurement_id"]}')
        grouped[model, view, r['measurement_id']].append(r)
    for model, model_views in views.items():
        for view in model_views:
            for r in records:
                rows = grouped[model, view, r['measurement_id']]
                if len(rows) != 3 or {int(x['seed']) for x in rows} != set(seeds):
                    raise ValueError(f'OOF coverage failure: {model}/{view}/{r["measurement_id"]}')

    weights = hierarchical_weights(records, list(range(205)), 'cpu').tolist()
    mass = {r['measurement_id']: w for r, w in zip(records, weights)}
    mixtures = ['equal_density_mixture', *map(str, seeds)]
    scores = []
    for model, model_views in views.items():
        for view in model_views:
            for mixture in mixtures:
                for r in records:
                    rows = grouped[model, view, r['measurement_id']]
                    selected = rows if mixture == 'equal_density_mixture' else [x for x in rows if x['seed'] == mixture]
                    p = parts(selected); mean = float(p[0] @ np.exp(p[1] + .5 * p[2] ** 2))
                    scores.append(dict(model=model, view=view, seed_mixture=mixture,
                        measurement_id=r['measurement_id'], formula=r['formula'], composition_key=r['composition_key'],
                        weight=mass[r['measurement_id']], mean_ev=mean, target_ev=r['gap_ev'], **score(r['gap_ev'], p)))

    metrics = []
    for subset in ('all', 'structure', 'natural_sg_only', 'natural_neither'):
        ix = [i for i in range(205) if subset == 'all' or subset == 'structure' and structure[i] or
              subset == 'natural_sg_only' and sg[i] and not structure[i] or
              subset == 'natural_neither' and not sg[i] and not structure[i]]
        if not ix: continue
        sw = hierarchical_weights(records, ix, 'cpu').tolist()
        wm = {records[i]['measurement_id']: w for i, w in zip(ix, sw)}
        for model, model_views in views.items():
            for view in model_views:
                for mixture in mixtures:
                    rs = [r for r in scores if r['model'] == model and r['view'] == view and
                          r['seed_mixture'] == mixture and r['measurement_id'] in wm]
                    metrics.append(dict(subset=subset, model=model, view=view, seed_mixture=mixture,
                        reports=len(rs), formulas=len({r['composition_key'] for r in rs}),
                        **{k: sum(wm[r['measurement_id']] * r[k] for r in rs) for k in METRICS}))

    # Complete same-formula actual-CIF set; no source-compatible-pair selection here.
    cifs = defaultdict(list)
    for i, r in enumerate(records):
        if structure[i]:
            digest = metadata[i].get('actual_cif_sha256')
            if not digest: raise ValueError('encoded CIF identity missing')
            cifs[r['composition_key'], digest].append(r)
    by_formula = defaultdict(list)
    for (key, digest), rs in cifs.items(): by_formula[key].append((digest, rs))
    pairs = {k: list(itertools.combinations(sorted(v), 2)) for k, v in by_formula.items() if len(v) > 1}
    pair_count, pair_scores = sum(map(len, pairs.values())), {}
    for r in scores:
        if r['model'] in PAIR_MODELS:
            pair_scores[r['model'], r['view'], r['seed_mixture'], r['measurement_id']] = r['mean_ev']
    pair_rows = []
    for model in PAIR_MODELS:
        for mixture in mixtures:
            for key, combinations in pairs.items():
                for (ha, aa), (hb, bb) in combinations:
                    wa = np.array([mass[r['measurement_id']] for r in aa]); wa /= wa.sum()
                    wb = np.array([mass[r['measurement_id']] for r in bb]); wb /= wb.sum()
                    observed = float(wb @ np.array([r['gap_ev'] for r in bb]) - wa @ np.array([r['gap_ev'] for r in aa]))
                    row = dict(model=model, composition_key=key, formula=aa[0]['formula'], cif_a=ha, cif_b=hb,
                        seed_mixture=mixture, observed_delta_ev=observed, zero_delta_error_ev=abs(observed),
                        weight=1 / len(pairs) / len(combinations))
                    for view in VIEWS:
                        delta = float(wb @ np.array([pair_scores[model, view, mixture, r['measurement_id']] for r in bb]) -
                                      wa @ np.array([pair_scores[model, view, mixture, r['measurement_id']] for r in aa]))
                        row[view + '_delta_ev'], row[view + '_delta_error_ev'] = delta, abs(delta - observed)
                    pair_rows.append(row)
    if pair_count != 22: raise ValueError(f'expected all 22 actual-CIF pairs, found {pair_count}')

    out = work / 'analysis'; out.mkdir(exist_ok=True)
    outputs = {'predictions.csv': raw, 'report_scores.csv': scores, 'metrics.csv': metrics,
               'fit_summary.csv': fit_rows, 'pair_response.csv': pair_rows}
    for name, rows in outputs.items(): rt.write_csv(out / name, rows)
    input_hashes = {str(p): rt.sha(p) for p in (rp, mp, fp)}
    parent_protocol = parent / 'training_protocol.json'
    rt.write_json(out / 'summary.json', dict(root=str(root), work=str(work), parent=str(parent),
        source_sha256=rt.sha(Path(__file__)), training_protocol_sha256=rt.sha(protocol_path),
        parent_protocol_sha256=rt.sha(parent_protocol) if parent_protocol.exists() else None,
        parent_input_sha256=input_hashes, fit_output_sha256=manifest,
        new_aggregate_fits=counts[0], new_report_fits=counts[1], reports=205,
        formulas=len({r['composition_key'] for r in records}), seeds=seeds, folds=nfolds, models=views,
        same_formula_actual_cif_pairs=pair_count, paired_formulas=len(pairs), pair_response_rows=len(pair_rows),
        contrast_boundary='Do not interpret cross-head F-to-SG differences as modality Shapley; SG-to-FULL is a same-conditional-head increment.',
        scope='Three-seed, three-fold grouped OOF with frozen references; no confirmation/reserved/auxiliary targets read.',
        output_sha256={name: rt.sha(out / name) for name in outputs}))
    print(dict(reports=205, raw_predictions=len(raw), scores=len(scores), metrics=len(metrics),
               actual_cif_pairs=pair_count, pair_response_rows=len(pair_rows),
               new_aggregate_fits=counts[0], new_report_fits=counts[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.work, args.parent)
