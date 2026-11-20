"""Matched joint versus frozen report-head fits from one aggregate prior."""
import argparse
import copy
import hashlib
import time
from pathlib import Path

import runtime as rt
import train
from training_data import build_fold

torch = rt.torch
RECIPE = dict(
    seeds=list(train.RECIPE['seeds']),
    folds=train.RECIPE['folds'],
    report_epochs=train.RECIPE['report_epochs'],
    patience=train.RECIPE['patience'],
    shared_lr=train.RECIPE['shared_lr'],
    head_lr=train.RECIPE['head_lr'],
    replay_coefficient=train.RECIPE['replay_coefficient'],
    report_selection=train.RECIPE['report_selection'],
    report_loss='hierarchical TRAIN LogNormal NLL averaged over four views',
    frozen_parameters='all parameters except report_head, including alpha',
    arms=['JOINT', 'FROZEN'],
    budget='15 aggregate fits + 30 report fits = 45 fits',
)


def fit_frozen(prior, data, standardized):
    model = copy.deepcopy(prior)
    model.mode = 'M'
    for parameter in model.parameters(): parameter.requires_grad_(False)
    head = list(model.report_head.parameters())
    for parameter in head: parameter.requires_grad_(True)

    def nll(indices, weights):
        loss = model.per_record_nll(data['views'][:, indices], data['targets'][indices])
        if loss.ndim == 2: loss = loss.mean(0)
        return (loss * weights).sum()

    def objective():
        replay = (model.aggregate(data['av'][:, data['fi']]).double()
                  - standardized[data['fi']]).square().mean()
        return nll(data['ti'], data['tw']) + train.RECIPE['replay_coefficient'] * replay

    optimizer = torch.optim.Adam(head, lr=train.RECIPE['head_lr'])
    fit = train.optimize(model, objective, lambda: nll(data['vi'], data['vw']),
                         optimizer, train.RECIPE['report_epochs'])
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if any(not name.startswith('report_head.') for name in trainable):
        raise ValueError('frozen_fit_unexpected_trainable_parameter')
    fit['trainable_parameter_names'] = trainable
    return model, fit


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_pair(root, work, seed, fold, device, modules, protocol_sha):
    out = work / f'seed_{seed}_fold_{fold + 1:02d}'
    out.mkdir(parents=True, exist_ok=True)
    complete_path = out / 'complete.json'
    if complete_path.exists():
        return

    started = time.time()
    data = build_fold(root, fold, device)
    prior, standardized, aggregate_fit, target_scalers = train.aggregate_prior(
        modules, data, seed, fold, device)
    prior_state = train.state(prior)
    prior_hash = _state_hash(prior_state)

    joint, joint_fit = train.adapt(prior, data, standardized, 'J')
    frozen, frozen_fit = fit_frozen(prior, data, standardized)
    joint_fit['initial_state_hash'] = prior_hash
    frozen_fit['initial_state_hash'] = prior_hash
    frozen_fit['selected_arm'] = 'FROZEN'
    joint_fit['selected_arm'] = 'JOINT'

    predictions = []
    with torch.no_grad():
        for name, model in (('JOINT', joint), ('FROZEN', frozen)):
            for view_index, view_name in enumerate(rt.VIEWS):
                parts = model.components(data['views'][view_index, data['oi']])
                predictions.extend(train.prediction_rows(
                    parts, data, seed, fold, name, view_name))

    if len(predictions) != 8 * len(data['oi']):
        raise ValueError(f'prediction_count_mismatch:{len(predictions)}')
    rt.write_csv(out / 'predictions.csv', predictions)
    rt.write_json(out / 'fits.json', dict(
        aggregate=aggregate_fit, JOINT=joint_fit, FROZEN=frozen_fit,
        fit_count=3, initial_prior_state_hash=prior_hash,
    ))
    models_path = out / 'models.pt'
    torch.save(dict(
        schema='frozen_fusion_compare_v1',
        seed=seed,
        fold=fold,
        models={'JOINT': train.state(joint), 'FROZEN': train.state(frozen)},
        scalers=data['scalers'],
        target_scalers=target_scalers,
        partition=data['provenance'],
        prior=prior_state,
    ), models_path)
    rt.write_json(complete_path, dict(
        status='complete', seed=seed, fold=fold,
        aggregate_fits=1, report_fits=2, total_fits=3,
        predictions=len(predictions),
        protocol_sha256=protocol_sha,
        predictions_sha256=rt.sha(out / 'predictions.csv'),
        fits_sha256=rt.sha(out / 'fits.json'),
        models_sha256=rt.sha(models_path),
        seconds=time.time() - started,
    ))
    print(f'COMPLETE seed={seed} fold={fold + 1} fits=3', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir.resolve()
    try:
        work.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError('work_dir_must_be_outside_formal_root')
    work.mkdir(parents=True, exist_ok=True)

    device = rt.configure()
    modules = rt.modules()
    sources = [Path(__file__).resolve(), Path(train.__file__).resolve(),
               Path(build_fold.__code__.co_filename).resolve(),
               Path(rt.__file__).resolve(),
               root / 'runtime' / 'anchored_transfer.py']
    sources += sorted((root / 'runtime' / 'modules').glob('*.py'))
    sources += [root / 'runtime' / 'modules' / 'symmetry_table.json']
    inputs = [root / 'data' / 'reports' / name for name in
              ('features.pt', 'records.jsonl', 'metadata.json')]
    inputs += [root / 'data' / 'composition' / name for name in
               ('features.pt', 'metadata.json')]
    binding = {str(path.relative_to(root)): rt.sha(path)
               for path in [*inputs, *sources]}
    protocol = RECIPE | dict(input_binding=binding, device=str(device),
                             gpu=torch.cuda.get_device_name(0))
    protocol_path = work / 'protocol.json'
    if protocol_path.exists() and rt.read_json(protocol_path) != protocol:
        raise ValueError('Input or recipe changed; existing run cannot be resumed')
    rt.write_json(protocol_path, protocol)
    protocol_sha = rt.sha(protocol_path)

    for seed in RECIPE['seeds']:
        for fold in range(RECIPE['folds']):
            run_pair(root, work, seed, fold, device, modules, protocol_sha)


if __name__ == '__main__':
    main()
