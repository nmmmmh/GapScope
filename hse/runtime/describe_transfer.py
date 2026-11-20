"""Formula-level and report-label summaries for the fixed transfer analysis."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import runtime as rt
from analyze_frozen import METRICS

SEEDS = ('17', '29', '41')
MIXTURES = ('equal_density_mixture', *SEEDS)
FOCAL = (('FROZEN', 'formula_only'), ('FROZEN', 'formula_sg'),
         ('FROZEN', 'formula_structure'), ('FROZEN', 'full'),
         ('FROZEN_SG', 'formula_only'), ('STRUCTURE_TRANSFER', 'formula_only'))
CONTRASTS = {
    'supervision_formula': (('FROZEN_SG', 'formula_only'), ('FROZEN', 'formula_only')),
    'supervision_sg': (('FROZEN_SG', 'formula_sg'), ('FROZEN', 'formula_sg')),
    'teacher_structure': (('FROZEN', 'formula_sg'), ('FROZEN', 'full')),
    'transfer_formula': (('FROZEN', 'formula_only'), ('STRUCTURE_TRANSFER', 'formula_only')),
}
EFFECT_METRICS = tuple(m for m in METRICS if m in ('nll', 'crps_ev', 'median_mae_ev'))


def _f(row, key):
    return float(row[key])


def _weighted(rows, key):
    total = sum(_f(r, 'weight') for r in rows)
    if total <= 0:
        raise ValueError('nonpositive_report_weight')
    return sum(_f(r, 'weight') * _f(r, key) for r in rows) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, analysis, out = Path(__file__).resolve().parents[1], args.analysis_dir.resolve(), args.output_dir.resolve()
    try:
        out.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError('output_dir_must_be_outside_formal_root')
    out.mkdir(parents=True, exist_ok=True)
    score_path, density_path = analysis / 'report_scores.csv', analysis / 'oof_densities.csv'
    scores, densities = rt.read_csv(score_path), rt.read_csv(density_path)
    focal=list(FOCAL); contrasts=dict(CONTRASTS)
    if any(r['model']=='FULL_DISTILL' for r in scores):
        focal.extend((m,'formula_only') for m in ('SG_DISTILL','FULL_DISTILL'))
        contrasts.update({
            'supervised_full':(('FROZEN','formula_only'),('FULL_DISTILL','formula_only')),
            'structure_distillation':(('SG_DISTILL','formula_only'),('FULL_DISTILL','formula_only'))})
    score_index, by_condition = {}, defaultdict(list)
    ids = set()
    for r in scores:
        key = (r['model'], r['view'], r['seed_mixture'], r['measurement_id'])
        if key in score_index or r['seed_mixture'] not in MIXTURES:
            raise ValueError(f'duplicate_or_unknown_score:{key}')
        score_index[key] = r
        by_condition[(r['model'], r['view'], r['seed_mixture'], r['composition_key'])].append(r)
        if r['seed_mixture'] == 'equal_density_mixture' and r['model'] == 'FROZEN' and r['view'] == 'formula_only':
            ids.add(r['measurement_id'])
    if len(ids) != 205:
        raise ValueError(f'expected_205_reports:{len(ids)}')
    density_seeds = defaultdict(set)
    for r in densities:
        density_seeds[(r['model'], r['view'], r['measurement_id'])].add(str(r['seed']))
    combinations = {(m, v) for m in ('FROZEN', 'FROZEN_SG')
                    for v in ('formula_only', 'formula_sg', 'formula_structure', 'full')}
    combinations.add(('STRUCTURE_TRANSFER', 'formula_only'))
    combinations.update((m,v) for m,v in focal)
    expected_density_keys = {(m, v, rid) for m, v in combinations for rid in ids}
    if set(density_seeds) != expected_density_keys or any(s != set(SEEDS) for s in density_seeds.values()):
        raise ValueError('oof_density_coverage_mismatch')
    for m, v in combinations:
        for mix in MIXTURES:
            if {rid for mm, vv, mi, rid in score_index if (mm, vv, mi) == (m, v, mix)} != ids:
                raise ValueError(f'score_coverage_mismatch:{m}:{v}:{mix}')

    formula_keys = sorted({r['composition_key'] for r in scores
                           if r['model'] == 'FROZEN' and r['view'] == 'formula_only'
                           and r['seed_mixture'] == MIXTURES[0]})
    if len(formula_keys) != 56:
        raise ValueError(f'expected_56_formulas:{len(formula_keys)}')
    effects = []
    for comp in formula_keys:
        base = next(r for r in by_condition[('FROZEN', 'formula_only', MIXTURES[0], comp)])
        row = dict(composition_key=comp, formula=base['formula'],
                   reports=len(by_condition[('FROZEN', 'formula_only', MIXTURES[0], comp)]))
        for name, (left, right) in contrasts.items():
            for mix in MIXTURES:
                for metric in EFFECT_METRICS:
                    a = _weighted(by_condition[(*left, mix, comp)], metric)
                    b = _weighted(by_condition[(*right, mix, comp)], metric)
                    suffix = '' if mix == MIXTURES[0] else f'_seed{mix}'
                    row[f'improvement_{name}_{metric}{suffix}'] = a - b
        effects.append(row)
    rt.write_csv(out / 'formula_effects.csv', effects)

    phase_groups = defaultdict(list)
    for rid in sorted(ids):
        r = score_index[('FROZEN', 'formula_only', MIXTURES[0], rid)]
        phase_groups[(r['composition_key'], r['phase_label'])].append(r)
    phase_rows = []
    for (comp, label), rows in sorted(phase_groups.items()):
        ids_in_group = [r['measurement_id'] for r in rows]
        row = dict(composition_key=comp, formula=rows[0]['formula'], phase_label=label,
                   phase_label_group='report phase label group; not a verified phase count',
                   reports=len(rows), observed_gap_ev=_weighted(rows, 'target_ev'))
        for model, view in focal:
            pred_rows = [score_index[(model, view, MIXTURES[0], rid)] for rid in ids_in_group]
            row[f'pred_median_{model}_{view}_ev'] = _weighted(pred_rows, 'median_ev')
        phase_rows.append(row)
    rt.write_csv(out / 'phase_response.csv', phase_rows)

    metric_rows, figure_rows = [], []
    for model, view in focal:
        condition = [r for comp in formula_keys
                     for r in by_condition[(model, view, MIXTURES[0], comp)]]
        entry = dict(model=model, view=view)
        for metric in ('nll', 'crps_ev'):
            value = _weighted(condition, metric)
            entry[metric] = value
            figure_rows.append(dict(panel='all205', model=model, view=view, metric=metric, value=value))
        metric_rows.append(entry)

    labels = [f'{m}\n{v}' for m, v in focal]
    x = list(range(len(focal)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 14), gridspec_kw={'width_ratios': [1, 1.6]})
    right_axis = left.twinx()
    width = .36
    bars_nll = left.bar([i - width / 2 for i in x], [r['nll'] for r in metric_rows], width,
                        color='#356b8c', label='NLL')
    bars_crps = right_axis.bar([i + width / 2 for i in x], [r['crps_ev'] for r in metric_rows], width,
                               color='#d98436', label='CRPS (eV)')
    left.set_xticks(x, labels, rotation=25, ha='right')
    left.set_ylabel('NLL')
    right_axis.set_ylabel('CRPS (eV)')
    left.set_title('All 205 reports: equal-density mixture')
    left.legend([bars_nll, bars_crps], ['NLL', 'CRPS (eV)'], loc='upper left')

    y = list(range(len(effects)))
    offsets = [-.25+.5*i/max(1,len(contrasts)-1) for i in range(len(contrasts))]
    for (name, _), offset, color in zip(contrasts.items(), offsets,
                                         ('#356b8c', '#5c9b69', '#bd5b50', '#8856a7','#d98436','#348f99')):
        key = f'improvement_{name}_crps_ev'
        right.scatter([r[key] for r in effects], [i + offset for i in y], s=18, color=color, label=name)
        for r in effects:
            figure_rows.append(dict(panel='formula_crps_effect', composition_key=r['composition_key'],
                                    formula=r['formula'], contrast=name, metric='crps_ev', value=r[key]))
    right.axvline(0, color='black', linewidth=.8)
    right.set_yticks(y, [r['formula'] for r in effects], fontsize=7)
    right.invert_yaxis()
    right.set_xlabel('CRPS improvement (eV; positive favors right-hand model)')
    right.set_title('Per-formula CRPS effects: all 56 formulas')
    right.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    png, pdf = out / 'transfer_summary.png', out / 'transfer_summary.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    rt.write_csv(out / 'figure_data.csv', figure_rows)
    outputs = [out / name for name in ('formula_effects.csv', 'phase_response.csv',
                                       'figure_data.csv', 'transfer_summary.png', 'transfer_summary.pdf')]
    rt.write_json(out / 'protocol.json', dict(
        analysis_inputs={score_path.name: rt.sha(score_path), density_path.name: rt.sha(density_path)},
        formula_count=len(formula_keys), report_count=len(ids), density_rows=len(densities),
        effect_convention='left metric minus right metric; positive favors right-hand model',
        effect_definitions={k: dict(left=f'{a[0]}/{a[1]}', right=f'{b[0]}/{b[1]}')
                            for k, (a, b) in contrasts.items()},
        phase_label_group='exact composition_key + original phase_label text; not a verified phase count',
        phase_medians='report-weighted mean of equal-density-mixture report medians',
        score_metrics=list(EFFECT_METRICS), output_binding={p.name: rt.sha(p) for p in outputs}))


if __name__ == '__main__':
    main()
