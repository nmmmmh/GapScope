"""Paired out-of-fold evidence for structural supervision and formula transfer."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from analyze_frozen import METRICS, _parts, _read_targets
from density_score import score
from training_data import hierarchical_weights


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--distill-work-dir',type=Path)
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    protocol=rt.read_json(args.work_dir/'protocol.json'); seeds=protocol['seeds']
    records=_read_targets(args.root); by_id={r['measurement_id']:r for r in records}
    weights=hierarchical_weights(records,list(range(len(records))),'cpu').tolist()
    weight_by_id={r['measurement_id']:w for r,w in zip(records,weights)}
    raw=[]; fits=[]; signals=[]; bindings={}
    for seed in seeds:
        for fold in range(5):
            path=args.work_dir/f'seed_{seed}_fold_{fold+1:02d}'
            receipt=rt.read_json(path/'complete.json')
            for name in ('predictions.csv','fits.json'):
                digest=rt.sha(path/name); bindings[f'{path.name}/{name}']=digest
                if receipt.get(name.replace('.csv','').replace('.json','')+'_sha256')!=digest:
                    raise ValueError(f'completion hash mismatch: {path.name}/{name}')
            raw.extend(rt.read_csv(path/'predictions.csv'))
            for model,fit in rt.read_json(path/'fits.json').items():
                fits.append(dict(seed=seed,fold=fold,model=model,**{k:v for k,v in fit.items() if k not in ('history','cdf_history')}))
            signals.extend([dict(seed=seed,fold=fold,**r) for r in rt.read_csv(path/'teacher_signal.csv')])
    if args.distill_work_dir:
        protocol['supervised_distillation']=rt.read_json(args.distill_work_dir/'protocol.json')
        for seed in seeds:
            for fold in range(5):
                path=args.distill_work_dir/f'seed_{seed}_fold_{fold+1:02d}'
                receipt=rt.read_json(path/'complete.json')
                for name in ('predictions.csv','fits.json'):
                    digest=rt.sha(path/name); bindings[f'direct/{path.name}/{name}']=digest
                    if receipt.get(name.replace('.csv','').replace('.json','')+'_sha256')!=digest:
                        raise ValueError(f'direct completion hash mismatch: {path.name}/{name}')
                raw.extend(r for r in rt.read_csv(path/'predictions.csv') if r['model']!='FROZEN')
                for model,fit in rt.read_json(path/'fits.json').items():
                    fits.append(dict(seed=seed,fold=fold,model=model,**{k:v for k,v in fit.items()
                        if k not in ('history','cdf_history')}))
    groups=defaultdict(list)
    for r in raw: groups[(r['model'],r['view'],r['measurement_id'])].append(r)
    combinations=[(m,v) for m in ('FROZEN','FROZEN_SG') for v in rt.VIEWS]+[('STRUCTURE_TRANSFER','formula_only')]
    if args.distill_work_dir:
        combinations += [(m,'formula_only') for m in ('SG_DISTILL','FULL_DISTILL')]
    if len(groups)!=len(combinations)*len(records): raise ValueError('incomplete OOF groups')
    scores=[]; indexed=defaultdict(dict)
    for (model,view,rid),rows in sorted(groups.items()):
        if (model,view) not in combinations or sorted(int(r['seed']) for r in rows)!=sorted(seeds):
            raise ValueError('missing or duplicate seed/model/view')
        if len({r['fold_id'] for r in rows})!=1: raise ValueError('fold differs across seeds')
        record=by_id[rid]
        for mixture,selected in [('equal_density_mixture',rows)]+[(str(r['seed']),[r]) for r in rows]:
            result=dict(model=model,view=view,seed_mixture=mixture,measurement_id=rid,
                formula=record['formula'],composition_key=record['composition_key'],
                phase_label=record.get('phase_label',''),target_ev=float(record['gap_ev']),
                sg_supported=int(rows[0]['sg_supported']),structure_supported=int(rows[0]['structure_supported']),
                weight=weight_by_id[rid],**score(float(record['gap_ev']),_parts(selected)))
            scores.append(result); indexed[(model,view,mixture)][rid]=result
    metrics=[]
    def selected_ids(rows,subset):
        return [rid for rid,r in rows.items() if subset=='all' or
                bool(r['structure_supported'])==(subset=='structure')]
    def average(rows,ids,metric):
        w=np.array([rows[i]['weight'] for i in ids]); w/=w.sum()
        return float(w@np.array([rows[i][metric] for i in ids]))
    for (model,view,mixture),rows in indexed.items():
        if len(rows)!=len(records): raise ValueError('OOF report count mismatch')
        for subset in ('all','structure','missing_structure'):
            ids=selected_ids(rows,subset)
            metrics.append(dict(model=model,view=view,seed_mixture=mixture,subset=subset,reports=len(ids),
                formulas=len({rows[i]['composition_key'] for i in ids}),
                **{metric:average(rows,ids,metric) for metric in METRICS}))
    definitions={
        'structural_report_supervision_formula':(('FROZEN_SG','formula_only'),('FROZEN','formula_only')),
        'structural_report_supervision_missing_structure':(('FROZEN_SG','formula_sg'),('FROZEN','formula_sg')),
        'total_available_structure_effect':(('FROZEN_SG','formula_sg'),('FROZEN','full')),
        'teacher_structure_at_inference':(('FROZEN','formula_sg'),('FROZEN','full')),
        'teacher_total_modal_at_inference':(('FROZEN','formula_only'),('FROZEN','full')),
        'structure_transfer_to_formula':(('FROZEN','formula_only'),('STRUCTURE_TRANSFER','formula_only'))}
    if args.distill_work_dir:
        definitions.update({
            'supervised_full_distillation':(('FROZEN','formula_only'),('FULL_DISTILL','formula_only')),
            'supervised_sg_distillation':(('FROZEN','formula_only'),('SG_DISTILL','formula_only')),
            'structure_distillation_increment':(('SG_DISTILL','formula_only'),('FULL_DISTILL','formula_only'))})
    contrasts=[]
    for name,(left,right) in definitions.items():
        for mixture in ['equal_density_mixture']+[str(seed) for seed in seeds]:
            a=indexed[(*left,mixture)]; b=indexed[(*right,mixture)]
            for subset in ('all','structure','missing_structure'):
                ids=selected_ids(a,subset)
                for metric in ('nll','crps_ev','median_mae_ev'):
                    contrasts.append(dict(contrast=name,seed_mixture=mixture,subset=subset,metric=metric,
                        improvement=average(a,ids,metric)-average(b,ids,metric),reports=len(ids)))
    lookup={(r['contrast'],r['seed_mixture'],r['subset'],r['metric']):r['improvement'] for r in contrasts}
    retention=[]
    for subset in ('all','structure'):
        for metric in ('nll','crps_ev'):
            teacher=lookup[('teacher_structure_at_inference','equal_density_mixture',subset,metric)]
            gain=lookup[('structure_transfer_to_formula','equal_density_mixture',subset,metric)]
            supervision=lookup[('structural_report_supervision_formula','equal_density_mixture',subset,metric)]
            retention.append(dict(subset=subset,metric=metric,teacher_structure_gain=teacher,
                formula_transfer_gain=gain,structure_supervision_formula_gain=supervision,
                descriptive_transfer_gain_ratio=gain/teacher if teacher>0 else '',
                interpretation='descriptive incremental transfer gain / same-teacher structure-at-inference gain; distinct prediction views'))
    rt.write_csv(out/'metrics.csv',metrics); rt.write_csv(out/'contrasts.csv',contrasts)
    rt.write_csv(out/'report_scores.csv',scores); rt.write_csv(out/'oof_densities.csv',raw)
    rt.write_csv(out/'fit_summary.csv',fits); rt.write_csv(out/'teacher_signal.csv',signals)
    rt.write_csv(out/'retention.csv',retention)
    rt.write_json(out/'protocol.json',dict(training=protocol,score='same analytic mixture LogNormal NLL/CRPS and hierarchical weights',
        input_binding=bindings,output_binding={p.name:rt.sha(p) for p in out.glob('*.csv')}))
    print(json.dumps(dict(reports=len(records),predictions=len(raw),fits=len(fits),metric_rows=len(metrics))))


if __name__=='__main__': main()
