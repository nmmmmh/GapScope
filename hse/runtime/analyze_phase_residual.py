"""Score the fixed phase-residual pair and all existing reference densities."""
import argparse
from pathlib import Path
from collections import defaultdict
import runtime as rt
from analyze_frozen import _read_targets, _score_rows, _metrics
from compare_cif_head_response import run as cif_response

ARMS=('PHASE_NLL','PHASE_PAIR')


def run(root,work,out):
    if out.resolve().is_relative_to(root.resolve()):raise ValueError('output must be outside formal root')
    out.mkdir(parents=True,exist_ok=True)
    protocol=rt.read_json(work/'protocol.json');raw=[];fits=[];binding={}
    for seed in (17,29,41):
        for fold in range(5):
            directory=work/f'seed_{seed}_fold_{fold+1:02d}'
            receipt=rt.read_json(directory/'complete.json')
            if receipt['protocol_sha256']!=rt.sha(work/'protocol.json'):raise ValueError('run protocol changed')
            for name in ('predictions.csv','fits.json','models.pt'):
                digest=rt.sha(directory/name);binding[directory.name+'/'+name]=digest
                if receipt[name.split('.')[0]+'_sha256']!=digest:raise ValueError('receipt mismatch')
            prediction=rt.read_csv(directory/'predictions.csv')
            if any(int(r['seed'])!=seed or int(r['fold_id'])!=fold or r['model'] not in ARMS for r in prediction):raise ValueError('prediction identity')
            raw+=prediction
            info=rt.read_json(directory/'fits.json')
            for arm in ARMS:
                fits.append(dict(seed=seed,fold=fold,model=arm,**{k:v for k,v in info[arm].items() if k!='history'}))
    raw += [r for r in rt.read_csv(root/'results/formula_and_modalities/oof_densities.csv') if r['model'] in ('FROZEN','SINGLE')]
    groups=defaultdict(list)
    for r in raw:
        r['seed']=int(r['seed']);r['fold_id']=int(r['fold_id'])
        groups[r['model'],r['view'],r['measurement_id']].append(r)
    if len(groups)!=4*4*205:raise ValueError('OOF report coverage')
    for key,rows in groups.items():
        if sorted(r['seed'] for r in rows)!=[17,29,41] or len({r['fold_id'] for r in rows})!=1:raise ValueError('OOF seed/fold coverage')
    scores,index=_score_rows(raw,_read_targets(root),[17,29,41]);metrics=_metrics(scores,[17,29,41])
    lookup={(r['model'],r['view'],r['seed_mixture'],r['subset']):r for r in metrics};contrasts=[]
    for name,left,right in [('pair_training_increment','PHASE_NLL','PHASE_PAIR'),
                            ('phase_nll_vs_frozen','FROZEN','PHASE_NLL'),
                            ('phase_pair_vs_frozen','FROZEN','PHASE_PAIR'),
                            ('phase_pair_vs_single','SINGLE','PHASE_PAIR')]:
        for view in rt.VIEWS:
            for mix in ('equal_density_mixture','17','29','41'):
                for subset in ('all','structure','missing_structure'):
                    a,b=lookup[left,view,mix,subset],lookup[right,view,mix,subset]
                    for metric in ('nll','crps_ev','median_mae_ev'):
                        contrasts.append(dict(contrast=name,view=view,seed_mixture=mix,subset=subset,metric=metric,
                            improvement=a[metric]-b[metric],reports=a['reports']))
    for name,rows in [('metrics.csv',metrics),('report_scores.csv',scores),('oof_densities.csv',raw),
                      ('fit_summary.csv',fits),('contrasts.csv',contrasts)]:rt.write_csv(out/name,rows)
    pair_groups=[];pair_rows=[];pair_summary=[]
    physical=root/'results/physical_cases'
    for name,target in [('cif_group_response.csv',pair_groups),('same_formula_cif_pairs.csv',pair_rows),
                        ('same_formula_pair_summary.csv',pair_summary)]:
        target.extend(rt.read_csv(physical/name))
    for arm in ARMS:
        destination=out/arm
        cif_response(root,out,physical,destination,arm)
        for name,target in [('cif_group_response.csv',pair_groups),('same_formula_cif_pairs.csv',pair_rows),
                            ('same_formula_pair_summary.csv',pair_summary)]:
            target.extend(r for r in rt.read_csv(destination/name) if r['model']==arm)
    for name,rows in [('cif_group_response.csv',pair_groups),('same_formula_cif_pairs.csv',pair_rows),
                      ('same_formula_pair_summary.csv',pair_summary)]:rt.write_csv(out/name,rows)
    rt.write_json(out/'protocol.json',dict(training=protocol,run_binding=binding,
        scoring='existing analytic lognormal-mixture NLL/CRPS; current hierarchical weights and same 18 formula / 22 CIF pair inventory',
        code_binding={p.name:rt.sha(p) for p in (Path(__file__),root/'runtime/compare_cif_head_response.py',
                                               root/'runtime/analyze_frozen.py',root/'runtime/density_score.py')},
        output_binding={p.name:rt.sha(p) for p in out.glob('*.csv')}))
    print({'new_fits':len(fits),'new_oof_rows':sum(r['model'] in ARMS for r in raw),'metrics':len(metrics)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.root,a.work_dir,a.output_dir)
