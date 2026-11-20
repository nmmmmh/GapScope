"""Score all grouped-study OOF predictions and actual-CIF phase pairs."""
import argparse,itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from density_score import score
from training_data import hierarchical_weights

METRICS=('nll','crps_ev','median_mae_ev','coverage_80','width_80_ev')
def run(root,work):
    protocol=rt.read_json(work/'training_protocol.json');metadata=rt.read_json(work/'inputs/metadata.json')
    features=rt.torch.load(work/'inputs/features.pt',map_location='cpu',weights_only=True)
    sg_supported=[bool(x) for x in features['sg_mask']];structure_supported=[bool(x) for x in features['structure_mask']]
    records=[dict(m,**json.loads(line)) for m,line in zip(metadata,(work/'inputs/records.jsonl').read_text(encoding='utf-8').splitlines())]
    raw=[];fit_rows=[];partitions=[]
    for seed in protocol['seeds']:
        for fold in range(protocol['folds']):
            path=work/'fits'/f'seed_{seed}_fold_{fold+1:02d}';complete=rt.read_json(path/'complete.json')
            for name,k in (('predictions.csv','predictions_sha256'),('models.pt','models_sha256'),('fits.json','fits_sha256')):
                if rt.sha(path/name)!=complete[k]:raise ValueError('fit output changed')
            raw+=rt.read_csv(path/'predictions.csv');partitions.append(rt.read_json(path/'partition.json'))
            fits=rt.read_json(path/'fits.json')
            for arm in protocol['arms']:
                f=fits[arm]
                if f['initial_state_hash']!=complete['prior_hash']:raise ValueError('unmatched parent')
                fit_rows.append(dict(seed=seed,fold=fold,model=arm,**{k:f[k] for k in ('initial_validation','selected_validation','selected_epoch','epochs_run','trainable_parameters')}))
    grouped=defaultdict(list)
    for r in raw:grouped[r['model'],r['view'],r['measurement_id']].append(r)
    weights=hierarchical_weights(records,list(range(len(records))),'cpu').tolist();mass={r['measurement_id']:w for r,w in zip(records,weights)}
    scores=[]
    for arm,views in protocol['evaluated_views'].items():
        for view in views:
            for mixture in ['equal_density_mixture',*map(str,protocol['seeds'])]:
                for r in records:
                    rows=grouped[arm,view,r['measurement_id']]
                    if len(rows)!=len(protocol['seeds']) or len({row['seed'] for row in rows})!=len(protocol['seeds']):raise ValueError('OOF coverage')
                    selected=rows if mixture=='equal_density_mixture' else [row for row in rows if row['seed']==mixture]
                    parts=tuple(np.concatenate([np.array(json.loads(row[k]),dtype=float) for row in selected]) for k in ('component_weights','log_mus','sigmas'))
                    parts=(parts[0]/len(selected),parts[1],parts[2]);mean=float(parts[0]@np.exp(parts[1]+.5*parts[2]**2))
                    scores.append(dict(model=arm,view=view,seed_mixture=mixture,measurement_id=r['measurement_id'],formula=r['formula'],
                        composition_key=r['composition_key'],weight=mass[r['measurement_id']],mean_ev=mean,target_ev=r['gap_ev'],**score(r['gap_ev'],parts)))
    metrics=[]
    for subset in ('all','structure','natural_sg_only','natural_neither','new_pairs'):
        ids=[i for i,r in enumerate(records) if subset=='all' or subset=='structure' and structure_supported[i]
            or subset=='natural_sg_only' and sg_supported[i] and not structure_supported[i]
            or subset=='natural_neither' and not sg_supported[i] and not structure_supported[i]
            or subset=='new_pairs' and r.get('development_stage')=='grouped_adaptation']
        if not ids:continue
        w=hierarchical_weights(records,ids,'cpu').tolist();wm={records[i]['measurement_id']:v for i,v in zip(ids,w)}
        for arm,views in protocol['evaluated_views'].items():
            for view in views:
                for mixture in ['equal_density_mixture',*map(str,protocol['seeds'])]:
                    rs=[r for r in scores if r['model']==arm and r['view']==view and r['seed_mixture']==mixture and r['measurement_id'] in wm]
                    metrics.append(dict(subset=subset,model=arm,view=view,seed_mixture=mixture,reports=len(rs),formulas=len({r['composition_key'] for r in rs}),
                        **{k:sum(wm[r['measurement_id']]*r[k] for r in rs) for k in METRICS}))
    pairs=[];cifs=defaultdict(list)
    for i,r in enumerate(records):
        if structure_supported[i]:
            digest=metadata[i].get('actual_cif_sha256')
            if not digest:raise ValueError('encoded CIF identity missing')
            cifs[r['composition_key'],digest].append(r)
    by_formula=defaultdict(list)
    for (key,digest),rs in cifs.items():by_formula[key].append((digest,rs))
    paired_formulas=sum(len(gs)>1 for gs in by_formula.values())
    for mixture in ['equal_density_mixture',*map(str,protocol['seeds'])]:
        si={(r['view'],r['measurement_id']):r for r in scores if r['model']=='FROZEN_MULTIVIEW' and r['seed_mixture']==mixture}
        for key,gs in by_formula.items():
            combinations=list(itertools.combinations(sorted(gs),2))
            for (ha,aa),(hb,bb) in combinations:
                wa=np.array([mass[r['measurement_id']] for r in aa]);wa/=wa.sum();wb=np.array([mass[r['measurement_id']] for r in bb]);wb/=wb.sum()
                observed=float(wb@np.array([r['gap_ev'] for r in bb])-wa@np.array([r['gap_ev'] for r in aa]))
                p=dict(composition_key=key,formula=aa[0]['formula'],cif_a=ha,cif_b=hb,seed_mixture=mixture,observed_delta_ev=observed,
                    zero_delta_error_ev=abs(observed),weight=1/paired_formulas/len(combinations))
                for view in rt.VIEWS:
                    delta=float(wb@np.array([si[view,r['measurement_id']]['mean_ev'] for r in bb])-wa@np.array([si[view,r['measurement_id']]['mean_ev'] for r in aa]))
                    p[view+'_delta_ev']=delta;p[view+'_delta_error_ev']=abs(delta-observed)
                pairs.append(p)
    out=work/'analysis';out.mkdir(exist_ok=True)
    for name,rs in (('predictions.csv',raw),('report_scores.csv',scores),('metrics.csv',metrics),('fit_summary.csv',fit_rows),('pair_response.csv',pairs)):rt.write_csv(out/name,rs)
    rt.write_json(out/'summary.json',dict(new_aggregate_fits=9,new_report_fits=27,reports=len(records),formulas=len({r['composition_key'] for r in records}),
        pairs=sum(len(gs)*(len(gs)-1)//2 for gs in by_formula.values()),paired_formulas=paired_formulas,partitions=partitions[:3],
        source_sha256=rt.sha(Path(__file__)),training_protocol_sha256=rt.sha(work/'training_protocol.json'),
        scope='source/element-set grouped development OOF; not a new independent confirmation; original confirmation unused',
        output_binding={p.name:rt.sha(p) for p in out.iterdir() if p.is_file() and p.name!='summary.json'}))
    print(dict(reports=len(records),raw_predictions=len(raw),scores=len(scores),metrics=len(metrics)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
