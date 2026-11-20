"""Evaluate a frozen, primary-source-confirmed cohort using saved weights only."""
import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from inference import _input_features,_restore_model
from distribution_head_compare import restore_single
from training_data import hierarchical_weights
from density_score import score

torch=rt.torch
METRICS=('nll','crps_ev','median_mae_ev','coverage_50','width_50_ev','coverage_80','width_80_ev','coverage_90','width_90_ev')

def run(root,work,device):
    frozen=rt.read_json(work/'fixed_protocol.json');cohort=rt.read_json(work/'cohort.json');rows=cohort['records']
    for name,digest in frozen['input_binding'].items():
        if rt.sha(root/name)!=digest:raise ValueError('frozen input changed: '+name)
    freeze=rt.read_json(work/'cohort_freeze.json')
    if rt.sha(work/'cohort.json')!=freeze['cohort_sha256']:raise ValueError('cohort changed after source freeze')
    if rt.sha(work/'fixed_protocol.json')!=freeze['fixed_protocol_sha256']:raise ValueError('evaluation design changed')
    if (work/'analysis/protocol.json').exists():raise ValueError('confirmation already evaluated; no silent new assay')
    features=rt.read_json(work/'encoded_features.json')
    if rt.sha(work/'encoded_features.json')!=rt.read_json(work/'encoding_protocol.json')['feature_sha256']:raise ValueError('encoded features changed')
    if [r['measurement_id'] for r in rows]!=[r['measurement_id'] for r in features]:raise ValueError('feature order')
    modules=rt.modules();torch.set_num_threads(2);device=torch.device(device)
    raw_inputs=[_input_features(r) for r in features]
    inputs={k:torch.cat([r[k] for r in raw_inputs]) for k in raw_inputs[0]};raw_predictions=[]
    predictions=defaultdict(list)
    for name,digest in frozen['checkpoint_binding'].items():
        path=root/name
        if rt.sha(path)!=digest:raise ValueError('checkpoint changed')
        cp=torch.load(path,map_location='cpu',weights_only=True)
        seed=int(path.parent.name.split('_')[1]);fold=int(path.parent.name.split('_')[3])-1
        if cp['partition']['fold']!=fold:raise ValueError('checkpoint fold mismatch')
        scalers=cp['scalers'];views=modules['t10'].four_views(inputs,
            *(scalers[k].to(device=device,dtype=torch.float32) for k in ('composition_mean','composition_std','structure_mean','structure_std')),device)
        for arm,names in frozen['evaluated_views'].items():
            state=cp['models'][arm]
            model=restore_single(modules,state,device) if arm=='SINGLE' else _restore_model(modules,state,'F' if arm=='F' else 'M',device)
            for view in names:
                vi=rt.VIEWS.index(view)
                with torch.inference_mode():parts=tuple(t.cpu().numpy() for t in model.components(views[vi]))
                for i,r in enumerate(rows):
                    p=tuple(t[i] for t in parts);predictions[arm,view,r['measurement_id']].append((str(seed),p))
                    raw_predictions.append(dict(model=arm,view=view,seed=seed,fold=fold,measurement_id=r['measurement_id'],
                        component_weights=p[0].tolist(),log_mus=p[1].tolist(),sigmas=p[2].tolist()))
        print(f'evaluated seed={seed} fold={fold}',flush=True)
    scored=[]
    for arm,names in frozen['evaluated_views'].items():
        for view in names:
            for mixture in ('equal_density_mixture','17','29','41'):
                for i,r in enumerate(rows):
                    candidates=predictions[arm,view,r['measurement_id']]
                    chunks=[p for seed,p in candidates if mixture=='equal_density_mixture' or seed==mixture]
                    parts=(np.concatenate([p[0] for p in chunks])/len(chunks),
                           np.concatenate([p[1] for p in chunks]),np.concatenate([p[2] for p in chunks]))
                    mean=float(parts[0]@np.exp(parts[1]+.5*parts[2]**2))
                    scored.append(dict(model=arm,view=view,seed_mixture=mixture,measurement_id=r['measurement_id'],
                        cohort=r['cohort'],formula=r['formula'],composition_key=r['composition_key'],target_ev=r['gap_ev'],
                        mean_ev=mean,sg_supported=int(features[i]['space_group'] is not None),
                        structure_supported=int(features[i]['structure'] is not None),aggregate_overlap=r['aggregate_overlap'],
                        **score(float(r['gap_ev']),parts)))
    metrics=[];weights_by_group={}
    for label in ('independent','same_sg_supplement'):
        base=[i for i,r in enumerate(rows) if r['cohort']==label]
        if not base:continue
        for subset in ('all','structure','missing_structure','aggregate_overlap','aggregate_nonoverlap'):
            selected=[i for i in base if subset=='all' or subset=='structure' and features[i]['structure'] is not None
                or subset=='missing_structure' and features[i]['structure'] is None
                or subset=='aggregate_overlap' and rows[i]['aggregate_overlap']
                or subset=='aggregate_nonoverlap' and not rows[i]['aggregate_overlap']]
            if not selected:continue
            weights=hierarchical_weights(rows,selected,'cpu').tolist();ids={rows[i]['measurement_id']:w for i,w in zip(selected,weights)}
            weights_by_group[label,subset]=ids
            for arm,names in frozen['evaluated_views'].items():
                for view in names:
                    for mixture in ('equal_density_mixture','17','29','41'):
                        rs=[r for r in scored if r['model']==arm and r['view']==view and r['seed_mixture']==mixture and r['measurement_id'] in ids]
                        metrics.append(dict(cohort=label,subset=subset,model=arm,view=view,seed_mixture=mixture,reports=len(rs),
                            formulas=len({r['composition_key'] for r in rs}),**{k:sum(ids[r['measurement_id']]*r[k] for r in rs) for k in METRICS}))
    for r in scored:r['weight']=weights_by_group[r['cohort'],'all'][r['measurement_id']]
    pair_rows=[]
    for arm,names in frozen['evaluated_views'].items():
        for mixture in ('equal_density_mixture','17','29','41'):
            groups=defaultdict(list)
            for i,r in enumerate(rows):
                if features[i].get('structure') is not None:groups[r['cohort'],r['composition_key'],features[i]['cif_sha256']].append(r)
            formulas=defaultdict(list)
            for (cohort_label,key,digest),rs in groups.items():formulas[cohort_label,key].append((digest,rs))
            counts={label:sum(len(gs)>1 for (lab,key),gs in formulas.items() if lab==label) for label in ('independent','same_sg_supplement')}
            score_index={(r['view'],r['measurement_id']):r for r in scored if r['model']==arm and r['seed_mixture']==mixture}
            for (label,key),gs in formulas.items():
                pairs=list(itertools.combinations(sorted(gs),2))
                for (ha,aa),(hb,bb) in pairs:
                    masses=weights_by_group[label,'all'];wa=np.array([masses[r['measurement_id']] for r in aa]);wa/=wa.sum();wb=np.array([masses[r['measurement_id']] for r in bb]);wb/=wb.sum()
                    observed=float(wb@np.array([r['gap_ev'] for r in bb])-wa@np.array([r['gap_ev'] for r in aa]))
                    result=dict(cohort=label,model=arm,seed_mixture=mixture,composition_key=key,formula=aa[0]['formula'],
                        cif_a=ha,cif_b=hb,sg_a=aa[0].get('space_group'),sg_b=bb[0].get('space_group'),
                        same_sg=aa[0].get('space_group') is not None and aa[0].get('space_group')==bb[0].get('space_group'),
                        observed_delta_ev=observed,weight=1/counts[label]/len(pairs))
                    for view in names:
                        delta=float(wb@np.array([score_index[view,r['measurement_id']]['mean_ev'] for r in bb])-wa@np.array([score_index[view,r['measurement_id']]['mean_ev'] for r in aa]))
                        result['pred_delta_'+view+'_ev']=delta;result['delta_error_'+view+'_ev']=abs(delta-observed)
                    pair_rows.append(result)
    out=work/'analysis';out.mkdir(exist_ok=True)
    for name,values in [('densities.csv',raw_predictions),('report_scores.csv',scored),('metrics.csv',metrics),('cif_pair_response.csv',pair_rows)]:rt.write_csv(out/name,values)
    rt.write_json(out/'protocol.json',dict(new_fits=0,cohort_sha256=rt.sha(work/'cohort.json'),fixed_protocol_sha256=rt.sha(work/'fixed_protocol.json'),
        source_sha256=rt.sha(Path(__file__)),feature_sha256=rt.sha(work/'encoded_features.json'),checkpoint_binding=frozen['checkpoint_binding'],
        ensemble=frozen['ensemble'],primary_view='formula_only',device=str(device),report_rows=len(rows),
        output_binding={p.name:rt.sha(p) for p in out.iterdir() if p.is_file() and p.name!='protocol.json'}))
    print(f'completed fixed-model confirmation: reports={len(rows)}, no fits')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.root,a.work,a.device)
