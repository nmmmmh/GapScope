"""Paired OOF score and actual density modes for one- and two-component heads."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import runtime as rt
from analyze_frozen import _read_targets,_score_rows,_metrics,_parts

def modes(parts):
    w,mu,sd=parts
    component_modes=mu-sd*sd
    lower,upper=float(component_modes.min()),float(component_modes.max())
    if upper-lower<1e-12:return [math.exp(lower)]
    def derivative(logx):
        xs=np.atleast_1d(logx)[:,None]
        logp=np.log(w)-xs-np.log(sd)-.5*((xs-mu)/sd)**2
        mixture=np.exp(logp-logsumexp(logp,axis=-1,keepdims=True))
        value=(mixture*(-1-(xs-mu)/sd**2)).sum(-1)
        return float(value[0]) if np.ndim(logx)==0 else value
    grid=np.linspace(lower-1e-6,upper+1e-6,4097)
    values=derivative(grid); crossings=np.where((values[:-1]>0)&(values[1:]<=0))[0]
    return [math.exp(brentq(derivative,float(grid[i]),float(grid[i+1]))) for i in crossings]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(root):raise ValueError('output must be outside formal root')
    protocol=rt.read_json(a.work_dir/'protocol.json');seeds=protocol['seeds'];raw=[];fits=[];binding={}
    for seed in seeds:
        for fold in range(5):
            directory=a.work_dir/f'seed_{seed}_fold_{fold+1:02d}'
            receipt=rt.read_json(directory/'complete.json')
            if receipt['protocol_sha256']!=rt.sha(a.work_dir/'protocol.json'):raise ValueError('receipt protocol mismatch')
            for name in ('predictions.csv','fits.json','models.pt'):
                digest=rt.sha(directory/name);binding[f'{directory.name}/{name}']=digest
                if receipt[name.split('.')[0]+'_sha256']!=digest:raise ValueError('receipt mismatch')
            raw+=rt.read_csv(directory/'predictions.csv')
            fit=rt.read_json(directory/'fits.json')['SINGLE']
            fits.append(dict(seed=seed,fold=fold,model='SINGLE',**{k:v for k,v in fit.items() if k!='history'}))
    grouped=defaultdict(list)
    for r in raw:grouped[(r['model'],r['view'],r['measurement_id'])].append(r)
    if len(grouped)!=2*4*205:raise ValueError('OOF coverage mismatch')
    for (m,v,rid),rs in grouped.items():
        if m not in ('SINGLE','FROZEN') or v not in rt.VIEWS or sorted(int(r['seed']) for r in rs)!=sorted(seeds):
            raise ValueError('OOF identity mismatch')
        if len({r['fold_id'] for r in rs})!=1:raise ValueError('fold mismatch')
    for r in raw:r['seed']=int(r['seed']);r['fold_id']=int(r['fold_id'])
    records=_read_targets(root);scores,index=_score_rows(raw,records,seeds);metrics=_metrics(scores,seeds)
    lookup={(r['model'],r['view'],r['seed_mixture'],r['subset']):r for r in metrics}
    contrasts=[]
    for view in rt.VIEWS:
        for mixture in ['equal_density_mixture']+[str(seed) for seed in seeds]:
            for subset in ('all','structure','missing_structure'):
                left=lookup['SINGLE',view,mixture,subset];right=lookup['FROZEN',view,mixture,subset]
                for metric in ('nll','crps_ev','median_mae_ev'):
                    contrasts.append(dict(contrast='two_vs_one_component',view=view,seed_mixture=mixture,
                        subset=subset,metric=metric,improvement=left[metric]-right[metric],reports=left['reports']))
    shape=[]
    for (model,view,rid),rs in grouped.items():
        for mixture,chosen in [('equal_density_mixture',rs)]+[(str(r['seed']),[r]) for r in rs]:
            peaks=modes(_parts(chosen));r=chosen[0]
            shape.append(dict(model=model,view=view,seed_mixture=mixture,measurement_id=rid,
                composition_key=r['composition_key'],formula=r['formula'],
                density_modes=len(peaks),mode_locations_ev=json.dumps(peaks),
                weight=index[model,view,mixture,rid]['weight']))
    shapes=defaultdict(list)
    for r in shape:shapes[r['model'],r['view'],r['seed_mixture']].append(r)
    summary=[]
    for (model,view,mix),rs in shapes.items():
        weight=sum(r['weight'] for r in rs)
        summary.append(dict(model=model,view=view,seed_mixture=mix,reports=len(rs),
            report_weighted_multimode_fraction=sum(r['weight'] for r in rs if r['density_modes']>=2)/weight,
            formulas_with_any_multimode_prediction=len({r['composition_key'] for r in rs if r['density_modes']>=2}),
            max_density_modes=max(r['density_modes'] for r in rs)))
    for name,rows in [('metrics.csv',metrics),('contrasts.csv',contrasts),('report_scores.csv',scores),
        ('oof_densities.csv',raw),('fit_summary.csv',fits),('density_modes.csv',shape),('mode_summary.csv',summary)]:rt.write_csv(out/name,rows)
    rt.write_json(out/'protocol.json',dict(training=protocol,input_binding=binding,
        scoring='same analytic lognormal-mixture NLL/CRPS on raw positive eV targets, global hierarchical weights renormalized per subset',
        shape='maxima of actual Eg-density, roots of d log(pdf(Eg))/d log(Eg); search between component mode extrema, 4097-point bracketing',
        interpretation='individual SINGLE model is unimodal; three-SINGLE density ensemble can be multimodal; two components do not imply two physical phases',
        code_binding={p.name:rt.sha(p) for p in (Path(__file__),root/'runtime/density_score.py',root/'runtime/analyze_frozen.py')},
        output_binding={p.name:rt.sha(p) for p in out.glob('*.csv')}))
    print(json.dumps(dict(reports=205,predictions=len(raw),new_fits=len(fits),metric_rows=len(metrics))))

if __name__=='__main__':main()
