"""TRAIN-only local CDF projection: centered rank-four head versus an intercept."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import runtime as rt
import train
import anchored_transfer as anchor
from inference import _restore_model
from training_data import build_fold

torch=rt.torch


def diagnose(root,seed,fold,device,modules):
    cp=torch.load(root/'models/checkpoints'/f'seed_{seed}_fold_{fold+1:02d}'/'models.pt',
                  map_location='cpu',weights_only=True)
    model=_restore_model(modules,cp['models']['FROZEN'],'M',device).requires_grad_(False)
    d=build_fold(root,fold,device); base=train.base_parts(model,d['views'][0]); rows=[]
    with torch.no_grad():
        for view,label in ((1,'SG_KD'),(3,'FULL_KD')):
            parts=model.components(d['views'][view,d['ti']])
            for j,i in enumerate(d['ti'].tolist()):
                r=d['records'][i]
                rows.append(dict(arm=label,measurement_id=r['measurement_id'],formula=r['formula'],
                    composition_key=r['composition_key'],train_weight=float(d['tw'][j]),
                    **{key:json.dumps(value[j].tolist()) for key,value in
                       zip(('component_weights','log_mus','sigmas'),parts)}))
    groups,ti,grid,target,mass,zero=anchor.teacher_targets(pd.DataFrame(rows),base,
        [d['records'][i]['measurement_id'] for i in d['ti'].tolist()],d['records'],device)
    b=anchor.subset_base(base,ti); hidden=b['hidden']; center=hidden.mean(0)
    _,s,vh=torch.linalg.svd(hidden-center,full_matrices=False)
    rank=min(4,int((s>torch.finfo(s.dtype).eps*s[0]*max(hidden.shape)).sum()))
    result=dict(seed=seed,fold=fold,formulas=len(groups),rank=rank,zero_teacher=zero)
    if zero or rank==0: return result
    features=(hidden-center)@vh[:rank].T/(s[:rank]/np.sqrt(len(hidden)-1))
    w,mu,sd=(b[k] for k in ('weights','mus','sigmas'))
    z=(grid[:,:,None]-mu[:,None,:])/sd[:,None,:]
    phi=torch.exp(-.5*z.square())/np.sqrt(2*np.pi)
    c=.5*(1+torch.erf(z/np.sqrt(2)))
    jac_gate=w[:,0,None]*w[:,1,None]*(c[:,:,0]-c[:,:,1])
    jac_mu=-w[:,None,:]*phi/sd[:,None,:]
    p=(sd-.05)/2.45
    jac_scale=jac_mu*z*(2.45*p*(1-p))[:,None,:]
    jac=torch.cat((jac_gate[:,:,None],jac_mu,jac_scale),-1)
    linear=(jac[:,:,:,None]*features[:,None,None,:]).reshape(len(groups),grid.shape[1],5*rank)
    endpoints=torch.ones_like(grid); endpoints[:,[0,-1]]=.5
    quadrature=mass[:,None]*(grid[:,-1]-grid[:,0])[:,None]/(grid.shape[1]-1)*endpoints
    sqrt_weight=quadrature.sqrt().reshape(-1,1)
    a=(linear.reshape(-1,5*rank)*sqrt_weight).cpu().numpy()
    y=(target.reshape(-1,1)*sqrt_weight).cpu().numpy()
    expanded=(torch.cat((linear,jac),-1).reshape(-1,5*rank+5)*sqrt_weight).cpu().numpy()
    energy=float(np.sum(y*y))
    residuals=[]
    for matrix in (a,expanded):
        beta=np.linalg.lstsq(matrix,y,rcond=1e-8)[0]
        residuals.append(float(np.sum((matrix@beta-y)**2)/energy))
    result.update(teacher_energy=energy,linearized_without_intercept_residual=residuals[0],
        linearized_with_intercept_residual=residuals[1],
        intercept_residual_reduction=residuals[0]-residuals[1],
        intercept_relative_reduction=(residuals[0]-residuals[1])/residuals[0] if residuals[0]>0 else 0.,
        interpretation='TRAIN-only unregularized first-order CDF approximation; not a fitted valid density or outer score')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    device=rt.configure(); modules=rt.modules()
    rows=[]
    for seed in (17,29,41):
        for fold in range(5):
            rows.append(diagnose(args.root,seed,fold,device,modules))
    rt.write_csv(args.output_dir/'capacity.csv',rows)
    rt.write_json(args.output_dir/'capacity_protocol.json',dict(
        scope='TRAIN formulas and TRAIN teacher increments only; no VALID or outer targets in projection',
        approximation='linearized CDF Jacobian around zero head, weighted by formula mass and log-grid trapezoidal quadrature',
        comparisons='5*rank weights versus those weights plus five intercepts; rcond1e-8',
        source_sha256=rt.sha(Path(__file__)),head_training='none'))
    print(json.dumps(dict(pairs=len(rows),mean_intercept_relative_reduction=float(np.mean([
        r.get('intercept_relative_reduction',0) for r in rows])))))


if __name__=='__main__': main()
