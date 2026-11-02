"""Frozen T48 F with a TRAIN-only low-rank, zero-start density increment head."""
import argparse
import json
import math
import os

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr



import runtime as rt
torch = rt.torch

RECIPE = dict(rank_max=4, outputs=5, bias=False, epochs_max=300, patience=30,
              learning_rate=1e-3, l2_coefficient=1e-3, grid_points=1025, grid_sigma=9,
              validation_min_delta=0, selection='original innerVALID weighted true-report NLL including epoch0',
              train_objective='formula-weighted integrated squared CDF increment residual / teacher increment energy + fixed L2',
              teacher_target='same frozen J TRAIN FULL marginal CDF minus SG marginal CDF',
              baseline='frozen T48 F; all encoders frozen', pca='unique TRAIN formulas, equal formula PCA',
              seeds=[17,29,41], folds=[f'fold_{x:02d}' for x in range(1,6)],
              outer_targets_read_in_training=False, new_six_reports_used=False)

def restore_f(m, cp, device):
    state = cp['state_dict']
    prefix = 'branches.0.source.report_head.'
    fake = dict(schema='t46_checkpoint_v1', node='t42_M', state_dict=state,
                beta=state[prefix+'raw_beta'], raw_feature_mean=float(state[prefix+'raw_feature_mean']),
                raw_feature_std=float(state[prefix+'raw_feature_std']))
    model = rt.restore(m, fake, device)
    model.mode = 'F'
    return model.requires_grad_(False).eval()

def formula_inputs(features, scalers, device):
    comp = (features['composition'].to(device=device, dtype=torch.float32) -
            scalers['composition_mean'].to(device)) / scalers['composition_std'].to(device)
    return torch.cat((comp, comp.new_zeros(len(comp),452)), -1)

def baseline(model, inputs):
    with torch.no_grad():
        weights, mus, sigmas = model.components(inputs)
        gate, _, _ = model.density_parameters(inputs)
        hidden = model.interaction.encode(inputs).double()
        raw_scale = torch.logit((sigmas-.05)/2.45)
    return dict(weights=weights, gate=gate, mus=mus, sigmas=sigmas,
                raw_scale=raw_scale, hidden=hidden)

def corrected(base, delta):
    gate = base['gate'] + delta[:,0]
    first = torch.sigmoid(gate)
    weights = torch.stack((first,1-first),-1)
    mus = base['mus'] + delta[:,1:3]
    # Subtract the same inversion/reconstruction roundoff so epoch0 is exactly F.
    scale0 = .05+2.45*torch.sigmoid(base['raw_scale'])
    sigmas = base['sigmas'] + (.05+2.45*torch.sigmoid(base['raw_scale']+delta[:,3:5])-scale0)
    return weights, mus, sigmas

def cdf(parts, grid):
    w,mu,sd=parts
    z=(grid[:,:,None]-mu[:,None,:])/sd[:,None,:]
    return (w[:,None,:]*(.5*(1+torch.erf(z/math.sqrt(2))))).sum(-1)

def nll(parts, y):
    w,mu,sd=parts
    log_y=y.double().log()[:,None]
    log_density=w.log()-log_y-sd.log()-.5*math.log(2*math.pi)-.5*((log_y-mu)/sd).square()
    return -torch.logsumexp(log_density,-1)

def teacher_targets(td, base, train_ids, records, device):
    sg=td[td.arm=='SG_KD'].set_index('measurement_id')
    full=td[td.arm=='FULL_KD'].set_index('measurement_id')
    assert set(sg.index)==set(full.index)==set(train_ids)
    keys=[records[i]['composition_key'] for i in range(len(records))]
    index={r['measurement_id']:i for i,r in enumerate(records)}
    groups=[]
    zero=True
    for key,rows in sg.groupby('composition_key',sort=True):
        ids=list(rows.index)
        mass=float(rows.train_weight.sum())
        parts=[]
        for table in (sg,full):
            ws,mus,sds=[],[],[]
            for rid in ids:
                r=table.loc[rid]
                ws.extend(np.asarray(json.loads(r.component_weights))*float(sg.loc[rid,'train_weight'])/mass)
                mus.extend(json.loads(r.log_mus)); sds.extend(json.loads(r.sigmas))
            parts.append((np.asarray(ws),np.asarray(mus),np.asarray(sds)))
        if not all(np.array_equal(a,b) for a,b in zip(*parts)):
            zero=False
        i=index[ids[0]]
        for rid in ids[1:]:
            j=index[rid]
            assert all(torch.equal(base[x][i],base[x][j]) for x in ('weights','mus','sigmas','hidden'))
        f=tuple(base[x][i].cpu().numpy() for x in ('weights','mus','sigmas'))
        lo=min(float((p[1]-9*p[2]).min()) for p in [*parts,f])
        hi=max(float((p[1]+9*p[2]).max()) for p in [*parts,f])
        grid=np.linspace(lo,hi,RECIPE['grid_points'])
        cf=[np.sum(p[0][None,:]*ndtr((grid[:,None]-p[1][None,:])/p[2][None,:]),-1) for p in parts]
        target=cf[1]-cf[0]
        groups.append(dict(composition_key=key,formula=rows.formula.iloc[0],mass=mass,report_index=i,
                           grid=grid,target=target,teacher_delta_energy=float(np.trapz(target**2,grid)),
                           teacher_log_w1=float(np.trapz(abs(target),grid)),
                           teacher_mean_increment_ev=float(sum(parts[1][0]*np.exp(parts[1][1]+.5*parts[1][2]**2))-
                                                           sum(parts[0][0]*np.exp(parts[0][1]+.5*parts[0][2]**2)))))
    ti=torch.tensor([x['report_index'] for x in groups],device=device)
    grid=torch.tensor(np.array([x['grid'] for x in groups]),dtype=torch.float64,device=device)
    target=torch.tensor(np.array([x['target'] for x in groups]),dtype=torch.float64,device=device)
    mass=torch.tensor([x['mass'] for x in groups],dtype=torch.float64,device=device)
    mass=mass/mass.sum()
    return groups,ti,grid,target,mass,zero

def subset_base(base,index):
    return {k:v[index] for k,v in base.items()}

def run_pair(m, seed, fold, device, input_dir, output_dir):
    name=f'seed_{seed}_{fold}'
    path=output_dir/name
    if (path/'summary.json').exists():
        print(f'{name}: already complete; no repeat fit',flush=True)
        return
    path.mkdir(parents=True,exist_ok=True)
    inner=torch.load(input_dir/'bundles'/f'{fold}_inner.pt',map_location='cpu',weights_only=True)
    outer=torch.load(input_dir/'bundles'/f'{fold}_outer_features.pt',map_location='cpu',weights_only=True)
    assert 'targets' not in outer
    cp=torch.load(input_dir/'baselines'/name/'F.pt',map_location='cpu',weights_only=True)
    assert cp['report_train_ids']==inner['train_ids'] and cp['report_validation_ids']==inner['valid_ids']
    assert cp['outer_ids']==outer['report_ids']
    model=restore_f(m,cp,device)
    x=formula_inputs(inner['report'],cp['scalers'],device)
    ox=formula_inputs(outer['report'],cp['scalers'],device)
    b=baseline(model,x); ob=baseline(model,ox)
    td=pd.read_csv(input_dir/'teachers'/name/'teacher_train_densities.csv')
    groups,ti,grid,target,mass,zero=teacher_targets(td,b,inner['train_ids'],inner['report_records'],device)
    tb=subset_base(b,ti)
    hidden=tb['hidden']; center=hidden.mean(0)
    u,s,vh=torch.linalg.svd(hidden-center,full_matrices=False)
    rank=min(RECIPE['rank_max'],int((s>torch.finfo(s.dtype).eps*s[0]*max(hidden.shape)).sum().item()))
    basis=vh[:rank].T
    scale=s[:rank]/math.sqrt(len(hidden)-1)
    features=(b['hidden']-center)@basis/scale
    outer_features=(ob['hidden']-center)@basis/scale
    head=torch.nn.Linear(rank,5,bias=False,dtype=torch.float64,device=device)
    torch.nn.init.zeros_(head.weight)
    initial_parts=corrected(b,head(features))
    assert all(torch.equal(a,bb) for a,bb in zip(initial_parts,(b['weights'],b['mus'],b['sigmas'])))
    vi=torch.tensor([inner['report_ids'].index(rid) for rid in inner['valid_ids']],device=device)
    vw=torch.tensor([inner['valid_weights'][rid] for rid in inner['valid_ids']],dtype=torch.float64,device=device)
    vw=vw/vw.sum(); y=inner['targets'].to(device)
    def valid():
        with torch.no_grad():
            return float((vw*nll(corrected(subset_base(b,vi),head(features[vi])),y[vi])).sum())
    base_cdf=cdf((tb['weights'],tb['mus'],tb['sigmas']),grid)
    energy=(mass*torch.trapezoid(target.square(),grid,dim=-1)).sum()
    def objective():
        increment=cdf(corrected(tb,head(features[ti])),grid)-base_cdf
        residual=(mass*torch.trapezoid((increment-target).square(),grid,dim=-1)).sum()
        normalized=residual/energy if not zero else residual
        reg=RECIPE['l2_coefficient']*head.weight.square().sum()
        return normalized+reg, normalized, increment
    best=valid(); initial_validation=best; best_epoch=0
    best_state={k:v.detach().clone() for k,v in head.state_dict().items()}
    history=[dict(epoch=0,validation_nll=best,normalized_cdf_residual=0. if zero else 1.,
                  objective=0. if zero else 1.,head_l2=0.,gradient_l2=0.,selected=True)]
    epochs=0; wait=0
    if not zero:
        optimizer=torch.optim.Adam(head.parameters(),lr=RECIPE['learning_rate'])
        for epoch in range(1,RECIPE['epochs_max']+1):
            optimizer.zero_grad(set_to_none=True)
            loss,residual,_=objective(); loss.backward()
            grad=float(head.weight.grad.norm())
            optimizer.step()
            current=valid()
            with torch.no_grad():
                loss,residual,_=objective()
            selected=current<best
            if selected:
                best=current; best_epoch=epoch; wait=0
                best_state={k:v.detach().clone() for k,v in head.state_dict().items()}
            else:
                wait+=1
            history.append(dict(epoch=epoch,validation_nll=current,normalized_cdf_residual=float(residual),
                                objective=float(loss),head_l2=float(head.weight.detach().norm()),
                                gradient_l2=grad,selected=selected))
            epochs=epoch
            if wait>=RECIPE['patience']:
                break
    head.load_state_dict(best_state)
    with torch.no_grad():
        _,selected_residual,increment=objective()
        out_parts=corrected(ob,head(outer_features))
        train_parts=corrected(tb,head(features[ti]))
        outer_delta=head(outer_features)
    predictions=[]
    for label,parts in [('T48_F',(ob['weights'],ob['mus'],ob['sigmas'])),('ANCHORED_INCREMENT',out_parts)]:
        for i,record in enumerate(outer['report_records']):
            predictions.append(dict(model=label,view='formula_only',seed=seed,fold_id=fold,**record,
                                    sg_supported=int(outer['report']['sg_mask'][i,0]),
                                    structure_supported=int(outer['report']['structure_mask'][i,0]),
                                    component_weights=json.dumps(parts[0][i].tolist()),
                                    log_mus=json.dumps(parts[1][i].tolist()),sigmas=json.dumps(parts[2][i].tolist())))
    diagnostics=[]
    for j,g in enumerate(groups):
        tp=tuple(p[j].cpu().numpy() for p in train_parts)
        bp=tuple(tb[k][j].cpu().numpy() for k in ('weights','mus','sigmas'))
        newmean=float(np.sum(tp[0]*np.exp(tp[1]+.5*tp[2]**2)))
        basemean=float(np.sum(bp[0]*np.exp(bp[1]+.5*bp[2]**2)))
        diagnostics.append({k:v for k,v in g.items() if k not in ('grid','target','report_index')} |
                           dict(seed=seed,fold_id=fold,learned_mean_increment_ev=newmean-basemean,
                                selected_increment_log_w1=float(torch.trapezoid(increment[j].abs(),grid[j])),
                                selected_residual_energy=float(torch.trapezoid((increment[j]-target[j]).square(),grid[j]))))
    rt.write_csv(path/'history.csv',history)
    rt.write_csv(path/'train_formula_diagnostics.csv',diagnostics)
    rt.write_csv(path/'outer_predictions.csv',predictions)
    baseline_path=input_dir/'baselines'/name/'F.pt'
    torch.save(dict(schema='anchored_increment_v1',baseline_relative=os.path.relpath(baseline_path, start=path).replace(os.sep, '/'),
                    head_state_dict={k:v.cpu() for k,v in head.state_dict().items()},pca_center=center.cpu(),
                    pca_basis=basis.cpu(),pca_scale=scale.cpu(),seed=seed,fold_id=fold,
                    best_epoch=best_epoch,train_ids=inner['train_ids'],valid_ids=inner['valid_ids'],recipe=RECIPE),path/'increment.pt')
    summary=dict(status='complete',seed=seed,fold_id=fold,zero_teacher=zero,new_fits=int(not zero),
                 train_reports=len(inner['train_ids']),valid_reports=len(inner['valid_ids']),train_formulas=len(groups),
                 nonzero_teacher_formulas=sum(g['teacher_delta_energy']>0 for g in groups),
                 pca_rank=rank,trainable_parameters=5*rank,best_epoch=best_epoch,epochs_run=epochs,
                 teacher_increment_energy=float(energy),initial_validation_nll=initial_validation,
                 selected_validation_nll=best,selected_normalized_cdf_residual=float(selected_residual),
                 final_normalized_cdf_residual=history[-1]['normalized_cdf_residual'],
                 minimum_normalized_cdf_residual=min(h['normalized_cdf_residual'] for h in history),
                 exact_epoch0_equals_F=True,baseline_encoder_parameters_updated=0,
                 zero_teacher_exact_output_F=all(torch.equal(a,bb) for a,bb in zip(out_parts,(ob['weights'],ob['mus'],ob['sigmas']))) if zero else None,
                 outer_max_raw_correction=float(outer_delta.abs().max()),outer_predictions=len(predictions),
                 outer_targets_read=False,sealed_targets_read=False,device=str(device))
    rt.write_json(path/'summary.json',summary)
    print(json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,required=True,help='Directory containing bundles/, baselines/, and teachers/.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=RECIPE['seeds'],required=True)
    parser.add_argument('--fold',choices=RECIPE['folds'],required=True)
    args=parser.parse_args()
    input_dir=args.input_dir.resolve()
    output_dir=args.output_dir.resolve()
    for child in ('bundles','baselines','teachers'):
        if not (input_dir/child).is_dir():
            parser.error(f'input directory is missing {child}/')
    output_dir.mkdir(parents=True,exist_ok=True)
    device=rt.configure(); m=rt.modules()
    rt.write_json(output_dir/'PROTOCOL.json',RECIPE|dict(device=str(device),gpu=torch.cuda.get_device_name(0)))
    run_pair(m,args.seed,args.fold,device,input_dir,output_dir)

if __name__=='__main__':
    main()


