"""Eighteen matched report fits; reuse nine grouped priors without aggregate fitting."""
import argparse,copy
from pathlib import Path
import runtime as rt
import train
from training_data import hierarchical_weights
from frozen_fusion_compare import _state_hash
from decoupled_distribution import DecoupledDistribution,cdf,means

torch=rt.torch
ARMS=('DECOUPLED_NLL','DECOUPLED_PAIR')
RECIPE=dict(seeds=[17,29,41],folds=3,arms=list(ARMS),aggregate_fits=0,report_fits=18,
    evaluated_views={a:list(rt.VIEWS) for a in ARMS},head_lr=.001,pair_coefficient=.1,kd_coefficient=.1,
    max_epochs=300,patience=30,pair_scale_floor_ev=.1,grid_points=257,grid='uniform logEg, TRAIN log-target min minus2 to max plus2',
    selection='innerVALID formula-only true-report hierarchical NLL; epoch0 eligible; one whole two-head state',
    teacher='current conditional FULL density on supported TRAIN records; fixed initial formula density otherwise; original full-formula hierarchical marginal',
    kd='mean squared CDF on fixed uniform log grid, weighted by supported TRAIN formulas; stopgrad teacher',
    pair_loss='source-compatible TRAIN experimental expected-gap differences, weighted MSE / weighted observed delta squared',
    initialization='parent matched FROZEN_FORMULA and FROZEN_MULTIVIEW selected states; warm-start; same for both arms',
    parameter_grid=False,default_changed=False,confirmation_used=False,new_snS_used_for_training=False)

def context(root,parent,cp,modules,device):
    meta=rt.read_json(parent/'inputs/metadata.json')
    records=[dict(m,**rt.json.loads(s)) for m,s in zip(meta,(parent/'inputs/records.jsonl').read_text(encoding='utf-8').splitlines())]
    raw=torch.load(parent/'inputs/features.pt',map_location='cpu',weights_only=True)
    hybrid=modules['t35'].transform(raw,'hybrid',modules['loader43'].symmetry_table(),set());s=cp['scalers']
    views=modules['t10'].four_views(hybrid,*(s[k].to(device=device,dtype=torch.float32) for k in ('composition_mean','composition_std','structure_mean','structure_std')),device)
    d=dict(records=records,views=views,targets=torch.tensor([r['gap_ev'] for r in records],dtype=torch.float64,device=device))
    for label,prefix in (('train','t'),('valid','v'),('outer','o')):
        keys=set(cp['partition'][f'report_{label}_composition_keys']);ids=[i for i,r in enumerate(records) if r['composition_key'] in keys]
        d[prefix+'i']=torch.tensor(ids,dtype=torch.long,device=device);d[prefix+'w']=hierarchical_weights(records,ids,device)
    assert sum(len(d[p+'i']) for p in ('t','v','o'))==len(records)==205
    return d

def fit(initial,d,coverage,fold,arm,device):
    model=copy.deepcopy(initial);ti=d['ti'];ids=ti.tolist();positions={i:j for j,i in enumerate(ids)}
    supported=d['views'][3,ti,633:].bool().any(-1)
    ci=ti[supported];cw=hierarchical_weights(d['records'],ci.tolist(),device)
    flags=d['views'][3,ci,633:].bool();viewmask=torch.stack((flags[:,0],flags[:,1],flags.all(-1))).double()
    selected=set(coverage['folds'][fold]['train_pair_ids']);pairs=[r for r in coverage['pairs'] if r['pair_id'] in selected]
    mass=hierarchical_weights(d['records'],list(range(len(d['records']))),device)
    matrix=torch.zeros(len(pairs),len(ids),dtype=torch.float64,device=device)
    for k,p in enumerate(pairs):
        for label,sign in (('indices_a',-1),('indices_b',1)):
            endpoint=p[label];w=mass[endpoint];w=w/w.sum()
            for i,value in zip(endpoint,w):matrix[k,positions[i]]=sign*value
    pw=torch.tensor([p['weight'] for p in pairs],dtype=torch.float64,device=device);pw/=pw.sum()
    observed=torch.tensor([p['observed_delta_ev'] for p in pairs],dtype=torch.float64,device=device)
    assert len(pairs)>0 and torch.allclose(matrix@d['targets'][ti],observed,atol=1e-6,rtol=1e-6)
    scale=(pw*observed.square()).sum().clamp_min(.01)
    groups={}
    for j,i in enumerate(ids):groups.setdefault(d['records'][i]['composition_key'],[]).append(j)
    groups={k:v for k,v in groups.items() if supported[v].any()}
    gm=torch.zeros(len(groups),len(ids),dtype=torch.float64,device=device);representatives=[];fm=[]
    for k,(key,js) in enumerate(groups.items()):
        w=d['tw'][js];fm.append(w.sum());gm[k,js]=w/w.sum();representatives.append(ids[js[0]])
    fm=torch.stack(fm);fm/=fm.sum();rho=gm@supported.double()
    representative=torch.tensor(representatives,dtype=torch.long,device=device)
    logs=d['targets'][ti].log();grid=torch.linspace(float(logs.min()-2),float(logs.max()+2),257,dtype=torch.float64,device=device)
    with torch.no_grad():fixed_cdf=cdf(initial.formula.components(d['views'][0,ti]),grid)
    coefficient=.1 if arm=='DECOUPLED_PAIR' else 0.
    def terms():
        fnll=(model.formula.per_record_nll(d['views'][0,ti],d['targets'][ti])*d['tw']).sum()
        loss=model.conditional.per_record_nll(d['views'][1:4,ci],d['targets'][ci])
        cnll=(((loss*viewmask).sum(0)/viewmask.sum(0))*cw).sum()
        with torch.no_grad():
            teacher_cdf=cdf(model.conditional.components(d['views'][3,ti]),grid)
            target=gm@torch.where(supported[:,None],teacher_cdf,fixed_cdf)
        student=cdf(model.formula.components(d['views'][0,representative]),grid)
        kd=(fm*(student-target).square().mean(-1)).sum()
        deltas=matrix@means(model.conditional.components(d['views'][3,ti]))
        pair=(pw*(deltas-observed).square()).sum()/scale
        return fnll,cnll,kd,pair
    formula_params=list(model.formula.report_head.parameters());conditional_params=list(model.conditional.report_head.parameters())
    initial_terms=terms()
    leaks={}
    for name,value,params in (('conditional_to_formula',initial_terms[1],formula_params),('pair_to_formula',initial_terms[3],formula_params),('kd_to_conditional',initial_terms[2],conditional_params)):
        leaks[name]=any(g is not None and bool(g.abs().max()>0) for g in torch.autograd.grad(value,params,retain_graph=True,allow_unused=True))
    assert not any(leaks.values())
    diagnostics=[]
    vflags=d['views'][3,d['vi'],633:].bool();vci=d['vi'][vflags.any(-1)]
    vcw=hierarchical_weights(d['records'],vci.tolist(),device)
    vf=d['views'][3,vci,633:].bool();vmask=torch.stack((vf[:,0],vf[:,1],vf.all(-1))).double()
    def diagnostic():
        values=terms();return dict(zip(('formula_train_nll','conditional_train_nll','kd_mse','pair_normalized_mse'),map(float,values)))
    def validation():
        vl=model.conditional.per_record_nll(d['views'][1:4,vci],d['targets'][vci])
        conditional_valid=(((vl*vmask).sum(0)/vmask.sum(0))*vcw).sum()
        diagnostics.append(dict(epoch=len(diagnostics),conditional_valid_nll=float(conditional_valid),**diagnostic()))
        return (model.formula.per_record_nll(d['views'][0,d['vi']],d['targets'][d['vi']])*d['vw']).sum()
    objective=lambda:sum(v*c for v,c in zip(terms(),(1.,1.,.1,coefficient)))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    result=train.optimize(model,objective,validation,optimizer,300)
    with torch.no_grad():final=diagnostic()
    result.update(initial_state_hash=_state_hash(train.state(initial)),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_pairs=len(pairs),train_pair_formulas=len({r['composition_key'] for r in pairs}),pair_scale_squared_ev=float(scale),
        teacher_reports=len(ci),kd_formulas=len(groups),rho_by_formula={k:float(v) for k,v in zip(groups,rho)},
        selected_diagnostics=final,selected_conditional_valid_nll=diagnostics[result['selected_epoch']]['conditional_valid_nll'],
        diagnostic_history=diagnostics,gradient_leaks=leaks,initial_diagnostics=diagnostics[0])
    return model,result

def run(root,parent,work):
    device=rt.configure();modules=rt.modules();work.mkdir(exist_ok=True)
    coverage=rt.read_json(work/'coverage/pairs.json')
    paths=[root/'runtime'/n for n in ('train_decoupled_distribution.py','decoupled_distribution.py','train.py','runtime.py','training_data.py','inference.py','frozen_fusion_compare.py')]
    paths+=sorted((root/'runtime/modules').glob('*.py'))+[root/'runtime/modules/symmetry_table.json']
    parents={f'models/report_adaptation/{d.name}/models.pt':rt.sha(d/'models.pt') for d in sorted((root/'models/report_adaptation').iterdir()) if d.is_dir()}
    protocol=RECIPE|dict(source_binding={p.relative_to(root).as_posix():rt.sha(p) for p in paths},parent_checkpoint_binding=parents,
        input_binding={n:rt.sha(parent/'inputs'/n) for n in ('features.pt','metadata.json','records.jsonl')},coverage_sha256=rt.sha(work/'coverage/pairs.json'),device=str(device))
    pp=work/'training_protocol.json'
    if pp.exists() and rt.read_json(pp)!=protocol:raise ValueError('closed study inputs or recipe changed')
    rt.write_json(pp,protocol)
    for seed in RECIPE['seeds']:
        for fold in range(3):
            name=f'seed_{seed}_fold_{fold+1:02d}';out=work/'fits'/name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            cp=torch.load(root/'models/report_adaptation'/name/'models.pt',map_location='cpu',weights_only=True)
            d=context(root,parent,cp,modules,device);initial=DecoupledDistribution(modules,cp['models']['FROZEN_FORMULA'],cp['models']['FROZEN_MULTIVIEW'],device)
            initial_hash=_state_hash(train.state(initial));states={};fits={};predictions=[]
            for arm in ARMS:
                model,fitted=fit(initial,d,coverage,fold,arm,device);states[arm]=train.state(model);fits[arm]=fitted
                with torch.no_grad():
                    for vi,view in enumerate(rt.VIEWS):predictions+=train.prediction_rows(model.components(d['views'][vi,d['oi']]),d,seed,fold,arm,view)
                print(f'seed={seed} fold={fold+1} {arm} epoch={fitted["selected_epoch"]} pairs={fitted["train_pairs"]}',flush=True)
            rt.write_json(out/'fits.json',fits);rt.write_json(out/'partition.json',cp['partition']);rt.write_csv(out/'predictions.csv',predictions)
            torch.save(dict(schema='decoupled_distribution_v1',seed=seed,fold=fold,models=states,initial=train.state(initial),scalers=cp['scalers'],partition=cp['partition']),out/'models.pt')
            rt.write_json(out/'complete.json',dict(aggregate_fits=0,report_fits=2,prior_hash=initial_hash,protocol_sha256=rt.sha(pp),
                parent_sha256=parents[f'models/report_adaptation/{name}/models.pt'],models_sha256=rt.sha(out/'models.pt'),
                predictions_sha256=rt.sha(out/'predictions.csv'),fits_sha256=rt.sha(out/'fits.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.parent,a.work)
