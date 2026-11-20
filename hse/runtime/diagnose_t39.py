"""Matched T39 trajectory: inner selection, evaluation-fold selection bias, frozen trunk."""
import argparse
import json
from pathlib import Path
import numpy as np
import runtime as rt
from density_score import score
torch=rt.torch
RECIPE=dict(epochs=300,patience=15,min_delta=1e-6,head_lr=.001,shared_lr=.0001,
            penalty=.01,replay=.1,arms=['structure_projection','readout_continue'],
            selectors=['inner','outer_diagnostic'],
            selection_warning='outer_diagnostic intentionally selects on evaluation labels; never used for delivery')


def optional(mu,sd,y,w,masks,m):
    loss=m['t10'].lognormal_nll(y.log()[None],mu,sd); terms=[]
    for key,views in [('sg',(1,3)),('structure',(2,3))]:
        supported=masks[key]>.5
        if supported.any():
            ws=w[supported]; ws=ws/ws.sum()
            terms.append((ws*loss[list(views)][:,supported].mean(0)).sum())
    return torch.stack(terms).mean()


def build(m,cp,arm,device):
    cp=m['model39'].complete_checkpoint(cp); state=cp['state_dict']; prefix='formula_base.report_head.'
    formula=m['Source']('simple_concat',state[prefix+'raw_beta'],float(state[prefix+'raw_feature_mean']),float(state[prefix+'raw_feature_std']))
    model=m['model39'].Adaptation(formula,m['t35'].SymmetryFusion('hybrid'),cp['basis'],arm).to(device)
    model.load_state_dict(state);return model.eval()


def run(args,m,device,seed,fold):
    out=args.work_dir/f'seed_{seed}_{fold}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    assets=args.input_dir
    inner=torch.load(assets/'bundles'/f'{fold}_inner.pt',map_location='cpu',weights_only=True)
    outer=torch.load(assets/'bundles'/f'{fold}_outer_features.pt',map_location='cpu',weights_only=True)
    cp=torch.load(assets/'parents'/f'seed_{seed}_{fold}'/'t36_sg.pt',map_location='cpu',weights_only=True)
    reference=torch.load(assets/'parents'/f'seed_{seed}_{fold}'/'t39_structure.pt',map_location='cpu',weights_only=True)
    assert cp['report_train_ids']==inner['train_ids'] and cp['report_validation_ids']==inner['valid_ids']
    assert set(inner['train_ids']).isdisjoint(outer['report_ids']) and set(inner['valid_ids']).isdisjoint(outer['report_ids'])
    table=m['loader43'].symmetry_table(); scalers=[cp[k].to(device) for k in ('composition_mean','composition_std','structure_mean','structure_std')]
    views,av,ov=[m['t10'].four_views(m['t35'].transform(f,'hybrid',table,set()),*scalers,device) for f in (inner['report'],inner['aggregate'],outer['report'])]
    y=inner['targets'].to(device);ay=inner['aggregate_targets'].to(device)
    standardized=(ay.log1p()-cp['aggregate_log1p_mean'].to(device))/cp['aggregate_log1p_std'].to(device)
    index={rid:i for i,rid in enumerate(inner['report_ids'])}
    ti=torch.tensor([index[r] for r in inner['train_ids']],device=device)
    vi=torch.tensor([index[r] for r in inner['valid_ids']],device=device)
    fi=torch.tensor(inner['aggregate_fit_indices'],device=device)
    tw=torch.tensor([inner['train_weights'][r] for r in inner['train_ids']],device=device,dtype=torch.float64)
    vw=torch.tensor([inner['valid_weights'][r] for r in inner['valid_ids']],device=device,dtype=torch.float64)
    truths={r['measurement_id']:r for r in [json.loads(s) for s in (assets/'records.jsonl').read_text(encoding='utf-8').splitlines()]}
    weights={r['measurement_id']:float(r['oof_weight']) for r in rt.read_csv(assets/'oof_weights.csv')}
    oy=torch.tensor([truths[r]['gap_ev'] for r in outer['report_ids']],device=device,dtype=torch.float64)
    ow=torch.tensor([weights[r] for r in outer['report_ids']],device=device,dtype=torch.float64);ow=ow/ow.sum()
    vm={key:inner['report'][key+'_mask'].to(device)[vi,0] for key in ('sg','structure')}
    om={key:outer['report'][key+'_mask'][:,0].to(device) for key in ('sg','structure')}
    predictions=[]; histories=[]; fits=[]
    for arm in RECIPE['arms']:
        model=build(m,cp,arm,device); groups=[dict(params=[model.location_weights,model.scale_weights],lr=.001)]
        if arm=='structure_projection':
            with torch.no_grad():pm,ps=model.report(ov,(.05,2.5))
            for view,name in enumerate(rt.VIEWS):
                for j,rid in enumerate(outer['report_ids']):
                    r=truths[rid]
                    predictions.append(dict(seed=seed,fold_id=fold,model='parent',selector='inner',view=name,
                        measurement_id=rid,composition_key=r['composition_key'],formula=r['formula'],
                        sg_supported=int(outer['report']['sg_mask'][j,0]),structure_supported=int(outer['report']['structure_mask'][j,0]),
                        component_weights='[1.0]',log_mus=json.dumps([float(pm[view,j])]),sigmas=json.dumps([float(ps[view,j])]),
                        target_ev=float(oy[j]),weight=weights[rid]))
        if arm=='structure_projection':groups.append(dict(params=model.shared_parameters(),lr=.0001))
        optimizer=torch.optim.Adam(groups)
        @torch.no_grad()
        def evaluate():
            mu,sd=model.report(views[:,vi],(.05,2.5));omu,osd=model.report(ov,(.05,2.5))
            return dict(inner=float(optional(mu,sd,y[vi],vw,vm,m)),outer_diagnostic=float(optional(omu,osd,oy,ow,om,m)))
        initial=evaluate();selectors={k:dict(best=v,epoch=0,bad=0,stopped=False,state=m['t10'].clone_state(model)) for k,v in initial.items()}
        for epoch in range(1,301):
            model.train();optimizer.zero_grad(set_to_none=True)
            features=model.fixed_features(views[:,ti]);mu,sd=model.report_from_features(features,(.05,2.5))
            loss=(tw*m['t10'].lognormal_nll(y[ti].log()[None],mu,sd).mean(0)).sum()
            dm,ds=model.residuals(features);penalty=(tw*(dm.square()+ds.square()).mean(0)).sum()
            replay=m['t10'].aggregate_loss(model.modal_base,av,standardized,fi)
            objective=loss+.01*penalty+.1*replay
            assert torch.isfinite(objective);objective.backward();optimizer.step();model.eval()
            values=evaluate();histories.append(dict(arm=arm,epoch=epoch,loss=float(loss.detach()),replay=float(replay.detach()),**values))
            for key,selected in selectors.items():
                if selected['stopped']:continue
                if values[key]<selected['best']-1e-6:
                    selected.update(best=values[key],epoch=epoch,bad=0,state=m['t10'].clone_state(model))
                else:selected['bad']+=1
                if selected['bad']>=15:selected['stopped']=True
            if all(s['stopped'] for s in selectors.values()):break
        for selector,selected in selectors.items():
            model.load_state_dict(selected['state']);model.eval()
            replay_error=None
            if selector=='inner' and arm=='structure_projection':
                other=rt.restore(m,reference,device)
                with torch.no_grad():replay_error=max(float((a-b).abs().max()) for a,b in zip(model.report(ov),other.report(ov)))
            fits.append(dict(seed=seed,fold_id=fold,arm=arm,selector=selector,epoch=selected['epoch'],validation=selected['best'],reference_parameter_output_max_difference=replay_error))
            with torch.no_grad():mu,sd=model.report(ov,(.05,2.5))
            for view,name in enumerate(rt.VIEWS):
                for j,rid in enumerate(outer['report_ids']):
                    r=truths[rid]
                    predictions.append(dict(seed=seed,fold_id=fold,model=arm,selector=selector,view=name,
                        measurement_id=rid,composition_key=r['composition_key'],formula=r['formula'],
                        sg_supported=int(outer['report']['sg_mask'][j,0]),structure_supported=int(outer['report']['structure_mask'][j,0]),
                        component_weights='[1.0]',log_mus=json.dumps([float(mu[view,j])]),sigmas=json.dumps([float(sd[view,j])]),target_ev=float(oy[j]),weight=weights[rid]))
    rt.write_csv(out/'predictions.csv',predictions);rt.write_csv(out/'fits.csv',fits);rt.write_csv(out/'history.csv',histories)
    rt.write_json(out/'complete.json',dict(seed=seed,fold=fold,new_fit_trajectories=2,selectors=4));print(f'complete {seed} {fold}',flush=True)


def summarize(args):
    from collections import defaultdict
    grouped=defaultdict(list);fits=[]
    for p in args.work_dir.glob('seed_*/predictions.csv'):
        for r in rt.read_csv(p):grouped[(r['model'],r['selector'],r['view'],r['measurement_id'])].append(r)
        fits+=rt.read_csv(p.with_name('fits.csv'))
    scores=[]
    for (model,selector,view,rid),rs in grouped.items():
        assert len(rs)==3
        parts=tuple(np.concatenate([np.array(json.loads(r[k])) for r in rs]) for k in ('component_weights','log_mus','sigmas'))
        parts=(parts[0]/3,parts[1],parts[2]);r=rs[0]
        scores.append(dict(model=model,selector=selector,view=view,measurement_id=rid,composition_key=r['composition_key'],
            structure_supported=int(r['structure_supported']),weight=float(r['weight']),**score(float(r['target_ev']),parts)))
    metrics=[]
    for model in RECIPE['arms']+['parent']:
        for selector in (['inner'] if model=='parent' else RECIPE['selectors']):
            for view in rt.VIEWS:
                for subset in ('all','structure'):
                    rs=[r for r in scores if r['model']==model and r['selector']==selector and r['view']==view and (subset=='all' or r['structure_supported'])]
                    w=np.array([r['weight'] for r in rs]);w/=w.sum()
                    metrics.append(dict(model=model,selector=selector,view=view,subset=subset,reports=len(rs),
                        **{k:float(w@np.array([r[k] for r in rs])) for k in ('nll','crps_ev','median_mae_ev')}))
    out=args.work_dir/'summary';out.mkdir(exist_ok=True)
    rt.write_csv(out/'metrics.csv',metrics);rt.write_csv(out/'report_scores.csv',scores);rt.write_csv(out/'fits.csv',fits)
    rt.write_json(out/'protocol.json',RECIPE|dict(device=torch.cuda.get_device_name(0),trained_reports=164,structure_reports=69,new_fit_trajectories=30))
    print(json.dumps([r for r in metrics if r['subset']=='structure' and r['view'] in ('formula_only','full')]),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True)
    args=p.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True);m=rt.modules();device=rt.configure()
    for seed in (17,29,41):
        for fold in (f'fold_{i:02d}' for i in range(1,6)):run(args,m,device,seed,fold)
    summarize(args)


if __name__=='__main__':main()
