"""Frozen formula density with a mean-preserving phase-conditioned residual."""
import argparse
import itertools
import math
import time
from collections import defaultdict
from pathlib import Path
import runtime as rt
import train
from inference import _restore_model
from training_data import build_fold
from phase_pair_inventory import condition_group_weights

torch=rt.torch
ARMS=('PHASE_NLL','PHASE_PAIR')
RECIPE=dict(seeds=[17,29,41],folds=5,arms=list(ARMS),new_fits=30,new_aggregate_fits=0,
    baseline='existing matching FROZEN formula density and backbone, all frozen',
    residual='bias-free 64-hidden-difference to two outputs; zero initialized; 2*tanh(raw/2)',
    parameters=128,learning_rate=.001,epochs=300,patience=30,pair_coefficient=.1,pair_scale_floor_ev=.1,
    report_loss='four-view hierarchical true lognormal mixture NLL',
    pair_loss='TRAIN-only CIF-group arithmetic mean differences; equal formula then equal pair; views2/3 average',
    pair_scale='max(TRAIN weighted RMS observed pair difference, 0.1 eV)',
    width='shared raw-scale shift; log-mu compensation keeps each component arithmetic mean unchanged by width',
    selection='four-view innerVALID NLL, epoch0 eligible, no pair loss in selection',
    formula='exact frozen formula components; no residual with missing SG and structure',parameter_search=[])


def prepare(baseline,views):
    with torch.no_grad():
        formula=baseline.formula_input(views[0])
        raw=baseline.raw_report(formula)
        w,mu,sd=baseline.components(formula)
        h=baseline.encode(formula)
        difference=(baseline.encode(views)-h.unsqueeze(0)).double()
        available=(views[..., -2:].sum(-1)>0).unsqueeze(-1)
        difference=difference*available
        difference[0].zero_()
        base=dict(weights=w,mus=mu,sigmas=sd,raw_scale=raw[...,3:5],
                  mean=(w*torch.exp(mu+.5*sd.square())).sum(-1))
    return base,difference


def corrected(base,delta):
    # Subtract the same forward reconstruction for exact equality at zero.
    raw_scale=base['raw_scale'].expand(*delta.shape[:-1],2).contiguous()
    original=.05+2.45*torch.sigmoid(raw_scale)
    sd=base['sigmas']+(.05+2.45*torch.sigmoid(raw_scale+delta[...,1:2])-original)
    mu=base['mus']+delta[...,0:1]+.5*(base['sigmas'].square()-sd.square())
    return base['weights'].expand_as(mu),mu,sd


def deltas(head,difference):return 2*torch.tanh(head(difference)/2)


def phase_components(baseline,inputs,head_state):
    head=torch.nn.Linear(64,2,bias=False,dtype=torch.float64,device=inputs.device)
    head.load_state_dict(head_state)
    formula=baseline.formula_input(inputs)
    base,difference=prepare(baseline,torch.stack((formula,inputs)))
    return tuple(x[1] for x in corrected(base,deltas(head,difference)))


def pair_targets(root,data):
    index={r['measurement_id']:r['actual_cif_sha256'] for r in rt.read_csv(root/'results/physical_cases/resolved_structure_index.csv')}
    split_weights={int(i):float(w) for i,w in zip(data['ti'].tolist(),data['tw'].tolist())}
    grouped=defaultdict(list)
    for i in data['ti'].tolist():
        if bool(data['views'][3,i,-1]):
            r=data['records'][i];grouped[r['composition_key'],index[r['measurement_id']]].append(i)
    formulas=defaultdict(list)
    for (key,cif_hash),indices in grouped.items():
        weights=condition_group_weights(indices,split_weights)
        formulas[key].append((cif_hash,indices,weights))
    pairs=[];count=sum(len(gs)>1 for gs in formulas.values())
    for key,gs in sorted(formulas.items()):
        combinations=list(itertools.combinations(sorted(gs),2))
        for a,b in combinations:
            row=dict(composition_key=key,cif_a=a[0],cif_b=b[0],weight=1/count/len(combinations))
            for side,g in [('a',a),('b',b)]:
                row[side+'_indices']=g[1];row[side+'_weights']=[g[2][i] for i in g[1]]
            row['target_ev']=sum(b[2][i]*float(data['targets'][i]) for i in b[1])-sum(a[2][i]*float(data['targets'][i]) for i in a[1])
            pairs.append(row)
    if not pairs:raise ValueError('no TRAIN CIF pairs')
    scale=max(math.sqrt(sum(r['weight']*r['target_ev']**2 for r in pairs)),RECIPE['pair_scale_floor_ev'])
    if any(i not in split_weights for r in pairs for side in ('a','b') for i in r[side+'_indices']):raise ValueError('pair target outside TRAIN')
    return pairs,scale


def fit(base,difference,data,pairs,scale,coefficient):
    head=torch.nn.Linear(64,2,bias=False,dtype=torch.float64,device=difference.device)
    with torch.no_grad():head.weight.zero_()
    def nll(indices,weights):
        w,mu,sd=parts(indices)
        log_y=data['targets'][indices].log()[None,:,None]
        losses=-torch.logsumexp(w.log()-log_y-sd.log()-.5*math.log(2*math.pi)-.5*((log_y-mu)/sd).square(),-1)
        return (losses.mean(0)*weights).sum()
    def parts(indices):
        selected={k:v[indices] for k,v in base.items()}
        return corrected(selected,deltas(head,difference[:,indices]))
    def pair_loss():
        delta=deltas(head,difference)
        # Analytic identity: the pair objective never receives width gradients.
        means=base['mean'][None,:]*torch.exp(delta[...,0])
        losses=[]
        for r in pairs:
            predicted=[]
            for side in ('a','b'):
                weights=torch.tensor(r[side+'_weights'],dtype=torch.float64,device=difference.device)
                predicted.append((means[2:4,r[side+'_indices']]*weights).sum(-1))
            losses.append(r['weight']*((predicted[1]-predicted[0]-r['target_ev'])/scale).square().mean())
        return torch.stack(losses).sum()
    def values():
        return dict(train_nll=float(nll(data['ti'],data['tw'])),pair_loss=float(pair_loss()),
                    validation_nll=float(nll(data['vi'],data['vw'])))
    with torch.no_grad():initial=values()
    gradient_nll=torch.autograd.grad(nll(data['ti'],data['tw']),head.weight)[0]
    gradient_pair=torch.autograd.grad(pair_loss(),head.weight)[0]
    diagnostics=dict(initial_nll_gradient_norm=float(gradient_nll.norm()),
        initial_pair_gradient_norm=float(gradient_pair.norm()),initial_pair_width_gradient_max=float(gradient_pair[1].abs().max()),
        initial_gradient_dot=float((gradient_nll*gradient_pair).sum()))
    optimizer=torch.optim.Adam(head.parameters(),lr=RECIPE['learning_rate'])
    fit_info=train.optimize(head,lambda:nll(data['ti'],data['tw'])+coefficient*pair_loss(),
        lambda:nll(data['vi'],data['vw']),optimizer,RECIPE['epochs'])
    with torch.no_grad():selected=values();delta=deltas(head,difference);output=corrected(base,delta)
    fitted_mean=(output[0]*torch.exp(output[1]+.5*output[2].square())).sum(-1)
    identity_error=float((fitted_mean-base['mean'][None,:]*torch.exp(delta[...,0])).abs().max())
    formula_error=max(float((output[j][0]-base[k]).abs().max()) for j,k in enumerate(('weights','mus','sigmas')))
    if formula_error!=0 or identity_error>1e-8 or diagnostics['initial_pair_width_gradient_max']!=0:raise ValueError('residual mathematical invariant')
    fit_info.update(initial=initial,selected=selected,**diagnostics,pair_coefficient=coefficient,
        trainable_parameters=128,pair_scale_ev=scale,train_pair_count=len(pairs),
        train_pair_formulas=len({r['composition_key'] for r in pairs}),mean_identity_max_error=identity_error,formula_component_max_error=formula_error)
    return head,output,fit_info


def run_fold(root,work,seed,fold,device,modules,protocol_sha):
    out=work/f'seed_{seed}_fold_{fold+1:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    start=time.time();data=build_fold(root,fold,device)
    source=root/'models/checkpoints'/out.name/'models.pt'
    cp=torch.load(source,map_location='cpu',weights_only=True)
    if cp['partition']!=data['provenance']:raise ValueError('baseline partition mismatch')
    baseline=_restore_model(modules,cp['models']['FROZEN'],'M',device).requires_grad_(False).eval()
    base,difference=prepare(baseline,data['views']);pairs,scale=pair_targets(root,data)
    # Pair targets are serialized separately; only TRAIN report indices occur.
    rt.write_json(out/'train_pairs.json',pairs)
    fits={};states={};predictions=[]
    for arm,coefficient in [('PHASE_NLL',0.),('PHASE_PAIR',RECIPE['pair_coefficient'])]:
        head,output,fits[arm]=fit(base,difference,data,pairs,scale,coefficient)
        states[arm]=train.state(head)
        for vi,view in enumerate(rt.VIEWS):
            predictions+=train.prediction_rows(tuple(p[vi,data['oi']] for p in output),data,seed,fold,arm,view)
        print(f'{out.name} {arm} selected={fits[arm]["selected_epoch"]}',flush=True)
    rt.write_csv(out/'predictions.csv',predictions);rt.write_json(out/'fits.json',fits)
    torch.save(dict(schema='phase_conditioned_residual_v1',seed=seed,fold=fold,heads=states,
        baseline=cp['models']['FROZEN'],partition=cp['partition'],scalers=cp['scalers'],
        baseline_package_sha256=rt.sha(source),recipe=RECIPE),out/'models.pt')
    rt.write_json(out/'complete.json',dict(seed=seed,fold=fold,new_fits=2,seconds=time.time()-start,
        protocol_sha256=protocol_sha,predictions_sha256=rt.sha(out/'predictions.csv'),
        fits_sha256=rt.sha(out/'fits.json'),models_sha256=rt.sha(out/'models.pt')))
    print(f'COMPLETE {out.name} fits=2',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--work-dir',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work_dir.resolve()
    if work.is_relative_to(root):raise ValueError('work must be outside formal root')
    work.mkdir(parents=True,exist_ok=True);device=rt.configure();modules=rt.modules()
    paths=[root/'data/reports'/n for n in ('features.pt','metadata.json','records.jsonl')]
    paths += [root/'data/composition'/n for n in ('features.pt','metadata.json')]
    paths += [root/'results/physical_cases/resolved_structure_index.csv']
    sources=[Path(__file__),root/'runtime/phase_pair_inventory.py',root/'runtime/train.py',root/'runtime/training_data.py',root/'runtime/inference.py',root/'runtime/runtime.py']
    sources += sorted((root/'runtime/modules').glob('*'))
    protocol=RECIPE|dict(device=str(device),gpu=torch.cuda.get_device_name(0),
        input_binding={p.relative_to(root).as_posix():rt.sha(p) for p in paths},
        code_binding={p.relative_to(root).as_posix():rt.sha(p) for p in sources if p.is_file()},
        checkpoint_binding={p.relative_to(root).as_posix():rt.sha(p) for p in sorted((root/'models/checkpoints').glob('seed_*/models.pt'))})
    target=work/'protocol.json'
    if target.exists() and rt.read_json(target)!=protocol:raise ValueError('existing run inputs or recipe changed')
    rt.write_json(target,protocol)
    for seed in RECIPE['seeds']:
        for fold in range(5):run_fold(root,work,seed,fold,device,modules,rt.sha(target))


if __name__=='__main__':main()
