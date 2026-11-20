"""One bounded matched study of report views and shared adaptation."""
import argparse
import copy
from pathlib import Path
import runtime as rt
import train
from grouped_adaptation_data import build_fold
from frozen_fusion_compare import _state_hash

torch=rt.torch
ARMS=('FROZEN_MULTIVIEW','FROZEN_FORMULA','JOINT_FORMULA')
def fit(prior,data,standardized,arm):
    model=copy.deepcopy(prior);model.mode='M' if arm=='FROZEN_MULTIVIEW' else 'F'
    head=list(model.report_head.parameters());head_ids={id(p) for p in head}
    if arm.startswith('FROZEN'):
        model.requires_grad_(False)
        for p in head:p.requires_grad_(True)
        optimizer=torch.optim.Adam(head,lr=.001)
    else:optimizer=torch.optim.Adam([dict(params=[p for p in model.parameters() if id(p) not in head_ids],lr=.0001),dict(params=head,lr=.001)])
    def nll(indices,weights,multiview=False):
        inputs=data['views'][:,indices] if multiview else data['views'][0,indices]
        loss=model.per_record_nll(inputs,data['targets'][indices])
        if loss.ndim==2:loss=loss.mean(0)
        return (loss*weights).sum()
    def objective():
        loss=nll(data['ti'],data['tw'],arm=='FROZEN_MULTIVIEW')
        if arm=='JOINT_FORMULA':
            loss=loss+.1*(model.aggregate(data['av'][:,data['fi']]).double()-standardized[data['fi']]).square().mean()
        return loss
    # The research endpoint is formula prediction; all arms share this selector.
    fitted=train.optimize(model,objective,lambda:nll(data['vi'],data['vw']),optimizer,300)
    fitted['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model,fitted

def run(root,work):
    work.mkdir(parents=True,exist_ok=True);device=rt.configure();modules=rt.modules()
    paths=[root/'runtime'/n for n in ('grouped_adaptation_compare.py','grouped_adaptation_data.py','train.py','training_data.py','runtime.py','frozen_fusion_compare.py')]
    paths+=sorted((root/'runtime/modules').glob('*.py'))+[root/'runtime/modules/symmetry_table.json']
    paths+=[root/'data/composition'/n for n in ('features.pt','metadata.json')]
    bindings={p.relative_to(root).as_posix():rt.sha(p) for p in paths}
    inputs={n:rt.sha(work/'inputs'/n) for n in ('features.pt','metadata.json','records.jsonl')}
    protocol=dict(seeds=[17,29,41],folds=3,arms=list(ARMS),primary_view='formula_only',
        evaluated_views={arm:list(rt.VIEWS if arm=='FROZEN_MULTIVIEW' else rt.VIEWS[:1]) for arm in ARMS},
        report_selection='innerVALID formula-only hierarchical NLL for all arms; epoch0 eligible',
        report_epochs=300,aggregate_epochs=train.RECIPE['aggregate_epochs'],aggregate_lr=train.RECIPE['aggregate_lr'],
        patience=30,head_lr=.001,shared_lr=.0001,replay_coefficient=.1,
        aggregate_fits=9,report_fits=27,parameter_grid=False,default_model_changed=False,
        grouping='connected formulas sharing primary source or element set; aggregate valid/outer element sets excluded',
        source_binding=bindings,input_binding=inputs,preflight_sha256=rt.sha(work/'preflight.json'),device=str(device),confirmation_used=False)
    protocol_path=work/'training_protocol.json'
    if protocol_path.exists() and rt.read_json(protocol_path)!=protocol:raise ValueError('study changed')
    rt.write_json(protocol_path,protocol)
    for seed in protocol['seeds']:
        for fold in range(3):
            out=work/'fits'/f'seed_{seed}_fold_{fold+1:02d}';out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            data=build_fold(root,work,fold,device)
            rt.write_json(out/'partition.json',data['provenance'])
            prior,standardized,afit,target_scalers=train.aggregate_prior(modules,data,seed,fold,device)
            prior_state=train.state(prior);prior_hash=_state_hash(prior_state);states={};fits={'aggregate':afit};predictions=[]
            for arm in ARMS:
                model,rf=fit(prior,data,standardized,arm);rf['initial_state_hash']=prior_hash;fits[arm]=rf;states[arm]=train.state(model)
                with torch.no_grad():
                    views=rt.VIEWS if arm=='FROZEN_MULTIVIEW' else rt.VIEWS[:1]
                    for vi,view in enumerate(views):predictions.extend(train.prediction_rows(model.components(data['views'][vi,data['oi']]),data,seed,fold,arm,view))
                print(f'fit seed={seed} fold={fold+1} arm={arm} epoch={rf["selected_epoch"]}',flush=True)
            rt.write_csv(out/'predictions.csv',predictions);rt.write_json(out/'fits.json',fits)
            torch.save(dict(schema='grouped_adaptation_v1',seed=seed,fold=fold,models=states,prior=prior_state,
                scalers=data['scalers'],target_scalers=target_scalers,partition=data['provenance']),out/'models.pt')
            rt.write_json(out/'complete.json',dict(aggregate_fits=1,report_fits=3,prior_hash=prior_hash,
                protocol_sha256=rt.sha(protocol_path),predictions_sha256=rt.sha(out/'predictions.csv'),models_sha256=rt.sha(out/'models.pt'),fits_sha256=rt.sha(out/'fits.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
