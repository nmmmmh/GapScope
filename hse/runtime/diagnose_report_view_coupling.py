"""Measure shared report-head gradients on the existing development folds; no fits."""
import argparse
from pathlib import Path
import runtime as rt
from inference import _restore_model
from training_data import build_fold

torch=rt.torch
def run(root,work):
    out=work/'view_coupling';out.mkdir(parents=True,exist_ok=True)
    modules=rt.modules();device=rt.configure();rows=[];loss_rows=[];bindings={}
    for seed in (17,29,41):
        for fold in range(5):
            path=root/f'models/checkpoints/seed_{seed}_fold_{fold+1:02d}/models.pt'
            bindings[path.relative_to(root).as_posix()]=rt.sha(path)
            cp=torch.load(path,map_location='cpu',weights_only=True);data=build_fold(root,fold,device)
            if cp['partition']['report_train_composition_keys']!=data['provenance']['report_train_composition_keys']:raise ValueError('fold differs')
            for arm in ('FROZEN','FROZEN_SG'):
                model=_restore_model(modules,cp['models'][arm],'M',device)
                model.requires_grad_(False);params=list(model.report_head.parameters())
                for p in params:p.requires_grad_(True)
                grads=[]
                for vi,view in enumerate(rt.VIEWS):
                    loss=(data['tw']*model.per_record_nll(data['views'][vi,data['ti']],data['targets'][data['ti']])).sum()
                    grad=torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,params)])
                    grads.append(grad.detach())
                    with torch.no_grad():
                        valid=(data['vw']*model.per_record_nll(data['views'][vi,data['vi']],data['targets'][data['vi']])).sum()
                    loss_rows.append(dict(seed=seed,fold=fold,model=arm,view=view,train_nll=float(loss),valid_nll=float(valid),head_parameters=grad.numel(),alpha=float(model.report_head.alpha)))
                # FROZEN_SG was trained on [formula,SG,formula,SG], not on structure/full.
                trained=(grads[0]+grads[1])/2 if arm=='FROZEN_SG' else torch.stack(grads).mean(0)
                conditional=grads[1] if arm=='FROZEN_SG' else torch.stack(grads[1:]).mean(0)
                def cosine(a,b):
                    den=float(a.norm()*b.norm())
                    return float(a@b)/den if den>0 else None
                rows.append(dict(seed=seed,fold=fold,model=arm,formula_conditional_cosine=cosine(grads[0],conditional),
                    formula_trained_objective_cosine=cosine(grads[0],trained),formula_gradient_norm=float(grads[0].norm()),
                    conditional_gradient_norm=float(conditional.norm()),objective_gradient_norm=float(trained.norm()),
                    formula_derivative_along_negative_objective_gradient=-float(grads[0]@trained),
                    formula_derivative_along_negative_conditional_gradient=-float(grads[0]@conditional)))
            print(f'diagnosed seed={seed} fold={fold+1}',flush=True)
    rt.write_csv(out/'gradients.csv',rows);rt.write_csv(out/'view_losses.csv',loss_rows)
    summary={arm:dict(packages=len(rs),formula_conditional_conflicts=sum(r['formula_conditional_cosine']<0 for r in rs),
        actual_objective_worsens_formula_locally=sum(r['formula_derivative_along_negative_objective_gradient']>0 for r in rs),
        mean_formula_conditional_cosine=sum(r['formula_conditional_cosine'] for r in rs)/len(rs))
        for arm in ('FROZEN','FROZEN_SG') for rs in [[r for r in rows if r['model']==arm]]}
    rt.write_json(out/'summary.json',dict(new_fits=0,development_only=True,summary=summary,
        scope='local gradients at selected states, not a causal explanation of external failure or a global optimization proof',
        source_sha256=rt.sha(Path(__file__)),checkpoint_binding=bindings,
        output_binding={p.name:rt.sha(p) for p in out.iterdir() if p.is_file() and p.name!='summary.json'}))
    print(summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
