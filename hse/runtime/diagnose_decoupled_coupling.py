"""Measure initial formula NLL/KD gradients, without optimization or held-out targets."""
import argparse
from pathlib import Path
import runtime as rt
from train_decoupled_distribution import context
from decoupled_distribution import DecoupledDistribution,cdf

def run(root,parent,work):
    torch=rt.torch;device=rt.configure();modules=rt.modules();result=[]
    for seed in (17,29,41):
        for fold in range(3):
            name=f'seed_{seed}_fold_{fold+1:02d}';path=root/'models/report_adaptation'/name/'models.pt'
            cp=torch.load(path,map_location='cpu',weights_only=True);d=context(root,parent,cp,modules,device)
            model=DecoupledDistribution(modules,cp['models']['FROZEN_FORMULA'],cp['models']['FROZEN_MULTIVIEW'],device)
            ti=d['ti'];ids=ti.tolist();available=d['views'][3,ti,633:].bool().any(-1);groups={}
            for j,i in enumerate(ids):groups.setdefault(d['records'][i]['composition_key'],[]).append(j)
            groups={k:js for k,js in groups.items() if available[js].any()}
            matrix=torch.zeros(len(groups),len(ids),dtype=torch.float64,device=device);mass=[];representatives=[]
            for k,js in enumerate(groups.values()):
                w=d['tw'][js];mass.append(w.sum());matrix[k,js]=w/w.sum();representatives.append(ids[js[0]])
            mass=torch.stack(mass);mass/=mass.sum();logs=d['targets'][ti].log()
            grid=torch.linspace(float(logs.min()-2),float(logs.max()+2),257,dtype=torch.float64,device=device)
            with torch.no_grad():
                f0=cdf(model.formula.components(d['views'][0,ti]),grid);m=cdf(model.conditional.components(d['views'][3,ti]),grid)
                teacher=matrix@torch.where(available[:,None],m,f0)
            predicted=cdf(model.formula.components(d['views'][0,representatives]),grid)
            kd=.1*(mass*(predicted-teacher).square().mean(-1)).sum()
            nll=(d['tw']*model.formula.per_record_nll(d['views'][0,ti],d['targets'][ti])).sum()
            parameters=list(model.formula.report_head.parameters())
            def gradient(loss):
                grads=torch.autograd.grad(loss,parameters,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).double() for p,g in zip(parameters,grads)])
            a=gradient(nll);b=gradient(kd);norma=float(a.norm());normb=float(b.norm())
            result.append(dict(seed=seed,fold=fold,formula_train_nll=float(nll.detach()),weighted_kd_loss=float(kd.detach()),
                nll_gradient_norm=norma,weighted_kd_gradient_norm=normb,kd_to_nll_gradient_ratio=normb/norma,
                nll_kd_gradient_cosine=float((a@b)/(a.norm()*b.norm()))))
    out=work/'coupling_diagnostic';out.mkdir(exist_ok=True)
    rt.write_csv(out/'initial_gradients.csv',result)
    rt.write_json(out/'protocol.json',dict(new_fits=0,scope='local gradients at warm-start TRAIN states, not global causality or Adam update decomposition',
        source_sha256=rt.sha(Path(__file__)),training_protocol_sha256=rt.sha(work/'training_protocol.json'),
        output_sha256=rt.sha(out/'initial_gradients.csv'),mean_gradient_ratio=sum(r['kd_to_nll_gradient_ratio'] for r in result)/len(result)))
    print(rt.read_json(out/'protocol.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.parent,a.work)
