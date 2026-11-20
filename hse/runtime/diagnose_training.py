"""GPU train/validation/outer modality diagnosis of the completed checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import runtime as rt
from training_data import build_fold
torch=rt.torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--work-dir',type=Path,required=True)
    args=p.parse_args(); device=rt.configure(); m=rt.modules(); rows=[]; phases=[]
    for fold in range(5):
        d=build_fold(args.root,fold,device)
        for seed in (17,29,41):
            cp=torch.load(args.work_dir/f'seed_{seed}_fold_{fold+1:02d}'/'models.pt',map_location=device,weights_only=False)
            with torch.no_grad():
                for arm in ('F','J','J_SG'):
                    state=cp['models'][arm]
                    model=m['interaction']('cross_attention',state['report_head.raw_beta'],
                        float(state['report_head.raw_feature_mean']),float(state['report_head.raw_feature_std']),
                        mode='F' if arm=='F' else 'M').to(device)
                    model.load_state_dict(state); model.eval()
                    for split,ik,wk in [('train','ti','tw'),('validation','vi','vw'),('outer','oi','ow')]:
                        idx,w=d[ik],d[wk]
                        for vi,view in enumerate(rt.VIEWS):
                            if arm=='F' and vi: continue
                            used=(0 if vi in (0,2) else 1) if arm=='J_SG' else vi
                            inputs=d['views'][used,idx]; parts=model.components(inputs)
                            nll=model.per_record_nll(inputs,d['targets'][idx])
                            for subset,mask in [('all',torch.ones(len(idx),dtype=torch.bool,device=device)),
                                                ('structure',d['views'][3,idx,-1]>.5)]:
                                if not mask.any(): continue
                                wt=w[mask]/w[mask].sum()
                                rows.append(dict(seed=seed,fold=fold,model=arm,view=view,split=split,subset=subset,
                                    reports=int(mask.sum()),nll=float((wt*nll[mask]).sum())))
                            if arm=='J' and split=='outer' and vi in (1,3):
                                expected_log=(parts[0]*parts[1]).sum(-1)
                                for j,i in enumerate(idx.tolist()):
                                    r=d['records'][i]
                                    phases.append(dict(seed=seed,fold=fold,view=view,measurement_id=r['measurement_id'],
                                        formula=r['formula'],phase=r.get('phase_label',''),
                                        structure_supported=int(d['views'][3,i,-1]),
                                        observed_log_gap=float(d['targets'][i].log()),predicted_log_gap=float(expected_log[j])))
            print(f'diagnosed seed={seed} fold={fold+1}',flush=True)
    out=args.work_dir/'summary'
    rt.write_csv(out/'split_diagnostics.csv',rows);rt.write_csv(out/'phase_responses.csv',phases)
    rt.write_json(out/'server_execution.json',dict(host='c137-13',device=torch.cuda.get_device_name(0),
        torch=torch.__version__,checkpoint_pairs=15,new_fits=0,
        purpose='train/validation/outer structure generalization and phase response diagnosis'))


if __name__=='__main__':main()
