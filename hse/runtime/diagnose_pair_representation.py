"""Read-only geometry of all current held-formula CIF pairs; no fitting."""
import argparse
from collections import defaultdict
from pathlib import Path
import runtime as rt
from training_data import build_fold,hierarchical_weights
from inference import _restore_model

torch=rt.torch

def run(root,out,device):
    out.mkdir(parents=True,exist_ok=True)
    modules=rt.modules();torch.set_num_threads(2)
    mapping=rt.read_csv(root/'results/physical_cases/resolved_structure_index.csv')
    hashes={r['measurement_id']:r['actual_cif_sha256'] for r in mapping}
    all_pairs=rt.read_csv(root/'results/physical_cases/same_formula_cif_pairs.csv')
    response={(r['seed_mixture'],r['composition_key'],r['cif_a'],r['cif_b']):r for r in all_pairs if r['model']=='FROZEN'}
    pair_rows=[r for r in all_pairs
               if r['model']=='FROZEN' and r['seed_mixture']=='equal_density_mixture']
    if len(pair_rows)!=22:raise ValueError('pair inventory differs')
    raw=torch.load(root/'data/reports/features.pt',map_location='cpu',weights_only=True)
    bindings={p.relative_to(root).as_posix():rt.sha(p) for p in [root/'data/reports/features.pt',
        root/'data/reports/metadata.json',root/'data/reports/records.jsonl',
        root/'results/physical_cases/resolved_structure_index.csv',root/'results/physical_cases/same_formula_cif_pairs.csv']}
    results=[];geometry=[]
    for fold in range(5):
        data=build_fold(root,fold,device)
        weights=hierarchical_weights(data['records'],list(range(len(data['records']))),'cpu')
        groups=defaultdict(list)
        for i,r in enumerate(data['records']):
            if r['measurement_id'] in hashes:groups[r['composition_key'],hashes[r['measurement_id']]].append(i)
        def average(x,key):
            ids=groups[key];w=weights[ids].to(x.device);w=w/w.sum()
            return (x[ids].double()*w[:,None]).sum(0)
        train_keys=set(data['provenance']['report_train_composition_keys'])
        outer_keys=set(data['provenance']['report_outer_composition_keys'])
        selected=[r for r in pair_rows if r['composition_key'] in outer_keys]
        train_pairs=[r for r in pair_rows if r['composition_key'] in train_keys]
        for seed in (17,29,41):
            path=root/f'models/checkpoints/seed_{seed}_fold_{fold+1:02d}/models.pt'
            cp=torch.load(path,map_location='cpu',weights_only=True);bindings[path.relative_to(root).as_posix()]=rt.sha(path)
            if cp['partition']!=data['provenance']:raise ValueError('partition mismatch')
            model=_restore_model(modules,cp['models']['FROZEN'],'M',device)
            with torch.inference_mode():
                hidden=model.encode(data['views']).double()
                structure_input=data['views'][2,:,473:633]
                token=model.activation(model.structure_projection(structure_input)).double()
                formula=hidden[0];differences=hidden-formula.unsqueeze(0)
            def span(matrix):
                _,singular,vh=torch.linalg.svd(matrix.cpu(),full_matrices=False)
                rank=int((singular>singular[0]*1e-6).sum()) if float(singular[0])>0 else 0
                return vh[:rank],rank,singular
            train_vectors={}
            for vi in (2,3):
                train_vectors[vi]=torch.stack([average(hidden[vi],(r['composition_key'],r['cif_b']))-
                    average(hidden[vi],(r['composition_key'],r['cif_a'])) for r in train_pairs])
            joint_basis,joint_rank,_=span(torch.cat(list(train_vectors.values())))
            nll_basis,nll_rank,_=span(differences[1:,data['ti']].reshape(-1,64))
            for vi in (2,3):
                view=rt.VIEWS[vi]
                basis,rank,singular=span(train_vectors[vi])
                geometry.append(dict(seed=seed,fold=fold,view=view,train_pairs=len(train_pairs),
                    train_pair_formulas=len({r['composition_key'] for r in train_pairs}),rank=rank,
                    joint_structure_full_pair_rank=joint_rank,nll_residual_input_rank=nll_rank,
                    singular_values=singular.tolist(),relative_rank_cutoff=1e-6))
                for r in selected:
                    key=r['composition_key'];a=(key,r['cif_a']);b=(key,r['cif_b'])
                    delta=(average(hidden[vi],b)-average(hidden[vi],a)).cpu()
                    norm=float(delta.norm());projected=basis.T@(basis@delta)
                    residual=float((delta-projected).norm())/norm if norm>0 else 0.
                    base=dict(seed=seed,fold=fold,view=view,composition_key=key,formula=r['formula'],
                        cif_a=r['cif_a'],cif_b=r['cif_b'],sg_a=r['sg_a'],sg_b=r['sg_b'],same_sg=r['sg_a']==r['sg_b'],
                        raw_structure_l2=float((average(raw['structure'],b)-average(raw['structure'],a)).norm()),
                        scaled_structure_l2=float((average(structure_input,b)-average(structure_input,a)).norm()),
                        structure_token_l2=float((average(token,b)-average(token,a)).norm()),hidden_pair_l2=norm,
                        hidden_pair_cosine=float(torch.nn.functional.cosine_similarity(average(hidden[vi],a),average(hidden[vi],b),dim=0)),
                        outside_train_pair_span_norm_fraction=residual,train_span_rank=rank,
                        outside_joint_pair_span_norm_fraction=float((delta-joint_basis.T@(joint_basis@delta)).norm())/norm if norm>0 else 0.,
                        outside_nll_residual_span_norm_fraction=float((delta-nll_basis.T@(nll_basis@delta)).norm())/norm if norm>0 else 0.,
                        joint_structure_full_pair_rank=joint_rank,nll_residual_input_rank=nll_rank,
                        frozen_prediction_delta_ev=float(response[str(seed),key,r['cif_a'],r['cif_b']]['pred_mean_delta_'+view+'_ev']),observed_delta_ev=float(r['observed_mean_delta_ev']),
                        weight=float(r['weight']))
                    for arm in ('PHASE_NLL','PHASE_PAIR'):
                        state=cp['phase_conditioned_residual'][arm];weight=state['weight'].to(device)
                        with torch.inference_mode():
                            d=2*torch.tanh(torch.nn.functional.linear(differences[vi],weight)/2)
                        base[arm.lower()+'_mean_row_weight_l2']=float(weight[0].norm())
                        base[arm.lower()+'_group_delta_d']=float((average(d,b)-average(d,a))[0])
                    results.append(base)
            print(f'seed={seed} fold={fold} held_pairs={len(selected)}',flush=True)
    if len(results)!=132:raise ValueError('expected 22 pairs x3seeds x2views')
    rt.write_csv(out/'pair_representation.csv',results);rt.write_json(out/'train_pair_geometry.json',geometry)
    summaries=[]
    for seed in (17,29,41):
        for view in rt.VIEWS[2:]:
            rows=[r for r in results if r['seed']==seed and r['view']==view]
            avg=lambda k:sum(r['weight']*r[k] for r in rows)
            summaries.append(dict(seed=seed,view=view,pairs=len(rows),formulas=len({r['composition_key'] for r in rows}),
                exact_raw_collapsed_pairs=sum(r['raw_structure_l2']==0 for r in rows),
                exact_token_collapsed_pairs=sum(r['structure_token_l2']==0 for r in rows),
                exact_hidden_collapsed_pairs=sum(r['hidden_pair_l2']==0 for r in rows),
                weighted_hidden_l2=avg('hidden_pair_l2'),weighted_outside_train_span_fraction=avg('outside_train_pair_span_norm_fraction'),
                weighted_outside_joint_pair_span_fraction=avg('outside_joint_pair_span_norm_fraction'),
                weighted_outside_nll_residual_span_fraction=avg('outside_nll_residual_span_norm_fraction'),
                same_sg_pairs=sum(r['same_sg'] for r in rows)))
    rt.write_csv(out/'representation_summary.csv',summaries)
    rt.write_json(out/'protocol.json',dict(new_fits=0,device=str(device),source_sha256=rt.sha(Path(__file__)),
        input_binding=bindings,pairs=22,formulas=18,seeds=[17,29,41],
        geometry='conditional hierarchical report-weight means within CIF; per-view and jointly structure/full TRAIN pair row spans plus all TRAIN NLL residual-input row span; geometry uses no target',
        scope='held-formula outer pairs only; frozen FROZEN 64-hidden representation; no changes to data or weights',
        interpretation='nonzero distances reject exact collapse but do not prove useful phase information; outside-span geometry is descriptive, not a learnability or irreducible-error estimate',
        output_binding={p.name:rt.sha(p) for p in out.iterdir() if p.is_file() and p.name!='protocol.json'}))
    print(summaries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError('work outside formal directory')
    run(a.root,a.out,torch.device(a.device))
