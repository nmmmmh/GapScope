"""One source-verified SnS probe of saved grouped-study models; no fitting."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from inference import _input_features,_restore_model
from grouped_adaptation_data import chemical_system
from density_score import score

def run(root,work):
    fixed=rt.read_json(work/'fixed_protocol.json');rows=rt.read_json(work/'cohort.json')['records'];freeze=rt.read_json(work/'cohort_freeze.json')
    assert rt.sha(work/'cohort.json')==freeze['cohort_sha256'] and rt.sha(work/'fixed_protocol.json')==freeze['fixed_protocol_sha256']
    out=work/'analysis';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('probe already evaluated')
    for name,digest in fixed['input_binding'].items():assert rt.sha(root/name)==digest
    encoded=rt.read_json(work/'encoded_features.json');assert [r['measurement_id'] for r in rows]==[r['measurement_id'] for r in encoded]
    assert rt.sha(work/'encoded_features.json')==rt.read_json(work/'encoding_protocol.json')['feature_sha256']
    rt.torch.set_num_threads(2);m=rt.modules();inputs=[_input_features(r) for r in encoded]
    features={k:rt.torch.cat([r[k] for r in inputs]) for k in inputs[0]};raw=[];grouped=defaultdict(list);selected=[]
    system=chemical_system(rows[0]['composition_key'])
    for name,digest in fixed['checkpoint_binding'].items():
        path=root/name;assert rt.sha(path)==digest
        cp=rt.torch.load(path,map_location='cpu',weights_only=True)
        if system not in {chemical_system(k) for k in cp['partition']['report_outer_composition_keys']}:continue
        assert system in cp['partition']['aggregate_removed_chemical_systems']
        selected.append(dict(seed=cp['seed'],fold=cp['fold'],checkpoint=name,sha256=digest))
        s=cp['scalers'];views=m['t10'].four_views(features,*(s[k].float() for k in ('composition_mean','composition_std','structure_mean','structure_std')),'cpu')
        for arm,names in fixed['evaluated_views'].items():
            model=_restore_model(m,cp['models'][arm],'M' if arm=='FROZEN_MULTIVIEW' else 'F','cpu')
            for view in names:
                with rt.torch.inference_mode():parts=tuple(t.numpy() for t in model.components(views[rt.VIEWS.index(view)]))
                for i,r in enumerate(rows):
                    p=tuple(t[i] for t in parts);grouped[arm,view,i].append((str(cp['seed']),p))
                    raw.append(dict(model=arm,view=view,seed=cp['seed'],fold=cp['fold'],measurement_id=r['measurement_id'],
                        component_weights=p[0].tolist(),log_mus=p[1].tolist(),sigmas=p[2].tolist()))
    assert len(selected)==3 and {s['seed'] for s in selected}=={17,29,41}
    scored=[];metrics=[]
    for arm,names in fixed['evaluated_views'].items():
        for view in names:
            for mixture in ('equal_density_mixture','17','29','41'):
                rs=[]
                for i,r in enumerate(rows):
                    chunks=[p for seed,p in grouped[arm,view,i] if mixture=='equal_density_mixture' or seed==mixture]
                    parts=tuple(np.concatenate([p[k] for p in chunks]) for k in range(3));parts=(parts[0]/len(chunks),parts[1],parts[2])
                    result=dict(model=arm,view=view,seed_mixture=mixture,measurement_id=r['measurement_id'],target_ev=r['gap_ev'],
                        mean_ev=float(parts[0]@np.exp(parts[1]+.5*parts[2]**2)),**score(r['gap_ev'],parts))
                    scored.append(result);rs.append(result)
                metrics.append(dict(model=arm,view=view,seed_mixture=mixture,reports=2,formulas=1,
                    **{k:sum(r[k] for r in rs)/2 for k in ('nll','crps_ev','median_mae_ev','coverage_80','width_80_ev')}))
    for name,rs in (('densities.csv',raw),('report_scores.csv',scored),('metrics.csv',metrics)):rt.write_csv(out/name,rs)
    rt.write_json(out/'protocol.json',dict(new_fits=0,selected_checkpoints=selected,cohort_sha256=rt.sha(work/'cohort.json'),
        fixed_protocol_sha256=rt.sha(work/'fixed_protocol.json'),encoded_sha256=rt.sha(work/'encoded_features.json'),source_sha256=rt.sha(Path(__file__)),
        scope=fixed['scope'],output_binding={p.name:rt.sha(p) for p in out.iterdir() if p.name!='protocol.json'}))
    print(dict(probe_reports=2,selected_checkpoints=len(selected),predictions=len(raw)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
