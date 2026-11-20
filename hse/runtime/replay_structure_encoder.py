"""Replay all 52 current encoded CIFs with the original frozen MEGNet."""
import argparse
import importlib.metadata
import json
import warnings
from collections import defaultdict
from pathlib import Path
import torch
from matgl.models import MEGNet
from matgl.ext.pymatgen import Structure2Graph
from pymatgen.core import Structure
import runtime as rt

def run(root,out):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    modeldir=root/'models/official/MEGNet-BandGap-mfi-MP-2019.4.1'
    expected={'model.json':'8022a635f11d585284b84d373253f06e14181faf15ecc9c66c8fea4acfa9555e',
              'state.pt':'2ab942707f0b4a4d03cabadd2f2f5e955555bf7e2d3aadfb7bbaece72099f40b'}
    for name,digest in expected.items():
        if rt.sha(modeldir/name)!=digest:raise ValueError('original frozen encoder hash differs')
    config=rt.read_json(modeldir/'model.json');model=MEGNet(**config['kwargs'])
    model.load_state_dict(torch.load(modeldir/'state.pt',map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
    converter=Structure2Graph(element_types=model.element_types,cutoff=model.cutoff)
    features=torch.load(root/'data/reports/features.pt',map_location='cpu',weights_only=True)
    metadata=rt.read_json(root/'data/reports/metadata.json');indices={r['measurement_id']:i for i,r in enumerate(metadata)}
    grouped=defaultdict(list)
    for r in rt.read_csv(root/'results/physical_cases/resolved_structure_index.csv'):grouped[r['actual_cif_sha256']].append(r)
    captured=[];hook=model.output_proj.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu()))
    results=[];bindings={}
    try:
        for digest,rows in sorted(grouped.items()):
            path=root/rows[0]['cif_path']
            if rt.sha(path)!=digest:raise ValueError('CIF mapping hash differs')
            bindings[rows[0]['cif_path']]=digest
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');structure=Structure.from_file(path)
                captured.clear()
                with torch.inference_mode():model.predict_structure(structure,state_attr=torch.tensor([0],dtype=torch.long),graph_converter=converter)
            if len(captured)!=1:raise ValueError('encoder hook count')
            vector=captured[0].reshape(-1)
            if vector.shape!=(160,) or not torch.isfinite(vector).all():raise ValueError('structure vector shape')
            differences=[float((vector-features['structure'][indices[r['measurement_id']]]).abs().max()) for r in rows]
            results.append(dict(cif_sha256=digest,cif_path=rows[0]['cif_path'],formula=rows[0]['formula'],reports=len(rows),
                max_component_difference=max(differences),warnings=json.dumps([str(w.message) for w in caught]),
                sites=len(structure),is_ordered=structure.is_ordered,valid_structure=structure.is_valid()))
            if len(results)%10==0:print(f'replayed CIFs={len(results)}',flush=True)
    finally:hook.remove()
    maximum=max(r['max_component_difference'] for r in results)
    rt.write_csv(out/'structure_encoder_replay.csv',results)
    protocol=dict(new_fits=0,cifs=len(results),reports=sum(r['reports'] for r in results),maximum_component_difference=maximum,
        matches_current_vectors=maximum<=1e-4,comparison_tolerance=1e-4,device='cpu',
        encoder='unchanged MEGNet-BandGap-mfi-MP-2019.4.1',state_attr=0,fidelity='PBE',cutoff_angstrom=float(model.cutoff),
        readout='160 inputs captured by output_proj.forward_pre_hook',model_binding=expected,cif_binding=bindings,
        input_binding={p.relative_to(root).as_posix():rt.sha(p) for p in [root/'data/reports/features.pt',root/'data/reports/metadata.json',root/'results/physical_cases/resolved_structure_index.csv']},
        versions={n:importlib.metadata.version(n) for n in ('torch','matgl','dgl','pymatgen')},source_sha256=rt.sha(Path(__file__)),
        output_binding={'structure_encoder_replay.csv':rt.sha(out/'structure_encoder_replay.csv')})
    rt.write_json(out/'encoder_replay_protocol.json',protocol);print(json.dumps({k:protocol[k] for k in ('cifs','reports','maximum_component_difference','matches_current_vectors')}))
    if not protocol['matches_current_vectors']:raise ValueError('cached encoder vectors differ from replay; inspect before changing inputs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError('replay outside formal root')
    run(a.root,a.out)
