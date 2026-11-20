"""Encode a source-verified frozen confirmation cohort without fitting."""
import argparse
import json
import warnings
from pathlib import Path
import runtime as rt
from encode import encode_formula

def run(root,work):
    import torch
    from pymatgen.core import Composition,Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from matgl.models import MEGNet
    from matgl.ext.pymatgen import Structure2Graph
    cohort=rt.read_json(work/'cohort.json');rows=cohort['records']
    if rt.sha(work/'cohort.json')!=rt.read_json(work/'cohort_freeze.json')['cohort_sha256']:raise ValueError('cohort changed')
    frozen=rt.read_json(work/'fixed_protocol.json')
    for name,digest in frozen['input_binding'].items():
        if rt.sha(root/name)!=digest:raise ValueError('frozen input changed: '+name)
    for name,digest in frozen['checkpoint_binding'].items():
        if rt.sha(root/name)!=digest:raise ValueError('model changed after freezing')
    modeldir=root/'models/official/MEGNet-BandGap-mfi-MP-2019.4.1'
    provenance=rt.read_json(modeldir/'PROVENANCE.json')
    for name,digest in provenance['model_binding'].items():
        if rt.sha(modeldir/name)!=digest:raise ValueError('MEGNet changed')
    model=MEGNet(**rt.read_json(modeldir/'model.json')['kwargs'])
    model.load_state_dict(torch.load(modeldir/'state.pt',map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
    converter=Structure2Graph(element_types=model.element_types,cutoff=model.cutoff)
    formula_cache={};cif_cache={};encoded=[];bindings={}
    def key(formula):
        amounts=Composition(formula).get_el_amt_dict();total=sum(amounts.values())
        return '|'.join(f'{e}:{amounts[e]/total:.12f}' for e in sorted(amounts))
    for r in rows:
        formula=r['formula'];composition_key=key(formula)
        if composition_key!=r['composition_key']:raise ValueError('confirmation formula key differs')
        if composition_key not in formula_cache:formula_cache[composition_key]=encode_formula(formula,device='cpu')
        raw=dict(formula_cache[composition_key]);raw['formula']=formula;raw['space_group']=r.get('space_group')
        raw['structure']=None;raw['structure_source']=None;raw['structure_encoding_issue']=None
        if r.get('cif_path'):
            path=work/r['cif_path'];digest=rt.sha(path);bindings[r['cif_path']]=digest
            if digest not in cif_cache:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');structure=Structure.from_file(path)
                    # The frozen encoder accepts element identities, not CIF oxidation labels.
                    structure.remove_oxidation_states()
                    vector=None;sg=None;issue=None
                    if key(structure.composition.formula)!=composition_key:raise ValueError('reference CIF composition mismatch')
                    if not structure.is_ordered:issue='partial/disordered CIF; frozen ordered encoder input unavailable'
                    elif not structure.is_valid():issue='invalid CIF geometry'
                    elif not set(str(e) for e in structure.composition.elements).issubset(model.element_types):issue='element outside frozen MEGNet vocabulary'
                    else:
                        sg=SpacegroupAnalyzer(structure,symprec=.01,angle_tolerance=5).get_space_group_number()
                        captured=[];hook=model.output_proj.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu()))
                        try:
                            with torch.inference_mode():model.predict_structure(structure,state_attr=torch.tensor([0],dtype=torch.long),graph_converter=converter)
                        finally:hook.remove()
                        if len(captured)!=1 or captured[0].numel()!=160:raise ValueError('MEGNet readout shape')
                        vector=captured[0].reshape(-1).tolist()
                cif_cache[digest]=dict(structure=vector,cif_sg=sg,issue=issue,warnings=[str(w.message) for w in caught])
            info=cif_cache[digest];raw['structure']=info['structure'];raw['cif_sha256']=digest
            raw['structure_source']=r.get('structure_relation','same_phase_reference')
            raw['structure_encoding_issue']=info['issue'];raw['cif_warnings']=info['warnings'];raw['cif_space_group']=info['cif_sg']
            if raw['space_group'] is None:
                raw['space_group']=info['cif_sg'];raw['space_group_origin']='matched reference CIF, symprec .01 A'
            else:raw['space_group_origin']='verified source assignment'
        encoded.append(dict(measurement_id=r['measurement_id'],**raw))
    rt.write_json(work/'encoded_features.json',encoded)
    rt.write_json(work/'encoding_protocol.json',dict(new_fits=0,records=len(encoded),formulas=len(formula_cache),cifs=len(cif_cache),
        structure_encoded=sum(r['structure'] is not None for r in encoded),sg_encoded=sum(r['space_group'] is not None for r in encoded),
        cohort_sha256=rt.sha(work/'cohort.json'),feature_sha256=rt.sha(work/'encoded_features.json'),cif_binding=bindings,
        encoder_binding=provenance['model_binding'],source_sha256=rt.sha(Path(__file__)),device='cpu original frozen encoders'))
    print(json.dumps(dict(encoded_reports=len(encoded),formulas=len(formula_cache),encoded_structures=sum(r['structure'] is not None for r in encoded))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
