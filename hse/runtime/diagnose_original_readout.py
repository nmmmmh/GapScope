"""Compare the original frozen PBE scalar with all current experimental phase pairs."""
import argparse
from collections import defaultdict
from pathlib import Path
import runtime as rt

def run(root,work):
    from pymatgen.core import Structure
    from matgl.models import MEGNet
    from matgl.ext.pymatgen import Structure2Graph
    rt.torch.set_num_threads(2)
    md=root/'models/official/MEGNet-BandGap-mfi-MP-2019.4.1';provenance=rt.read_json(md/'PROVENANCE.json')
    for name,digest in provenance['model_binding'].items():assert rt.sha(md/name)==digest
    model=MEGNet(**rt.read_json(md/'model.json')['kwargs'])
    model.load_state_dict(rt.torch.load(md/'state.pt',map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
    converter=Structure2Graph(element_types=model.element_types,cutoff=model.cutoff)
    inputs=root/'results/physical_cases/resolved_structure_index.csv';grouped=defaultdict(list)
    for r in rt.read_csv(inputs):grouped[r['actual_cif_sha256']].append(r)
    output=[];binding={}
    for digest,rs in sorted(grouped.items()):
        path=root/rs[0]['cif_path'];assert rt.sha(path)==digest;binding[rs[0]['cif_path']]=digest
        structure=Structure.from_file(path)
        with rt.torch.inference_mode():value=model.predict_structure(structure,state_attr=rt.torch.tensor([0],dtype=rt.torch.long),graph_converter=converter)
        value=rt.torch.as_tensor(value);assert value.numel()==1 and rt.torch.isfinite(value).all()
        output.append(dict(cif_sha256=digest,formula=rs[0]['formula'],cif_path=rs[0]['cif_path'],reports=len(rs),pbe_gap_ev=float(value.item())))
    lookup={r['cif_sha256']:r['pbe_gap_ev'] for r in output};pairs=[]
    for r in rt.read_csv(work/'analysis/pair_response.csv'):
        if r['seed_mixture']!='equal_density_mixture':continue
        delta=lookup[r['cif_b']]-lookup[r['cif_a']];observed=float(r['observed_delta_ev'])
        pairs.append(dict(composition_key=r['composition_key'],formula=r['formula'],cif_a=r['cif_a'],cif_b=r['cif_b'],
            observed_delta_ev=observed,pbe_delta_ev=delta,pbe_delta_error_ev=abs(delta-observed),zero_delta_error_ev=abs(observed),weight=float(r['weight'])))
    out=work/'readout_diagnostic';out.mkdir(exist_ok=True)
    rt.write_csv(out/'scalar_predictions.csv',output);rt.write_csv(out/'pair_baseline.csv',pairs)
    weight=sum(r['weight'] for r in pairs);assert len(output)==52 and len(pairs)==22 and abs(weight-1)<1e-6
    rt.write_json(out/'summary.json',dict(status='complete',new_fits=0,cif_predictions=len(output),mapped_reports=sum(r['reports'] for r in output),pairs=len(pairs),
        pbe_delta_mae_ev=sum(r['weight']*r['pbe_delta_error_ev'] for r in pairs),zero_delta_mae_ev=sum(r['weight']*r['zero_delta_error_ev'] for r in pairs),
        mean_abs_pbe_delta_ev=sum(r['weight']*abs(r['pbe_delta_ev']) for r in pairs),direction_agreement=sum(r['weight']*(r['pbe_delta_ev']*r['observed_delta_ev']>0) for r in pairs),
        encoder_binding=provenance['model_binding'],cif_binding=binding,index_sha256=rt.sha(inputs),pair_input_sha256=rt.sha(work/'analysis/pair_response.csv'),source_sha256=rt.sha(Path(__file__)),
        scalar_predictions_sha256=rt.sha(out/'scalar_predictions.csv'),pair_baseline_sha256=rt.sha(out/'pair_baseline.csv'),
        scope='unchanged original MEGNet PBE scalar, no experimental fitting; descriptive readout diagnostic, not a replacement candidate',
        execution_note='Two prior wrapper calls failed before any inference: incorrect 60 vs96 map count. Root corrected scope and b-minus-a direction; one completed 52-CIF CPU pass.'))
    (out/'README.md').write_text('# 原结构标量诊断\n\n全52CIF原MEGNet state0(PBE)标量与全部18式22对实验差作参照。差值方向统一b−a，沿用现有按式/对权重。没有拟合或新选择；PBE计算值不等同实验gap。先前封装错误在推理前退出，root修正全96结构报告范围与差值方向后一次完成。\n',encoding='utf-8')
    print({k:v for k,v in rt.read_json(out/'summary.json').items() if k in ('status','cif_predictions','pairs','pbe_delta_mae_ev','zero_delta_mae_ev')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.root,a.work)
