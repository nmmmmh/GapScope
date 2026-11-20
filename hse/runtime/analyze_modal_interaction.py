"""OOF density interaction, pretrained attention routing and same-formula CIF response."""
import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr
from pymatgen.core import Structure
import runtime as rt
from analyze_frozen import _parts, _read_targets
from density_score import score
from inference import _restore_model
from training_data import build_fold

torch=rt.torch
MIXTURES=('equal_density_mixture','17','29','41')

def average(rows,key):
    mass=sum(float(r['weight']) for r in rows)
    return sum(float(r['weight'])*float(r[key]) for r in rows)/mass

def cdf(parts,x):
    w,mu,sd=parts
    return ndtr((np.log(x)[:,None]-mu)/sd)@w

def tails(parts,low,high):
    w,mu,sd=parts; means=np.exp(mu+.5*sd*sd)
    below=low*ndtr((math.log(low)-mu)/sd)-means*ndtr((math.log(low)-mu-sd*sd)/sd)
    above=means*ndtr((mu+sd*sd-math.log(high))/sd)-high*ndtr((mu-math.log(high))/sd)
    return float(w@(np.maximum(below,0)+np.maximum(above,0)))

def density_effects(parts):
    lower=min(np.min(mu-9*sd) for w,mu,sd in parts)
    upper=max(np.max(mu+9*sd) for w,mu,sd in parts)
    values=[]
    for size in (2049,4097):
        x=np.exp(np.linspace(lower,upper,size)); fs=[cdf(p,x) for p in parts]
        f,sg,st,full=fs
        values.append(dict(cdf_interaction_l1_ev=float(np.trapz(abs(full-sg-st+f),x)),
            sg_w1_given_structure_ev=float(np.trapz(abs(full-st),x)),
            structure_w1_given_sg_ev=float(np.trapz(abs(full-sg),x)),
            total_modal_w1_ev=float(np.trapz(abs(full-f),x))))
    result=values[-1]
    result['max_grid_refinement_change_ev']=max(abs(values[0][k]-result[k]) for k in result)
    result['cdf_interaction_tail_upper_bound_ev']=sum(tails(p,math.exp(lower),math.exp(upper)) for p in parts)
    return result

def describe_cif(root,meta):
    path=root/meta['cif_path']; cif_hash=meta['cif_sha256']
    if rt.sha(path)!=cif_hash: raise ValueError('CIF hash mismatch')
    s=Structure.from_file(path)
    neighbors=s.get_all_neighbors(6.)
    nearest=[]; hetero=[]; coordination=[]; per_element=defaultdict(list)
    for site,near in zip(s,neighbors):
        distances=[float(n.nn_distance) for n in near if n.nn_distance>1e-6]
        if not distances: raise ValueError(f'no periodic neighbor within 6 A:{path}')
        nearest.append(min(distances))
        different=[float(n.nn_distance) for n in near if n.nn_distance>1e-6
                   and {e.symbol for e in n.species}!={e.symbol for e in site.species}]
        if different:
            shortest=min(different); count=sum(d<=1.2*shortest for d in different)
            hetero.append(shortest); coordination.append(count)
            for element in site.species: per_element[element.symbol].append((shortest,count))
    return dict(cif_sha256=cif_hash,cif_path=meta['cif_path'],sites=len(s),ordered=bool(s.is_ordered),
        volume_per_atom_a3=float(s.volume/float(s.composition.num_atoms)),density_g_cm3=float(s.density),
        nearest_site_distance_a=float(np.mean(nearest)),
        nearest_unlike_species_distance_a=float(np.mean(hetero)) if hetero else '',
        unlike_species_neighbor_proxy=float(np.mean(coordination)) if coordination else '',
        per_element_neighbor_proxy_json=json.dumps({e:dict(nearest_distance_a=float(np.mean([p[0] for p in ps])),
            count_within_1p2_nearest=float(np.mean([p[1] for p in ps]))) for e,ps in per_element.items()},sort_keys=True))

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    records=_read_targets(root); by_id={r['measurement_id']:r for r in records}
    metas=rt.read_json(root/'data/reports/metadata.json'); metadata={r['measurement_id']:dict(r) for r in metas}
    for meta in metadata.values():
        path=meta.get('cif_path') or meta.get('model_sg_reference_cif_path')
        if meta.get('structure_encoded') and path:
            digest=rt.sha(root/path)
            reference=meta.get('cif_sha256') or meta.get('model_sg_reference_cif_sha256') or digest
            if digest!=reference:raise ValueError(f'CIF content differs from reference:{path}')
            meta['cif_path']=path;meta['cif_sha256']=digest
    score_path=root/'results/formula_and_modalities/report_scores.csv'
    density_path=root/'results/formula_and_modalities/oof_densities.csv'
    scores=[r for r in rt.read_csv(score_path) if r['model']=='FROZEN']
    raw=[r for r in rt.read_csv(density_path) if r['model']=='FROZEN']
    indexed={(r['seed_mixture'],r['view'],r['measurement_id']):r for r in scores}
    if len(indexed)!=4*4*205 or len(raw)!=3*4*205: raise ValueError('FROZEN OOF incomplete')
    densities=defaultdict(list)
    for r in raw: densities[(r['measurement_id'],r['view'])].append(r)
    modules=rt.modules(); attention=[]; checkpoints={}
    for seed in (17,29,41):
        for fold in range(5):
            path=root/'models/checkpoints'/f'seed_{seed}_fold_{fold+1:02d}'/'models.pt'
            checkpoints[path.relative_to(root).as_posix()]=rt.sha(path)
            cp=torch.load(path,map_location='cpu',weights_only=True)
            d=build_fold(root,fold,'cpu')
            if cp['partition']!=d['provenance']: raise ValueError('partition mismatch')
            model=_restore_model(modules,cp['models']['FROZEN'],'M','cpu')
            with torch.inference_mode():
                _,weights=model.encode(d['views'][:,d['oi']],return_attention=True)
            weights=weights.reshape(4,len(d['oi']),4,3).cpu().numpy()
            for vi,view in enumerate(rt.VIEWS):
                for j,i in enumerate(d['oi'].tolist()):
                    r=d['records'][i]
                    for h in range(4):
                        attention.append(dict(seed=seed,fold_id=fold,view=view,head=h,
                            measurement_id=r['measurement_id'],composition_key=r['composition_key'],
                            formula=r['formula'],attention_formula=float(weights[vi,j,h,0]),
                            attention_sg=float(weights[vi,j,h,1]),attention_structure=float(weights[vi,j,h,2])))
    att=defaultdict(list)
    for r in attention: att[(str(r['seed']),r['view'],r['measurement_id'])].append(r)
    effects=[]
    for rid,r in by_id.items():
        for mixture in MIXTURES:
            rows=[indexed[(mixture,v,rid)] for v in rt.VIEWS]
            parts=[_parts(densities[(rid,v)] if mixture==MIXTURES[0] else
                          [x for x in densities[(rid,v)] if str(x['seed'])==mixture]) for v in rt.VIEWS]
            result=dict(measurement_id=rid,composition_key=r['composition_key'],formula=r['formula'],
                phase_label=r.get('phase_label',''),seed_mixture=mixture,weight=float(rows[0]['weight']),
                sg_supported=int(rows[0]['sg_supported']),structure_supported=int(rows[0]['structure_supported']),
                cif_sha256=metadata[rid].get('cif_sha256',''))
            for metric in ('nll','crps_ev'):
                f,sg,st,full=[float(x[metric]) for x in rows]
                result['sg_shapley_'+metric]=.5*(f-sg+st-full)
                result['structure_shapley_'+metric]=.5*(f-st+sg-full)
                result['synergy_'+metric]=sg+st-f-full
                result['total_modal_gain_'+metric]=f-full
                if abs(result['sg_shapley_'+metric]+result['structure_shapley_'+metric]-(f-full))>1e-10:
                    raise ValueError('Shapley identity mismatch')
            result.update(density_effects(parts))
            chosen=[a for seed in ('17','29','41') if mixture==MIXTURES[0] or mixture==seed
                    for a in att[(seed,'full',rid)]]
            for key in ('attention_formula','attention_sg','attention_structure'):
                result[key]=float(np.mean([a[key] for a in chosen]))
            effects.append(result)
    rt.write_csv(out/'report_interactions.csv',effects); rt.write_csv(out/'attention_heads.csv',attention)
    metrics=[k for k in effects[0] if k.startswith(('sg_shapley_','structure_shapley_','synergy_','total_modal_gain_'))]
    metrics+=['cdf_interaction_l1_ev','sg_w1_given_structure_ev','structure_w1_given_sg_ev','total_modal_w1_ev']
    summaries=[]
    subsets={'all':lambda r:True,'structure':lambda r:r['structure_supported'],
             'sg_only':lambda r:r['sg_supported'] and not r['structure_supported'],
             'neither':lambda r:not r['sg_supported'] and not r['structure_supported']}
    for mixture in MIXTURES:
        for subset,select in subsets.items():
            rs=[r for r in effects if r['seed_mixture']==mixture and select(r)]
            summaries.append(dict(seed_mixture=mixture,subset=subset,reports=len(rs),
                formulas=len({r['composition_key'] for r in rs}),**{k:average(rs,k) for k in metrics}))
    rt.write_csv(out/'interaction_summary.csv',summaries)
    correl=[]
    for mixture in MIXTURES:
        grouping=defaultdict(list)
        for r in effects:
            if r['seed_mixture']==mixture and r['structure_supported']: grouping[r['composition_key']].append(r)
        for modality,x,y in [('sg','attention_sg','sg_w1_given_structure_ev'),
                             ('structure','attention_structure','structure_w1_given_sg_ev')]:
            a=[average(rs,x) for rs in grouping.values()]; b=[average(rs,y) for rs in grouping.values()]
            correl.append(dict(seed_mixture=mixture,modality=modality,formulas=len(a),
                descriptive_spearman=float(spearmanr(a,b).statistic),unit='within-support hierarchical report mean per formula; no p-value'))
    rt.write_csv(out/'attention_effect_correlations.csv',correl)
    cif_groups=defaultdict(list)
    for rid in by_id:
        if int(indexed[(MIXTURES[0],'full',rid)]['structure_supported']):
            cif_groups[(by_id[rid]['composition_key'],metadata[rid]['cif_sha256'])].append(rid)
    descriptors={h:describe_cif(root,metadata[ids[0]]) for (_,h),ids in cif_groups.items()}
    rt.write_csv(out/'structure_descriptors.csv',list(descriptors.values()))
    groups=[]
    for (key,cif_hash),ids in cif_groups.items():
        original=[by_id[rid] for rid in ids]
        base=dict(composition_key=key,formula=original[0]['formula'],cif_sha256=cif_hash,reports=len(ids),
            phase_labels_json=json.dumps(sorted({r.get('phase_label','') for r in original}),ensure_ascii=False),
            source_groups_json=json.dumps(sorted({r['source_group_id'] for r in original})),
            model_space_group=metadata[ids[0]].get('model_space_group_number'),
            structure_relation_json=json.dumps(sorted({metadata[rid].get('structure_relation','') for rid in ids})))
        ws=np.array([float(indexed[(MIXTURES[0],'formula_only',rid)]['weight']) for rid in ids]);ws/=ws.sum()
        base['observed_mean_ev']=float(ws@np.array([float(r['gap_ev']) for r in original]))
        for mixture in MIXTURES:
            entry=base|dict(seed_mixture=mixture)
            for view in rt.VIEWS:
                ps=[_parts(densities[(rid,view)] if mixture==MIXTURES[0] else
                    [x for x in densities[(rid,view)] if str(x['seed'])==mixture]) for rid in ids]
                means=[float(w@np.exp(mu+.5*sd*sd)) for w,mu,sd in ps]
                medians=[float(indexed[(mixture,view,rid)]['median_ev']) for rid in ids]
                entry['pred_mean_'+view+'_ev']=float(ws@means)
                entry['pred_median_'+view+'_ev']=float(ws@medians)
                entry['within_cif_prediction_span_'+view+'_ev']=max(means)-min(means)
            groups.append(entry)
    for g in groups:
        g['same_cif_input_consistent']=max(g['within_cif_prediction_span_'+v+'_ev'] for v in rt.VIEWS)<1e-4
    rt.write_csv(out/'cif_group_response.csv',groups)
    by_formula=defaultdict(list)
    for g in groups: by_formula[(g['seed_mixture'],g['composition_key'])].append(g)
    pairs=[]
    for (mixture,key),gs in by_formula.items():
        for a,b in itertools.combinations(sorted(gs,key=lambda x:x['cif_sha256']),2):
            row=dict(seed_mixture=mixture,composition_key=key,formula=a['formula'],
                cif_a=a['cif_sha256'],cif_b=b['cif_sha256'],phase_a=a['phase_labels_json'],phase_b=b['phase_labels_json'],
                sg_a=a['model_space_group'],sg_b=b['model_space_group'],
                shared_source=bool(set(json.loads(a['source_groups_json']))&set(json.loads(b['source_groups_json']))),
                observed_mean_delta_ev=b['observed_mean_ev']-a['observed_mean_ev'])
            for view in rt.VIEWS:
                for statistic in ('mean','median'):
                    row['pred_'+statistic+'_delta_'+view+'_ev']=b['pred_'+statistic+'_'+view+'_ev']-a['pred_'+statistic+'_'+view+'_ev']
                row['mean_delta_error_'+view+'_ev']=abs(row['pred_mean_delta_'+view+'_ev']-row['observed_mean_delta_ev'])
            row['structure_conditional_mean_delta_ev']=row['pred_mean_delta_full_ev']-row['pred_mean_delta_formula_sg_ev']
            for k in ('volume_per_atom_a3','nearest_site_distance_a','nearest_unlike_species_distance_a','unlike_species_neighbor_proxy'):
                x,y=descriptors[a['cif_sha256']][k],descriptors[b['cif_sha256']][k]
                row['delta_'+k]=float(y)-float(x) if x!='' and y!='' else ''
            pairs.append(row)
    pair_summary=[]
    for mixture in MIXTURES:
        rs=[r for r in pairs if r['seed_mixture']==mixture]; counts=defaultdict(int)
        for r in rs: counts[r['composition_key']]+=1
        for r in rs: r['weight']=1/len(counts)/counts[r['composition_key']]
        pair_summary.append(dict(seed_mixture=mixture,formulas=len(counts),pairs=len(rs),
            **{k:average(rs,k) for k in rs[0] if k.startswith('mean_delta_error_')},
            full_direction_agreement=average([r|dict(direction=float(r['pred_mean_delta_full_ev']*r['observed_mean_delta_ev']>0)) for r in rs],'direction'),
            max_formula_mean_delta_ev=max(abs(r['pred_mean_delta_formula_only_ev']) for r in rs),
            weighting='equal formula then equal within-formula CIF pair'))
    rt.write_csv(out/'same_formula_cif_pairs.csv',pairs)
    rt.write_csv(out/'same_formula_pair_summary.csv',pair_summary)
    rt.write_json(out/'protocol.json',dict(fits=0,model='current FROZEN held-formula OOF only',
        input_binding={p.relative_to(root).as_posix():rt.sha(p) for p in
            (root/'data/reports/metadata.json',root/'data/reports/records.jsonl',score_path,density_path)},
        checkpoint_binding=checkpoints,source_sha256=rt.sha(Path(__file__)),
        attention='pretrained frozen cross attention, four heads, one formula query, three keys; direct modality tokens also enter fusion',
        interaction='Shapley score allocation and signed four-CDF nonadditivity; neither implies physical causality',
        cif_pair='all encoded same-formula distinct CIF reference pairs; original source phase text retained; no controlled experimental claim',
        neighbor_proxy='periodic neighbors within 6 A; count unlike-species neighbors within 1.2 * nearest unlike-species distance, not assigned bonds',
        integral_grid='2049 vs 4097 equally spaced log points, min(mu-9sigma) to max(mu+9sigma), integral over Eg=exp(loggrid)',
        max_grid_refinement_change_ev=max(r['max_grid_refinement_change_ev'] for r in effects),
        max_cdf_interaction_tail_upper_bound_ev=max(r['cdf_interaction_tail_upper_bound_ev'] for r in effects),
        same_cif_input_inconsistent_groups=sum(not g['same_cif_input_consistent'] for g in groups),
        output_binding={p.name:rt.sha(p) for p in out.glob('*.csv')}))
    print(json.dumps(dict(reports=205,cif_pairs=len(pairs)//4,pair_formulas=pair_summary[0]['formulas'],
        supported_cifs=len(descriptors),fits=0)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output_dir.resolve()
    if out.is_relative_to(root):raise ValueError('output must be outside formal root')
    torch.set_num_threads(2);run(root,out)

if __name__=='__main__':main()
