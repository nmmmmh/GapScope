"""Fresh source/chemical-system grouped folds for the adaptation study."""
import hashlib
import math
import random
from collections import defaultdict
import runtime as rt
import training_data as td

torch=rt.torch
def chemical_system(key):return '|'.join(part.split(':')[0] for part in key.split('|'))

def partition(records):
    keys=sorted({r['composition_key'] for r in records});parent={k:k for k in keys}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def join(a,b):parent[find(b)]=find(a)
    linked=defaultdict(list)
    for r in records:
        linked['source',r['source_group_id']].append(r['composition_key'])
        linked['chemistry',chemical_system(r['composition_key'])].append(r['composition_key'])
    for group in linked.values():
        for key in group[1:]:join(group[0],key)
    groups=defaultdict(set)
    for key in keys:groups[find(key)].add(key)
    groups=sorted(groups.values(),key=lambda g:(-len(g),hashlib.sha256('|'.join(sorted(g)).encode()).hexdigest()))
    bins=[set() for _ in range(3)];component_bins=[[] for _ in range(3)]
    for group in groups:
        slot=min(range(3),key=lambda i:(len(bins[i]),i));bins[slot]|=group;component_bins[slot].append(sorted(group))
    return bins,component_bins

def build_fold(root,work,fold,device):
    reports_dir=work/'inputs';modules=rt.modules()
    raw=torch.load(reports_dir/'features.pt',map_location='cpu',weights_only=True)
    metadata=rt.read_json(reports_dir/'metadata.json');records=td._load_records(reports_dir/'records.jsonl')
    if len(metadata)!=len(records) or len({r['measurement_id'] for r in records})!=len(records):raise ValueError('report identity')
    td._check_features(raw,metadata,'reports')
    for r,m in zip(records,metadata):
        if any(r[k]!=m[k] for k in ('measurement_id','composition_key','formula')):raise ValueError('metadata order')
        if not isinstance(r.get('gap_ev'),(int,float)) or not math.isfinite(r['gap_ev']) or r['gap_ev']<=0:raise ValueError('invalid positive report target')
    records=[dict(m,**r) for m,r in zip(metadata,records)]
    bins,components=partition(records);outer=bins[fold]
    remaining=[g for i,gs in enumerate(components) if i!=fold for g in gs]
    random.Random(20261012+fold).shuffle(remaining)
    target=max(1,len(set().union(*[set(g) for g in remaining]))//4);valid=set()
    for group in remaining:
        if len(valid)>=target:break
        valid.update(group)
    train=set().union(*bins)-outer-valid
    if not train or not valid or not outer:raise ValueError('empty grouped split')
    split_keys={'train':train,'valid':valid,'outer':outer}
    sources={label:{r['source_group_id'] for r in records if r['composition_key'] in ks} for label,ks in split_keys.items()}
    systems={label:{chemical_system(k) for k in ks} for label,ks in split_keys.items()}
    for a,b in (('train','valid'),('train','outer'),('valid','outer')):
        if sources[a]&sources[b] or systems[a]&systems[b]:raise ValueError('group collision')
    aggregate_features_path=root/'data/composition/features.pt'
    aggregate_metadata_path=root/'data/composition/metadata.json'
    aggregate_all=torch.load(aggregate_features_path,map_location='cpu',weights_only=True)
    ameta_all=rt.read_json(aggregate_metadata_path)
    symmetry_completion_provenance=None
    symmetry_completion_path=root/'data/composition/symmetry_completion.json'
    if symmetry_completion_path.is_file():
        from candidate_symmetry_completion import apply_completion
        source_sg_rows=int(aggregate_all['sg_mask'].sum().item())
        source_structure_rows=int(aggregate_all['structure_mask'].sum().item())
        aggregate_all,ameta_all,overlay=apply_completion(
            aggregate_all,ameta_all,rt.read_json(symmetry_completion_path))
        symmetry_completion_provenance=dict(
            completion_file_sha256=rt.sha(symmetry_completion_path),
            base_features_file_sha256=rt.sha(aggregate_features_path),
            completion_payload_sha256=overlay['completion_payload_sha256'],
            patch_rows_applied=overlay['rows_patched'],
            original_sg_supported_rows=source_sg_rows,
            original_structure_supported_rows=source_structure_rows,
            effective_sg_supported_rows=int(aggregate_all['sg_mask'].sum().item()),
            effective_structure_supported_rows=int(aggregate_all['structure_mask'].sum().item()))
    td._check_features(aggregate_all,ameta_all,'aggregate',has_targets=True)
    held_systems=systems['valid']|systems['outer']
    retained=[i for i,r in enumerate(ameta_all) if chemical_system(r['composition_key']) not in held_systems]
    aggregate=td._feature_subset(aggregate_all,retained,len(ameta_all));ameta=[ameta_all[i] for i in retained]
    audit=td._audit_formula_features(aggregate,ameta,raw,metadata)
    ai=[i for i,r in enumerate(ameta) if r['composition_key'] not in train and td._hash_validation_key(r['composition_key'])]
    aset=set(ai);fi=[i for i in range(len(ameta)) if i not in aset]
    if not ai or not fi:raise ValueError('empty aggregate split')
    ids={label:[i for i,r in enumerate(records) if r['composition_key'] in ks] for label,ks in split_keys.items()}
    sym=modules['loader43'].symmetry_table();rh=modules['t35'].transform(raw,'hybrid',sym,set());ah=modules['t35'].transform(aggregate,'hybrid',sym,set())
    index={r['measurement_id']:i for i,r in enumerate(records)}
    cm,cs,sm,ss,stats=modules['t10'].fold_scalers(ah,ameta,fi,rh,metadata,[records[i]['measurement_id'] for i in ids['train']],records,index,device)
    views=modules['t10'].four_views(rh,cm,cs,sm,ss,device);av=modules['t10'].four_views(ah,cm,cs,sm,ss,device)
    scalers=dict(composition_mean=cm,composition_std=cs,structure_mean=sm,structure_std=ss,**stats)
    provenance=dict(fold=fold,folds=3,report_split_rule='same primary source or same element set connects formulas; whole components grouped',
        report_fold_composition_keys=[sorted(b) for b in bins],report_train_composition_keys=sorted(train),
        report_valid_composition_keys=sorted(valid),report_outer_composition_keys=sorted(outer),
        report_counts={label:len(is_) for label,is_ in ids.items()},grouped_components=sum(len(gs) for gs in components),
        report_source_overlap=0,report_chemical_system_overlap=0,aggregate_removed=len(ameta_all)-len(ameta),
        aggregate_removed_chemical_systems=sorted(held_systems),aggregate_counts=dict(retained=len(ameta),fit=len(fi),valid=len(ai)),formula_feature_audit=audit)
    if symmetry_completion_provenance is not None:
        provenance['candidate_symmetry_completion']=symmetry_completion_provenance
    result=dict(views=views,av=av,targets=torch.tensor([r['gap_ev'] for r in records],dtype=torch.float64,device=device),
        ay=aggregate['targets'].reshape(-1).to(device=device,dtype=torch.float64),records=records,report_metadata=metadata,aggregate_metadata=ameta,scalers=scalers,provenance=provenance)
    for label,prefix in (('train','t'),('valid','v'),('outer','o')):
        result[prefix+'i']=torch.tensor(ids[label],dtype=torch.long,device=device);result[prefix+'w']=td.hierarchical_weights(records,ids[label],device)
    result['fi']=torch.tensor(fi,dtype=torch.long,device=device);result['ai']=torch.tensor(ai,dtype=torch.long,device=device)
    return result

