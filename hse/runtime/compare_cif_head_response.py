"""Compare both fitted density heads on the same complete CIF-pair inventory."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def run(root, analysis, interaction, out, model='SINGLE'):
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve().is_relative_to(root.resolve()):
        raise ValueError('output must be outside formal root')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metas = json.loads((root/'data/reports/metadata.json').read_text(encoding='utf-8-sig'))
    scores = [r for r in read(analysis/'report_scores.csv') if r['model']==model]
    raw = [r for r in read(analysis/'oof_densities.csv') if r['model']==model]
    index = {(r['seed_mixture'],r['view'],r['measurement_id']):r for r in scores}
    densities = defaultdict(list)
    for r in raw: densities[r['measurement_id'],r['view']].append(r)
    ids = defaultdict(list)
    for meta in metas:
        rid = meta['measurement_id']
        s = index['equal_density_mixture','full',rid]
        if int(s['structure_supported']):
            path = meta.get('cif_path') or meta['model_sg_reference_cif_path']
            ids[s['composition_key'],sha(root/path)].append(rid)
    old_groups = [r for r in read(interaction/'cif_group_response.csv') if r.get('model','FROZEN')=='FROZEN']
    single_groups = []
    views = ('formula_only','formula_sg','formula_structure','full')
    for original in old_groups:
        r = dict(original,model=model); mix=r['seed_mixture']
        members = ids[r['composition_key'],r['cif_sha256']]
        mass = sum(float(index[mix,'full',rid]['weight']) for rid in members)
        for view in views:
            means=[]; medians=[]
            for rid in members:
                ds=densities[rid,view]
                if mix!='equal_density_mixture': ds=[d for d in ds if d['seed']==mix]
                mean=sum(sum(w*math.exp(mu+.5*sd*sd) for w,mu,sd in zip(
                    json.loads(d['component_weights']),json.loads(d['log_mus']),json.loads(d['sigmas']))) for d in ds)/len(ds)
                weight=float(index[mix,view,rid]['weight'])/mass
                means.append((weight,mean)); medians.append((weight,float(index[mix,view,rid]['median_ev'])))
            r['pred_mean_'+view+'_ev']=sum(w*x for w,x in means)
            r['pred_median_'+view+'_ev']=sum(w*x for w,x in medians)
            r['within_cif_prediction_span_'+view+'_ev']=max(x for w,x in means)-min(x for w,x in means)
        r['same_cif_input_consistent']=max(r['within_cif_prediction_span_'+v+'_ev'] for v in views)<1e-4
        single_groups.append(r)
    groups={(r['seed_mixture'],r['composition_key'],r['cif_sha256']):r for r in single_groups}
    old_pairs=[r for r in read(interaction/'same_formula_cif_pairs.csv') if r.get('model','FROZEN')=='FROZEN']; single_pairs=[]
    for original in old_pairs:
        r=dict(original,model=model); key=(r['seed_mixture'],r['composition_key'])
        a,b=groups[key+(r['cif_a'],)],groups[key+(r['cif_b'],)]
        for view in views:
            for statistic in ('mean','median'):
                r['pred_'+statistic+'_delta_'+view+'_ev']=b['pred_'+statistic+'_'+view+'_ev']-a['pred_'+statistic+'_'+view+'_ev']
            r['mean_delta_error_'+view+'_ev']=abs(r['pred_mean_delta_'+view+'_ev']-float(r['observed_mean_delta_ev']))
        r['structure_conditional_mean_delta_ev']=r['pred_mean_delta_full_ev']-r['pred_mean_delta_formula_sg_ev']
        single_pairs.append(r)
    summaries=[]
    for mix in ('equal_density_mixture','17','29','41'):
        rs=[r for r in single_pairs if r['seed_mixture']==mix]
        avg=lambda k:sum(float(r['weight'])*float(r[k]) for r in rs)
        summaries.append(dict(model=model,seed_mixture=mix,formulas=len({r['composition_key'] for r in rs}),pairs=len(rs),
            **{'mean_delta_error_'+v+'_ev':avg('mean_delta_error_'+v+'_ev') for v in views},
            full_direction_agreement=sum(float(r['weight'])*float(r['pred_mean_delta_full_ev']*float(r['observed_mean_delta_ev'])>0) for r in rs),
            max_formula_mean_delta_ev=max(abs(r['pred_mean_delta_formula_only_ev']) for r in rs),weighting='equal formula then equal within-formula CIF pair'))
    for name,old,new in [('cif_group_response.csv',old_groups,single_groups),('same_formula_cif_pairs.csv',old_pairs,single_pairs),
                         ('same_formula_pair_summary.csv',[r for r in read(interaction/'same_formula_pair_summary.csv') if r.get('model','FROZEN')=='FROZEN'],summaries)]:
        write(out/name,[dict(r,model='FROZEN') for r in old]+new)
    protocol=dict(fits=0,model=model,selection='same complete 18-formula / 22-CIF-pair inventory; no new routing rule',
        source_sha256=sha(Path(__file__)),input_binding={str(p):sha(p) for p in [analysis/'report_scores.csv',analysis/'oof_densities.csv',
            interaction/'cif_group_response.csv',interaction/'same_formula_cif_pairs.csv',root/'data/reports/metadata.json']})
    (out/'head_response_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(summaries[0]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--interaction-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',default='SINGLE')
    a=p.parse_args();run(a.root,a.analysis_dir,a.interaction_dir,a.output_dir,a.model)
