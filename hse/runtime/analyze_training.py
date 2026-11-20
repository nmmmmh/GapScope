"""Score the single completed current-data run, keeping comparisons paired."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import runtime as rt
from density_score import score
from training_data import hierarchical_weights

METRICS = ['nll', 'crps_ev', 'median_mae_ev', 'coverage_50', 'width_50_ev',
           'coverage_80', 'width_80_ev', 'coverage_90', 'width_90_ev']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    protocol = rt.read_json(args.work_dir/'protocol.json'); expected_seeds = protocol['seeds']
    records = [json.loads(line) for line in (args.root/'data/reports/records.jsonl').read_text(encoding='utf-8').splitlines() if line]
    by_id = {r['measurement_id']: r for r in records}
    weights = hierarchical_weights(records, list(range(len(records))), 'cpu').tolist()
    weight_by_id = {r['measurement_id']: w for r,w in zip(records,weights)}
    grouped = defaultdict(list); raw = []; fits = []; diagnostics=[]
    for seed in expected_seeds:
        for fold in range(5):
            path=args.work_dir/f'seed_{seed}_fold_{fold+1:02d}'
            if not (path/'complete.json').exists(): raise ValueError(f'Incomplete run: {path.name}')
            rows = rt.read_csv(path/'predictions.csv'); raw += rows
            for r in rows: grouped[(r['model'],r['view'],r['measurement_id'])].append(r)
            for arm,fit in rt.read_json(path/'fits.json').items():
                fits.append(dict(seed=seed,fold=fold,model=arm,**{k:v for k,v in fit.items() if k not in ('history','cdf_history')}))
            diagnostics += [dict(seed=seed,fold=fold,**r) for r in rt.read_csv(path/'teacher_signal.csv')]
    scores=[]
    for (model,view,rid),rows in sorted(grouped.items()):
        if sorted(int(r['seed']) for r in rows)!=sorted(expected_seeds): raise ValueError('Missing or duplicate seed')
        r=rows[0]; y=float(by_id[rid]['gap_ev'])
        for mixture, subset in [('equal_density_mixture',rows)]+[(str(q['seed']),[q]) for q in rows]:
            parts=tuple(np.concatenate([np.asarray(json.loads(q[k])) for q in subset]) for k in ('component_weights','log_mus','sigmas'))
            parts=(parts[0]/len(subset),parts[1],parts[2])
            scores.append(dict(model=model,view=view,seed_mixture=mixture,measurement_id=rid,
                formula=r['formula'],composition_key=r['composition_key'],phase_label=r['phase_label'],
                sg_supported=int(r['sg_supported']),structure_supported=int(r['structure_supported']),
                target_ev=y,weight=weight_by_id[rid],**score(y,parts)))
    score_groups=defaultdict(list)
    for r in scores: score_groups[(r['model'],r['view'],r['seed_mixture'])].append(r)
    metrics=[]
    for (model,view,mixture),rows in score_groups.items():
        if len(rows)!=len(records): raise ValueError('OOF report coverage incomplete')
        for subset,filtered in [('all',rows),('structure',[r for r in rows if r['structure_supported']]),
                                 ('missing_structure',[r for r in rows if not r['structure_supported']])]:
            if not filtered: continue
            w=np.array([r['weight'] for r in filtered]); w/=w.sum()
            metrics.append(dict(model=model,view=view,seed_mixture=mixture,subset=subset,reports=len(filtered),
                formulas=len({r['composition_key'] for r in filtered}),
                **{k:float(w@np.array([r[k] for r in filtered])) for k in METRICS}))
    definitions={
        'teacher_structure_at_inference':(('J','formula_sg'),('J','full')),
        'teacher_total_modal_at_inference':(('J','formula_only'),('J','full')),
        'transfer_to_formula':(('F','formula_only'),('ANCHORED','formula_only')),
        'structure_training_to_formula':(('J_SG','formula_only'),('J','formula_only')),
        'structure_training_full':(('J_SG','formula_sg'),('J','full')),
        'structure_training_missing_structure':(('J_SG','formula_sg'),('J','formula_sg')),
        'structure_training_missing_sg':(('J_SG','formula_only'),('J','formula_structure'))}
    if any(key[0] == 'FROZEN' for key in score_groups):
        definitions.update({
            'frozen_vs_joint_formula':(('J','formula_only'),('FROZEN','formula_only')),
            'frozen_vs_joint_full':(('J','full'),('FROZEN','full')),
            'frozen_structure_at_inference':(('FROZEN','formula_sg'),('FROZEN','full')),
            'frozen_total_modal_at_inference':(('FROZEN','formula_only'),('FROZEN','full')),
            'frozen_formula_vs_F':(('F','formula_only'),('FROZEN','formula_only'))})
    contrasts=[]
    for name,(left,right) in definitions.items():
        for mixture in ['equal_density_mixture']+[str(s) for s in expected_seeds]:
            a={r['measurement_id']:r for r in score_groups[(*left,mixture)]}
            b={r['measurement_id']:r for r in score_groups[(*right,mixture)]}
            for subset in ['all','structure']:
                ids=[rid for rid in a if subset=='all' or a[rid]['structure_supported']]
                w=np.array([a[rid]['weight'] for rid in ids]); w/=w.sum()
                for metric in ['nll','crps_ev','median_mae_ev']:
                    delta=np.array([a[rid][metric]-b[rid][metric] for rid in ids])
                    contrasts.append(dict(contrast=name,seed_mixture=mixture,subset=subset,metric=metric,
                        improvement=float(w@delta),reports=len(ids)))
    lookup={(r['contrast'],r['seed_mixture'],r['subset'],r['metric']):r['improvement'] for r in contrasts}
    retention=[]
    for subset in ['all','structure']:
        for metric in ['nll','crps_ev']:
            denom=lookup[('teacher_structure_at_inference','equal_density_mixture',subset,metric)]
            gain=lookup[('transfer_to_formula','equal_density_mixture',subset,metric)]
            positive=sum(lookup[('teacher_structure_at_inference',str(s),subset,metric)]>0 for s in expected_seeds)
            retention.append(dict(subset=subset,metric=metric,teacher_structure_gain=denom,
                formula_transfer_gain=gain,positive_teacher_seeds=positive,
                ratio=gain/denom if denom>0 and positive>=2 else '',
                interpretation='conditional same-teacher structural increment; descriptive ratio across baseline/teacher'))
    rt.write_csv(out/'oof_densities.csv',raw); rt.write_csv(out/'report_scores.csv',scores)
    rt.write_csv(out/'metrics.csv',metrics); rt.write_csv(out/'contrasts.csv',contrasts)
    rt.write_csv(out/'retention.csv',retention); rt.write_csv(out/'fit_summary.csv',fits)
    rt.write_csv(out/'teacher_signal.csv',diagnostics); rt.write_json(out/'protocol.json',protocol)
    selected=[r for r in metrics if r['seed_mixture']=='equal_density_mixture' and r['subset']=='all'
              and (r['model'],r['view']) in [('F','formula_only'),('J','full'),('J','formula_only'),('ANCHORED','formula_only'),('FROZEN','formula_only'),('FROZEN','full')]]
    lines=['# 当前数据统一训练结果','',f'{len(records)}条报告；5折按化学式划分；3个随机种子。组成监督参与预训练及报告阶段回放。',
           '所有指标来自折外预测；下表为三个种子的等权密度混合。','',
           '|模型|输入|NLL|CRPS (eV)|MAE (eV)|80%覆盖率|80%宽度 (eV)|',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"|{r['model']}|{r['view']}|{r['nll']:.3f}|{r['crps_ev']:.3f}|{r['median_mae_ev']:.3f}|{r['coverage_80']:.1%}|{r['width_80_ev']:.3f}|")
    lines+=['','F为报告阶段仅化学式预测；J为完整模态联合训练；J_SG在报告阶段与回放中移除结构；ANCHORED为冻结F上的结构增量迁移。',
            'FROZEN只更新报告分布头，共同多模态组成预训练主干冻结；当前默认采用FROZEN，公式模式屏蔽可选模态。',
            '三者共同的组成预训练使用可用模态，因此J/J_SG对照度量后续结构信息增量，不能解释为全生命周期去除结构。',
            '当前冷启动协议与历史164条结果初始化及划分不同，不将二者差值归因于补充数据。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(selected,ensure_ascii=False))


if __name__=='__main__': main()
