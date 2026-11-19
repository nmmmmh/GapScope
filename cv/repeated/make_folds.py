"""Fold builder of cv/build_data.py ('groups, folds' section copied verbatim), with SEED from argv and the partition-independent
tier-A / tier-B tables read from the existing cv/data files instead of being rebuilt. Usage: make_folds.py SEED OUTDIR"""
import sys, json, random, os
from pathlib import Path
import pandas as pd, torch
SRC = Path('/data/czx/GapScope_Revision_20261113/cv')
SEED = int(sys.argv[1]); CV = Path(sys.argv[2])
(CV/'data').mkdir(parents=True, exist_ok=True)
for f in ('tierA.pt', 'tierB_all_rows.csv', 'tierB_features.pt', 'tierB_rows.csv'):
    if not (CV/'data'/f).exists(): os.symlink(SRC/'data'/f, CV/'data'/f)
def system(k): return '|'.join(x.split(':')[0] for x in k.split('|'))
recsl = torch.load(SRC/'data/tierA.pt', weights_only=False)['records']
ex = pd.read_csv(SRC/'data/tierB_all_rows.csv')
core = [r for r in recsl if r['origin'] == 'core']; ext = [r for r in recsl if r['origin'] == 'external']
Akeys = sorted({r['composition_key'] for r in recsl}); Bkeys = sorted(set(ex.key))
# ---------------- groups, folds (verbatim from build_data.py)
Aset, Bset = set(Akeys), set(Bkeys)
coreK = {r['composition_key'] for r in core}; extK = {r['composition_key'] for r in ext}
allkeys = sorted(Aset | Bset)
def assign(group_of):
    gA = sorted({group_of[k] for k in Aset}); gB = sorted({group_of[k] for k in allkeys} - set(gA))
    random.Random(SEED).shuffle(gA); random.Random(SEED + 1).shuffle(gB)
    fg = {x: i % 5 for i, x in enumerate(gA)}
    fg.update({x: (i + len(gA)) % 5 for i, x in enumerate(gB)})
    return {k: fg[group_of[k]] for k in allkeys}
def inner_val(fold_of, group_of, f):
    gA = sorted({group_of[k] for k in Aset if fold_of[k] != f})
    n = max(1, round(0.1 * len(gA))); random.Random(SEED + 100 + f).shuffle(gA)
    vg = set(gA[:n]); return sorted(k for k in Aset if fold_of[k] != f and group_of[k] in vg)
nA = {k: sum(1 for r in recsl if r['composition_key'] == k) for k in Akeys}
nB = ex.groupby('key').size().to_dict()
summary = {}
for variant, gf in (('comp', lambda k: k), ('chem', system)):
    group_of = {k: gf(k) for k in allkeys}; fold_of = assign(group_of)
    iv = {f: inner_val(fold_of, group_of, f) for f in range(5)}
    df = pd.DataFrame([dict(composition_key=k, chem_system=system(k), group=group_of[k], fold=fold_of[k] + 1, in_tierA=k in Aset, in_tierB=k in Bset,
                            n_reports_A=nA.get(k, 0), n_rows_B=int(nB.get(k, 0))) for k in allkeys])
    df.to_csv(CV/f'folds_{variant}.csv', index=False)
    json.dump({str(f+1): v for f, v in iv.items()}, open(CV/f'data/inner_val_{variant}.json', 'w'))
    sm = dict(per_fold=[])
    for f in range(5):
        d = df[df.fold == f + 1]
        sm['per_fold'].append(dict(fold=f+1, comps_total=len(d), A_comps=int(d.in_tierA.sum()), A_reports=int(d.n_reports_A.sum()), B_comps=int(d.in_tierB.sum()), B_rows=int(d.n_rows_B.sum()),
                                   both=int((d.in_tierA & d.in_tierB).sum()), inner_val_A_comps=len(iv[f])))
    sm['total'] = dict(A_comps=int(df.in_tierA.sum()), A_reports=int(df.n_reports_A.sum()), B_comps=int(df.in_tierB.sum()), B_rows=int(df.n_rows_B.sum()), both=int((df.in_tierA & df.in_tierB).sum()), union=len(df))
    summary[variant] = sm
json.dump(dict(seed=SEED, **summary), open(CV/'fold_summary.json', 'w'), indent=1)
print(SEED, 'comp A per fold', [p['A_comps'] for p in summary['comp']['per_fold']], 'chem A per fold', [p['A_comps'] for p in summary['chem']['per_fold']])
