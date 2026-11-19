import sys, json
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/roost')
from roostlib import *
import pandas as pd
CV = __import__('os').environ['CVDIR']
A = torch.load(CV + '/data/tierA.pt', weights_only=False)['records']
Ball = pd.read_csv(CV + '/data/tierB_all_rows.csv'); Bf = torch.load(CV + '/data/tierB_features.pt', weights_only=False)
forms = dict(zip(Bf['keys'], Bf['formulas']))
import glob
rm = {}
for p in glob.glob(W + '/original_inputs/fold_*_outer.csv'):
    d_ = pd.read_csv(p); rm.update(dict(zip(d_.id, d_.composition)))
emb_ = json.load(open(EMB))
for r in A:
    r['rf'] = rm.get(r['measurement_id'], r['formula'])
    el, _ = parse_roost(r['rf']); assert all(e in emb_ for e in el), r
print('roost formulas replaced for', sum(r['rf'] != r['formula'] for r in A))
import os
os.makedirs(CV + '/roost_in', exist_ok=True)
for variant in ('comp', 'chem'):
    fd = pd.read_csv(f'{CV}/folds_{variant}.csv').set_index('composition_key').fold.to_dict()
    for f in range(1, 6):
        tr = [dict(id=100000 + i, composition=r['rf'], Eg=r['gap_ev']) for i, r in enumerate(A) if fd[r['composition_key']] != f]
        tr += [dict(id=int(x.id), composition=x.composition, Eg=x.Eg) for x in Ball.itertuples() if fd[x.key] != f]
        te = [dict(id=100000 + i, composition=r['rf'], Eg=0.0) for i, r in enumerate(A) if fd[r['composition_key']] == f]
        te += [dict(id=200000 + j, composition=forms[k], Eg=0.0) for j, k in enumerate(Bf['keys']) if fd[k] == f]
        pd.DataFrame(tr).to_csv(f'{CV}/roost_in/{variant}_f{f}_train.csv', index=False)
        pd.DataFrame(te).to_csv(f'{CV}/roost_in/{variant}_f{f}_test.csv', index=False)
        print(variant, f, len(tr), len(te))
