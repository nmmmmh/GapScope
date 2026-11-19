"""Self-contained data package for the fine-tuned GS arms (GS_FT, GS_FT_HSE) and ROOST_CAL; runs on c137-13, read-only on existing cv files."""
import sys, json, glob
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse/runtime')
import numpy as np, pandas as pd, torch
from pathlib import Path
from training_data import hierarchical_weights
CV = Path(__import__('os').environ['CVDIR'])
W = '/home/czx/2024/BandgapFusion_RoostReproductionWork'
A = torch.load(CV / 'data/tierA.pt', weights_only=False)
recs, feat = A['records'], A['features']
rm = {}
for p in glob.glob(W + '/original_inputs/fold_*_outer.csv'):
    d_ = pd.read_csv(p); rm.update(dict(zip(d_.id, d_.composition)))
wA = hierarchical_weights(recs, list(range(len(recs))), 'cpu').numpy().astype(float)
key = [r['composition_key'] for r in recs]
wc = np.zeros(len(recs))
for k in set(key):
    ii = [i for i, x in enumerate(key) if x == k]
    wc[ii] = wA[ii] / wA[ii].sum()
sg = feat['sg_features'].float(); sgm = feat['sg_mask'].float().reshape(len(recs), -1)[:, 0]
print('sg_features', tuple(sg.shape), 'mask on', int(sgm.sum()), 'range', float(sg.min()), float(sg.max()))
Arows = [dict(idx=i, key=key[i], formula=rm.get(r['measurement_id'], r['formula']), y=float(r['gap_ev']), wc=float(wc[i]), has_sg=bool(sgm[i] > 0)) for i, r in enumerate(recs)]
rowsB = pd.read_csv(CV / 'data/tierB_rows.csv', dtype={'target_ev': str})
Bf = torch.load(CV / 'data/tierB_features.pt', weights_only=False)
folds = {v: pd.read_csv(CV / f'folds_{v}.csv').set_index('composition_key').fold.to_dict() for v in ('comp', 'chem')}
iv = {v: json.load(open(CV / f'data/inner_val_{v}.json')) for v in ('comp', 'chem')}
hp = pd.read_csv('/data/czx/GapScope_Revision_20261113/hse/hse_pool.csv')
out = dict(A=Arows, sg=sg, sg_mask=sgm, B=dict(key=list(rowsB.composition_key), formula=list(rowsB.formula), y=[float(x) for x in rowsB.target_ev], w=list(rowsB.within_composition_weight)),
           Bkeys=list(Bf['keys']), Bformulas=list(Bf['formulas']), folds=folds, inner_val=iv,
           hse=dict(key=list(hp.key), formula=list(hp.formula), y=list(hp.target.astype(float))))
(CV / 'ft').mkdir(exist_ok=True)
torch.save(out, CV / 'ft/ft_data.pt')
print('A', len(Arows), 'B rows', len(rowsB), 'B min y', rowsB.target_ev.astype(float).min(), 'Bkeys', len(out['Bkeys']), 'hse', len(hp), 'hse zero', int((hp.target <= 0).sum()))
