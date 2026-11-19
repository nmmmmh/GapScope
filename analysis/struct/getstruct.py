import json, sys, csv
import pandas as pd, numpy as np
P = '/data/czx/GapScope_Revision_20261113/struct/pairs.csv'
d = pd.read_csv(P)
cols = [c for c in ('d_cn_cat_mean','d_bond_cat_an_mean','d_dim_max') if c in d.columns]
print('match cols', cols, file=sys.stderr)
m = d.copy()
for c, t in (('d_cn_cat_mean',0.25), ('d_bond_cat_an_mean',0.10), ('d_dim_max',0.5)):
    if c in m.columns: m = m[m[c].abs() <= t]
print('matched', len(m), 'mean |dgap|', m.d_Band_gap_HSE.abs().mean(), file=sys.stderr)
mm = m.reindex(m.d_Band_gap_HSE.abs().sort_values(ascending=False).index).head(8)
print(mm[['formula','sg_a','sg_b','id_a','id_b','d_Band_gap_HSE','Band_gap_HSE_a','Band_gap_HSE_b']
      + cols].to_string(), file=sys.stderr)

WANT_FORM = ['Ga2O3','CsNiBr3','CrSBr','Ag2WO4','SnSe','SiC','FAPbI3']
ids = set()
for _, r in mm.head(3).iterrows():
    ids.add(str(r.id_a)); ids.add(str(r.id_b))
sel = d[d.formula.isin(WANT_FORM)]
for _, r in sel.iterrows():
    ids.add(str(r.id_a)); ids.add(str(r.id_b))
print('ids wanted', len(ids), file=sys.stderr)

out = {}
data = json.load(open('/home/czx/2024/doped/data/snumat.json'))
for idx, rec in enumerate(data):
    rid = str(rec.get('SNUMAT_id', rec.get('ICSD_number', idx)))
    if rid in ids:
        a = rec.get('atoms') or {}
        out[rid] = dict(lattice=a.get('lattice_mat'), coords=a.get('coords'),
                        elements=a.get('elements'),
                        sg=rec.get('Space_group_rlx'), name=rec.get('Structure_rlx'),
                        gap=rec.get('Band_gap_HSE'))
print('found', len(out), file=sys.stderr)
json.dump(out, open('/data/czx/GapScope_Revision_20261113/struct/sel_structures.json','w'))
mm.head(8).to_csv('/data/czx/GapScope_Revision_20261113/struct/matched_top.csv', index=False)
sel.to_csv('/data/czx/GapScope_Revision_20261113/struct/sel_pairs.csv', index=False)
