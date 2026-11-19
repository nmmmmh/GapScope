import json, sys
import pandas as pd
from collections import Counter
d = pd.read_csv('/data/czx/GapScope_Revision_20261113/struct/descriptors.csv')
print([c for c in d.columns][:12], file=sys.stderr)
WANT = ['CsNiBr3','CrSBr','Ag2WO4','Ag2O4W','BN','Ga2O3','SiC','SnSe']
fc = 'formula' if 'formula' in d.columns else 'reduced_formula'
sub = d[d[fc].isin(WANT)]
print(Counter(sub[fc]), file=sys.stderr)
idc = 'snumat_id'
print('idcol', idc, file=sys.stderr)
keep = {}
for f, g in sub.groupby(fc):
    g = g.sort_values('Band_gap_HSE') if 'Band_gap_HSE' in g.columns else g
    for _, r in g.iterrows():
        keep[str(r[idc])] = f
print('ids', len(keep), file=sys.stderr)
out = {}
data = json.load(open('/home/czx/2024/doped/data/snumat.json'))
for idx, rec in enumerate(data):
    rid = str(rec.get('SNUMAT_id', rec.get('ICSD_number', idx)))
    if rid in keep:
        a = rec.get('atoms') or {}
        out[rid] = dict(lattice=a.get('lattice_mat'), coords=a.get('coords'),
                        elements=a.get('elements'), formula=keep[rid],
                        sg=rec.get('Space_group_rlx'), gap=rec.get('Band_gap_HSE'))
print('found', len(out), file=sys.stderr)
json.dump(out, open('/data/czx/GapScope_Revision_20261113/struct/sel_structures2.json','w'))
cols = [c for c in (idc, fc, 'sg_number', 'crystal_system', 'Band_gap_HSE', 'cn_cat_mean', 'cn_cat_spread', 'dim_max') if c in sub.columns]
sub[cols].to_csv('/data/czx/GapScope_Revision_20261113/struct/sel_desc2.csv', index=False)
