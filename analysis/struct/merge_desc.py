"""Merge the descriptor shards into descriptors.csv and report the failures."""
import glob, json
import pandas as pd
from pathlib import Path

D = Path('/data/czx/GapScope_Revision_20261113/struct')
f = sorted(glob.glob(str(D / 'shards' / 'd*[0-9].csv')))
d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True).sort_values('idx')
fl = [pd.read_csv(x) for x in sorted(glob.glob(str(D / 'shards' / 'd*_fail.csv'))) if Path(x).stat().st_size > 5]
fl = pd.concat(fl, ignore_index=True) if fl else pd.DataFrame(columns=['idx', 'snumat_id', 'error'])
print('shards', len(f), 'rows', len(d), 'unique idx', d.idx.nunique(), 'failures', len(fl))
assert d.idx.nunique() == len(d)
d.to_csv(D / 'descriptors.csv', index=False)
if len(fl):
    fl.to_csv(D / 'descriptor_failures.csv', index=False)
    fl['kind'] = fl.error.str.split(':').str[0]
    print(fl.kind.value_counts().to_string())
    print(fl.head(10).to_string())
print('total records expected 10481; covered', len(d) + len(fl))
print('nan rate per descriptor (top 12):')
print((d.isna().mean().sort_values(ascending=False).head(12) * 100).round(2).to_string())
print('oxi/BVS coverage %.1f%%' % (100 * d.oxi_ok.mean()))
print('dim_cat_an distribution:'); print(d.dim_cat_an.value_counts(dropna=False).to_string())
print('crystal systems:'); print(d.crystal_system.value_counts().to_string())
print('spglib sg == Space_group_rlx: %.3f' % (d.sg_number == d.Space_group_rlx).mean())
