"""Write energy-only D_c per traced composition (for Fig 3b) in the format of descr/d2_per_composition.csv."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0].replace("print('cells", "pass #"))
subE = sub[sub.Free_energy.notna()]
D = pd.read_csv(R + 'descr/d2_per_composition.csv')
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig')
fm = P.drop_duplicates('composition_key').set_index('composition_key').formula
out = []
for k in keys:
    c = subE[subE.composition_key == k]
    c = c[c.Free_energy - c.Free_energy.min() <= 0.10] if len(c) else c
    sp = lambda s: (s.max() - s.min()) if s.notna().sum() >= 2 else 0.0
    out.append(dict(formula=fm.get(k, k), D_struct=sp(c.cn_cat_mean) + sp(c.mean_shared_vertices), D_elec=sp(c.Band_gap_HSE), n_acc=len(c)))
O = pd.DataFrame(out)
M = D[['formula', 'floor', 'rng']].merge(O, on='formula', how='left')
print('rows', len(M), 'missing', M.D_elec.isna().sum())
M.to_csv(R + 'descr/d2_per_composition_energy.csv', index=False)
print(M.sort_values('D_elec', ascending=False).head(5).round(3).to_string(index=False))
