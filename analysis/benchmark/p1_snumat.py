import json, numpy as np, pandas as pd, warnings
from collections import Counter
from pymatgen.core import Composition
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'

def ckey(comp):
    c=comp.fractional_composition
    return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))

J=json.load(open('/home/czx/2024/doped/data/snumat.json'))
rows=[]
for r in J:
    els=Counter(r['atoms']['elements'])
    comp=Composition({k:v for k,v in els.items()})
    mat=''.join(f'{e}{int(els[e])}' for e in sorted(els))
    rows.append(dict(SNUMAT_id=r['SNUMAT_id'], ICSD_number=r['ICSD_number'],
        Material=mat, formula=comp.reduced_formula, ckey=ckey(comp), natoms=sum(els.values()),
        hse=float(r['Band_gap_HSE']), hse_opt=float(r['Band_gap_HSE_optical']),
        gga=float(r['Band_gap_GGA']), gga_opt=float(r['Band_gap_GGA_optical']),
        dir_gga=r['Direct_or_indirect'], dir_hse=r['Direct_or_indirect_HSE'],
        soc=str(r['SOC']), sg=int(r['Space_group_rlx']), mag=r['Magnetic_ordering']))
S=pd.DataFrame(rows)
S.to_csv(B+'snumat_tidy.csv', index=False)
print('SNUMAT json rows', len(S), 'unique SNUMAT_id', S.SNUMAT_id.nunique(), 'unique ckey', S.ckey.nunique())
print('SOC', S.soc.value_counts().to_dict(), '| mag', S.mag.value_counts().to_dict())
print('dir_hse', S.dir_hse.value_counts().to_dict())

C=pd.read_csv('/home/czx/2024/spinels/data/hse_data.csv', index_col=0)
print('csv rows', len(C), 'unique Material', C.Material.nunique())
# relation: match on Material + rounded gaps
C['k']=C.Material+'|'+C.Band_gap_HSE.round(6).astype(str)+'|'+C.Band_gap_GGA.round(6).astype(str)
S['k']=S.Material+'|'+S.hse.round(6).astype(str)+'|'+S.gga.round(6).astype(str)
print('json rows whose (Material,HSE,GGA) appears in csv:', S.k.isin(set(C.k)).sum(), '/', len(S))
print('csv rows whose (Material,HSE,GGA) appears in json:', C.k.isin(set(S.k)).sum(), '/', len(C))
# material-level
print('json Materials in csv:', S.Material.isin(set(C.Material)).sum(),'/',len(S),
      '| csv Materials in json:', C.Material.isin(set(S.Material)).sum(),'/',len(C))
# duplicate Material within each
print('csv dup Material rows', (C.Material.duplicated(keep=False)).sum(), 'json dup Material rows',(S.Material.duplicated(keep=False)).sum())
# space group agreement on unique-Material matches
a=S.drop_duplicates('Material',keep=False)[['Material','sg','hse','gga','dir_hse']]
b=C.drop_duplicates('Material',keep=False)[['Material','Space_group_cif','Band_gap_HSE','Band_gap_GGA','Direct_or_indirect_HSE','Free_energy']]
m=a.merge(b,on='Material')
print('unique-Material merge n',len(m),'| sg equal', (m.sg==m.Space_group_cif).mean().round(4),
      '| max|dHSE|', (m.hse-m.Band_gap_HSE).abs().max(), '| max|dGGA|',(m.gga-m.Band_gap_GGA).abs().max(),
      '| dir equal',(m.dir_hse==m.Direct_or_indirect_HSE).mean().round(4))
