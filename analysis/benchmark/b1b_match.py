"""Matching: SNUMAT (json, primary) to provenance set (237 records, suppdata) and compilation set (tierB)."""
import json, numpy as np, pandas as pd, warnings
from pymatgen.core import Composition
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
S=pd.read_csv(B+'snumat_tidy.csv'); S['soc']=S.soc.astype(str)
def ck(f):
    try: c=Composition(f).fractional_composition
    except Exception: return None
    return '|'.join(f'{e}:{c[e]:.12f}' for e in sorted(c.as_dict()))
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
P205=pd.read_csv(R+'polymorph/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
Bt=pd.read_csv(R+'cv/data/tierB_rows.csv')
bad=[(f,k) for f,k in zip(P.formula,P.composition_key) if ck(f)!=k]
print('provenance 237 rows, %d compositions; key reproduced for %d/237; exceptions %s'%(P.composition_key.nunique(),237-len(bad),bad[:4]))
sk=set(S.ckey)
out={}
for lab,keys in [('provenance_237',set(P.composition_key)),('provenance_205',set(P205.composition_key)),('compilation',set(Bt.composition_key))]:
    out[lab]=dict(n_comp=len(keys),n_matched=len(keys&sk))
    print('%-16s %d of %d compositions matched to SNUMAT'%(lab,len(keys&sk),len(keys)))
allk=set(P.composition_key)|set(Bt.composition_key); ov=sorted(allk&sk)
out['union']=dict(n_comp=len(allk),n_matched=len(ov))
print('union %d of %d'%(len(ov),len(allk)))
nsg=S[S.ckey.isin(ov)].groupby('ckey').sg.nunique()
print('overlapping compositions with >1 SNUMAT space group: %d of %d (%.1f%%)'%((nsg>1).sum(),len(nsg),100*(nsg>1).mean()))
pA=set(P.composition_key)&sk
nsgA=S[S.ckey.isin(pA)].groupby('ckey').sg.nunique()
print('provenance-matched with >1 sg: %d of %d'%((nsgA>1).sum(),len(nsgA)))
print('provenance RECORDS on matched compositions: %d of 237'%P.composition_key.isin(pA).sum())
print('compilation ROWS on matched compositions: %d of %d'%(Bt.composition_key.isin(sk).sum(),len(Bt)))
out['multi_sg_union']=int((nsg>1).sum()); out['multi_sg_prov']=int((nsgA>1).sum())
json.dump(out,open(B+'match_counts.json','w'),indent=1)
pd.Series(ov,name='ckey').to_csv(B+'matched_keys.csv',index=False)
# existing snumat_overlap.csv comparison
O=pd.read_csv(R+'cv/snumat_overlap.csv')
print('\nexisting snumat_overlap.csv: %d rows, %d compositions; of these in my json match set: %d'%(len(O),O.ckey.nunique(),len(set(O.ckey)&sk)))
