"""Item 1 (optical vs fundamental on SNUMAT) + composition matching of SNUMAT to the two experimental sets."""
import json, numpy as np, pandas as pd, warnings
from pymatgen.core import Composition
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; F=B+'fig_csv/'
R='/data/czx/GapScope_Revision_20261113/'
S=pd.read_csv(B+'snumat_tidy.csv'); S['soc']=S.soc.astype(str)

def q(x):
    x=np.asarray(x,float)
    return dict(n=len(x), mean=x.mean(), median=np.median(x), q25=np.percentile(x,25), q75=np.percentile(x,75),
                p90=np.percentile(x,90), f_gt_0p1=(x>0.1).mean(), f_gt_0p3=(x>0.3).mean(), f_gt_0p5=(x>0.5).mean(),
                max=x.max(), min=x.min())
S['d_hse']=S.hse_opt-S.hse; S['d_gga']=S.gga_opt-S.gga
rows=[]
for lab,col,dircol,gapcol in [('HSE','d_hse','dir_hse','hse'),('GGA','d_gga','dir_gga','gga')]:
    for sub,m in [('all',np.ones(len(S),bool)), ('gap>0',S[gapcol]>0), ('gap>0,noSOC',(S[gapcol]>0)&(S.soc=='False')),
                  ('gap>0,NM',(S[gapcol]>0)&(S.mag=='NM'))]:
        rows.append(dict(functional=lab,subset=sub,**q(S.loc[m,col])))
    for dv in ['Direct','Indirect']:
        m=(S[gapcol]>0)&(S[dircol]==dv)
        rows.append(dict(functional=lab,subset=f'gap>0,{dv}',**q(S.loc[m,col])))
T=pd.DataFrame(rows); T.to_csv(B+'item1_optical_vs_fundamental.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(T.round(4).to_string(index=False))
print('\nnegative deltas (optical < fundamental): HSE',(S.d_hse<-1e-9).sum(),'GGA',(S.d_gga<-1e-9).sum())
print('exactly zero delta: HSE',(S.d_hse.abs()<1e-9).mean().round(4),'GGA',(S.d_gga.abs()<1e-9).mean().round(4))
# direct/indirect agreement and delta=0 relation
x=S[S.hse>0]
print('HSE direct & delta==0:',((x.dir_hse=="Direct")&(x.d_hse.abs()<1e-9)).sum(),'of',(x.dir_hse=="Direct").sum(),
      '| indirect & delta==0:',((x.dir_hse=="Indirect")&(x.d_hse.abs()<1e-9)).sum(),'of',(x.dir_hse=="Indirect").sum())
S[['SNUMAT_id','ckey','formula','sg','hse','hse_opt','d_hse','gga','gga_opt','d_gga','dir_hse','dir_gga','soc','mag']].to_csv(F+'fig_item1_deltas.csv',index=False)

# ---------------- matching ----------------
def ck(f):
    try: c=Composition(f).fractional_composition
    except Exception: return None
    return '|'.join(f'{e}:{c[e]:.12f}' for e in sorted(c.as_dict()))
P=pd.read_csv(R+'polymorph/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
print('\nprovenance rows',len(P),'compositions',P.composition_key.nunique())
print('key reproduced from formula:',np.mean([ck(f)==k for f,k in zip(P.formula,P.composition_key)]).round(4))
Bt=pd.read_csv(R+'cv/data/tierB_rows.csv')
print('compilation rows',len(Bt),'compositions',Bt.composition_key.nunique())
print('provenance keys also in compilation:',P.composition_key.isin(set(Bt.composition_key)).sum(),
      'compositions',P[P.composition_key.isin(set(Bt.composition_key))].composition_key.nunique())
sk=set(S.ckey)
pA=sorted(set(P.composition_key)&sk); pB=sorted(set(Bt.composition_key)&sk)
allk=sorted(set(P.composition_key)|set(Bt.composition_key)); ov=sorted(set(allk)&sk)
print('MATCH provenance: %d of %d compositions'%(len(pA),P.composition_key.nunique()))
print('MATCH compilation: %d of %d compositions'%(len(pB),Bt.composition_key.nunique()))
print('MATCH union: %d of %d compositions'%(len(ov),len(allk)))
nsg=S[S.ckey.isin(ov)].groupby('ckey').sg.nunique()
print('overlapping compositions with >1 SNUMAT space group: %d of %d'%((nsg>1).sum(),len(nsg)))
print('provenance overlapping with >1 sg: %d of %d'%((S[S.ckey.isin(pA)].groupby("ckey").sg.nunique()>1).sum(),len(pA)))
print('records per overlapping composition: median %.1f max %d'%(S[S.ckey.isin(ov)].groupby("ckey").size().median(),S[S.ckey.isin(ov)].groupby("ckey").size().max()))
json.dump(dict(prov_comp=int(P.composition_key.nunique()),prov_matched=len(pA),comp_comp=int(Bt.composition_key.nunique()),
    comp_matched=len(pB),union=len(allk),union_matched=len(ov),multi_sg=int((nsg>1).sum())),open(B+'match_counts.json','w'),indent=1)
pd.Series(ov,name='ckey').to_csv(B+'matched_keys.csv',index=False)
