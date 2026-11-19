"""Does the SNUMAT space-group count reproduce the multi/single contrast on the 85 provenance compositions?"""
import numpy as np, pandas as pd
R='/data/czx/GapScope_Revision_20261113/'
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
S=pd.read_csv(R+'benchmark/snumat_tidy.csv')[['ckey','sg']]
C=pd.read_csv(R+'benchmark/hse_csv_keyed.csv')[['ckey','Space_group_cif']].rename(columns={'Space_group_cif':'sg'})
SG=pd.concat([S,C]).dropna(); nsg=SG.groupby('ckey').sg.nunique()
P['ph']=P.phase_label.fillna('').astype(str).str.strip().str.lower()
P['med']=P.groupby('composition_key').bandgap_ev.transform('median'); P['ad']=(P.bandgap_ev-P.med).abs()
pc=P.groupby('composition_key').agg(n=('ad','size'),mad=('ad','mean'),nph=('ph',lambda s:s[s!=''].nunique())).reset_index()
pc['nsg']=pc.composition_key.map(nsg)
pc['labels']=np.where(pc.nph>=2,'multi(labels)','single(labels)')
pc['proxy']=np.where(pc.nsg.isna(),'unmatched',np.where(pc.nsg>=2,'multi(SNUMAT)','single(SNUMAT)'))
d=pc[pc.n>=2]
print(pd.crosstab(d.labels,d.proxy,margins=True))
print(d.groupby('labels').mad.agg(['size','mean','median']).round(3))
print(d.groupby('proxy').mad.agg(['size','mean','median']).round(3))
# compilation: same-source information absent; how close are duplicate values? (rounding / transcription signature)
B=pd.read_csv(R+'compfloor/per_composition.csv'); A=pd.read_csv(R+'cv/data/tierB_all_rows.csv')
g=A.groupby('key').Eg; r=(g.transform('max')-g.transform('min'))
print('compilation duplicate compositions: span distribution (eV)')
dd=A.assign(r=r).drop_duplicates('key'); dd=dd[g.transform('size')[dd.index]>=2]
print(dd.r.describe(percentiles=[.25,.5,.75,.9]).round(3))
print('multi-label provenance compositions present in compilation with >=2 values:',
      len(set(pc[pc.nph>=2].composition_key)&set(B[B.n>=2].ckey)))
