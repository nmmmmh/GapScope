"""Same compositions, two sources: provenance-traced spread vs compilation spread."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R='/data/czx/GapScope_Revision_20261113/'
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
P['ph']=P.phase_label.fillna('').astype(str).str.strip().str.lower()
def st(df,k,y):
    m=df.groupby(k)[y].transform('median'); a=(df[y]-m).abs()
    return df.assign(a=a).groupby(k).agg(n=(y,'size'),mad=('a','mean'),lo=(y,'min'),hi=(y,'max'))
p=st(P,'composition_key','bandgap_ev'); p['nph']=P.groupby('composition_key').ph.agg(lambda s:s[s!=''].nunique())
A=pd.read_csv(R+'cv/data/tierB_all_rows.csv').drop_duplicates(['key','Eg']); c=st(A,'key','Eg')
J=p.join(c,lsuffix='_prov',rsuffix='_comp',how='inner'); J=J[(J.n_comp>=2)]
fm=P.drop_duplicates('composition_key').set_index('composition_key').formula
J['formula']=fm.reindex(J.index).values
for lab,j in [('all overlap',J),('multi-label',J[J.nph>=2])]:
    print('%s: %d compositions, floor prov %.3f comp %.3f, Wilcoxon P=%.2g'%(lab,len(j),j.mad_prov.mean(),j.mad_comp.mean(),wilcoxon(j.mad_prov,j.mad_comp).pvalue if len(j)>5 else np.nan))
print(J.sort_values('mad_prov',ascending=False)[['formula','nph','n_prov','mad_prov','lo_prov','hi_prov','n_comp','mad_comp','lo_comp','hi_comp']].round(2).to_string())
J.to_csv(R+'compfloor/overlap.csv')
