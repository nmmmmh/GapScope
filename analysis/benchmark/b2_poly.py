"""Item 2: polymorph spread of computed HSE/GGA vs experimental spread; ground-state vs space-group-matched choice."""
import json, numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; F=B+'fig_csv/'; R='/data/czx/GapScope_Revision_20261113/'
S=pd.read_csv(B+'snumat_tidy.csv'); S['soc']=S.soc.astype(str)
C=pd.read_csv(B+'hse_csv_keyed.csv')
# Free_energy (eV/atom) joined from the tidy csv export on exact (ckey, space group, HSE, GGA)
C['K']=C.ckey+'|'+C.Space_group_cif.astype('Int64').astype(str)+'|'+C.Band_gap_HSE.round(6).astype(str)+'|'+C.Band_gap_GGA.round(6).astype(str)
S['K']=S.ckey+'|'+S.sg.astype(str)+'|'+S.hse.round(6).astype(str)+'|'+S.gga.round(6).astype(str)
fe=C.drop_duplicates('K').set_index('K').Free_energy
S['Free_energy']=S.K.map(fe)
print('SNUMAT records with Free_energy joined: %d of %d (%.1f%%)'%(S.Free_energy.notna().sum(),len(S),100*S.Free_energy.notna().mean()))

# one computed value per (composition, space group): median over SNUMAT records sharing it
g=S.groupby(['ckey','sg']).agg(hse=('hse','median'),hse_opt=('hse_opt','median'),gga=('gga','median'),gga_opt=('gga_opt','median'),
    n_rec=('hse','size'),dir_hse=('dir_hse',lambda s:s.mode().iat[0]),Free_energy=('Free_energy','min')).reset_index()
g.to_csv(B+'snumat_by_comp_sg.csv',index=False)
print('distinct (composition, space group) cells %d over %d compositions'%(len(g),g.ckey.nunique()))
spread=g.groupby('ckey').agg(n_sg=('sg','size'),hse_min=('hse','min'),hse_max=('hse','max'),hse_sd=('hse','std'),
    gga_min=('gga','min'),gga_max=('gga','max'),gga_sd=('gga','std')).reset_index()
spread['hse_range']=spread.hse_max-spread.hse_min; spread['gga_range']=spread.gga_max-spread.gga_min
m=spread[spread.n_sg>=2]
def qd(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return dict(n=len(x),mean=x.mean(),median=np.median(x),q25=np.percentile(x,25),q75=np.percentile(x,75),p90=np.percentile(x,90))
rows=[dict(quantity=k,**qd(m[k])) for k in ['hse_range','hse_sd','gga_range','gga_sd']]
print(pd.DataFrame(rows).round(4).to_string(index=False))

# experimental spread per composition, both tiers
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
Bt=pd.read_csv(R+'cv/data/tierB_rows.csv')
E=pd.concat([P[['composition_key','bandgap_ev']].set_axis(['ckey','y'],axis=1).assign(tier='provenance'),
             Bt[['composition_key','target_ev']].set_axis(['ckey','y'],axis=1).assign(tier='compilation')])
ex=E.groupby(['tier','ckey']).y.agg(n_exp='size',exp_min='min',exp_max='max',exp_sd='std',exp_median='median',exp_mean='mean',n_dist=lambda s:s.nunique()).reset_index()
ex['exp_range']=ex.exp_max-ex.exp_min
exb=E.groupby('ckey').y.agg(n_exp='size',exp_min='min',exp_max='max',exp_sd='std',exp_median='median',exp_mean='mean',n_dist=lambda s:s.nunique()).reset_index().assign(tier='both')
exb['exp_range']=exb.exp_max-exb.exp_min
EX=pd.concat([ex,exb])
T=EX.merge(spread,on='ckey')
T.to_csv(F+'fig_item2_spread_by_composition.csv',index=False)
res=[]
for t in ['provenance','compilation','both']:
    q=T[(T.tier==t)&(T.n_sg>=2)&(T.n_dist>=2)]
    r=dict(tier=t,n=len(q),exp_range_median=q.exp_range.median(),hse_range_median=q.hse_range.median(),gga_range_median=q.gga_range.median())
    for a in ['hse_range','gga_range','hse_sd','gga_sd']:
        b='exp_range' if a.endswith('range') else 'exp_sd'
        qq=q.dropna(subset=[a,b])
        rho,p=spearmanr(qq[a],qq[b]); r['rho_'+a]=rho; r['p_'+a]=p; r['n_'+a]=len(qq)
    res.append(r)
Rk=pd.DataFrame(res); Rk.to_csv(B+'item2_rank_correlation.csv',index=False)
print(Rk.round(4).to_string(index=False))
# single-sg compositions for contrast
for t in ['provenance','compilation','both']:
    q=T[(T.tier==t)&(T.n_dist>=2)]
    print('%s: median exp_range with >=2 SNUMAT sg %.3f (n=%d) vs single sg %.3f (n=%d)'%(t,
        q[q.n_sg>=2].exp_range.median(),(q.n_sg>=2).sum(),q[q.n_sg==1].exp_range.median(),(q.n_sg==1).sum()))
spread.to_csv(B+'snumat_spread_by_composition.csv',index=False)
