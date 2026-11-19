"""Items 5 and 6: does the HSE-vs-GGA verdict change, and what the cleaning conventions do."""
import json, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
D=pd.read_csv(B+'per_record.csv')
g=pd.read_csv(B+'snumat_by_comp_sg.csv')
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
Bt=pd.read_csv(R+'cv/data/tierB_rows.csv')

# ---------- (a) composition level, experimental median, one computed number per composition ----------
sp=g.groupby('ckey').agg(n_sg=('sg','size'),hse_med=('hse','median'),gga_med=('gga','median')).reset_index()
gs=g.dropna(subset=['Free_energy']).sort_values('Free_energy').drop_duplicates('ckey')[['ckey','hse','gga']].rename(columns={'hse':'hse_gs','gga':'gga_gs'})
ex=P.groupby('composition_key').bandgap_ev.agg(exp_n='size',exp_med='median',exp_min='min',exp_max='max',exp_sd='std').reset_index().rename(columns={'composition_key':'ckey'})
A=ex.merge(sp,on='ckey').merge(gs,on='ckey',how='left')
rows=[]
def mae(x,y):
    v=(np.asarray(x,float)-np.asarray(y,float)); v=v[~np.isnan(v)]; return np.abs(v).mean(),len(v)
for ref,lab in [('hse_med','HSE polymorph median'),('gga_med','GGA polymorph median'),('hse_gs','HSE ground state'),('gga_gs','GGA ground state')]:
    m,n=mae(A[ref],A.exp_med); rows.append(dict(view='(a) composition median',computed=lab,n=n,MAE=m,
        bias=np.nanmean(A[ref]-A.exp_med)))
# ---------- (b) space-group matched record level ----------
S=D[D.sg_matched]
for f,lab in [('hse','HSE sg-matched cell'),('gga','GGA sg-matched cell')]:
    rows.append(dict(view='(b) sg-matched record',computed=lab,n=len(S),MAE=S['abs_err_'+f].mean(),bias=(S[f+'_ref']-S.bandgap_ev).mean()))
for f,lab in [('hse','HSE polymorph median'),('gga','GGA polymorph median')]:
    rows.append(dict(view='(b) same records, polymorph median',computed=lab,n=len(S),
        MAE=(S[f+'_med_poly']-S.bandgap_ev).abs().mean(),bias=(S[f+'_med_poly']-S.bandgap_ev).mean()))
# all 196 records, per-report (no median)
for f,lab in [('hse','HSE'),('gga','GGA')]:
    rows.append(dict(view='(b2) all matched records, per report',computed=lab,n=len(D),MAE=D['abs_err_'+f].mean(),bias=(D[f+'_ref']-D.bandgap_ev).mean()))

# ---------- (c) CRPS / PIT of the computed value under the model predictive distribution ----------
PK=pickle.load(open(R+'cv/easyuq_comp_GS_SELCAL_PRE_NZ.pkl','rb'))
TA=torch.load(R+'cv/data/tierA.pt',weights_only=False)['records']
aid=[r['measurement_id'] for r in TA]; akey=[r['composition_key'] for r in TA]
GRID=np.arange(-0.5,14.0,0.002)
def lap_cdf(z): return np.where(z<0,0.5*np.exp(z),1-0.5*np.exp(-z))
def cdf_wlap(d,x):
    _,w,c,b=d
    return float((w*lap_cdf((x-c)/b)).sum())
def crps_wlap(d,x):
    _,w,c,b=d
    F=(w[None,:]*lap_cdf((GRID[:,None]-c[None,:])/b[None,:])).sum(1)
    H=(GRID>=x).astype(float)
    return float(np.trapz((F-H)**2,GRID))
crows=[]
for nm in ['EASYUQ_ON_GS_SELCAL_PRE_NZ_IDR','EASYUQ_ON_GS_SELCAL_PRE_NZ_KS']:
    Adist=PK[nm][0][0]
    idx={aid[i]:i for i in range(len(aid))}
    sub=D[D.record_id.isin(idx)]
    for f,lab in [('hse','HSE'),('gga','GGA')]:
        cr=[];pit=[];cry=[]
        for _,r in sub.iterrows():
            d=Adist[idx[r.record_id]]
            cr.append(crps_wlap(d,r[f+'_ref'])); pit.append(cdf_wlap(d,r[f+'_ref'])); cry.append(crps_wlap(d,r.bandgap_ev))
        cr=np.array(cr);pit=np.array(pit);cry=np.array(cry)
        crows.append(dict(arm=nm,functional=lab,n=len(cr),CRPS_of_computed=cr.mean(),CRPS_of_experiment=cry.mean(),
            median_CRPS_computed=np.median(cr),frac_PIT_in_10_90=float(((pit>0.1)&(pit<0.9)).mean()),
            frac_PIT_in_05_95=float(((pit>0.05)&(pit<0.95)).mean()),median_PIT=float(np.median(pit))))
CR=pd.DataFrame(crows); CR.to_csv(B+'item5c_crps.csv',index=False)
pd.set_option('display.width',280); pd.set_option('display.max_columns',22); pd.set_option('display.max_colwidth',44)
print('--- (c) CRPS of the computed value under the main model predictive distribution (out-of-fold) ---')
print(CR.round(4).to_string(index=False))

# view contrast: composition only vs composition + space group, raw GS lognormal mixtures
VN=['formula_only','formula_sg','formula_structure','full']
per={}
for s in (17,29,41):
    for fo in range(1,6):
        z=np.load(R+'cv/gs_out/comp_GS_seed%d_fold%d.npz'%(s,fo))
        for j,i in enumerate(z['A_idx']):
            for v in range(4):
                per.setdefault((v,int(i)),[]).append(tuple(z['A%d'%v][t][j] for t in range(3)))
def lnorm_cdf_mix(parts,x):
    if x<=0: return 0.0
    tot=0.
    for w,mu,sd in parts:
        from math import erf,sqrt,log
        tot+=float((w*0.5*(1+np.vectorize(erf)((np.log(x)-mu)/(sd*np.sqrt(2))))).sum())
    return tot/len(parts)
def lnorm_crps_mix(parts,x):
    F=np.zeros_like(GRID)
    pos=GRID>0
    from math import erf
    ef=np.vectorize(erf)
    for w,mu,sd in parts:
        F[pos]+=(w[None,:]*0.5*(1+ef((np.log(GRID[pos])[:,None]-mu[None,:])/(sd[None,:]*np.sqrt(2))))).sum(1)
    F/=len(parts)
    H=(GRID>=x).astype(float)
    return float(np.trapz((F-H)**2,GRID))
idx={aid[i]:i for i in range(len(aid))}
sub=D[D.record_id.isin(idx)]
vrows=[]
for v in (0,1):
    for f,lab in [('hse','HSE'),('gga','GGA')]:
        cr=[];cry=[];pit=[]
        for _,r in sub.iterrows():
            i=idx[r.record_id]
            if (v,i) not in per or len(per[(v,i)])!=3: continue
            pt=per[(v,i)]
            cr.append(lnorm_crps_mix(pt,r[f+'_ref'])); cry.append(lnorm_crps_mix(pt,r.bandgap_ev)); pit.append(lnorm_cdf_mix(pt,r[f+'_ref']))
        vrows.append(dict(view=VN[v],functional=lab,n=len(cr),CRPS_of_computed=np.mean(cr),CRPS_of_experiment=np.mean(cry),
            frac_PIT_in_10_90=float(np.mean((np.array(pit)>0.1)&(np.array(pit)<0.9)))))
V=pd.DataFrame(vrows); V.to_csv(B+'item5c_view_contrast.csv',index=False)
print('--- (c) view contrast, raw GS 3-seed lognormal mixtures (out-of-fold) ---')
print(V.round(4).to_string(index=False))

T5=pd.DataFrame(rows); T5.to_csv(B+'item5_verdict.csv',index=False)
print('\n--- (a)/(b) apparent error under the different references ---')
print(T5.round(4).to_string(index=False))
print('\ninside the experimental spread (196 matched records): HSE %d (%.1f%%), GGA %d (%.1f%%); any polymorph+definition HSE %d (%.1f%%), GGA %d (%.1f%%)'%(
  D.inside_exp_range_hse.sum(),100*D.inside_exp_range_hse.mean(),D.inside_exp_range_gga.sum(),100*D.inside_exp_range_gga.mean(),
  D.inside_any_polymorph_hse.sum(),100*D.inside_any_polymorph_hse.mean(),D.inside_any_polymorph_gga.sum(),100*D.inside_any_polymorph_gga.mean()))
cc=A[A.exp_n>=2]
print('compositions with >=2 reports (n=%d): computed inside [min,max]: HSE %d, GGA %d'%(len(cc),
  ((cc.hse_med>=cc.exp_min)&(cc.hse_med<=cc.exp_max)).sum(),((cc.gga_med>=cc.exp_min)&(cc.gga_med<=cc.exp_max)).sum()))

# ---------- item 6: cleaning conventions ----------
E=pd.concat([P[['composition_key','bandgap_ev']].set_axis(['ckey','y'],axis=1).assign(tier='provenance'),
             Bt[['composition_key','target_ev']].set_axis(['ckey','y'],axis=1).assign(tier='compilation')])
orows=[]
for tier,Ed in [('provenance',E[E.tier=='provenance']),('compilation',E[E.tier=='compilation']),('both',E)]:
    st=Ed.groupby('ckey').y.agg(n='size',med='median',mn='min',mx='max').reset_index()
    st['rng']=st.mx-st.mn
    M=st.merge(sp,on='ckey')
    allrep=Ed.merge(sp,on='ckey')
    for f,lab in [('hse_med','HSE'),('gga_med','GGA')]:
        e_all=(allrep[f]-allrep.y).abs()
        e_med=(M[f]-M.med).abs()
        keep=M[M.rng<=0.1]; keepall=allrep[allrep.ckey.isin(set(keep.ckey))]
        orows.append(dict(tier=tier,functional=lab,
            n_reports=len(allrep),n_comps=len(M),
            MAE_every_report=e_all.mean(),
            MAE_composition_median=e_med.mean(),
            delta_median_convention=e_med.mean()-e_all.mean(),
            n_comps_multi=int((M.n>1).sum()),
            n_comps_kept_span_le_0p1=len(keep),n_comps_removed=len(M)-len(keep),
            MAE_median_after_discard=(keep[f]-keep.med).abs().mean(),
            delta_discard=(keep[f]-keep.med).abs().mean()-e_med.mean(),
            MAE_every_report_after_discard=(keepall[f]-keepall.y).abs().mean()))
O=pd.DataFrame(orows); O.to_csv(B+'item6_conventions.csv',index=False)
print('\n--- item 6: what the two cleaning conventions do ---')
print(O.round(4).to_string(index=False))
