"""Item 4: error budget on matched provenance records. Writes per_record.csv and budget.csv."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
g=pd.read_csv(B+'snumat_by_comp_sg.csv')
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
meta=json.load(open('/home/czx/2024/BandgapFusion_MeanTargetControl_20261023/source/data/reports/metadata.json'))
msg={m['measurement_id']:m.get('model_space_group_number') for m in meta}
for c in ['phase_label','gap_definition','measurement_method','source_group_id','specimen_group_id']:
    P[c]=P[c].fillna('NA').astype(str).str.strip()
def mcat(r):
    t=(r.gap_definition+' '+r.measurement_method).lower()
    if 'thermoreflect' in t: return 'thermoreflectance'
    if 'photolum' in t or ' pl ' in t or 'emission' in t: return 'PL'
    if 'ellipso' in t: return 'ellipsometry'
    if 'tauc' in t or 'kubelka' in t or '(alpha' in t:
        return 'Tauc-indirect' if 'indirect' in t or '1/2' in t else 'Tauc-direct'
    if 'absorption edge' in t or 'onset' in t or 'absorption' in t or 'reflect' in t or 'transmit' in t: return 'absorption-edge'
    return 'other'
P['mcat']=P.apply(mcat,axis=1)
# what the measurement probes: a direct-allowed analysis targets the lowest DIPOLE-ALLOWED DIRECT transition
# (SNUMAT "optical"); an indirect Tauc or a bare absorption edge targets the fundamental gap (SNUMAT "fundamental").
DIRECT={'Tauc-direct','ellipsometry','thermoreflectance'}
FUND={'Tauc-indirect','absorption-edge','PL','other'}
P['probes']=np.where(P.mcat.isin(DIRECT),'optical_direct','fundamental')
print('method categories (237):',P.mcat.value_counts().to_dict())
print('probes:',P.probes.value_counts().to_dict())
sgr=pd.to_numeric(P.space_group,errors='coerce')
sgm=pd.to_numeric(pd.Series([msg.get(r) for r in P.record_id]),errors='coerce')
P['sg_use']=sgr.fillna(sgm)
P['sg_source']=np.where(sgr.notna(),'reported',np.where(P.sg_use.notna(),'model_reference','none'))
ex=P.groupby('composition_key').bandgap_ev.agg(exp_n='size',exp_med='median',exp_min='min',exp_max='max',exp_sd='std').reset_index()
ex['exp_range']=ex.exp_max-ex.exp_min
sp=g.groupby('ckey').agg(n_sg=('sg','size'),hse_med_poly=('hse','median'),gga_med_poly=('gga','median'),
    hse_min=('hse','min'),hse_max=('hse','max'),gga_min=('gga','min'),gga_max=('gga','max'),
    hse_opt_med=('hse_opt','median'),gga_opt_med=('gga_opt','median'),
    hse_opt_min=('hse_opt','min'),hse_opt_max=('hse_opt','max'),gga_opt_min=('gga_opt','min'),gga_opt_max=('gga_opt','max')).reset_index()
sp['hse_poly_range']=sp.hse_max-sp.hse_min; sp['gga_poly_range']=sp.gga_max-sp.gga_min
gs=g.dropna(subset=['Free_energy']).sort_values('Free_energy').drop_duplicates('ckey')[['ckey','sg','hse','gga','hse_opt','gga_opt']]
gs.columns=['ckey','sg_gs','hse_gs','gga_gs','hse_opt_gs','gga_opt_gs']
cell=g.rename(columns={'sg':'sg_use','hse':'hse_cell','gga':'gga_cell','hse_opt':'hse_opt_cell','gga_opt':'gga_opt_cell',
    'dir_hse':'dir_hse_cell','n_rec':'n_snumat_rec'})[['ckey','sg_use','hse_cell','gga_cell','hse_opt_cell','gga_opt_cell','dir_hse_cell','n_snumat_rec']]
D=P.merge(ex,on='composition_key').merge(sp,left_on='composition_key',right_on='ckey',how='inner')
D=D.merge(gs,on='ckey',how='left').merge(cell,on=['ckey','sg_use'],how='left')
D['sg_matched']=D.hse_cell.notna()
print('provenance records on SNUMAT-matched compositions: %d (compositions %d); space-group matched: %d (compositions %d)'%(
    len(D),D.ckey.nunique(),D.sg_matched.sum(),D[D.sg_matched].ckey.nunique()))

def dist_to_interval(v,lo,hi):
    return np.maximum(0.0,np.maximum(lo-v,v-hi))

for f in ['hse','gga']:
    # (0) conventional: fundamental gap of the sg-matched cell, else the polymorph median
    D[f+'_ref']=D[f+'_cell'].fillna(D[f+'_med_poly'])
    D[f+'_opt_ref']=D[f+'_opt_cell'].fillna(D[f+'_opt_med'])
    D['abs_err_'+f]=(D[f+'_ref']-D.bandgap_ev).abs()
    # (1) optical/fundamental term: magnitude, and the definition-matched value
    D['term_optical_'+f]=np.where(D.probes=='optical_direct',D[f+'_opt_ref']-D[f+'_ref'],0.0)
    D[f+'_defmatched']=np.where(D.probes=='optical_direct',D[f+'_opt_ref'],D[f+'_ref'])
    D['abs_err_defmatched_'+f]=(D[f+'_defmatched']-D.bandgap_ev).abs()
    # (2) polymorph-choice term: half the range across the composition's space groups, for the probed quantity
    rng_f=D[f+'_max']-D[f+'_min']; rng_o=D[f+'_opt_max']-D[f+'_opt_min']
    D['term_polymorph_'+f]=np.where(D.probes=='optical_direct',rng_o,rng_f)/2.0
    # (3) specimen/source term: experimental SD within the composition
    D['term_specimen']=D.exp_sd.fillna(0.0)
    # (4) residuals
    # upper bound: nothing explained away
    D['residual_upper_'+f]=D['abs_err_'+f]
    # mid: definition matched + sg matched, experiment taken as its interval of reports
    D['residual_mid_'+f]=dist_to_interval(D[f+'_defmatched'].values,D.exp_min.values,D.exp_max.values)
    # lower bound: best polymorph x best definition, against the experimental interval
    lo=np.minimum(D[f+'_min'],D[f+'_opt_min']); hi=np.maximum(D[f+'_max'],D[f+'_opt_max'])
    best=np.clip(D.exp_med.values,lo.values,hi.values)   # closest attainable computed value
    D['residual_lower_'+f]=dist_to_interval(best,D.exp_min.values,D.exp_max.values)
    D['inside_exp_range_'+f]=(D[f+'_ref']>=D.exp_min)&(D[f+'_ref']<=D.exp_max)
    D['inside_exp_range_defmatched_'+f]=(D[f+'_defmatched']>=D.exp_min)&(D[f+'_defmatched']<=D.exp_max)
    D['inside_any_polymorph_'+f]=(lo<=D.exp_max)&(hi>=D.exp_min)
D.to_csv(B+'per_record.csv',index=False)
print('per_record.csv written:',len(D),'rows')
rows=[]
def add(term,x,src,note=''):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    rows.append(dict(term=term,n=len(x),point_estimate_ev=x.mean(),median_ev=np.median(x),q25_ev=np.percentile(x,25),
        q75_ev=np.percentile(x,75),p90_ev=np.percentile(x,90),source=src,note=note))
for f,lab in [('hse','HSE'),('gga','GGA')]:
    add(lab+': total |computed fundamental - experiment|',D['abs_err_'+f],'per_record.csv','conventional quantity; sg-matched cell else polymorph median')
    add(lab+': optical/fundamental term (magnitude)',D['term_optical_'+f],'SNUMAT optical minus fundamental, matched cell','zero where the analysis targets the fundamental gap (86 of 237 records)')
    add(lab+': total after definition matching',D['abs_err_defmatched_'+f],'per_record.csv','effect of term 1 on the budget')
    add(lab+': polymorph-choice term (magnitude)',D['term_polymorph_'+f],'half the range of the probed quantity over the composition SNUMAT space groups','')
    add(lab+': specimen/source term (magnitude)',D.term_specimen,'experimental SD within the composition (Supp Data 1)','0 where only one report')
    add(lab+': residual, upper bound',D['residual_upper_'+f],'per_record.csv','no term assigned; equals the total')
    add(lab+': residual, mid',D['residual_mid_'+f],'per_record.csv','definition and space group matched, experiment as the interval of its reports')
    add(lab+': residual, lower bound',D['residual_lower_'+f],'per_record.csv','best polymorph and best gap definition vs the experimental interval')
T=pd.DataFrame(rows); T.to_csv(B+'budget.csv',index=False)
pd.set_option('display.width',280); pd.set_option('display.max_columns',20); pd.set_option('display.max_colwidth',58)
print(T.round(4).to_string(index=False))
print()
for f,lab in [('hse','HSE'),('gga','GGA')]:
    print('%s: computed inside the experimental spread: fundamental %d/%d (%.1f%%), definition-matched %d/%d (%.1f%%), some polymorph+definition %d/%d (%.1f%%)'%(
      lab,D['inside_exp_range_'+f].sum(),len(D),100*D['inside_exp_range_'+f].mean(),
      D['inside_exp_range_defmatched_'+f].sum(),len(D),100*D['inside_exp_range_defmatched_'+f].mean(),
      D['inside_any_polymorph_'+f].sum(),len(D),100*D['inside_any_polymorph_'+f].mean()))
    print('   bias (computed - experiment) %+.4f eV; sg-matched subset MAE %.4f (n=%d)'%(
      (D[f+'_ref']-D.bandgap_ev).mean(),D[D.sg_matched]['abs_err_'+f].mean(),D.sg_matched.sum()))
# optical term by HSE direct/indirect character
o=D.dropna(subset=['dir_hse_cell'])
print('\nHSE optical-minus-fundamental on the matched cells, by HSE character: '+
      '; '.join('%s n=%d median %.3f mean %.3f'%(k,len(v),(v.hse_opt_cell-v.hse_cell).median(),(v.hse_opt_cell-v.hse_cell).mean()) for k,v in o.groupby('dir_hse_cell')))
D[['record_id','formula','composition_key','mcat','probes','bandgap_ev','exp_n','exp_med','exp_min','exp_max','exp_sd',
   'sg_use','sg_source','sg_matched','n_sg','hse_ref','hse_opt_ref','hse_defmatched','gga_ref','gga_opt_ref','gga_defmatched',
   'abs_err_hse','abs_err_gga','term_optical_hse','term_polymorph_hse','term_specimen',
   'residual_lower_hse','residual_mid_hse','residual_upper_hse']].to_csv(B+'fig_csv/fig_item4_budget_terms.csv',index=False)
