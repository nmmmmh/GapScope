"""Item 2b: ground-state polymorph vs space-group-matched polymorph as the computed reference."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
g=pd.read_csv(B+'snumat_by_comp_sg.csv')
P=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
meta=json.load(open('/home/czx/2024/BandgapFusion_MeanTargetControl_20261023/source/data/reports/metadata.json'))
msg={m['measurement_id']:m.get('model_space_group_number') for m in meta}
P['sg_reported']=pd.to_numeric(P.space_group,errors='coerce')
P['sg_model']=[msg.get(r) for r in P.record_id]
P['sg_use']=P.sg_reported.fillna(pd.to_numeric(P.sg_model,errors='coerce'))
print('provenance records with reported sg %d, with model sg %d, with either %d of 237'%(P.sg_reported.notna().sum(),P.sg_model.notna().sum(),P.sg_use.notna().sum()))
gs=g.sort_values(['Free_energy']).dropna(subset=['Free_energy']).drop_duplicates('ckey')[['ckey','sg','hse','gga','hse_opt','gga_opt']]
gs.columns=['ckey','sg_gs','hse_gs','gga_gs','hse_opt_gs','gga_opt_gs']
lo=g.sort_values('hse').drop_duplicates('ckey')[['ckey','hse']].rename(columns={'hse':'hse_lowest'})
med=g.groupby('ckey').agg(hse_med_poly=('hse','median'),gga_med_poly=('gga','median'),n_sg=('sg','size')).reset_index()
cell=g.rename(columns={'sg':'sg_use','hse':'hse_sgm','gga':'gga_sgm','hse_opt':'hse_opt_sgm','gga_opt':'gga_opt_sgm','dir_hse':'dir_hse_sgm','n_rec':'n_rec_sgm'})
D=P.merge(cell[['ckey','sg_use','hse_sgm','gga_sgm','hse_opt_sgm','gga_opt_sgm','dir_hse_sgm','n_rec_sgm']],
          left_on=['composition_key','sg_use'],right_on=['ckey','sg_use'],how='inner')
D=D.merge(gs,on='ckey',how='left').merge(med,on='ckey',how='left').merge(lo,on='ckey',how='left')
print('provenance records matched to a SNUMAT cell with the SAME space group: %d (compositions %d)'%(len(D),D.ckey.nunique()))
D.to_csv(B+'item2b_sg_matched_records.csv',index=False)
a=D.dropna(subset=['hse_gs'])
print('of those, with a Free_energy ground state available: %d (compositions %d)'%(len(a),a.ckey.nunique()))
rows=[]
for lab,ch,cg in [('HSE','hse_sgm','hse_gs'),('GGA','gga_sgm','gga_gs')]:
    e_sgm=(a[ch]-a.bandgap_ev).abs(); e_gs=(a[cg]-a.bandgap_ev).abs()
    rows.append(dict(functional=lab,n=len(a),MAE_sg_matched=e_sgm.mean(),MAE_ground_state=e_gs.mean(),
        delta_MAE=e_gs.mean()-e_sgm.mean(),median_abs_change=(e_gs-e_sgm).abs().median(),
        p90_abs_change=np.percentile((e_gs-e_sgm).abs(),90),frac_same_cell=float((a.sg_use==a.sg_gs).mean()),
        bias_sg_matched=(a[ch]-a.bandgap_ev).mean(),bias_ground_state=(a[cg]-a.bandgap_ev).mean()))
    # polymorph-median reference too
    mcol='hse_med_poly' if lab=='HSE' else 'gga_med_poly'
    rows[-1]['MAE_polymorph_median']=(a[mcol]-a.bandgap_ev).abs().mean()
T=pd.DataFrame(rows); T.to_csv(B+'item2b_choice_effect.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(T.round(4).to_string(index=False))
# restricted to compositions where ground state != matched cell
b=a[a.sg_use!=a.sg_gs]
print('\nsubset where the ground-state cell differs from the matched cell: n=%d'%len(b))
for lab,ch,cg in [('HSE','hse_sgm','hse_gs'),('GGA','gga_sgm','gga_gs')]:
    print('  %s MAE matched %.4f -> ground state %.4f (delta %+.4f); median |change| %.4f'%(lab,
        (b[ch]-b.bandgap_ev).abs().mean(),(b[cg]-b.bandgap_ev).abs().mean(),
        (b[cg]-b.bandgap_ev).abs().mean()-(b[ch]-b.bandgap_ev).abs().mean(),((b[cg]-b.bandgap_ev).abs()-(b[ch]-b.bandgap_ev).abs()).abs().median()))
