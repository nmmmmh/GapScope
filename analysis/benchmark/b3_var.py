"""Item 3: experimental spread expressed in eV (phase / source / specimen components)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
d=pd.read_csv(R+'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv',encoding='utf-8-sig')
for c in ['phase_label','gap_definition','measurement_method','sample_form','source_group_id','specimen_group_id']:
    d[c]=d[c].fillna('NA').astype(str).str.strip()
def comps(cells,col):
    """one-way random-effects moment estimates pooled over cells; returns (sigma_between, sigma_within) in eV"""
    SSB=SSW=K=N=0
    for _,gg in cells:
        m=gg.bandgap_ev.mean()
        if gg[col].nunique()<2:
            SSW+=((gg.bandgap_ev-m)**2).sum(); N+=len(gg)-1; continue
        SSB+=sum(len(h)*(h.bandgap_ev.mean()-m)**2 for _,h in gg.groupby(col))
        SSW+=sum(((h.bandgap_ev-h.bandgap_ev.mean())**2).sum() for _,h in gg.groupby(col))
        K+=gg[col].nunique()-1; N+=len(gg)-gg[col].nunique()
    MSW=SSW/N if N else np.nan; MSB=SSB/K if K else np.nan
    nbar=sum(len(g) for _,g in cells)/max(1,sum(g[col].nunique() for _,g in cells))
    vb=max(0.,(MSB-MSW)/nbar)
    return dict(sd_between=np.sqrt(vb),sd_within=np.sqrt(MSW),share_raw=SSB/(SSB+SSW),K=K,N=N,
                n_reports=sum(len(g) for _,g in cells),n_cells=len(cells))
out=[]
# level 1: composition -> phase
cellC=[(k,g) for k,g in d.groupby('composition_key') if len(g)>=2 and g.phase_label.nunique()>=1]
r=comps(cellC,'phase_label'); r.update(level='within composition: between phases / within phase',grouping='phase_label'); out.append(r)
tot=np.sqrt(sum(((g.bandgap_ev-g.bandgap_ev.mean())**2).sum() for _,g in cellC)/sum(len(g)-1 for _,g in cellC))
print('total within-composition SD %.4f eV over %d reports in %d compositions (n>=2)'%(tot,sum(len(g) for _,g in cellC),len(cellC)))
# level 2: within (composition, phase) cells
cellP=[(k,g) for k,g in d.groupby(['composition_key','phase_label']) if len(g)>=2]
for col,lab in [('source_group_id','within phase: between sources / within source'),
                ('specimen_group_id','within phase: between specimens / within specimen'),
                ('measurement_method','within phase: between methods / within method'),
                ('gap_definition','within phase: between gap definitions / within')]:
    r=comps(cellP,col); r.update(level=lab,grouping=col); out.append(r)
wp=np.sqrt(sum(((g.bandgap_ev-g.bandgap_ev.mean())**2).sum() for _,g in cellP)/sum(len(g)-1 for _,g in cellP))
print('total within-(composition,phase) SD %.4f eV over %d reports in %d cells'%(wp,sum(len(g) for _,g in cellP),len(cellP)))
T=pd.DataFrame(out)[['level','grouping','n_cells','n_reports','sd_between','sd_within','share_raw','K','N']]
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(T.round(4).to_string(index=False))
T.to_csv(B+'item3_variance_components_ev.csv',index=False)
# per-composition experimental SD and range, for the budget
pc=d.groupby('composition_key').bandgap_ev.agg(n='size',sd='std',rng=lambda s:s.max()-s.min(),med='median').reset_index()
pc.to_csv(B+'fig_csv/fig_item3_per_composition_exp.csv',index=False)
mm=pc[pc.n>=2]
print('\nper-composition experimental SD (n>=2, %d compositions): median %.4f, mean %.4f, q75 %.4f, p90 %.4f eV'%(
    len(mm),mm.sd.median(),mm.sd.mean(),mm.sd.quantile(.75),mm.sd.quantile(.90)))
print('per-composition experimental range: median %.4f, p90 %.4f eV'%(mm.rng.median(),mm.rng.quantile(.90)))
# specimen-level: single-specimen repeat spread
sp=d.groupby(['composition_key','phase_label','specimen_group_id']).bandgap_ev.agg(n='size',sd='std')
print('specimen cells with >=2 reports: %d, pooled within-specimen SD %.4f eV'%((sp.n>=2).sum(),np.sqrt((sp[sp.n>=2].sd**2*(sp[sp.n>=2].n-1)).sum()/(sp[sp.n>=2].n-1).sum())))
