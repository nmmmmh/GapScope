"""Per-record CRPS and PIT of the computed value under the main model's calibrated distribution (figure-ready)."""
import pickle, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
B='/data/czx/GapScope_Revision_20261113/benchmark/'; R='/data/czx/GapScope_Revision_20261113/'
D=pd.read_csv(B+'per_record.csv')
PK=pickle.load(open(R+'cv/easyuq_comp_GS_SELCAL_PRE_NZ.pkl','rb'))['EASYUQ_ON_GS_SELCAL_PRE_NZ_KS'][0][0]
TA=torch.load(R+'cv/data/tierA.pt',weights_only=False)['records']
idx={r['measurement_id']:i for i,r in enumerate(TA)}
GRID=np.arange(-0.5,14.0,0.002)
def lap_cdf(z): return np.where(z<0,0.5*np.exp(z),1-0.5*np.exp(-z))
def F_of(d,x):
    _,w,c,b=d; return float((w*lap_cdf((x-c)/b)).sum())
def crps(d,x):
    _,w,c,b=d
    F=(w[None,:]*lap_cdf((GRID[:,None]-c[None,:])/b[None,:])).sum(1)
    return float(np.trapz((F-(GRID>=x).astype(float))**2,GRID))
rows=[]
for _,r in D[D.record_id.isin(idx)].iterrows():
    d=PK[idx[r.record_id]]
    rows.append(dict(record_id=r.record_id,formula=r.formula,composition_key=r.composition_key,
        experiment_ev=r.bandgap_ev,hse_ev=r.hse_ref,gga_ev=r.gga_ref,
        crps_hse=crps(d,r.hse_ref),crps_gga=crps(d,r.gga_ref),crps_experiment=crps(d,r.bandgap_ev),
        pit_hse=F_of(d,r.hse_ref),pit_gga=F_of(d,r.gga_ref),pit_experiment=F_of(d,r.bandgap_ev)))
T=pd.DataFrame(rows); T.to_csv(B+'fig_csv/fig_item5_crps_per_record.csv',index=False)
print('rows',len(T),'| mean CRPS hse %.4f gga %.4f exp %.4f'%(T.crps_hse.mean(),T.crps_gga.mean(),T.crps_experiment.mean()))
