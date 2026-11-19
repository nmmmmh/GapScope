import sys, csv, math
sys.path.insert(0,'/data/czx/GapScope_Revision_20261113/cv')
sys.path.insert(0,'/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, evaluate_cv_v4 as E
from scipy.stats import norm
ev=E.ev
g=E.load_gs_dir('comp','GS_SELCAL_PRE_NZ',seeds=(17,29,41))
fac=E.fit_scale_oof('comp',g[0]); gA,gB=E.apply_scale('comp',*g,fac)
WANT=['Ga2O3','CH5N2PbI3','SiC','TiO2','In2Se3','CrSBr','CsNiBr3','SnSe']
byf={}
for i,r in enumerate(ev.A): byf.setdefault(r['formula'],[]).append(i)
def cdf(y,w,mu,sd):
    y=np.atleast_1d(y); out=np.zeros_like(y,dtype=float)
    for wk,mk,sk in zip(w,mu,sd): out+=wk*norm.cdf((np.log(y)-mk)/sk)
    return out
def quant(p,w,mu,sd):
    lo,hi=1e-4,200.0
    for _ in range(80):
        m=math.sqrt(lo*hi)
        if cdf(m,w,mu,sd)[0]<p: lo=m
        else: hi=m
    return math.sqrt(lo*hi)
rows=[];meta=[]
for f in WANT:
    if f not in byf: print('missing', f); continue
    idxs=byf[f]; have=[i for i in idxs if i in gA[0]]
    kind,w,mu,sd=gA[0][have[0]]
    gaps=sorted(ev.A[i]['gap_ev'] for i in idxs)
    q={p:quant(p,w,mu,sd) for p in (0.01,0.10,0.50,0.90,0.99)}
    lo=min(q[0.01],min(gaps)); hi=max(q[0.99],max(gaps))
    pad=0.12*(hi-lo); x=np.linspace(max(lo-pad,1e-3),hi+pad,600)
    d=np.zeros_like(x)
    for wk,mk,sk in zip(w,mu,sd):
        d+=wk*np.exp(-((np.log(x)-mk)**2)/(2*sk*sk))/(x*sk*math.sqrt(2*math.pi))
    for xi,di in zip(x,d): rows.append((f,round(float(xi),5),float(di)))
    inside=sum(1 for v in gaps if q[0.10]<=v<=q[0.90])
    meta.append(dict(formula=f,n=len(gaps),gaps=';'.join('%g'%z for z in gaps),
        p10=q[0.10],median=q[0.50],p90=q[0.90],reported_lo=gaps[0],reported_hi=gaps[-1],
        reported_range=gaps[-1]-gaps[0],width80=q[0.90]-q[0.10],inside80=inside))
    print("%-9s n=%-3d reported %.2f-%.2f (%.2f)  pred %.2f [%.2f,%.2f] w80=%.2f inside %d/%d"%(
        f,len(gaps),gaps[0],gaps[-1],gaps[-1]-gaps[0],q[0.50],q[0.10],q[0.90],q[0.90]-q[0.10],inside,len(gaps)))
csv.writer(open('/data/czx/GapScope_Revision_20261113/figs_v4/fig5_density.csv','w',newline='')).writerows([('formula','x','density')]+rows)
wm=csv.DictWriter(open('/data/czx/GapScope_Revision_20261113/figs_v4/fig5_meta.csv','w',newline=''),fieldnames=list(meta[0])); wm.writeheader(); wm.writerows(meta)
