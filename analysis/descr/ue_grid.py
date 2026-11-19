import numpy as np
from scipy.special import ndtr
def install(ev, e4):
    YL, DY = ev.YL, ev.DY
    pos = YL > 0; LY = np.log(np.where(pos, YL, 1.0))
    old = ev.unit_eval
    def ue(dist, ys):
        if dist[0] != "ln": return old(dist, ys)
        _, w, mu, s = dist
        G = np.where(pos, (w[None] * ndtr((LY[:, None] - mu[None]) / s[None])).sum(1), 0.0)
        q = lambda a: float(np.interp(a, G, YL))
        med, lo, hi = q(.5), q(.1), q(.9)
        ys = np.asarray(ys, float)
        crps = np.array([float(((G - (YL >= y)) ** 2).sum() * DY) for y in ys])
        return dict(crps=crps, nll=-np.log(e4.dens(dist, ys) + 1e-300), mae=np.abs(ys - med), cov80=((lo <= ys) & (ys <= hi)).astype(float), median=med, width80=hi - lo, pit=np.interp(ys, YL, G))
    ev.unit_eval = ue
