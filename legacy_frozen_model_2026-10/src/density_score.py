import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr

def score(y, parts):
    w, mu, sigma = parts
    assert abs(w.sum() - 1) < 1e-7 and np.all(w > 0) and np.all(sigma > 0)
    z = (math.log(y) - mu) / sigma
    nll = -float(logsumexp(np.log(w) - math.log(y) - np.log(sigma)
                          - .5 * math.log(2 * math.pi) - .5 * z * z))
    means = np.exp(mu + .5 * sigma * sigma)
    absolute = y * (2 * ndtr(z) - 1) + means * (1 - 2 * ndtr(z - sigma))
    distance = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    minimum = (means[:, None] * ndtr((mu[None, :] - mu[:, None] - sigma[:, None] ** 2) / distance)
               + means[None, :] * ndtr((mu[:, None] - mu[None, :] - sigma[None, :] ** 2) / distance))
    pair = means[:, None] + means[None, :] - 2 * minimum
    crps = float(w @ absolute - .5 * np.sum(w[:, None] * w[None, :] * pair))
    def quantile(p):
        return brentq(lambda x: float(w @ ndtr((math.log(x) - mu) / sigma)) - p,
                      1e-200, float(np.exp(np.max(mu + 12 * sigma))), xtol=1e-9)
    qs = {p: quantile(p) for p in (.05, .1, .25, .5, .75, .9, .95)}
    result = dict(nll=nll, crps_ev=crps, median_mae_ev=abs(y - qs[.5]), median_ev=qs[.5])
    for level, lo, hi in ((50, .25, .75), (80, .1, .9), (90, .05, .95)):
        result['coverage_' + str(level)] = float(qs[lo] <= y <= qs[hi])
        result['width_' + str(level) + '_ev'] = qs[hi] - qs[lo]
    assert all(math.isfinite(v) for v in result.values()) and crps >= -1e-8
    return result
