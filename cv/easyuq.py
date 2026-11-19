"""EasyUQ baseline (Walz, Henzi, Ziegel, Gneiting 2024): isotonic distributional regression (IDR) of observed values on the
ROOST point prediction (mean of the 3-seed ROOST mu), fitted per outer fold on training-fold data only, i.e. on the
out-of-fold ROOST predictions of the four other folds (each scored by a ROOST ensemble that never saw it). Evaluation only.
Thresholds are a 0.02 eV grid (pool-adjacent-violators per threshold on 1{y<=t}); the IDR CDF becomes atoms at bin centres.
EASYUQ_IDR: atoms with a 0.01 eV Laplace kernel (numerically a step distribution). EASYUQ_KS: kernel-smoothed (Laplace kernel)
with the kernel scale chosen per outer fold by 5-fold CV log score inside the training data."""
import sys, pickle, json
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
import numpy as np
import evaluate_cv_v2 as E
from sklearn.isotonic import IsotonicRegression
ev = E.ev
A, Akey, Bkeys, Bvals = ev.A, ev.Akey, ev.Bkeys, ev.Bvals
H = 0.02
BGRID = [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]


def idr_fit_predict(X, Y, W, Xt, edges):
    F = np.zeros((len(Xt), len(edges)))
    for j, e in enumerate(edges):
        z = (Y <= e).astype(float)
        if z.min() == z.max():
            F[:, j] = z[0]
        else:
            F[:, j] = IsotonicRegression(increasing=False, out_of_bounds='clip', y_min=0., y_max=1.).fit(X, z, sample_weight=W).predict(Xt)
    F = np.clip(np.maximum.accumulate(F, axis=1), 0, 1)
    F[:, -1] = 1.0
    return np.diff(F, axis=1, prepend=0.0)          # atom masses on (e_{j-1}, e_j]


def lap_nll(P, C, b, y):
    dens = (P * np.exp(-np.abs(y[:, None] - C[None]) / b) / (2 * b)).sum(1)
    return -np.log(dens + 1e-300)


def run(variant):
    rA, rB = ev.load_roost(variant)
    xA = {i: float(np.mean(d[1])) for i, d in rA[0].items()}
    xB = {k: float(np.mean(d[1])) for k, d in rB.items()}
    fd = E.FOLDS[variant]
    # comp-level pair table: (comp id, x, y, weight) with composition mass 1
    rows = []
    wcomp = {}
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]
        for i in ii:
            rows.append((k, xA[i], float(A[i]['gap_ev']), ev.wA[i] / ev.wA[ii].sum()))
    for k in Bkeys:
        v = Bvals[k]
        for y in v:
            rows.append((k, xB[k], float(y), 1.0 / len(v)))
    comp = np.array([r[0] for r in rows]); X = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows]); W = np.array([r[3] for r in rows])
    # a composition present in both tiers (A and B) counts mass 1 per tier unit; keep as is
    edges = np.arange(np.floor(Y.min() / H) * H - H, Y.max() + 2 * H, H)
    centers = (edges[1:] + edges[:-1]) / 2
    # drop first edge column (mass below lowest edge is 0): atoms live on edges[1:]
    out = {'EASYUQ_IDR': ({v: {} for v in range(4)}, {}), 'EASYUQ_KS': ({v: {} for v in range(4)}, {})}
    info = {}
    rng = np.random.default_rng(20261113)
    for f in range(1, 6):
        trm = np.array([fd[k] != f for k in comp])
        Xtr, Ytr, Wtr, Ctr = X[trm], Y[trm], W[trm], comp[trm]
        # kernel scale by inner 5-fold CV (random composition groups) mean composition-weighted log score
        ucomp = np.array(sorted(set(Ctr))); perm = rng.permutation(len(ucomp)); grp = {k: g % 5 for g, k in zip(range(len(ucomp)), ucomp[perm])}
        gi = np.array([grp[k] for k in Ctr])
        sc = np.zeros(len(BGRID)); wsum = 0.
        for g in range(5):
            te = gi == g
            P = idr_fit_predict(Xtr[~te], Ytr[~te], Wtr[~te], Xtr[te], edges)[:, 1:]
            P = P / P.sum(1, keepdims=True)
            for bi, b in enumerate(BGRID):
                sc[bi] += (lap_nll(P, centers, b, Ytr[te]) * Wtr[te]).sum()
            wsum += Wtr[te].sum()
        bsel = BGRID[int(np.argmin(sc))]
        info[f] = dict(kernel_scale=bsel, cv_nll=[float(x / wsum) for x in sc], n_train_pairs=int(trm.sum()), n_thresholds=len(edges))
        # test units of this fold
        iA = [i for i in range(len(A)) if fd[Akey[i]] == f]
        kB = [k for k in Bkeys if fd[k] == f]
        xt = np.array([xA[i] for i in iA] + [xB[k] for k in kB])
        P = idr_fit_predict(Xtr, Ytr, Wtr, xt, edges)[:, 1:]
        P = P / P.sum(1, keepdims=True)
        for n, (nm, b) in enumerate((('EASYUQ_IDR', 0.01), ('EASYUQ_KS', bsel))):
            for r, key in enumerate([('A', i) for i in iA] + [('B', k) for k in kB]):
                p = P[r]; m = p > 1e-5
                d = ('wlap', p[m] / p[m].sum(), centers[m].copy(), np.full(int(m.sum()), b))
                if key[0] == 'A':
                    for v in range(4):
                        out[nm][0][v][key[1]] = d
                else:
                    out[nm][1][key[1]] = d
        print(variant, 'fold', f, 'kernel scale', bsel, 'train pairs', int(trm.sum()), flush=True)
    return out, info


if __name__ == '__main__':
    for variant in sys.argv[1:]:
        out, info = run(variant)
        pickle.dump(out, open(E.CV / f'easyuq_{variant}.pkl', 'wb'))
        json.dump(info, open(E.CV / f'easyuq_{variant}_info.json', 'w'), indent=1)
        print('saved', variant)
