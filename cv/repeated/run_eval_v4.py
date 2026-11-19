"""evaluate_cv_v4.main() with scipy brentq maxiter raised (2000) in density_score.quantile: very wide components (sd up to ~10 after width factor)
need more than the default 100 iterations on the [1e-200, exp(mu+12 sd)] bracket; converged cases are unchanged."""
import sys
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv_rep/code')
import evaluate_cv
import density_score as ds
from functools import partial
ds.brentq = partial(ds.brentq, maxiter=2000)
import evaluate_cv_v4 as E
E.main()
