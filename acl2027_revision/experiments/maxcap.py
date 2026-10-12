"""Per-replacement (maximum) churn caps, Appendix S.

Mask C is feasible for every single replacement h = (j, r) when
    |(S0(P) xor S0(P^{j->r})) cap C| <= c := floor(2 k alpha).
Under the completion model, an unobserved entry (i, r) can flip row i under
every replacement j in pot[i, j] simultaneously (candidate score at its
max-flip endpoint), and under none (zero-flip endpoint).  Hence for any fixed
mask the per-replacement worst and best counts are attained jointly, and
V-_max / V+_max are the optima of two 0/1 programs.  Both are NP-hard in
general (Proposition S.2), so we solve them with HiGHS.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def indicator(flips, pot, obs_rows, sel_idx, upper=True):
    """(n_sel, M*T) 0/1 matrix of per-replacement flip indicators on selected
    rows. obs_rows: (n_sel, T) bool observed entries."""
    F = flips[sel_idx]                 # (n, M, T)
    P = pot[sel_idx][:, :, None]       # (n, M, 1)
    O = obs_rows[:, None, :]           # (n, 1, T)
    if upper:
        X = np.where(O, F, np.broadcast_to(P, F.shape))
    else:
        X = np.where(O, F, False)
    return X.reshape(len(sel_idx), -1).astype(np.int8)


def max_retention(X, cap, time_limit=60.0):
    """max |C| s.t. X[C].sum(0) <= cap.  Rows with zero vectors are free."""
    free = ~X.any(1)
    rows = np.where(~free)[0]
    if len(rows) == 0:
        return int(free.sum()), np.where(free)[0], True
    A = X[rows].T.astype(float)
    keep = A.sum(1) > cap
    if not keep.any():
        return int(len(X)), np.arange(len(X)), True
    A = A[keep]
    res = milp(-np.ones(len(rows)), constraints=[LinearConstraint(csr_matrix(A), -np.inf, cap)],
               integrality=np.ones(len(rows)), bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit, mip_rel_gap=0.0, disp=False))
    chosen = rows[np.round(res.x).astype(bool)]
    closed = res.status == 0
    return int(free.sum() + len(chosen)), np.concatenate([np.where(free)[0], chosen]), closed


def per_replacement_counts(flips, sel_idx_in_mask):
    """True per-replacement flip counts of a returned mask (array of row ids)."""
    if len(sel_idx_in_mask) == 0:
        return np.zeros(flips.shape[1] * flips.shape[2])
    return flips[sel_idx_in_mask].reshape(len(sel_idx_in_mask), -1).sum(0)
