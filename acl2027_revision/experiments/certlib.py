"""Certificate library for retention under evaluator churn (revision experiments).

All certification quantities follow Sections 3-5 of the paper.  An *instance*
is stored after removing invariant selected rows (they cancel from the count
gap) and non-selected rows (they never enter the returned count):

    c[i, r] >= v[i, r] >= 0   integer endpoint cap and true contribution of
                              candidate r on active selected row i
    b                         integer budget in the same units
    obs[i, r]                 bool, entry already observed (history)

Uniform single-member families have c[i, r] = p_i.  Fixed nonnegative
replacement weights (Appendix R) give column-dependent caps after scaling the
weights to integers.  Odd-median rows are endpoint-only: v in {0, c}.

Functions here are offline constructors: they use the true matrix to choose
evidence, exactly like Theorems 1-3.  The verifier only needs `bounds`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

INF = float("inf")


# ----------------------------------------------------------------------------
# Instances, bounds and capacities
# ----------------------------------------------------------------------------
@dataclass
class Instance:
    c: np.ndarray            # (m, T) int caps
    v: np.ndarray            # (m, T) int true contributions
    b: int                   # integer budget
    obs: np.ndarray = None   # (m, T) bool history
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=np.int64)
        self.v = np.asarray(self.v, dtype=np.int64)
        if self.obs is None:
            self.obs = np.zeros_like(self.c, dtype=bool)
        assert self.c.shape == self.v.shape
        assert np.all(self.v >= 0) and np.all(self.v <= self.c)

    @property
    def m(self):
        return self.c.shape[0]

    @property
    def T(self):
        return self.c.shape[1]


def capacity(w, b):
    """V(w): largest number of rows whose costs fit in budget b (sorted prefix)."""
    w = np.sort(np.asarray(w, dtype=np.int64))
    cs = np.cumsum(w)
    return int(np.searchsorted(cs, b, side="right"))


def bounds(inst: Instance, obs=None):
    obs = inst.obs if obs is None else obs
    lo = np.where(obs, inst.v, 0).sum(axis=1)
    up = lo + np.where(obs, 0, inst.c).sum(axis=1)
    return lo, up


def count_bounds(inst: Instance, obs=None):
    lo, up = bounds(inst, obs)
    return capacity(up, inst.b), capacity(lo, inst.b)   # (V-, V+) on active rows


def true_optimum(inst: Instance):
    return capacity(inst.v.sum(axis=1), inst.b)


def gap_ok(inst: Instance, obs, eps):
    vm, vp = count_bounds(inst, obs)
    return vp - vm <= eps


# ----------------------------------------------------------------------------
# Brute force (tiny instances): joint, safety, exclusion, best union, split
# ----------------------------------------------------------------------------
def _safe_ok(inst, obs, s):
    lo, up = bounds(inst, obs)
    return capacity(up, inst.b) >= s


def _excl_ok(inst, obs, p):
    if p > inst.m:
        return True
    lo, up = bounds(inst, obs)
    return capacity(lo, inst.b) <= p - 1


def brute_force(inst: Instance, eps: int):
    """Exhaustive minima over additional query subsets (|unobserved| <= ~16)."""
    free = [(i, r) for i in range(inst.m) for r in range(inst.T) if not inst.obs[i, r]]
    L = true_optimum(inst)
    targets = list(range(max(0, L - eps), L + 1))
    n = len(free)
    best_joint = INF
    S = {s: INF for s in targets}
    E = {}
    safe_sets = {s: [] for s in targets}
    excl_sets = {}
    for s in targets:
        E[s + eps + 1] = INF
        excl_sets[s + eps + 1] = []
    masks_by_size = sorted(range(1 << n), key=lambda x: bin(x).count("1"))
    for mask in masks_by_size:
        k = bin(mask).count("1")
        obs = inst.obs.copy()
        for j in range(n):
            if mask >> j & 1:
                obs[free[j]] = True
        if k < best_joint and gap_ok(inst, obs, eps):
            best_joint = k
        for s in targets:
            if k <= S[s] and _safe_ok(inst, obs, s):
                if k < S[s]:
                    S[s] = k
                    safe_sets[s] = []
                safe_sets[s].append(mask)
            p = s + eps + 1
            if k <= E[p] and _excl_ok(inst, obs, p):
                if k < E[p]:
                    E[p] = k
                    excl_sets[p] = []
                excl_sets[p].append(mask)
    split = min(S[s] + E[s + eps + 1] for s in targets)
    union = INF
    for s in targets:
        for a in safe_sets[s]:
            for e in excl_sets[s + eps + 1]:
                union = min(union, bin(a | e).count("1"))
    return dict(joint=best_joint, union=union, split=split, L=L)


# ----------------------------------------------------------------------------
# Outcome-group MILP (Theorem 3 / Proposition 4), generalised to (cap, value)
# ----------------------------------------------------------------------------
def _groups(inst: Instance):
    """Per row: list of (c, v, count) over unobserved entries."""
    out = []
    for i in range(inst.m):
        d = {}
        for r in range(inst.T):
            if not inst.obs[i, r]:
                key = (int(inst.c[i, r]), int(inst.v[i, r]))
                d[key] = d.get(key, 0) + 1
        out.append(sorted(d.items()))
    return out


class _Model:
    def __init__(self):
        self.lb, self.ub, self.integ, self.obj = [], [], [], []
        self.rows = []  # (dict col->coef, lo, hi)

    def var(self, lb, ub, integer, obj=0.0):
        self.lb.append(lb)
        self.ub.append(ub)
        self.integ.append(1 if integer else 0)
        self.obj.append(obj)
        return len(self.lb) - 1

    def con(self, coefs, lo, hi):
        self.rows.append((coefs, lo, hi))

    def solve(self, time_limit=60.0, presolve=True):
        n = len(self.lb)
        A = lil_matrix((max(1, len(self.rows)), n))
        lo = np.full(max(1, len(self.rows)), -np.inf)
        hi = np.full(max(1, len(self.rows)), np.inf)
        for k, (coefs, l, h) in enumerate(self.rows):
            for j, a in coefs.items():
                A[k, j] = a
            lo[k], hi[k] = l, h
        cons = [LinearConstraint(A.tocsr(), lo, hi)] if self.rows else []
        res = milp(np.array(self.obj, float), constraints=cons,
                   integrality=np.array(self.integ),
                   bounds=Bounds(np.array(self.lb, float), np.array(self.ub, float)),
                   options=dict(time_limit=time_limit, mip_rel_gap=0.0, presolve=presolve, disp=False))
        return res


def _add_safety(M, inst, groups, a, s, lo0, up0):
    """Adds y, z and safety constraints for target s; returns y."""
    m = inst.m
    y = [M.var(0, 1, True) for _ in range(m)]
    M.con({j: 1.0 for j in y}, s, s)
    coefs = {}
    for i in range(m):
        coefs[y[i]] = coefs.get(y[i], 0.0) + float(up0[i])
        for g, ((c, v), d) in enumerate(groups[i]):
            if c - v == 0:
                continue
            z = M.var(0, d, False)
            ag = a[i][g]
            M.con({z: 1.0, ag: -1.0}, -np.inf, 0)               # z <= a
            M.con({z: 1.0, y[i]: -float(d)}, -np.inf, 0)        # z <= d y
            M.con({z: 1.0, ag: -1.0, y[i]: -float(d)}, -float(d), np.inf)  # z >= a - d(1-y)
            coefs[z] = coefs.get(z, 0.0) - float(c - v)
    M.con(coefs, -np.inf, inst.b)
    return y


def _add_exclusion(M, inst, groups, a, p, lo0, tmax):
    m = inst.m
    t = M.var(0, tmax, True)
    ws = []
    for i in range(m):
        w = M.var(0, tmax, False)
        coefs = {w: 1.0, t: -1.0}
        for g, ((c, v), d) in enumerate(groups[i]):
            if v:
                coefs[a[i][g]] = float(v)
        M.con(coefs, -float(lo0[i]), np.inf)   # w - t + sum v a >= -lo0
        ws.append(w)
    coefs = {t: float(p)}
    for w in ws:
        coefs[w] = -1.0
    M.con(coefs, inst.b + 1, np.inf)


def _new_a(M, groups, obj=1.0):
    return [[M.var(0, d, True, obj) for ((c, v), d) in groups[i]] for i in range(len(groups))]


def _extract(res, a, groups, inst):
    """Turns group counts into a concrete set of observed entries."""
    obs = inst.obs.copy()
    for i in range(inst.m):
        for g, ((c, v), d) in enumerate(groups[i]):
            k = int(round(res.x[a[i][g]]))
            if k == 0:
                continue
            for r in range(inst.T):
                if k == 0:
                    break
                if not obs[i, r] and not inst.obs[i, r] and inst.c[i, r] == c and inst.v[i, r] == v:
                    obs[i, r] = True
                    k -= 1
    return obs


@dataclass
class OptResult:
    value: float
    lower: float
    closed: bool
    obs: np.ndarray = None


def _targets(inst, eps):
    L = true_optimum(inst)
    return L, list(range(max(0, L - eps), L + 1))


def joint_min(inst: Instance, eps: int, time_limit=60.0, presolve=True):
    L, targets = _targets(inst, eps)
    groups = _groups(inst)
    lo0, up0 = bounds(inst)
    if gap_ok(inst, inst.obs, eps):
        return OptResult(0, 0, True, inst.obs.copy())
    tmax = int(inst.v.sum(axis=1).max())
    best, best_obs, lower, closed = INF, None, INF, True
    for s in targets:
        p = s + eps + 1
        M = _Model()
        a = _new_a(M, groups)
        _add_safety(M, inst, groups, a, s, lo0, up0)
        if p <= inst.m:
            _add_exclusion(M, inst, groups, a, p, lo0, tmax)
        res = M.solve(time_limit, presolve)
        if res.x is None:
            if res.status == 2:   # infeasible target
                continue
            closed = False
            lower = min(lower, 0)
            continue
        val = int(round(res.fun))
        lb = getattr(res, "mip_dual_bound", res.fun)
        lb = max(0, int(np.ceil(lb - 1e-6))) if lb is not None and np.isfinite(lb) else 0
        lower = min(lower, lb)
        if val > lb:
            closed = False
        if val < best:
            best, best_obs = val, _extract(res, a, groups, inst)
    closed = closed and lower >= best
    return OptResult(best, min(lower, best), closed, best_obs)


def safety_min(inst: Instance, s: int, time_limit=60.0):
    groups = _groups(inst)
    lo0, up0 = bounds(inst)
    if capacity(up0, inst.b) >= s:
        return OptResult(0, 0, True, inst.obs.copy())
    M = _Model()
    a = _new_a(M, groups)
    _add_safety(M, inst, groups, a, s, lo0, up0)
    res = M.solve(time_limit)
    if res.x is None:
        return OptResult(INF, INF, res.status == 2)
    return OptResult(int(round(res.fun)), res.fun, True, _extract(res, a, groups, inst))


def exclusion_min(inst: Instance, p: int):
    """Exact greedy: for each threshold t, per-row marginal gains of
    min(l_i, t) are nonincreasing when entries are taken in decreasing v."""
    m = inst.m
    if p > m:
        return 0
    lo0, _ = bounds(inst)
    if capacity(lo0, inst.b) <= p - 1:
        return 0
    tmax = int(inst.v.sum(axis=1).max())
    if tmax > 4000:          # large weighted units: use the exact MILP instead
        return exclusion_min_milp(inst, p)
    vals = [sorted((int(inst.v[i, r]) for r in range(inst.T) if not inst.obs[i, r]), reverse=True)
            for i in range(m)]
    best = INF
    for t in range(0, tmax + 1):
        need = inst.b + 1 + (m - p) * t - int(np.minimum(lo0, t).sum())
        if need <= 0:
            return 0
        gains = []
        for i in range(m):
            cur = int(lo0[i])
            for x in vals[i]:
                g = min(cur + x, t) - min(cur, t)
                if g <= 0:
                    break
                gains.append(g)
                cur += x
        gains.sort(reverse=True)
        cs = np.cumsum(gains) if gains else np.array([0])
        if cs[-1] >= need:
            best = min(best, int(np.searchsorted(cs, need)) + 1)
    return best


def exclusion_min_milp(inst: Instance, p: int, time_limit=60.0):
    if p > inst.m:
        return 0
    groups = _groups(inst)
    lo0, _ = bounds(inst)
    if capacity(lo0, inst.b) <= p - 1:
        return 0
    M = _Model()
    a = _new_a(M, groups)
    _add_exclusion(M, inst, groups, a, p, lo0, int(inst.v.sum(axis=1).max()))
    res = M.solve(time_limit)
    return INF if res.x is None else int(round(res.fun))


def split_min(inst: Instance, eps: int, time_limit=60.0):
    L, targets = _targets(inst, eps)
    best = INF
    for s in targets:
        S = safety_min(inst, s, time_limit).value
        E = exclusion_min(inst, s + eps + 1)
        best = min(best, S + E)
    return best


def union_min(inst: Instance, eps: int, time_limit=60.0):
    """Best union of separately minimum witnesses (Proposition 5 exchange form)."""
    L, targets = _targets(inst, eps)
    groups = _groups(inst)
    lo0, up0 = bounds(inst)
    tmax = int(inst.v.sum(axis=1).max())
    best, closed = INF, True
    for s in targets:
        p = s + eps + 1
        S = safety_min(inst, s, time_limit).value
        E = exclusion_min(inst, p)
        if S == INF or E == INF:
            continue
        if E == 0 or S == 0:
            best = min(best, S + E)
            continue
        M = _Model()
        a = _new_a(M, groups, obj=0.0)
        e = _new_a(M, groups, obj=0.0)
        _add_safety(M, inst, groups, a, s, lo0, up0)
        _add_exclusion(M, inst, groups, e, p, lo0, tmax)
        M.con({x: 1.0 for row in a for x in row}, S, S)
        M.con({x: 1.0 for row in e for x in row}, E, E)
        for i in range(inst.m):
            for g, ((c, v), d) in enumerate(groups[i]):
                u = M.var(0, d, True, 1.0)
                M.con({u: 1.0, a[i][g]: -1.0}, 0, np.inf)
                M.con({u: 1.0, e[i][g]: -1.0}, 0, np.inf)
        res = M.solve(time_limit)
        if res.x is None:
            closed = False
            continue
        best = min(best, int(round(res.fun)))
    return best


# ----------------------------------------------------------------------------
# Score-level construction: pooling, flips, caps
# ----------------------------------------------------------------------------
def consensus(scores, pool):
    if pool == "mean":
        return scores.mean(axis=-1)
    return np.median(scores, axis=-1)


def replacement_flips(base, cand, t0, pool):
    """Returns sel (N,), flip tensor (N, M, T) bool for true candidate scores,
    potential tensor (N, M) bool = flips possible at the max-flip endpoint."""
    N, M = base.shape
    T = cand.shape[1]
    sel = consensus(base, pool) >= t0
    flips = np.zeros((N, M, T), dtype=bool)
    pot = np.zeros((N, M), dtype=bool)
    for j in range(M):
        others = np.delete(base, j, axis=1)
        for r in range(T):
            pool_scores = np.concatenate([others, cand[:, r:r + 1]], axis=1)
            flips[:, j, r] = (consensus(pool_scores, pool) >= t0) != sel
        lo_end = np.where(sel, 0.0, 1.0)[:, None]
        pot[:, j] = (consensus(np.concatenate([others, lo_end], axis=1), pool) >= t0) != sel
    return sel, flips, pot


def instance_from_scores(base, cand, t0, pool, k, alpha, weights=None, scale=None):
    """Builds the active-row instance. `weights` (M, T) nonnegative integers
    (replacement weights up to a common factor); budget uses real 2k*alpha
    times the weight normaliser, floored in integer units."""
    sel, flips, pot = replacement_flips(base, cand, t0, pool)
    N, M = base.shape
    T = cand.shape[1]
    if weights is None:
        weights = np.ones((M, T), dtype=np.int64)
    W = int(weights.sum())
    # contribution of entry (i,r): sum_j w_jr flip_ijr ; cap: sum_j w_jr pot_ij
    v = np.einsum("imt,mt->it", flips.astype(np.int64), weights)
    c = np.einsum("im,mt->it", pot.astype(np.int64), weights)
    # budget: R_w(C) = sum_i contrib / (2k W) <= alpha  ->  sum contrib <= 2 k alpha W
    b = int(np.floor(2 * k * alpha * W + 1e-9))
    selected = np.where(sel)[0]
    active = selected[c[selected].sum(axis=1) > 0]
    inst = Instance(c[active], v[active], b,
                    meta=dict(N=N, M=M, T=T, S0=int(sel.sum()), m0=int(len(selected) - len(active)),
                              active=active, k=k, alpha=alpha))
    return inst, dict(sel=sel, flips=flips, pot=pot)


# ----------------------------------------------------------------------------
# Odd-median closed forms (Theorem 1, Theorem 2, Corollary 1), opposition units
# ----------------------------------------------------------------------------
def median_closed_form(z, T, b, eps):
    z = np.sort(np.asarray(z, dtype=np.int64))
    m = len(z)
    L = capacity(z, b)

    def E(p):
        if p > m:
            return 0
        Dp = lambda t: int(np.minimum(z[:p], t).sum())
        tp = next(t for t in range(T + 1) if Dp(t) > b)
        return b + 1 + (m - p) * tp

    return min(max(0, s * T - b) + E(s + eps + 1) for s in range(max(0, L - eps), L + 1))
