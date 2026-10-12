"""Online entry acquisition policies (Section 6 and the new odd-median
closed-form policy OM-CF of Appendix T).  Policies see only base information,
reference-fitted predictions and revealed entries; certification uses only the
revealed entries (certlib.bounds)."""
from __future__ import annotations

import numpy as np

from certlib import capacity


def _cap_sorted(w, b):
    return capacity(w, b)


def run_policy(inst, pred, policy, eps, seed=0, max_queries=None, oracle=False, record=None):
    """Runs an entry policy until V+ - V- <= eps. Returns number of queries.
    pred: (m,T) predicted contributions in [0, c]. With oracle=True the policy
    receives true contributions as predictions (diagnostic only)."""
    m, T = inst.c.shape
    c, v, b = inst.c, inst.v, inst.b
    P = v.astype(float) if oracle else np.asarray(pred, float)
    obs = inst.obs.copy()
    lo = np.where(obs, v, 0).sum(1).astype(np.int64)
    up = lo + np.where(obs, 0, c).sum(1)
    rng = np.random.default_rng(seed)
    tie = rng.permutation(m)                     # seeded label-blind row priority
    rank = np.empty(m, int)
    rank[tie] = np.arange(m)
    rand_order = [(i, r) for i in range(m) for r in range(T) if not obs[i, r]]
    rng.shuffle(rand_order)
    rptr = 0
    nq = 0
    pmax = c.max(1) if T else np.zeros(m)
    while True:
        vm, vp = _cap_sorted(up, b), _cap_sorted(lo, b)
        if record is not None:
            record.append((nq, vm, vp))
        if vp - vm <= eps:
            return nq
        if max_queries is not None and nq >= max_queries:
            return None
        unq = ~obs
        elig = unq.any(1)
        if policy == "fixed":
            i = min(np.where(elig)[0], key=lambda x: rank[x])
            r = int(np.where(unq[i])[0][0])
        elif policy == "random":
            while obs[rand_order[rptr]]:
                rptr += 1
            i, r = rand_order[rptr]
        elif policy == "width":
            w = np.where(elig, up - lo, -1)
            best = w.max()
            i = min(np.where(w == best)[0], key=lambda x: rank[x])
            r = int(np.where(unq[i])[0][0])
        elif policy == "reference":
            est = lo + np.where(unq, P, 0).sum(1)
            est = np.where(elig, est, np.inf)
            best = est.min()
            i = min(np.where(np.isclose(est, best))[0], key=lambda x: rank[x])
            red = np.where(unq[i], c[i] - P[i], -np.inf)
            r = int(np.argmax(red))
        elif policy in ("frontier", "omcf"):
            i, r = _frontier_choice(policy, c, P, obs, lo, up, b, vm, vp, eps, rank)
        else:
            raise ValueError(policy)
        obs[i, r] = True
        lo[i] += v[i, r]
        up[i] -= c[i, r] - v[i, r]
        nq += 1


def _frontier_choice(policy, c, P, obs, lo, up, b, vm, vp, eps, rank):
    m, T = c.shape
    unq = ~obs
    Fhat = lo + np.where(unq, P, 0).sum(1)
    order = np.lexsort((rank, Fhat))
    if policy == "frontier":
        # Appendix P.3 frontier priority (mean pooling), exclusion target s+1
        s = min(vp, max(vm + 1, capacity(np.ceil(Fhat - 1e-9).astype(np.int64), b)))
        p = s + 1
        safe_on = True
    else:
        # OM-CF: closed-form target from Corollary 1 on predicted opposition counts
        s, p, safe_on = _omcf_target(c, P, obs, lo, up, b, vm, vp, eps, Fhat, rank)
    A = np.zeros(m, bool)
    if safe_on and s > 0:
        A[order[:s]] = True
    Ds = up[A].sum() - b if A.any() else 0
    pri = np.zeros((m, T))
    if safe_on and Ds > 0:
        pri += A[:, None] * (c - P) / Ds
    if p <= m:
        if policy == "frontier":
            t = int(np.ceil(np.sort(Fhat)[p - 1] - 1e-9))
            De = np.maximum(t - lo, 0).sum() - p * t + b + 1
            if De > 0:
                pri += np.minimum(P, np.maximum(t - lo, 0)[:, None]) / De
        else:
            t = _omcf_level(c, P, obs, lo, b, p)
            De = b + 1 + (m - p) * t - np.minimum(lo, t).sum()
            if De > 0:
                pri += (lo < t)[:, None] * np.minimum(P, np.maximum(t - lo, 0)[:, None]) / De
    pri = np.where(unq, pri, -np.inf)
    if not np.isfinite(pri).any() or pri.max() <= 0:
        # fall back to largest width
        w = np.where(unq.any(1), up - lo, -1)
        i = min(np.where(w == w.max())[0], key=lambda x: rank[x])
        return i, int(np.where(unq[i])[0][0])
    best = pri.max()
    cand = np.argwhere(np.isclose(pri, best))
    i, r = min(cand, key=lambda x: (rank[x[0]], x[1]))
    return int(i), int(r)


def _pred_exclusion(c, P, obs, lo, b, p):
    """Predicted minimum number of entries to exclude every p-row set, and the
    Lemma-1 threshold attaining it: for each integer t, per-row capped gains
    min(lo_i + x, t) are concave when entries are taken by decreasing
    predicted contribution, so a global greedy is optimal for that t."""
    m, T = c.shape
    unq = ~obs
    if p > m:
        return 0, 0
    tmax = int(np.ceil((lo + np.where(unq, P, 0).sum(1)).max() - 1e-9)) if m else 0
    vals = [sorted(P[i][unq[i]], reverse=True) for i in range(m)]
    best = (10 ** 9, 0)
    for t in range(0, tmax + 1):
        De = b + 1 + (m - p) * t - np.minimum(lo, t).sum()
        if De <= 0:
            return 0, t
        gains = []
        for i in range(m):
            cur = lo[i]
            for x in vals[i]:
                g = min(cur + x, t) - min(cur, t)
                if g <= 1e-12:
                    break
                gains.append(g)
                cur += x
        gains.sort(reverse=True)
        cg = np.cumsum(gains) if gains else np.array([0.0])
        if cg[-1] >= De - 1e-9:
            q = int(np.searchsorted(cg, De - 1e-9)) + 1
            if q < best[0]:
                best = (q, t)
    return best


def _omcf_level(c, P, obs, lo, b, p):
    return _pred_exclusion(c, P, obs, lo, b, p)[1]


def _omcf_target(c, P, obs, lo, up, b, vm, vp, eps, Fhat, rank):
    """Choose the safe target by the Corollary-1 objective evaluated on
    predicted costs (residual form: agreements needed = predicted upper-cost
    reduction count, oppositions needed from Lemma 1)."""
    m, T = c.shape
    unq = ~obs
    Fr = np.maximum(lo, np.minimum(up, np.rint(Fhat))).astype(np.int64)
    L = capacity(Fr, b)
    L = min(max(L, vm), vp)
    best = None
    for s in range(max(vm, L - eps), L + 1):
        p = s + eps + 1
        # safety: rows with smallest predicted cost; count predicted agreements needed
        order = np.lexsort((rank, Fr))
        need = up[order[:s]].sum() - b if s > 0 else 0
        red = []
        for i in order[:s]:
            red += list((c[i] - P[i])[unq[i]])
        red.sort(reverse=True)
        cs = np.cumsum(red) if red else np.array([0.0])
        qs = 0 if need <= 0 else (int(np.searchsorted(cs, need - 1e-9)) + 1 if cs[-1] >= need - 1e-9 else 10 ** 9)
        qe = _pred_exclusion(c, P, obs, lo, b, p)[0] if p <= m else 0
        cost = qs + qe
        if best is None or cost < best[0]:
            best = (cost, s, p)
    _, s, p = best
    return s, p, s > vm
