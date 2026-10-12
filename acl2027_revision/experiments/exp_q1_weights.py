"""Q1: nonuniform replacement weights.

(a) Weighted odd-median rows are endpoint-only (v in {0, c_ir} with
    column-dependent caps): exhaustive check that joint = best union = split.
(b) Factor-two family under perturbed weights (Proposition R.2): realised
    Proposition-9 scores with alpha = (T+1/2)/(2MT); every weight vector with
    |w_jr / mean(w) - 1| < 1/(2MT) keeps Q_joint = T+1 and Q_union = Q_split = 2T.
    Larger perturbations are reported descriptively.
(c) Panel sensitivity: Dirichlet replacement weights of decreasing concentration
    on the RB-like mean panel.
"""
import json
import sys

import numpy as np

from certlib import (Instance, brute_force, count_bounds, instance_from_scores, joint_min,
                     split_min, true_optimum, union_min)
from panels import build_task
from suite import get_panel, PANELS

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q1_weights.json"
res = {}

# (a) ---------------------------------------------------------------------
rng = np.random.default_rng(7)
n, bad = 0, 0
for trial in range(3000):
    m = int(rng.integers(1, 4)); T = int(rng.integers(1, 4))
    c = rng.integers(1, 6, size=(m, T))
    v = np.where(rng.random((m, T)) < rng.random(), c, 0)
    obs = rng.random((m, T)) < 0.2
    b = int(rng.integers(0, c.sum() + 1))
    inst = Instance(c, v, b, obs)
    for eps in range(m + 1):
        r = brute_force(inst, eps)
        n += 1
        bad += not (r["joint"] == r["union"] == r["split"])
res["a_weighted_endpoint_only"] = dict(cases=n, unequal=bad)
print(res, flush=True)


# (b) ---------------------------------------------------------------------
def prop9_scores(T):
    M = T + 1
    a = np.array([0.5 + j / (4 * M * M) for j in range(M)])
    bb = np.array([1 / (16 * M * M)] + [M / (2 * (M - 1)) + j / (4 * M * M) for j in range(M - 1)])
    base = np.vstack([a, bb])
    cand = np.ones((2, T))
    for row, x in ((0, a), (1, bb)):
        tau = np.sort(M * 0.5 - x.sum() + x)       # replacement thresholds
        pos = tau[tau > 0]
        mid = (pos[-1] + pos[-2]) / 2              # exactly one flip
        if row == 0:
            cand[0, :] = mid
        else:
            cand[1, 0] = mid
    return base, cand, M


rows_b = []
for T in (2, 4, 6):
    base, cand, M = prop9_scores(T)
    alpha = (T + 0.5) / (2 * M * T)
    for scale_mult in (0.9, 2, 5, 10, 20):
        delta = scale_mult / (2 * M * T)
        kept, ratios = 0, []
        draws = 100 if T < 6 else 40
        for d in range(draws):
            omega = 1 + delta * rng.uniform(-1, 1, size=(M, T))
            omega = np.maximum(omega, 0)
            w = np.rint(omega * 10 ** 4).astype(np.int64)
            inst, aux = instance_from_scores(base, cand, 0.5, "mean", 1, alpha, weights=w)
            if inst.m < 2:
                ratios.append(np.nan)
                continue
            j = joint_min(inst, 0).value
            u = union_min(inst, 0)
            s = split_min(inst, 0)
            kept += (j == T + 1 and u == 2 * T and s == 2 * T)
            ratios.append(u / j if j else np.nan)
        rows_b.append(dict(T=T, M=M, delta_over_bound=scale_mult, draws=draws, preserved=kept,
                           mean_union_over_joint=float(np.nanmean(ratios))))
        print(rows_b[-1], flush=True)
    # uniform reference
    inst, _ = instance_from_scores(base, cand, 0.5, "mean", 1, alpha)
    rows_b.append(dict(T=T, M=M, delta_over_bound=0, uniform=True, joint=joint_min(inst, 0).value,
                       union=union_min(inst, 0), split=split_min(inst, 0)))
    print(rows_b[-1], flush=True)
res["b_prop9_perturbation"] = rows_b

# (c) ---------------------------------------------------------------------
rows_c = []
pan = get_panel("RB-like")
spec = PANELS["RB-like"]
H = spec["M"] * spec["T"]
for kappa in ("uniform", 10.0, 1.0, 0.3):
    for q in (0.05, 0.30):
        for split in range(10):
            wr = np.random.default_rng(5000 + split)
            if kappa == "uniform":
                w = np.full((spec["M"], spec["T"]), 1000, dtype=np.int64)
            else:
                w = np.rint(wr.dirichlet(np.full(H, kappa)) * 1000 * H).astype(np.int64).reshape(spec["M"], spec["T"])
            task = build_task(pan, 1000 + split, q, 0.10, "mean", M=spec["M"], T=spec["T"], weights=w)
            inst = task.inst
            eps = int(np.ceil(0.02 * task.S0))
            vm, vp = count_bounds(inst)
            m0 = inst.meta["m0"]
            rec = dict(kappa=kappa, q=q, split=split, S0=task.S0, m=inst.m, m0=m0,
                       zero_cov=(vm + m0) / task.S0, full_cov=(true_optimum(inst) + m0) / task.S0)
            for stop, e in (("primary", eps), ("exact", 0)):
                jr = joint_min(inst, e, time_limit=60)
                rec[f"joint_{stop}"] = jr.value
                rec[f"closed_{stop}"] = jr.closed
                rec[f"union_{stop}"] = union_min(inst, e, time_limit=60)
                rec[f"split_{stop}"] = split_min(inst, e, time_limit=60)
            rows_c.append(rec)
            print(json.dumps(rec), flush=True)
res["c_panel_dirichlet"] = rows_c
json.dump(res, open(OUT, "w"), indent=1)
