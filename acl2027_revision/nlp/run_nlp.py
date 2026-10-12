"""Revision experiments on the NLP panels.

  python3 run_nlp.py validate  --panels panels/      # replication check vs. paper numbers (run first)
  python3 run_nlp.py maxcap    --panels panels/      # Q4: per-replacement caps          (Appendix R.2)
  python3 run_nlp.py weights   --panels panels/      # Q1: nonuniform replacement weights (Appendix R.1)
  python3 run_nlp.py screen    --panels panels/      # Q7: pre-scoring screen for joint gains (Appendix U)
  python3 run_nlp.py solvers   --panels panels/      # W3: HiGHS vs CBC on the joint program (Appendix W)

Options: --only rewardbench2,wmt23_13  (subset of panels), --splits 0-39.
Every record is appended as one JSON line to ../results/nlp/<exp>.jsonl, so
runs can be resumed and parallelised by panel.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, HERE)
from acquire import run_policy  # noqa: E402
from certlib import (bounds, capacity, count_bounds, joint_min, median_closed_form, split_min,  # noqa: E402
                     true_optimum, union_min)
from maxcap import indicator, max_retention, per_replacement_counts  # noqa: E402
from protocol import M, PANELS, build, load  # noqa: E402

DEPTHS = (0.05, 0.10, 0.20, 0.30)
OUT = os.path.join(HERE, "..", "results", "nlp")


def parse_splits(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return range(int(a), int(b) + 1)


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l)
            keys.add(d["key"])
    return keys


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float)[:300], flush=True)


def eps_primary(inst):
    return int(np.ceil(0.02 * inst.meta["S0"]))


def ref_policy(inst, pred, aux, eps):
    if inst.m == 0:
        return 0, inst.obs.copy()
    return run_policy(inst, pred, "reference", eps, rank=aux["rank_active"], return_obs=True)


# --------------------------------------------------------------------------- validate
def cmd_validate(a):
    path = os.path.join(OUT, "validate.jsonl")
    keys = done_keys(path)
    for name in a.only:
        pan = load(a.panels, name)
        for pool in ("median", "mean"):
            for r in a.splits:
                key = f"{name}|{pool}|{r}"
                if key in keys:
                    continue
                inst, aux, pred = build(pan, name, r, 0.05, 0.10, pool)
                S0, m0 = inst.meta["S0"], inst.meta["m0"]
                vm, vp = count_bounds(inst)
                L = true_optimum(inst)
                eps = eps_primary(inst)
                nq, _ = ref_policy(inst, pred, aux, eps)
                rec = dict(key=key, panel=name, pool=pool, split=r, N=inst.meta["N"], S0=S0, m=inst.m,
                           zero_cov=(m0 + vm) / S0, full_cov=(m0 + L) / S0,
                           ref_insp=nq / (inst.m * inst.T) if inst.m else 0.0, zero_stop=bool(S0 - m0 - vm <= eps))
                if pool == "median" and inst.m:
                    h1 = int(inst.c[0, 0])
                    rec["Qexact"] = int(median_closed_form(inst.v.sum(1) // h1, inst.T, inst.b // h1, 0))
                emit(path, rec)


PAPER = {  # q = .05, alpha = .10 (Tables 2, 8; Appendix M.2, H)
    ("rewardbench2", "median"): dict(zero_cov=80.9, full_cov=99.6, ref_insp=61.4, Qexact=196.6),
    ("wmt23_13", "median"): dict(zero_cov=84.9, full_cov=99.5, ref_insp=45.1, Qexact=415.9),
    ("wmt23_12", "median"): dict(zero_cov=99.2, full_cov=99.9, ref_insp=3.8),
    ("rewardbench2", "mean"): dict(ref_insp=86.5),
    ("wmt23_13", "mean"): dict(ref_insp=85.7),
    ("wmt23_12", "mean"): dict(ref_insp=85.4),
}


def cmd_report(a):
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT, "validate.jsonl"))])
    for (p, pool), g in d.groupby(["panel", "pool"]):
        ours = dict(zero_cov=100 * g.zero_cov.mean(), full_cov=100 * g.full_cov.mean(),
                    ref_insp=100 * g.ref_insp.mean(), Qexact=g.Qexact.mean() if "Qexact" in g else np.nan)
        ref = PAPER.get((p, pool), {})
        print(f"{p:14s} {pool:6s} n={len(g):2d} " + "  ".join(
            f"{k}={ours[k]:.1f}" + (f" (paper {ref[k]})" if k in ref else "") for k in ours))


# --------------------------------------------------------------------------- Q4 maxcap
def cmd_maxcap(a):
    path = os.path.join(OUT, "maxcap.jsonl")
    keys = done_keys(path)
    for name in a.only:
        pan = load(a.panels, name)
        for pool in ("median", "mean"):
            for q in DEPTHS:
                for r in a.splits:
                    key = f"{name}|{pool}|{q}|{r}"
                    if key in keys:
                        continue
                    inst, aux, pred = build(pan, name, r, q, 0.10, pool)
                    S0, m0, k, T = inst.meta["S0"], inst.meta["m0"], inst.meta["k"], inst.meta["T"]
                    cap = (2 * k * 1) // 10              # floor(2 k alpha), alpha = .10
                    sel_idx = np.where(aux["sel"])[0]
                    pos = {g: i for i, g in enumerate(sel_idx)}
                    act = np.array([pos[x] for x in inst.meta["active"]], int)
                    inv = np.setdiff1d(np.arange(len(sel_idx)), act)
                    rec = dict(key=key, panel=name, pool=pool, q=q, split=r, S0=S0, m=inst.m, cap=cap, H=M * T)

                    def obs_sel(obs_act):
                        o = np.ones((len(sel_idx), T), bool)
                        if len(act):
                            o[act] = obs_act
                        return o

                    full = obs_sel(np.ones((inst.m, T), bool))
                    vstar_max, _, c1 = max_retention(indicator(aux["flips"], aux["pot"], full, sel_idx), cap)
                    rec.update(full_avg=(m0 + true_optimum(inst)) / S0, full_max=vstar_max / S0, full_closed=c1)
                    zero = obs_sel(np.zeros((inst.m, T), bool))
                    vm0, _, c2 = max_retention(indicator(aux["flips"], aux["pot"], zero, sel_idx), cap)
                    vm_avg0, _ = count_bounds(inst)
                    lo, up = bounds(inst)
                    rec.update(zero_avg=(m0 + vm_avg0) / S0, zero_max=vm0 / S0,
                               zero_red=(m0 + capacity(up, cap)) / S0, zero_closed=c2)
                    # history of reference priority at the primary stop (the returned average certificate)
                    eps = eps_primary(inst)
                    nq, obs = ref_policy(inst, pred, aux, eps)
                    lo, up = bounds(inst, obs)
                    order = np.argsort(up, kind="stable")[:capacity(up, inst.b)]
                    rows = np.concatenate([sel_idx[inv], sel_idx[act[order]]]) if len(act) else sel_idx[inv]
                    cnt = per_replacement_counts(aux["flips"], rows)
                    o = obs_sel(obs)
                    vmx, _, c3 = max_retention(indicator(aux["flips"], aux["pot"], o, sel_idx, upper=True), cap)
                    vpx, _, c4 = max_retention(indicator(aux["flips"], aux["pot"], o, sel_idx, upper=False), cap)
                    rec.update(ref_queries=nq, ref_avg=len(rows) / S0, ref_avg_viol=bool((cnt > cap).any()),
                               ref_avg_maxchurn=float(cnt.max() / (2 * k)) if len(cnt) else 0.0,
                               ref_avg_over=float((cnt > cap).mean()) if len(cnt) else 0.0,
                               ref_max=vmx / S0, ref_max_up=vpx / S0, ref_closed=bool(c3 and c4))
                    emit(path, rec)


# --------------------------------------------------------------------------- Q1 weights
def weight_schemes(pan, name, r, T):
    """uniform; 'agree': candidate weight proportional to its reference
    correlation with base consensus (a deployment-likelihood prior);
    'dirichlet': symmetric Dirichlet(1) over all M*T replacements (seeded)."""
    from protocol import ecdf_num, split
    raw = pan["raw"]
    is_ref = split(pan, name, r)[0]
    A = np.column_stack([ecdf_num(raw[is_ref, e], raw[is_ref, e]) for e in range(M + T)]).astype(float)
    cons = A[:, :M].mean(1)
    corr = np.array([np.corrcoef(A[:, M + t], cons)[0, 1] for t in range(T)])
    corr = np.clip(np.nan_to_num(corr), 1e-3, None)
    H = M * T
    agree = np.rint(np.tile(corr / corr.sum() * 1000 * T, (M, 1))).astype(np.int64)
    rng = np.random.default_rng(5000 + r)
    dirich = np.rint(rng.dirichlet(np.ones(H)) * 1000 * H).astype(np.int64).reshape(M, T)
    return {"uniform": None, "agree": agree, "dirichlet": dirich}


def cmd_weights(a):
    path = os.path.join(OUT, "weights.jsonl")
    keys = done_keys(path)
    for name in a.only:
        pan = load(a.panels, name)
        T = pan["raw"].shape[1] - M
        for q in DEPTHS:
            for r in a.splits:
                for scheme, w in weight_schemes(pan, name, r, T).items():
                    key = f"{name}|{q}|{r}|{scheme}"
                    if key in keys:
                        continue
                    inst, aux, pred = build(pan, name, r, q, 0.10, "mean", weights=w)
                    S0, m0 = inst.meta["S0"], inst.meta["m0"]
                    vm, vp = count_bounds(inst)
                    L = true_optimum(inst)
                    rec = dict(key=key, panel=name, q=q, split=r, scheme=scheme, S0=S0, m=inst.m,
                               zero_cov=(m0 + vm) / S0, full_cov=(m0 + L) / S0)
                    for stop, eps in (("exact", 0), ("primary", eps_primary(inst))):
                        t = time.time()
                        jr = joint_min(inst, eps, time_limit=a.tl)
                        sp = split_min(inst, eps, time_limit=a.tl)
                        un = union_min(inst, eps, time_limit=a.tl) if jr.value < sp else jr.value
                        rec.update({f"{stop}_joint": jr.value, f"{stop}_lb": jr.lower, f"{stop}_closed": jr.closed,
                                    f"{stop}_union": un, f"{stop}_split": sp, f"{stop}_sec": time.time() - t})
                    emit(path, rec)


# --------------------------------------------------------------------------- Q7 screen
def cmd_screen(a):
    path = os.path.join(OUT, "screen.jsonl")
    keys = done_keys(path)
    for name in a.only:
        pan = load(a.panels, name)
        for q in DEPTHS:
            for r in a.splits:
                key = f"{name}|{q}|{r}"
                if key in keys:
                    continue
                inst, aux, pred = build(pan, name, r, q, 0.10, "mean")
                S0, m0 = inst.meta["S0"], inst.meta["m0"]
                L = true_optimum(inst)
                vhat = m0 + capacity(np.rint(pred.sum(1)).astype(np.int64), inst.b)
                rec = dict(key=key, panel=name, q=q, split=r, S0=S0, m=inst.m, Vstar=m0 + L, Vhat=vhat,
                           full=bool(L == inst.m))
                for stop, eps in (("exact", 0), ("primary", eps_primary(inst))):
                    rec[f"{stop}_flag"] = bool(vhat + eps < S0)
                    rec[f"{stop}_excl_req"] = bool(m0 + L + eps < S0)
                    if L == inst.m:     # full retention: exclusion automatic, no gain (Appendix M.9)
                        rec.update({f"{stop}_gain": 0, f"{stop}_closed": True})
                        continue
                    jr = joint_min(inst, eps, time_limit=a.tl)
                    sp = split_min(inst, eps, time_limit=a.tl)
                    un = union_min(inst, eps, time_limit=a.tl) if jr.value < sp else jr.value
                    rec.update({f"{stop}_joint": jr.value, f"{stop}_union": un, f"{stop}_split": sp,
                                f"{stop}_gain": un - jr.value, f"{stop}_closed": jr.closed})
                emit(path, rec)


# --------------------------------------------------------------------------- W3 solvers
def cmd_solvers(a):
    from exp_w3_solvers import build_exact_model, solve_cbc, solve_highspy, solve_scipy
    path = os.path.join(OUT, "solvers.jsonl")
    keys = done_keys(path)
    for name in a.only:
        pan = load(a.panels, name)
        for q in DEPTHS:
            for r in a.splits:
                key = f"{name}|{q}|{r}"
                if key in keys:
                    continue
                inst, aux, pred = build(pan, name, r, q, 0.10, "mean")
                if inst.m == 0:
                    continue
                Mdl = build_exact_model(inst)
                rec = dict(key=key, panel=name, q=q, split=r, m=inst.m, full=bool(true_optimum(inst) == inst.m),
                           nvar=len(Mdl.lb))
                rec["highs-p1"] = solve_scipy(Mdl, True)
                rec["highs-p0"] = solve_scipy(Mdl, False)
                rec["highs-s1"] = solve_highspy(Mdl, 1)
                rec["highs-s2"] = solve_highspy(Mdl, 2)
                rec["cbc"] = solve_cbc(Mdl)
                emit(path, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "report", "maxcap", "weights", "screen", "solvers"])
    ap.add_argument("--panels", default=os.path.join(HERE, "panels"))
    ap.add_argument("--only", default=",".join(PANELS))
    ap.add_argument("--splits", default="0-39")
    ap.add_argument("--tl", type=float, default=60.0)
    a = ap.parse_args()
    a.only = a.only.split(",")
    a.splits = parse_splits(a.splits)
    os.makedirs(OUT, exist_ok=True)
    globals()["cmd_" + a.cmd](a)
