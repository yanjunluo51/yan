"""W3: solver and formulation sensitivity of the joint outcome-count program.

The exact-stop joint program (single safe target s = L*, Proposition 4) is
exported to several backends on identical instances:
  highs-p1     SciPy/HiGHS, presolve on (paper's backend family)
  highs-p0     SciPy/HiGHS, presolve off
  highs-s1/s2  highspy, random_seed 1 / 2
  cbc          COIN-OR CBC via PuLP
  cpsat        OR-Tools CP-SAT (pure-integer version: products and deficits
               are integral), one worker
Instances: (i) synthetic panels (RB-like, MT-like, alpha in {.02,.10}, depth
.05/.30, splits 0-2); (ii) a constructed uniform-score grid in the spirit of
Appendix M.7 (mean pooling, cutoff .5, q = .30, alpha = .10), N in {512, 2048},
M in {7, 15}, T in {17, 33}.  Nominal 60 s per backend and target.
"""
import json
import sys
import time

import numpy as np

from certlib import _Model, _add_exclusion, _add_safety, _groups, _new_a, bounds, instance_from_scores, true_optimum
from panels import build_task
from suite import PANELS, get_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/w3_solvers.json"
TL = 60.0


def build_exact_model(inst):
    L = true_optimum(inst)
    groups = _groups(inst)
    lo0, up0 = bounds(inst)
    M = _Model()
    a = _new_a(M, groups)
    _add_safety(M, inst, groups, a, L, lo0, up0)
    if L + 1 <= inst.m:
        _add_exclusion(M, inst, groups, a, L + 1, lo0, int(inst.v.sum(1).max()))
    return M


def solve_scipy(M, presolve):
    t = time.time()
    res = M.solve(TL, presolve)
    dt = time.time() - t
    val = res.fun if res.x is not None else None
    lb = getattr(res, "mip_dual_bound", None)
    return dict(val=val, lb=lb, status=int(res.status), sec=dt)


def solve_highspy(M, seed):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", TL)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("threads", 1)
    n = len(M.lb)
    inf = highspy.kHighsInf
    for j in range(n):
        h.addVar(M.lb[j], M.ub[j])
    h.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(M.obj, float))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous for x in M.integ]))
    for coefs, lo, hi in M.rows:
        idx = np.array(list(coefs.keys()), dtype=np.int32)
        val = np.array(list(coefs.values()), float)
        h.addRow(lo if np.isfinite(lo) else -inf, hi if np.isfinite(hi) else inf, len(idx), idx, val)
    t = time.time()
    h.run()
    dt = time.time() - t
    info = h.getInfo()
    st = h.getModelStatus()
    val = info.objective_function_value if info.primal_solution_status == 2 else None
    return dict(val=val, lb=info.mip_dual_bound, status=str(st), sec=dt)


def solve_cbc(M):
    import pulp
    prob = pulp.LpProblem("j", pulp.LpMinimize)
    xs = [pulp.LpVariable(f"x{j}", M.lb[j], M.ub[j], cat="Integer" if M.integ[j] else "Continuous")
          for j in range(len(M.lb))]
    prob += pulp.lpSum(M.obj[j] * xs[j] for j in range(len(xs)) if M.obj[j])
    for k, (coefs, lo, hi) in enumerate(M.rows):
        e = pulp.lpSum(a * xs[j] for j, a in coefs.items())
        if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
            prob += e == lo
        else:
            if np.isfinite(lo):
                prob += e >= lo
            if np.isfinite(hi):
                prob += e <= hi
    t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=TL, gapRel=0.0, threads=1))
    dt = time.time() - t
    status = pulp.LpStatus[prob.status]
    val = pulp.value(prob.objective) if status == "Optimal" or prob.sol_status in (1, 2) else None
    return dict(val=val, lb=None, status=status, sec=dt, solstatus=prob.sol_status)


def solve_cpsat(M):
    import pickle, subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump((M.lb, M.ub, M.integ, M.obj, M.rows, TL), f)
        path = f.name
    out = subprocess.run([sys.executable, "cpsat_worker.py", path], capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return dict(val=None, lb=None, status="error: " + out.stderr[-300:], sec=None)


def instances():
    for name, spec in PANELS.items():
        pan = get_panel(name)
        for alpha in (0.02, 0.10):
            for q in (0.05, 0.30):
                for split in range(3):
                    task = build_task(pan, 1000 + split, q, alpha, "mean", M=spec["M"], T=spec["T"])
                    yield dict(src=name, alpha=alpha, q=q, split=split), task.inst
    for N in (512, 2048):
        for Mb in (7, 15):
            for T in (17, 33):
                rng = np.random.default_rng(731191 + N + 100 * Mb + T)
                base = rng.uniform(0, 1, (N, Mb))
                cand = rng.uniform(0, 1, (N, T))
                k = max(1, int(round(0.30 * N)))
                inst, _ = instance_from_scores(base, cand, 0.5, "mean", k, 0.10)
                yield dict(src="uniform-grid", N=N, M=Mb, T=T), inst


rows = []
for meta, inst in instances():
    if inst.m == 0:
        continue
    L = true_optimum(inst)
    Mdl = build_exact_model(inst)
    rec = dict(meta, m=inst.m, L=L, full=L == inst.m, nvar=len(Mdl.lb), ncon=len(Mdl.rows),
               G=sum(len(g) for g in _groups(inst)))
    rec["highs-p1"] = solve_scipy(Mdl, True)
    rec["highs-p0"] = solve_scipy(Mdl, False)
    rec["highs-s1"] = solve_highspy(Mdl, 1)
    rec["highs-s2"] = solve_highspy(Mdl, 2)
    rec["cbc"] = solve_cbc(Mdl)
    rec["cpsat"] = solve_cpsat(Mdl)
    rows.append(rec)
    print(json.dumps(rec, default=str), flush=True)
json.dump(rows, open(OUT, "w"), indent=1, default=str)
