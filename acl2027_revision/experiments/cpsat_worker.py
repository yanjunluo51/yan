"""Solves a pickled certlib._Model with OR-Tools CP-SAT in a separate process
(OR-Tools and highspy ship incompatible HiGHS symbols)."""
import json, pickle, sys, time
import numpy as np
from ortools.sat.python import cp_model

lb, ub, integ, obj, rows, TL = pickle.load(open(sys.argv[1], "rb"))
md = cp_model.CpModel()
xs = [md.NewIntVar(int(np.floor(lb[j])), int(np.ceil(ub[j])), f"x{j}") for j in range(len(lb))]
for coefs, lo, hi in rows:
    e = sum(int(round(a)) * xs[j] for j, a in coefs.items())
    if np.isfinite(lo):
        md.Add(e >= int(np.ceil(lo - 1e-9)))
    if np.isfinite(hi):
        md.Add(e <= int(np.floor(hi + 1e-9)))
md.Minimize(sum(int(obj[j]) * xs[j] for j in range(len(xs)) if obj[j]))
s = cp_model.CpSolver()
s.parameters.max_time_in_seconds = TL
s.parameters.num_workers = 1
t = time.time()
st = s.Solve(md)
dt = time.time() - t
ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
print(json.dumps(dict(val=s.ObjectiveValue() if ok else None, lb=s.BestObjectiveBound() if ok else None,
                      status=s.StatusName(st), sec=dt)))
