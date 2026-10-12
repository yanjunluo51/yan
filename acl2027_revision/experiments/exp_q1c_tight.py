"""Q1(c'): Dirichlet replacement weights at the tighter budget alpha = .02
(exact stopping), where exclusion evidence is needed and joint gains can occur."""
import json, sys
import numpy as np
from certlib import count_bounds, joint_min, split_min, true_optimum, union_min
from panels import build_task
from suite import get_panel, PANELS
OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q1c_tight.json"
pan = get_panel("RB-like"); spec = PANELS["RB-like"]; H = spec["M"] * spec["T"]
rows = []
for kappa in ("uniform", 10.0, 1.0, 0.3):
    for q in (0.05, 0.30):
        for split in range(10):
            wr = np.random.default_rng(5000 + split)
            w = (np.full((spec["M"], spec["T"]), 1000, dtype=np.int64) if kappa == "uniform" else
                 np.rint(wr.dirichlet(np.full(H, kappa)) * 1000 * H).astype(np.int64).reshape(spec["M"], spec["T"]))
            task = build_task(pan, 1000 + split, q, 0.02, "mean", M=spec["M"], T=spec["T"], weights=w)
            inst = task.inst
            vm, vp = count_bounds(inst); m0 = inst.meta["m0"]
            jr = joint_min(inst, 0, time_limit=60); sp = split_min(inst, 0, time_limit=60)
            un = union_min(inst, 0, time_limit=60) if jr.value < sp else jr.value
            rec = dict(kappa=kappa, q=q, split=split, S0=task.S0, m=inst.m, zero_cov=(vm + m0) / task.S0,
                       full_cov=(true_optimum(inst) + m0) / task.S0, joint=jr.value, closed=jr.closed, union=un, split_q=sp)
            rows.append(rec); print(json.dumps(rec), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
