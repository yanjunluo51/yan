"""Q7: when does joint optimisation beat the best separate union, and can it
be anticipated before candidate-test scoring?

Mean pooling, two synthetic panel families, four depths, ten splits, three
budgets, primary and exact stopping (480 targets).  For each target we compute
Q_joint and, when Q_joint < Q_split, the best separate union (otherwise the
sandwich Q_joint <= Q_union <= Q_split forces equality).

Pre-query quantities (no candidate-test score used):
  zero_stop   |S0| - V-(0) <= eps                      (no queries, no gain)
  stair_rows  active rows with >= 2 distinct positive replacement thresholds
              (necessary for an intermediate contribution, Corollary 3)
  plug_excl   Vhat* + eps < |S0|, Vhat* from reference-ridge predicted costs
  plug_slack  |S0| - eps - Vhat*  (continuous score)
Ex post: excl_req = V* + eps < |S0| (exclusion evidence necessary).
"""
import json
import sys

import numpy as np

from certlib import capacity, count_bounds, joint_min, split_min, true_optimum, union_min
from panels import build_task
from suite import PANELS, DEPTHS, SPLITS, get_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q7_predictors.json"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
ALPHAS = [float(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.02, 0.05, 0.10]
rows = []
for name, spec in PANELS.items():
    if ONLY and name != ONLY:
        continue
    pan = get_panel(name)
    for alpha in ALPHAS:
        for q in DEPTHS:
            for split in SPLITS:
                task = build_task(pan, 1000 + split, q, alpha, "mean", M=spec["M"], T=spec["T"])
                inst = task.inst
                m0, S0 = inst.meta["m0"], task.S0
                act = inst.meta["active"]
                base = task.base_test[act]
                M = base.shape[1]
                tau = M * task.t0 - base.sum(1)[:, None] + base
                tau_pos = [np.unique(np.round(t[(t > 0) & (t <= 1)], 12)) for t in tau]
                stair = int(sum(len(t) >= 2 for t in tau_pos))
                Fhat = np.rint(task.pred.sum(1)).astype(np.int64)
                vhat = m0 + capacity(Fhat, inst.b)
                vm0, vp0 = count_bounds(inst)
                L = true_optimum(inst)
                inter = float(((inst.v > 0) & (inst.v < inst.c)).mean()) if inst.m else 0.0
                for stop, eps in (("exact", 0), ("primary", int(np.ceil(0.02 * S0)))):
                    rec = dict(panel=name, alpha=alpha, q=q, split=split, stop=stop, eps=eps, S0=S0, m=inst.m,
                               m0=m0, Vstar=m0 + L, V0m=m0 + vm0, Vhat=vhat, stair_rows=stair,
                               inter_frac=inter,
                               zero_stop=bool(S0 - (m0 + vm0) <= eps),
                               plug_excl=bool(vhat + eps < S0), plug_slack=S0 - eps - vhat,
                               excl_req=bool(m0 + L + eps < S0))
                    if rec["zero_stop"]:
                        rec.update(joint=0, union=0, qsplit=0, closed=True)
                    else:
                        tl = 60 if eps == 0 else 15
                        jr = joint_min(inst, eps, time_limit=tl)
                        sp = split_min(inst, eps, time_limit=tl)
                        un = union_min(inst, eps, time_limit=tl) if jr.value < sp else jr.value
                        rec.update(joint=jr.value, joint_lb=jr.lower, union=un, qsplit=sp, closed=bool(jr.closed))
                    rec["gain"] = rec["union"] - rec["joint"]
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
