"""Q6: can acquisition exploit the odd-median closed forms?

Policies: fixed, random, largest width, reference priority (Section 6) and the
new OM-CF policy (Appendix T), plus OM-CF with oracle predictions (diagnostic).
Reports mean queries and mean splitwise ratio to the matching closed-form
minimum Q_eps (Corollary 1) at primary (ceil(.02|S0|)) and exact stopping.
"""
import json
import sys

import numpy as np

from acquire import run_policy
from certlib import median_closed_form, count_bounds
from panels import build_task
from suite import PANELS, DEPTHS, SPLITS, get_panel

POLICIES = ["fixed", "random", "width", "reference", "omcf", "omcf_oracle"]


def main(out):
    rows = []
    for name, spec in PANELS.items():
        pan = get_panel(name)
        for q in DEPTHS:
            for split in SPLITS:
                task = build_task(pan, 1000 + split, q, 0.10, "median", M=spec["M"], T=spec["T"])
                inst = task.inst
                h1 = int(inst.c[0, 0]) if inst.m else 1
                z = inst.v.sum(1) // h1 if inst.m else np.zeros(0, int)
                bopp = inst.b // h1
                for stop, eps in (("primary", int(np.ceil(0.02 * task.S0))), ("exact", 0)):
                    Q = median_closed_form(z, inst.T, bopp, eps) if inst.m else 0
                    rec = dict(panel=name, q=q, split=split, stop=stop, eps=eps, m=inst.m, S0=task.S0, Qmin=int(Q))
                    for pol in POLICIES:
                        oracle = pol.endswith("oracle")
                        base = pol.replace("_oracle", "")
                        rec[pol] = run_policy(inst, task.pred, base, eps, seed=split, oracle=oracle) if inst.m else 0
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../results/q6_median_acquisition.json")
