"""Q4: certifying every single replacement (maximum churn) versus the average.

For each synthetic task (two panel families, two pooling rules, four depths,
ten splits, alpha = .10) we report selected coverage of
  * the average-budget certificate (paper),
  * the per-replacement cap certificate (Appendix S, exact 0/1 programs),
  * the sufficient reduction "average budget alpha/H" (sortable),
at 0%, 25%, 50% and 100% of active selected entries revealed in a seeded
random order, plus how often average-certified masks exceed alpha for some
single replacement.
"""
import json
import sys

import numpy as np

from certlib import bounds, capacity
from maxcap import indicator, max_retention, per_replacement_counts
from panels import build_task
from suite import PANELS, DEPTHS, SPLITS, get_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q4_maxcap.json"
FRACS = (0.0, 0.25, 0.5, 1.0)


def avg_mask(inst, obs, budget):
    lo, up = bounds(inst, obs)
    order = np.argsort(up, kind="stable")
    cs = np.cumsum(up[order])
    n = int(np.searchsorted(cs, budget, side="right"))
    return order[:n]


rows = []
for name, spec in PANELS.items():
    pan = get_panel(name)
    for pool in ("median", "mean"):
        for q in DEPTHS:
            for split in SPLITS:
                task = build_task(pan, 1000 + split, q, 0.10, pool, M=spec["M"], T=spec["T"])
                inst, aux = task.inst, task.aux
                sel_idx = np.where(aux["sel"])[0]
                act = inst.meta["active"]
                pos = {g: i for i, g in enumerate(sel_idx)}
                act_in_sel = np.array([pos[a] for a in act], dtype=int)
                cap = int(np.floor(2 * task.k * 0.10 + 1e-9))
                H = spec["M"] * spec["T"]
                S0, m0 = task.S0, inst.meta["m0"]
                rng = np.random.default_rng(split)
                order = rng.permutation(inst.m * inst.T)
                rec = dict(panel=name, pool=pool, q=q, split=split, S0=S0, m=inst.m, cap=cap, H=H)
                # full information
                Xtrue = indicator(aux["flips"], aux["pot"], np.ones((len(sel_idx), spec["T"]), bool), sel_idx)
                vstar_max, _, cl = max_retention(Xtrue, cap)
                Fi = inst.v.sum(1)
                vstar_avg = m0 + capacity(Fi, inst.b)
                rec.update(full_avg=vstar_avg / S0, full_max=vstar_max / S0, full_max_closed=cl)
                # average-optimal full-information mask: per-replacement violation
                fo = np.argsort(Fi, kind="stable")[:capacity(Fi, inst.b)]
                inv = np.setdiff1d(np.arange(len(sel_idx)), act_in_sel)
                mask_rows = np.concatenate([sel_idx[inv], sel_idx[act_in_sel[fo]]])
                cnt = per_replacement_counts(aux["flips"], mask_rows)
                rec.update(full_avg_mask_maxchurn=float(cnt.max() / (2 * task.k)),
                           full_avg_mask_over=float((cnt > cap).mean()))
                for f in FRACS:
                    nq = int(round(f * inst.m * inst.T))
                    obs = np.zeros((inst.m, inst.T), bool)
                    for e in order[:nq]:
                        obs[e // inst.T, e % inst.T] = True
                    obs_sel = np.zeros((len(sel_idx), spec["T"]), bool)
                    obs_sel[act_in_sel] = obs
                    obs_sel[inv] = True          # invariant rows: no flips possible either way
                    Xu = indicator(aux["flips"], aux["pot"], obs_sel, sel_idx, upper=True)
                    Xl = indicator(aux["flips"], aux["pot"], obs_sel, sel_idx, upper=False)
                    vm_max, cmask, c1 = max_retention(Xu, cap)
                    vp_max, _, c2 = max_retention(Xl, cap)
                    am = avg_mask(inst, obs, inst.b)
                    vm_avg = m0 + len(am)
                    vm_red = m0 + len(avg_mask(inst, obs, cap))        # alpha/H reduction
                    lo, up = bounds(inst, obs)
                    vp_avg = m0 + capacity(lo, inst.b)
                    mrows = np.concatenate([sel_idx[inv], sel_idx[act_in_sel[am]]])
                    cnt = per_replacement_counts(aux["flips"], mrows)
                    tag = f"f{int(f * 100)}"
                    rec.update({f"{tag}_avg": vm_avg / S0, f"{tag}_avg_up": vp_avg / S0,
                                f"{tag}_max": vm_max / S0, f"{tag}_max_up": vp_max / S0,
                                f"{tag}_red": vm_red / S0, f"{tag}_closed": bool(c1 and c2),
                                f"{tag}_avgmask_viol": bool((cnt > cap).any()),
                                f"{tag}_avgmask_maxchurn": float(cnt.max() / (2 * task.k))})
                rows.append(rec)
                print(json.dumps(rec), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
