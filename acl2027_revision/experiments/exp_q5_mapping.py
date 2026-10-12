"""Q5: mapping protocols and a reference-only difficulty diagnostic.

For six reference protocols (full / half / quarter reference groups x right /
midrank ECDF) we compute, before any candidate-test score is used,
  ref_boundary  share of reference-selected outputs on the odd-median boundary
                (exactly h+1 base votes at or above the cutoff)
  ref_ties      share of reference outputs whose consensus equals the cutoff
and, on the test batch, zero-query selected coverage, the exact closed-form
minimum Q_exact and reference-priority queries at the primary stop.
Median pooling, alpha = .10, q in {.05, .10}, ten splits, both panel families.
"""
import json
import sys

import numpy as np

from acquire import run_policy
from certlib import consensus, count_bounds, instance_from_scores, median_closed_form, replacement_flips
from panels import (cutoff_from_reference, fit_ridge, group_split, midrank_ecdf, nominal_k, predict_ridge,
                    ridge_features, right_ecdf)
from suite import PANELS, SPLITS, get_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q5_mapping.json"
rows = []
for name, spec in PANELS.items():
    pan = get_panel(name)
    M, T = spec["M"], spec["T"]
    h = (M - 1) // 2
    for q in (0.05, 0.10):
        for split in SPLITS:
            is_ref, is_cal, is_test = group_split(pan.group, 1000 + split)
            ref_groups = []
            for g in pan.group[is_ref]:
                if g not in ref_groups:
                    ref_groups.append(g)
            for frac in (1.0, 0.5, 0.25):
                keep = set(ref_groups[:max(1, int(len(ref_groups) * frac))])
                ref_mask = np.array([g in keep for g in pan.group]) & is_ref
                for mapping in ("right", "midrank"):
                    f = right_ecdf if mapping == "right" else midrank_ecdf
                    cols = list(range(M + T))
                    nr = np.column_stack([f(pan.raw[ref_mask, e], pan.raw[ref_mask, e]) for e in cols])
                    nt = np.column_stack([f(pan.raw[ref_mask, e], pan.raw[is_test, e]) for e in cols])
                    br, cr, bt, ct = nr[:, :M], nr[:, M:], nt[:, :M], nt[:, M:]
                    t0 = cutoff_from_reference(br, q, "median")
                    cons_r = consensus(br, "median")
                    votes_r = (br >= t0).sum(1)
                    sel_r = cons_r >= t0
                    ref_boundary = float((votes_r[sel_r] == h + 1).mean()) if sel_r.any() else 0.0
                    ref_ties = float(np.isclose(cons_r, t0).mean())
                    N = int(is_test.sum())
                    k = nominal_k(q, N)
                    inst, aux = instance_from_scores(bt, ct, t0, "median", k, 0.10)
                    S0 = int(aux["sel"].sum())
                    m0 = inst.meta["m0"]
                    vm, vp = count_bounds(inst)
                    eps = int(np.ceil(0.02 * S0))
                    if inst.m:
                        h1 = int(inst.c[0, 0])
                        z = inst.v.sum(1) // h1
                        Qx = median_closed_form(z, T, inst.b // h1, 0)
                        Qp = median_closed_form(z, T, inst.b // h1, eps)
                        sel_rr, flips_r, pot_r = replacement_flips(br, cr, t0, "median")
                        W = fit_ridge(ridge_features(br, t0, "median", pot_r.sum(1)), flips_r.sum(1))
                        act = inst.meta["active"]
                        pred = np.clip(predict_ridge(W, ridge_features(bt[act], t0, "median", aux["pot"].sum(1)[act])), 0, inst.c)
                        qref = run_policy(inst, pred, "reference", eps, seed=split)
                    else:
                        Qx = Qp = qref = 0
                    rec = dict(panel=name, q=q, split=split, ref_frac=frac, mapping=mapping,
                               ref_boundary=ref_boundary, ref_ties=ref_ties, S0=S0, m=inst.m,
                               test_boundary=inst.m / S0 if S0 else 0.0,
                               zero_cov=(m0 + vm) / S0 if S0 else 1.0, Qexact=int(Qx), Qprimary=int(Qp),
                               ref_queries=int(qref), active_insp=qref / (inst.m * T) if inst.m else 0.0)
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
