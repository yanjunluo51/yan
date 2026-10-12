"""Q2: lightweight cross-entry constraints (Appendix U).

Constraint family: for every candidate r and outputs i, i'
    |s_r(i) - s_r(i')| <= L * |z_i - z_i'| + eta ,   s_r(i) in [0, 1],
a difference-constraint system whose bound delta satisfies the triangle
inequality.  The feasible completion set is a lattice; given observed entries
its greatest / least elements are the envelopes
    U_r(i) = min(1, min_o s_r(o) + L|z_i - z_o| + eta),
    Lam_r(i) = max(0, max_o s_r(o) - L|z_i - z_o| - eta),
so l_i = sum_obs f + sum_unobs f_i(U), u_i = sum_obs f + sum_unobs f_i(Lam)
are sharp (Proposition U.1).

Controlled design: base scores are the ECDF-normalised RB-like panel; the
seven candidates' normalised test scores are generated as an L-Lipschitz
function of a 1-D output feature plus bounded noise of width eta, so the
declared constraint is valid by construction.  A misspecified run uses
(L/2, eta/2), which is invalid, to show what is lost when the constraint is
wrong.  Mean pooling, alpha = .10.
"""
import json
import sys

import numpy as np

from certlib import capacity
from panels import build_task
from suite import get_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "../results/q2_lipschitz.json"


def smooth_fn(rng, L, knots=12):
    xs = np.linspace(0, 1, knots)
    steps = rng.uniform(-L, L, knots - 1) * (xs[1] - xs[0])
    ys = np.concatenate([[0], np.cumsum(steps)])
    ys = ys - ys.min()
    ys = ys + rng.uniform(0, max(0.0, 1 - ys.max())) if ys.max() < 1 else ys / ys.max()
    return lambda z: np.interp(z, xs, ys)


class Envelope:
    """Rectangular (L=None) or Lipschitz-with-slack completion bounds on mean rows."""

    def __init__(self, tau, z, cand, b, L=None, eta=0.0):
        self.tau = tau            # (m, M) replacement thresholds: flip iff x < tau
        self.z = z
        self.cand = cand          # (m, T) true candidate scores (revealed on query)
        self.b = b
        self.L, self.eta = L, eta
        m, T = cand.shape
        self.obs = np.zeros((m, T), bool)
        self.U = np.ones((m, T))
        self.Lam = np.zeros((m, T))
        self.fobs = np.zeros(m, np.int64)

    def f(self, x, rows):
        return (x[:, None] < self.tau[rows]).sum(1)

    def query(self, i, r):
        self.obs[i, r] = True
        s = self.cand[i, r]
        self.fobs[i] += int((s < self.tau[i]).sum())
        if self.L is not None:
            d = self.L * np.abs(self.z - self.z[i]) + self.eta
            self.U[:, r] = np.minimum(self.U[:, r], np.minimum(1.0, s + d))
            self.Lam[:, r] = np.maximum(self.Lam[:, r], np.maximum(0.0, s - d))

    def row_bounds(self):
        m, T = self.cand.shape
        lo = self.fobs.copy()
        up = self.fobs.copy()
        for r in range(T):
            un = ~self.obs[:, r]
            rows = np.where(un)[0]
            lo[rows] += self.f(self.U[rows, r], rows)
            up[rows] += self.f(self.Lam[rows, r], rows)
        return lo, up

    def counts(self):
        lo, up = self.row_bounds()
        return capacity(up, self.b), capacity(lo, self.b), up


def run(env, eps, policy, rng, max_q=None):
    m, T = env.cand.shape
    order = rng.permutation(m * T)
    ptr, nq = 0, 0
    while True:
        vm, vp, up = env.counts()
        if vp - vm <= eps or (max_q is not None and nq >= max_q) or env.obs.all():
            return nq, vm, vp, up
        if policy == "random":
            while env.obs.flat[order[ptr]]:
                ptr += 1
            i, r = divmod(int(order[ptr]), T)
        else:   # width: widest row, then widest unobserved envelope in that row
            lo, upb = env.row_bounds()
            w = np.where(env.obs.all(1), -1, upb - lo)
            i = int(np.argmax(w))
            ew = np.where(env.obs[i], -1, env.U[i] - env.Lam[i])
            r = int(np.argmax(ew))
        env.query(i, r)
        nq += 1


rows = []
pan = get_panel("RB-like")
for L, eta in ((1.0, 0.05), (1.0, 0.2), (4.0, 0.05), (4.0, 0.2)):
    for q in (0.05, 0.30):
        for split in range(10):
            task = build_task(pan, 1000 + split, q, 0.10, "mean")
            act = task.inst.meta["active"]
            rng = np.random.default_rng(900 + split)
            zt = task.feat_test
            fns = [smooth_fn(rng, L) for _ in range(7)]
            cand_all = np.column_stack([np.clip(fn(zt) + rng.uniform(-eta / 2, eta / 2, len(zt)), 0, 1) for fn in fns])
            base = task.base_test[act]
            M = base.shape[1]
            tau = M * task.t0 - base.sum(1)[:, None] + base           # mean thresholds
            cand = cand_all[act]
            sel = np.ones(len(act), bool)
            Ftrue = (cand[:, :, None] < tau[:, None, :]).sum(2).sum(1)
            b = int(np.floor(2 * task.k * 0.10 * M * 7 + 1e-9))
            m0 = task.S0 - len(act)
            Vstar = m0 + capacity(Ftrue, b)
            eps = int(np.ceil(0.02 * task.S0))
            rec = dict(L=L, eta=eta, q=q, split=split, S0=task.S0, m=len(act), full_cov=Vstar / task.S0)
            for model, (LL, ee) in (("rect", (None, 0)), ("lip", (L, eta)), ("lip_mis", (L / 2, eta / 2))):
                for stop, e in (("primary", eps), ("exact", 0)):
                    env = Envelope(tau, zt[act], cand, b, LL, ee)
                    nq, vm, vp, up = run(env, e, "width", np.random.default_rng(split))
                    order = np.argsort(up, kind="stable")[:vm]
                    true_churn = Ftrue[order].sum() / (2 * task.k * M * 7)
                    rec[f"{model}_{stop}_queries"] = nq
                    rec[f"{model}_{stop}_frac"] = nq / max(1, len(act) * 7)
                    rec[f"{model}_{stop}_invalid"] = bool(true_churn > 0.10 + 1e-12 or m0 + vm > Vstar)
                    outside = ((cand < env.Lam - 1e-12) | (cand > env.U + 1e-12)) & ~env.obs
                    rec[f"{model}_{stop}_outside"] = float(outside.mean())
                    rec[f"{model}_{stop}_true_churn"] = float(true_churn)
                for frac in (0.05, 0.10, 0.25):
                    env = Envelope(tau, zt[act], cand, b, LL, ee)
                    nq, vm, vp, up = run(env, -1, "random", np.random.default_rng(split),
                                         max_q=int(round(frac * len(act) * 7)))
                    rec[f"{model}_cov{int(frac * 100)}"] = (m0 + vm) / task.S0
            rows.append(rec)
            print(json.dumps(rec), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
