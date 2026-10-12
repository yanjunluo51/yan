"""Synthetic latent-quality evaluator panels and the paper's selection protocol.

The generator imitates the structure of an NLP evaluator panel: grouped
outputs (prompts / source segments), a latent output quality, a shared
nuisance factor (e.g. length) and evaluator-specific scale, bias, noise and
optional score discretisation (ties, as for surface-overlap metrics).  It is
used only for controlled experiments; it is not a substitute for the NLP
panels.  The protocol functions (`right_ecdf`, `cutoff_from_reference`,
`nominal_k`, group splits, ridge features) follow Appendix B of the paper and
can be applied unchanged to cached NLP score matrices.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from certlib import consensus, instance_from_scores


@dataclass
class Panel:
    raw: np.ndarray        # (n_outputs, n_evaluators) raw scores
    group: np.ndarray      # (n_outputs,) group id
    feat: np.ndarray       # (n_outputs,) 1-D output feature (for Lipschitz runs)


def make_panel(n_groups=600, group_size=4, n_eval=14, seed=0, shared=0.35,
               discrete_frac=0.0, lipschitz_cands=None, L=2.0):
    """Latent-quality panel. If `lipschitz_cands` is a set of evaluator indices,
    those evaluators score outputs by an L-Lipschitz function of `feat`."""
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    group = np.repeat(np.arange(n_groups), group_size)
    g_eff = rng.normal(0, 0.6, n_groups)[group]
    q = g_eff + rng.normal(0, 1.0, n)
    nuis = rng.normal(0, 1.0, n)
    feat = rng.uniform(0, 1, n)
    raw = np.empty((n, n_eval))
    for e in range(n_eval):
        lam = rng.uniform(0.6, 1.0)
        gam = rng.normal(0, shared)
        sig = rng.uniform(0.35, 0.9)
        s = lam * q + gam * nuis + sig * rng.normal(0, 1, n)
        if rng.random() < discrete_frac:
            s = np.round(s * 2) / 2      # ties
        raw[:, e] = s
    if lipschitz_cands:
        for e in lipschitz_cands:
            # random smooth function of feat with slope bounded by L (in [0,1] units)
            knots = np.linspace(0, 1, 9)
            vals = np.cumsum(rng.uniform(-L, L, 9) * (knots[1] - knots[0]))
            vals = 0.5 + vals - vals.mean()
            raw[:, e] = np.interp(feat, knots, vals)
    return Panel(raw, group, feat)


def group_split(group, seed, fracs=(1 / 3, 1 / 3)):
    ids = sorted(set(group.tolist()))
    random.Random(seed).shuffle(ids)
    d = len(ids)
    n_ref = int(d * fracs[0])
    n_cal = int(d * fracs[1])
    ref = set(ids[:n_ref])
    cal = set(ids[n_ref:n_ref + n_cal])
    is_ref = np.array([g in ref for g in group])
    is_cal = np.array([g in cal for g in group])
    return is_ref, is_cal, ~(is_ref | is_cal)


def right_ecdf(ref_scores, scores):
    srt = np.sort(ref_scores)
    return np.searchsorted(srt, scores, side="right") / len(srt)


def midrank_ecdf(ref_scores, scores):
    srt = np.sort(ref_scores)
    lt = np.searchsorted(srt, scores, side="left")
    le = np.searchsorted(srt, scores, side="right")
    return (lt + le) / (2 * len(srt))


def nominal_k(q, n):
    return max(1, int(round(q * n)))       # Python round: half to even


def cutoff_from_reference(ref_norm, q, pool):
    cons = np.sort(consensus(ref_norm, pool))
    n = len(cons)
    return cons[n - max(1, int(round(q * n)))]


def normalise(panel: Panel, base_idx, cand_idx, is_ref, test_mask, mapping="right"):
    f = right_ecdf if mapping == "right" else midrank_ecdf
    cols = list(base_idx) + list(cand_idx)
    norm_ref = np.column_stack([f(panel.raw[is_ref, e], panel.raw[is_ref, e]) for e in cols])
    norm_test = np.column_stack([f(panel.raw[is_ref, e], panel.raw[test_mask, e]) for e in cols])
    M = len(base_idx)
    return norm_ref[:, :M], norm_ref[:, M:], norm_test[:, :M], norm_test[:, M:]


def ridge_features(base, t0, pool, pcap):
    M = base.shape[1]
    cons = consensus(base, pool)
    return np.column_stack([base, cons - t0, base.max(1) - base.min(1), pcap / M])


def fit_ridge(X, Y, lam=1.0):
    """Ridge with unpenalised intercept; Y may have several columns."""
    Xc = np.column_stack([np.ones(len(X)), X])
    P = lam * np.eye(Xc.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ Y)


def predict_ridge(W, X):
    return np.column_stack([np.ones(len(X)), X]) @ W


@dataclass
class Task:
    inst: object          # certlib.Instance on active rows (uniform family)
    aux: dict
    pred: np.ndarray      # (m, T) clipped reference-ridge predicted contributions
    base_test: np.ndarray
    cand_test: np.ndarray
    t0: float
    k: int
    pool: str
    S0: int
    feat_test: np.ndarray


def build_task(panel: Panel, split_seed, q, alpha, pool, M=7, T=7, mapping="right",
               eval_order=None, weights=None):
    order = list(range(panel.raw.shape[1])) if eval_order is None else list(eval_order)
    base_idx, cand_idx = order[:M], order[M:M + T]
    is_ref, is_cal, is_test = group_split(panel.group, split_seed)
    b_ref, c_ref, b_test, c_test = normalise(panel, base_idx, cand_idx, is_ref, is_test, mapping)
    t0 = cutoff_from_reference(b_ref, q, pool)
    N = int(is_test.sum())
    k = nominal_k(q, N)
    inst, aux = instance_from_scores(b_test, c_test, t0, pool, k, alpha, weights=weights)
    # reference-fitted predictions of per-candidate flip counts
    from certlib import replacement_flips
    sel_r, flips_r, pot_r = replacement_flips(b_ref, c_ref, t0, pool)
    pcap_r = pot_r.sum(1)
    Xr = ridge_features(b_ref, t0, pool, pcap_r)
    Yr = flips_r.sum(1)                        # (n_ref, T) flip counts f_ir
    W = fit_ridge(Xr, Yr)
    act = inst.meta["active"]
    pcap_t = aux["pot"].sum(1)
    Xt = ridge_features(b_test[act], t0, pool, pcap_t[act])
    pred = np.clip(predict_ridge(W, Xt), 0, inst.c) if len(act) else np.zeros((0, T))
    return Task(inst, aux, pred, b_test, c_test, t0, k, pool, int(aux["sel"].sum()),
                panel.feat[is_test])
