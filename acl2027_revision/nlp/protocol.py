"""Selection protocol of the paper on cached NLP panels (Appendix B, H), with
exact integer arithmetic: right-ECDF values are k / n_ref, so consensus
comparisons are done on integer numerators."""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from certlib import Instance  # noqa: E402

PANELS = {
    "rewardbench2": dict(kind="rb", seed0=20260815, sizes=(574, 574)),
    "wmt23_13": dict(kind="wmt23", seed0=87000, sizes=(185, 185)),
    "wmt23_12": dict(kind="wmt23", seed0=87000, sizes=(185, 185)),
    "wmt24_ende": dict(kind="wmt24", seed0=924000, sizes=None),
    "wmt24_enes": dict(kind="wmt24", seed0=924000, sizes=None),
    "wmt24_jazh": dict(kind="wmt24", seed0=924000, sizes=None),
}
M = 7


def load(panel_dir, name):
    z = np.load(os.path.join(panel_dir, name + ".npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def split(panel, name, r):
    spec = PANELS[name]
    ids = [int(g) for g in panel["group_order"]]
    seed = spec["seed0"] + r
    random.Random(seed).shuffle(ids)
    if spec["sizes"] is None:
        d = len(ids) // 3
        nref = ncal = d
    else:
        nref, ncal = spec["sizes"]
    ref, cal = set(ids[:nref]), set(ids[nref:nref + ncal])
    g = panel["group"]
    is_ref = np.isin(g, list(ref))
    is_cal = np.isin(g, list(cal))
    return is_ref, is_cal, ~(is_ref | is_cal), seed


def ecdf_num(ref, x):
    """Right-ECDF numerator: #{reference scores <= x}."""
    return np.searchsorted(np.sort(ref), x, side="right")


def consensus_num(S, pool):
    """Integer consensus: sum for mean pooling, median numerator for odd medians."""
    if pool == "mean":
        return S.sum(axis=-1)
    return np.sort(S, axis=-1)[..., S.shape[-1] // 2]


def build(panel, name, r, q, alpha, pool, weights=None):
    raw = panel["raw"]
    T = raw.shape[1] - M
    is_ref, is_cal, is_test, seed = split(panel, name, r)
    nref = int(is_ref.sum())
    cols = range(M + T)
    A_ref = np.column_stack([ecdf_num(raw[is_ref, e], raw[is_ref, e]) for e in cols]).astype(np.int64)
    A_test = np.column_stack([ecdf_num(raw[is_ref, e], raw[is_test, e]) for e in cols]).astype(np.int64)
    cons_ref = np.sort(consensus_num(A_ref[:, :M], pool))
    cut = int(cons_ref[nref - max(1, int(round(q * nref)))])
    B, C = A_test[:, :M], A_test[:, M:]
    N = B.shape[0]
    k = max(1, int(round(q * N)))
    sel = consensus_num(B, pool) >= cut
    # flips[i, j, r]: replacing base member j by candidate r changes membership
    flips = np.zeros((N, M, T), bool)
    pot = np.zeros((N, M), bool)
    for j in range(M):
        others = np.delete(B, j, axis=1)
        for rr in range(T):
            flips[:, j, rr] = (consensus_num(np.column_stack([others, C[:, rr]]), pool) >= cut) != sel
        end = np.where(sel, 0, nref)                        # max-flip endpoint (score 0 or 1)
        pot[:, j] = (consensus_num(np.column_stack([others, end]), pool) >= cut) != sel
    if weights is None:
        weights = np.ones((M, T), np.int64)
    W = int(weights.sum())
    v = np.einsum("imt,mt->it", flips.astype(np.int64), weights)
    c = np.einsum("im,mt->it", pot.astype(np.int64), weights)
    b = (2 * k * W * alpha_num(alpha)) // alpha_den(alpha)   # exact floor(2 k alpha W)
    selected = np.where(sel)[0]
    active = selected[c[selected].sum(1) > 0]
    inst = Instance(c[active], v[active], int(b),
                    meta=dict(N=N, T=T, S0=int(sel.sum()), m0=int(len(selected) - len(active)), active=active,
                              k=k, alpha=alpha, nref=nref, cut=cut, seed=seed))
    # reference ridge priorities (Appendix B): features use normalized scores
    pred = ridge_predictions(A_ref, A_test, cut, pool, nref, T, active, pot)
    tie = np.random.default_rng(seed + 700001).permutation(N)
    rank = np.empty(N, int)
    rank[tie] = np.arange(N)
    aux = dict(sel=sel, flips=flips, pot=pot, is_test=is_test, rank_active=rank[active])
    return inst, aux, pred


def alpha_num(a):
    from fractions import Fraction
    return Fraction(a).limit_denominator(1000).numerator


def alpha_den(a):
    from fractions import Fraction
    return Fraction(a).limit_denominator(1000).denominator


def ridge_predictions(A_ref, A_test, cut, pool, nref, T, active, pot_test):
    Br, Cr = A_ref[:, :M], A_ref[:, M:]
    sel_r = consensus_num(Br, pool) >= cut
    fl = np.zeros((len(Br), T))
    pr = np.zeros(len(Br))
    for j in range(M):
        others = np.delete(Br, j, axis=1)
        end = np.where(sel_r, 0, nref)
        pr += (consensus_num(np.column_stack([others, end]), pool) >= cut) != sel_r
        for rr in range(T):
            fl[:, rr] += (consensus_num(np.column_stack([others, Cr[:, rr]]), pool) >= cut) != sel_r

    def feats(Bn, p):
        x = Bn / nref
        cons = consensus_num(Bn, pool) / (M if pool == "mean" else 1) / nref
        t0 = cut / (M if pool == "mean" else 1) / nref
        return np.column_stack([x, cons - t0, x.max(1) - x.min(1), p / M])

    X = np.column_stack([np.ones(len(Br)), feats(Br, pr)])
    P = np.eye(X.shape[1])
    P[0, 0] = 0
    Wt = np.linalg.solve(X.T @ X + P, X.T @ fl)
    Bt = A_test[active, :M]
    Xt = np.column_stack([np.ones(len(active)), feats(Bt, pot_test[active].sum(1))])
    pcap = pot_test[active].sum(1)
    return np.clip(Xt @ Wt, 0, pcap[:, None]) if len(active) else np.zeros((0, T))
