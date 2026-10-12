"""Rebuild the six NLP score panels of the paper (Appendix B, H) from public
sources and save them as compressed .npz files.

  RewardBench 2 : allenai/reward-bench-2-results @ b19c7033e964187d12e74a43a07f2d727a3d37e5
  WMT23 en-de   : MTME archive (mt-metrics-eval-v2), 13 and 12 output streams
  WMT24 en-de/en-es/ja-zh : same archive, document splits

If you already have the cached panels from the original pipeline, skip this
script and export them in the same .npz format (see README.md); that is the
exact-replication route.  This script follows the written protocol and prints
the checks (prompt / segment / system / metric counts) that must match the
paper before any experiment is run.

Usage:
  python3 build_panels.py rewardbench  --out panels/
  python3 build_panels.py wmt --mtme /path/to/mt-metrics-eval-v2 --out panels/
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

RB_REPO = "allenai/reward-bench-2-results"
RB_REV = "b19c7033e964187d12e74a43a07f2d727a3d37e5"
# Frozen evaluator order (Appendix B): first seven = base pool, last seven = candidates.
RB_ORDER = ["Tulu-3-8B", "QRM-Gemma-27B", "WorldPM-72B", "SkyworkV2-8B", "DeBERTa-v3", "InternLM2-20B",
            "Starling-34B", "UltraRM-13B", "URM-LLaMa-8B", "OffsetBias-8B", "GRM-Llama3-8B", "Skywork-V2-3B",
            "LDL-Gemma-27B", "BTRM-Qwen2-7B"]
# Regular expressions used to locate each model's result file. The original
# mapping is evidence/source-model-files.json; check the printed matches.
RB_PATTERNS = {
    "Tulu-3-8B": r"tulu-3-8b-rm", "QRM-Gemma-27B": r"qrm-gemma-2-27b", "WorldPM-72B": r"worldpm-72b",
    "SkyworkV2-8B": r"skywork-reward-v2-llama-3\.1-8b(?!-40m)", "DeBERTa-v3": r"deberta-v3-large-v2",
    "InternLM2-20B": r"internlm2-20b-reward", "Starling-34B": r"starling-rm-34b", "UltraRM-13B": r"ultrarm-13b",
    "URM-LLaMa-8B": r"urm-llama-3\.1-8b", "OffsetBias-8B": r"offsetbias-rm-8b", "GRM-Llama3-8B": r"grm-llama3-8b",
    "Skywork-V2-3B": r"skywork-reward-v2-llama-3\.2-3b", "LDL-Gemma-27B": r"ldl-reward-gemma-2-27b",
    "BTRM-Qwen2-7B": r"btrm_qwen2_7b",
}
WMT_ORDER = ["BERTscore", "BLEU", "BLEURT-20", "COMET", "Calibri-COMET22", "MEE4", "MaTESe", "MetricX-23",
             "MetricX-23-b", "MetricX-23-c", "XCOMET-Ensemble", "XCOMET-XL", "XCOMET-XXL", "XLsim", "YiSi-1", "chrF",
             "docWMT22CometDA", "eBLEU", "embed_llama", "f200spBLEU", "instructscore", "mre-score-labse-regular",
             "prismRef", "tokengram_F"]


def build_rewardbench(out):
    from huggingface_hub import snapshot_download
    path = snapshot_download(RB_REPO, repo_type="dataset", revision=RB_REV, allow_patterns=["*.json"])
    files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    chosen = {}
    for name, pat in RB_PATTERNS.items():
        hits = [f for f in files if re.search(pat, f.lower())]
        print(f"{name:16s} -> {[os.path.relpath(h, path) for h in hits]}")
        if len(hits) != 1:
            sys.exit(f"ambiguous or missing result file for {name}; fix RB_PATTERNS")
        chosen[name] = hits[0]
    per_model = {}
    for name in RB_ORDER:
        d = json.load(open(chosen[name]))
        ids, subsets, scores = d["id"], d["subset"], d["scores"]
        keep = {}
        for i, sub, sc in zip(ids, subsets, scores):
            if sub == "Ties":
                continue
            if isinstance(sc, list) and len(sc) == 1 and isinstance(sc[0], list):
                sc = sc[0]
            if not (isinstance(sc, list) and len(sc) == 4 and all(isinstance(x, (int, float)) for x in sc)):
                continue
            keep[int(i)] = [float(x) for x in sc]      # last eligible occurrence wins
        per_model[name] = keep
    common = sorted(set.intersection(*[set(v) for v in per_model.values()]))
    print("common prompts:", len(common), "(paper: 1,723)")
    raw = np.array([[per_model[n][pid][c] for n in RB_ORDER] for pid in common for c in range(4)])
    assert np.isfinite(raw).all()
    group = np.repeat(np.arange(len(common)), 4)
    label = np.tile([1, 0, 0, 0], len(common))         # chosen completion first
    np.savez_compressed(os.path.join(out, "rewardbench2.npz"), raw=raw, group=group, label=label,
                        evaluators=np.array(RB_ORDER), group_order=np.arange(len(common)), kind="rb")


def read_seg_scores(path):
    """MTME *.seg.score: lines 'system<TAB>score' grouped by system."""
    by = {}
    for line in open(path):
        sys_, val = line.rstrip("\n").split("\t")
        by.setdefault(sys_, []).append(None if val == "None" else float(val))
    return by


def build_wmt(mtme, year, lp, ref, out, name, drop_streams=(), doc_split=False):
    base = os.path.join(mtme, year)
    files = sorted(glob.glob(os.path.join(base, "metric-scores", lp, f"*-{ref}.seg.score")))
    metrics, seen = {}, {}
    for f in files:
        m = os.path.basename(f)[: -len(f"-{ref}.seg.score")]
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h in seen:
            continue
        seen[h] = m
        metrics[m] = read_seg_scores(f)
    refs = {os.path.basename(p).split(".")[1] for p in glob.glob(os.path.join(base, "references", f"{lp}.*.txt"))}
    systems = sorted(s for s in next(iter(metrics.values())) if s not in refs and s not in drop_streams)
    order = [m for m in WMT_ORDER if m in metrics] if year == "wmt23" else sorted(metrics)
    if year == "wmt23":
        missing = [m for m in WMT_ORDER if m not in metrics]
        print("missing metrics:", missing)
    nseg = len(metrics[order[0]][systems[0]])
    ok = [all(metrics[m][s][j] is not None for m in order for s in systems) for j in range(nseg)]
    segs = [j for j in range(nseg) if ok[j]]
    raw = np.array([[metrics[m][s][j] for m in order] for j in segs for s in systems])
    print(f"{name}: segments {len(segs)}, streams {len(systems)}, metrics {len(order)}, outputs {raw.shape[0]}")
    docs = None
    dfile = os.path.join(base, "documents", f"{lp}.docs")
    if os.path.exists(dfile):
        docs = [l.split("\t")[-1].strip() for l in open(dfile)]
    if doc_split:
        doc_of_seg = [docs[j] for j in segs]
        uniq = list(dict.fromkeys(doc_of_seg))
        gid = {d: i for i, d in enumerate(uniq)}
        group = np.repeat([gid[d] for d in doc_of_seg], len(systems))
        group_order = np.arange(len(uniq))
    else:
        group = np.repeat(np.arange(len(segs)), len(systems))
        group_order = np.arange(len(segs))
    mqm = None
    hfile = os.path.join(base, "human-scores", f"{lp}.mqm.seg.score")
    if os.path.exists(hfile):
        h = read_seg_scores(hfile)
        mqm = np.array([np.nan if h.get(s, [None] * nseg)[j] is None else h[s][j] for j in segs for s in systems])
    np.savez_compressed(os.path.join(out, f"{name}.npz"), raw=raw, group=group, evaluators=np.array(order),
                        systems=np.array(systems), group_order=group_order,
                        mqm=mqm if mqm is not None else np.full(raw.shape[0], np.nan),
                        kind="wmt24" if doc_split else "wmt23")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rewardbench", "wmt"])
    ap.add_argument("--mtme", default="mt-metrics-eval-v2")
    ap.add_argument("--out", default="panels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == "rewardbench":
        build_rewardbench(a.out)
    else:
        build_wmt(a.mtme, "wmt23", "en-de", "refA", a.out, "wmt23_13")
        build_wmt(a.mtme, "wmt23", "en-de", "refA", a.out, "wmt23_12", drop_streams=("synthetic_ref",))
        for lp, ref in (("en-de", "refB"), ("en-es", "refA"), ("ja-zh", "refA")):
            build_wmt(a.mtme, "wmt24", lp, ref, a.out, f"wmt24_{lp.replace('-', '')}", doc_split=True)
