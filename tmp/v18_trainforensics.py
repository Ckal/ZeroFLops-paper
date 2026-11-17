"""Training-side forensics: how did this run get to a static constant policy?

1. Full metric timeline (all eval rows) for this run.
2. Did the model's column preference ever change? (per-eval holdout pick dist)
3. The train.log line-by-line trend (bpb rising from step 0, LEGAL flat).
4. Checkpoint comparison: do best_hard/best_soft/hard_model_final/final_float
   all show the SAME static column preference? (=> not a stale-export problem)
5. Compare with the OTHER recent run (perply_v24_full) that reportedly
   converged: is IT also static, or does it have real signal?
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path("runs/c4/v7mixed_v18_seed32")
OUT = []


def P(s=""):
    OUT.append(str(s))
    print(s)


P("=" * 78)
P("1. FULL METRIC TIMELINE")
P("=" * 78)
rows = [json.loads(l) for l in (RUN / "metrics.jsonl").read_text().splitlines() if l.strip()]
P("metric rows: %d" % len(rows))
P("keys present: %s" % sorted({k for r in rows for k in r}))
P()
hdr = ("step", "hard_acc", "soft_acc", "legal_acc", "hard_bpb", "soft_bpb",
       "holdout_optimal_pct", "edge_optimal_pct", "vs_opp")
P("%-6s %-11s %-11s %-10s %-10s %-10s %-10s %-10s %s"
  % ("step", "hard_acc", "soft_acc", "legal_acc", "hard_bpb", "soft_bpb",
     "hold_opt%", "edge_opt%", "vs_opp"))
for r in rows:
    v = r.get("vs_opp")
    vs = ""
    if isinstance(v, dict):
        vs = "W%s L%s D%s" % (v.get("w"), v.get("l"), v.get("d"))
    elif v is not None:
        vs = str(v)
    P("%-6s %-11s %-11s %-10s %-10s %-10s %-10s %-10s %s"
      % (r.get("step"),
         r.get("hard_acc"), r.get("soft_acc"), r.get("legal_acc"),
         r.get("hard_bpb"), r.get("soft_bpb"),
         r.get("holdout_optimal_pct"), r.get("edge_optimal_pct"), vs))
P()

P("=" * 78)
P("2. DISTINCT VALUES (plateau detection)")
P("=" * 78)
for k in ("hard_acc", "soft_acc", "legal_acc", "holdout_optimal_pct",
          "edge_optimal_pct", "hard_bpb", "soft_bpb"):
    vals = [r.get(k) for r in rows if r.get(k) is not None]
    uniq = sorted(set(vals))
    P("%-22s n=%-4d distinct=%-4d  %s"
      % (k, len(vals), len(uniq),
         uniq if len(uniq) <= 8 else ("%s ... %s" % (uniq[:4], uniq[-3:]))))
P()

P("=" * 78)
P("3. HOLDOUT PICK DISTRIBUTION OVER TIME")
P("=" * 78)
for r in rows:
    d = r.get("holdout_pick_dist")
    t = r.get("holdout_pick_tot")
    if d is None:
        continue
    P("step %-6s dist=%-34s tot=%s" % (r.get("step"), d, t))
P()

P("=" * 78)
P("4. CHECKPOINT CROSS-COMPARISON (stale export?)")
P("=" * 78)
from src.gf256_markov_lm_v6 import GFMarkovLM

CKPTS = [
    ("checkpoints/best_hard.pt", "best_hard"),
    ("checkpoints/best_soft.pt", "best_soft"),
    ("hard_model_final.pt", "hard_final"),
    ("final_float.pt", "final_float"),
]
for rel, tag in CKPTS:
    p = RUN / rel
    if not p.exists():
        P("%-14s MISSING" % tag)
        continue
    raw = torch.load(p, map_location="cpu", weights_only=False)
    meta = raw.get("meta") or {}
    P("%-14s step=%-6s keys=%s" % (tag, meta.get("step"), sorted(raw.keys())))
    for k in ("best_hard_acc", "best_hard_bpb", "best_vs_ref_score",
              "best_distinct_games", "transcript", "table_bias", "lags"):
        if k in meta:
            v = meta[k]
            P("      meta.%-20s %r" % (k, v if not isinstance(v, list) or len(v) < 20 else str(v[:12]) + "..."))
    # static column bias from the state dict is architecture-specific; instead
    # compare a hash of the state to detect divergence
    st = raw.get("state") or raw.get("model") or {}
    if isinstance(st, dict) and st:
        import hashlib
        h = hashlib.sha256()
        for k in sorted(st.keys()):
            v = st[k]
            if isinstance(v, torch.Tensor):
                h.update(k.encode())
                h.update(v.detach().float().numpy().tobytes())
        P("      state_sha256[:16] = %s  (tensors=%d)"
          % (h.hexdigest()[:16], sum(1 for v in st.values() if isinstance(v, torch.Tensor))))
    P()

P("=" * 78)
P("5. COMPARISON RUN: runs/c4/perply_v24_full")
P("=" * 78)
alt = Path("runs/c4/perply_v24_full")
if alt.exists():
    P("entries: %s" % sorted(p.name for p in alt.iterdir())[:20])
    mj = alt / "metrics.jsonl"
    if mj.exists():
        arows = [json.loads(l) for l in mj.read_text().splitlines() if l.strip()]
        P("alt metric rows: %d" % len(arows))
        P("alt keys: %s" % sorted({k for r in arows for k in r}))
        for r in arows[-8:]:
            P("   step %-6s hard_acc=%-11s legal_acc=%-9s hold_opt=%-9s edge_opt=%-9s hard_bpb=%s"
              % (r.get("step"), r.get("hard_acc"), r.get("legal_acc"),
                 r.get("holdout_optimal_pct"), r.get("edge_optimal_pct"),
                 r.get("hard_bpb")))
        for k in ("hard_acc", "legal_acc", "holdout_optimal_pct", "edge_optimal_pct"):
            vals = [r.get(k) for r in arows if r.get(k) is not None]
            if vals:
                P("   alt distinct %-22s = %d  last=%s"
                  % (k, len(set(vals)), vals[-1]))
    fj = alt / "final.json"
    if fj.exists():
        P("alt final.json: %s" % json.dumps(json.loads(fj.read_text()))[:800])
else:
    P("alt run dir missing")
P()

Path("tmp/v18_train_forensics.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_train_forensics.txt")
