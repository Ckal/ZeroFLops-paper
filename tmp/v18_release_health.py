import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
L = []
def P(s=""):
    L.append(str(s)); print(s)

P("=" * 78)
P("IS THE RELEASED v24 MODEL (dense lags) STILL COLLAPSED?")
P("=" * 78)

for run in ("runs/c4/perply_v24_full", "runs/_probe/v24"):
    p = Path(run)
    mj = p / "metrics.jsonl"
    if not mj.exists():
        P("no metrics.jsonl in %s" % run); continue
    rows = [json.loads(l) for l in open(mj) if l.strip()]
    P("--- %s : %d rows ---" % (run, len(rows)))
    for r in rows[-5:]:
        P("  step %-6s hard_acc=%-12s legal=%-12s hold_opt=%-8s edge_opt=%-8s bpb=%s"
          % (r.get("step"), r.get("hard_acc"), r.get("legal_acc"),
             r.get("holdout_optimal_pct"), r.get("edge_optimal_pct"), r.get("hard_bpb")))
        P("      pick_tot %s" % (r.get("holdout_pick_tot"),))
        P("      pick_dist %s" % (r.get("holdout_pick_dist"),))
    for k in ("hard_acc", "legal_acc", "holdout_optimal_pct"):
        v = [r.get(k) for r in rows if r.get(k) is not None]
        if v:
            P("  distinct %-20s n=%-4s last=%s  max=%s" % (k, len(set(v)), v[-1], max(v)))
    P()

P("=" * 78)
P("COLUMN-CONCENTRATION TEST ON THE RELEASED CHECKPOINT")
P("=" * 78)
try:
    import torch
    from src.c4 import fmt as fmtmod
    ck = Path("runs/_probe/v24/checkpoints/best_hard.pt")
    r = torch.load(ck, map_location="cpu", weights_only=False)
    P("lags: %s" % r.get("lags"))
    P("holdout_optimal_pct: %s" % r.get("holdout_optimal_pct"))
    P("hard_bpb: %s" % r.get("hard_bpb"))
    P("ladder_net: %s" % r.get("ladder_net"))
except Exception as e:
    P("torch probe failed: %s" % e)
P()

P("=" * 78)
P("STATIC-POLICY REFERENCE (always-column baselines on the same holdout)")
P("=" * 78)
for f in ("v18_baselines.txt", "v18_boardblind.txt"):
    p = Path(f)
    if p.exists():
        P("--- %s ---" % f)
        for ln in p.read_text(errors="replace").splitlines():
            P("  " + ln)
        P()

Path("tmp/v18_release_health.txt").write_text("\n".join(L), encoding="utf-8")
print("\n[written] tmp/v18_release_health.txt")
