import json, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path("tmp/v18_final_prov.txt")
L = []
def P(s=""):
    L.append(str(s)); print(s)

RUN = Path("runs/c4/v7mixed_v18_seed32")
rows = [json.loads(l) for l in open(RUN / "metrics.jsonl") if l.strip()]
f = json.load(open(RUN / "final.json"))

P("=" * 78); P("1. final.json AS WRITTEN BY THE RUN (2026-10-23)"); P("=" * 78)
P(json.dumps(f, indent=2))
P()

P("=" * 78); P("2. CANDIDATE SLOT MAPPING (4-tuple = ladder_net, hold_opt, -hard_bpb, distinct)"); P("=" * 78)
for r in rows:
    if r.get("step") == 1300:
        vs = r.get("vs_ref") or {}
        net = vs.get("wins", 0) - vs.get("losses", 0)
        P("step 1300 row:")
        P("  ladder_net (wins-losses)      = %s - %s = %s" % (vs.get("wins"), vs.get("losses"), net))
        P("  holdout_optimal_pct           = %s" % r.get("holdout_optimal_pct"))
        P("  hard_bpb                      = %s" % r.get("hard_bpb"))
        P("  best_distinct (from metrics?) = %s" % r.get("distinct_games"))
        P()
        P("final.json best_vs_ref_score = %s" % f["best_vs_ref_score"])
        P("  == -hard_bpb                 ? %s" % (abs(f["best_vs_ref_score"] - (-r["hard_bpb"])) < 1e-6))
        P("  == ladder_net                ? %s" % (abs(f["best_vs_ref_score"] - net) < 1e-6))
        P("final.json best_hard_acc      = %s" % f["best_hard_acc"])
        P("  == ladder_net                ? %s" % (abs(f["best_hard_acc"] - net) < 1e-6))
        P("  == -(hard_acc*100)           ? %s" % (abs(f["best_hard_acc"] - (-r["hard_acc"] * 100)) < 1e-6))
        P("final.json best_hard_bpb      = %s" % f["best_hard_bpb"])
        P("  == -(holdout_optimal_pct)    ? %s" % (abs(f["best_hard_bpb"] - (-r["holdout_optimal_pct"])) < 1e-6))
        P("  == -hard_bpb                 ? %s" % (abs(f["best_hard_bpb"] - (-r["hard_bpb"])) < 1e-6))
        P("  == -(distinct*100)           ? %s" % (abs(f["best_hard_bpb"] - (-(r.get("distinct_games") or 0) * 100)) < 1e-6))
P()
P("distinct_games value in the step-1300 row: " + repr(rows[-1].get("distinct_games")))
for r in rows:
    if r.get("step") == 1300:
        P("  step1300 distinct_games: " + repr(r.get("distinct_games")))
        P("  step1300 keys: " + str(sorted(r.keys())))
P()

P("=" * 78); P("3. GIT: WHEN WERE THE final.json KEYS RENAMED?"); P("=" * 78)
g = lambda *a: subprocess.run(["git"] + list(a), capture_output=True, text=True, cwd=".").stdout
for c in g("log", "--oneline", "-S", "best_vs_ref_score", "--", "src/c4/train.py").splitlines():
    P("  log -S best_vs_ref_score : " + c)
for c in g("log", "--oneline", "-S", "best_ladder_net", "--", "src/c4/train.py").splitlines():
    P("  log -S best_ladder_net  : " + c)
P()
P("  current HEAD writer (src/c4/train.py 3801-3812):")
P("    keys = n_params, corpus_games, corpus_bytes, best_ladder_net,")
P("           best_ladder_opp, best_holdout_optimal_pct, best_hard_bpb,")
P("           best_distinct_games, transcript, elapsed_s")
P("  -> the run's final.json still has the OLD keys best_vs_ref_score /")
P("     best_hard_acc, so it was written by pre-rename code.")
P()

P("=" * 78); P("4. LAG SET OF THE RELEASED / SERVED ARTIFACTS"); P("=" * 78)
cands = [
    "release/artifacts/connect_four_gf256_best_hard_step18000.pt",
    "runs/_probe/v24/checkpoints/best_hard.pt",
    "runs/c4/v7mixed_v18_seed32/checkpoints/best_hard.pt",
    "runs/c4/v7mixed_v18_seed32/hard_model_final.pt",
    "runs/c4/v7mixed_v18_seed32/final_float.pt",
    "runs/c4/v7mixed_v23/checkpoints/best_hard.pt",
    "runs/c4/v7mixed_v19/checkpoints/best_hard.pt",
]
for rel in cands:
    p = Path(rel)
    if not p.exists():
        P("%-58s MISSING" % rel); continue
    try:
        raw = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        P("%-58s LOAD-ERR %s" % (rel, e)); continue
    meta = raw.get("meta") or {}
    lags = meta.get("lags")
    cfg = meta.get("config") or {}
    P("%-58s step=%-7s lags=%s" % (rel, meta.get("step"), lags))
    if isinstance(cfg, dict) and cfg:
        P("     cfg.lags=%s  d=%s layers=%s" % (cfg.get("lags"), cfg.get("d"), cfg.get("layers")))
P()

P("=" * 78); P("5. LAG SET USED BY EACH SIBLING RUN (from train.log)"); P("=" * 78)
for d in sorted(Path("runs/c4").iterdir()):
    lg = d / "train.log"
    if not lg.is_dir() and lg.exists():
        txt = lg.read_text(errors="replace")
        head = txt.splitlines()[:6]
        lag = None
        for ln in head:
            if "lags" in ln.lower():
                lag = ln.strip(); break
        P("%-34s %s" % (d.name, (lag or head[0] if head else "")[:150]))
P()

P("=" * 78); P("6. IS THE PUBLISHED MODEL HEALTHY? (release artifact metrics)"); P("=" * 78)
for rel in ("release/RELEASE_RECORD.md",):
    p = Path(rel)
    if p.exists():
        txt = p.read_text(errors="replace")
        P("--- %s (last 60 lines) ---" % rel)
        for ln in txt.splitlines()[-60:]:
            P("  " + ln)

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n[written]", OUT)
