"""DECISIVE PROBE: byte-lag receptive field vs the 7 column bytes.

Hypothesis H (from tmp/v18_lagcause.txt):
  The model's byte lags are [0,1,2,4,8,16,32,64]. Under the v2 layout the
  read position for ply p's MOVE is p*10+8, so ply p's 7 column bytes sit at
  distances {8,7,6,5,4,3,2} from it. Lags 3,5,6,7 are ABSENT, so columns
  1,2,3,5 occupancy bytes are UNREACHABLE at the decision point.

This probe measures, per header byte:
  (a) the L1 change in the FULL 256-way logit vector
  (b) the L1 change restricted to the 7 COLUMN logits (what actually picks
      the move)  <-- the decisive measurement
  (c) the change in the model's CHOSEN column
and it does this for the CURRENT ply and the PREVIOUS ply.

It also confirms the run's own checkpoint lag configuration and compares the
lag sets of the sibling runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.fmt import PLY_LEN, MOVE_OFFSET, COL_BYTE_OFFSET  # noqa: E402
from src.gf256_markov_lm_v6 import GFMarkovLM  # noqa: E402

RUN = Path("runs/c4/v7mixed_v18_seed32")
OUT = []


def P(s=""):
    OUT.append(str(s))
    print(s)


def load_corpus(n=400):
    """Reconstruct the same holdout rows the trainer scored."""
    import glob
    cands = sorted(glob.glob("runs/c4/posdata_split/*.bin")) + \
        sorted(glob.glob("runs/c4/posdata_split/*.raw"))
    return cands


P("=" * 78)
P("0. LAYOUT CONSTANTS")
P("=" * 78)
P("PLY_LEN = %d   MOVE_OFFSET = %d   COL_BYTE_OFFSET = 0x%02X"
  % (PLY_LEN, MOVE_OFFSET, COL_BYTE_OFFSET))
P()

P("=" * 78)
P("1. LAG SETS OF EVERY CHECKPOINT IN THIS RUN")
P("=" * 78)
for rel in ("checkpoints/best_hard.pt", "checkpoints/best_soft.pt",
            "hard_model_final.pt", "final_float.pt"):
    p = RUN / rel
    if not p.exists():
        P("%-24s MISSING" % rel)
        continue
    raw = torch.load(p, map_location="cpu", weights_only=False)
    meta = raw.get("meta") or {}
    lags = meta.get("lags")
    P("%-24s step=%-6s lags=%s" % (rel, meta.get("step"), lags))
    if lags is None:
        # dig it out of the state dict shape
        for k, v in (raw.get("state") or {}).items():
            if "lag" in k.lower():
                P("      state[%s] shape=%s" % (k, tuple(v.shape)))
P()

P("=" * 78)
P("2. REQUIRED LAGS vs CONFIGURED LAGS (v2 layout)")
P("=" * 78)
LAGS = [0, 1, 2, 4, 8, 16, 32, 64]
READ_OFF = 8  # turn byte offset inside a ply == read position
required = {}
for c in range(7):
    off = c  # column byte offset inside the ply
    dist = READ_OFF - off
    required[c] = dist
P("read position for ply p's move = p*%d + %d" % (PLY_LEN, READ_OFF))
P("column byte offset inside ply    : 0..6")
P()
P("col : byte_off : lag_needed : lag_present : reachable")
for c in range(7):
    dist = required[c]
    present = dist in LAGS
    P(" %d  :    %d     :    %2d      :   %-5s     : %s"
      % (c, c, dist, str(present), "YES" if present else "NO  <== BLIND"))
P()
P("reachable columns : %s" % [c for c in range(7) if required[c] in LAGS])
P("blind columns     : %s" % [c for c in range(7) if required[c] not in LAGS])
P()

P("=" * 78)
P("3. DECISIVE MEASUREMENT: COLUMN-SLICE SENSITIVITY")
P("=" * 78)

# Rebuild the eval input the same way the trainer does, from the holdout
# corpus rows, and perturb single header bytes.
sys.path.insert(0, str(ROOT))
try:
    from src.c4 import transcript as T
except Exception:
    T = None

HOLD = Path("runs/c4/posdata_split")
files = sorted(HOLD.glob("*"))
P("posdata_split entries: %s" % [f.name for f in files][:20])
P()

# Load the model
raw = torch.load(RUN / "checkpoints/best_hard.pt", map_location="cpu",
                 weights_only=False)
sd = raw.get("state") or raw.get("model")
meta = raw.get("meta") or {}
P("best_hard.pt top-level keys: %s" % sorted(raw.keys()))
P("meta: %s" % json.dumps({k: (v if isinstance(v, (int, float, str, list, type(None))) else str(v))
                             for k, v in meta.items()})[:1500])
P()

cfg = meta.get("config") or meta.get("cfg") or {}
P("config keys: %s" % sorted(cfg.keys()) if isinstance(cfg, dict) else "no config")

if isinstance(cfg, dict):
    m = GFMarkovLM(
        d=cfg.get("d", cfg.get("dim", 32)),
        layers=cfg.get("layers", 3),
        fan_in=cfg.get("fan_in", 3),
        lags=cfg.get("lags", LAGS),
    )
    m.load_state_dict(sd)
    m.eval()
else:
    P("!! no config in meta; cannot rebuild model -- skipping sensitivity")
    Path("tmp/v18_decisive.txt").write_text("\n".join(OUT), encoding="utf-8")
    raise SystemExit(0)

P("rebuilt model: lags=%s" % (m.lags if hasattr(m, "lags") else "?"))
P()

# ---- build inputs from the actual holdout corpus -----------------------
import src.c4.train as TR  # noqa: E402

# find how the trainer loads holdout bytes
def _load_holdout_bytes(n=400):
    """Try the corpus files the trainer used."""
    cands = []
    for pat in ("runs/c4/posdata_split/*.bin", "runs/c4/posdata_split/*.dat",
                "runs/c4/posdata_split/*.raw", "runs/c4/posdata_split/holdout*"):
        cands += sorted(Path(".").glob(pat))
    return cands


cands = _load_holdout_bytes()
P("candidate holdout byte files: %s" % [str(c) for c in cands][:10])
P()

Path("tmp/v18_decisive.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_decisive.txt")
