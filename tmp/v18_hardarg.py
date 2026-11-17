"""Probe the HARD-path argmax byte and the soft-path argmax byte.

hard_acc froze at step 1000 at 0.06955460646735814. If the hard path always
argmaxes to the SAME byte, hard_acc is exactly the corpus frequency of that
byte -- which is what we measured.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")


def rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


raw = torch.load(RUN / "final_float.pt", map_location="cpu", weights_only=False)
meta = raw.get("meta") or {}
print("final_float meta keys:", sorted(meta.keys()))
for k in ("step", "d", "layers", "fan_in", "seed", "transcript", "table_bias",
          "lags", "ladder_opp", "ladder_greedy"):
    if k in meta:
        print("   %s = %r" % (k, meta[k]))

from src.gf256_markov_lm_v6 import GFMarkovLM
m = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
               fan_in=meta.get("fan_in", 3), bigram_logp=None,
               seed=meta.get("seed", 32))
m.load_state_dict(raw["state"])
m.eval()
m.set_ste(False)

seqs = rows(SPLIT / "holdout_corpus.txt")
print("holdout rows:", len(seqs))

# ---- import the trainer's own move-byte mask if available -------------------
mask_fn = None
try:
    from src.c4 import train as c4train
    for nm in ("_move_byte_mask", "move_byte_mask", "_target_mask"):
        if hasattr(c4train, nm):
            mask_fn = getattr(c4train, nm)
            print("using trainer mask fn:", nm)
            break
except Exception as e:
    print("trainer mask import failed:", e)

from src.c4.agent import hard_to_cpu, hard_min_len_for
hcpu = hard_to_cpu(m.export_hard())
hl = max(hard_min_len_for(hcpu["lags"]), 1)
print("hard_len:", hl, "lags:", list(hcpu["lags"])[:20])

hard_argb = Counter()
soft_argb = Counter()
hard_hits = 0
soft_hits = 0
ntok = 0
for i in range(0, 600):
    b = seqs[i]
    arr = np.frombuffer(b, dtype=np.uint8)
    x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        hout = m.hard_forward(x, hcpu).float()[0]
        sout = m(x)[0]
    ha = hout[:-1].argmax(-1)
    sa = sout[:-1].argmax(-1)
    nxt = arr[1:]
    # move byte sits at ply_start+9; positions in the flattened stream
    for p in range(len(nxt)):
        if (p + 1) % 10 == 0:      # move byte of each ply (0-indexed ply start p)
            ntok += 1
            hb = int(ha[p]); sb = int(sa[p]); tgt = int(nxt[p])
            hard_argb[hb] += 1
            soft_argb[sb] += 1
            hard_hits += (hb == tgt)
            soft_hits += (sb == tgt)

print()
print("move positions scored:", ntok)
print("HARD argmax byte distribution:", hard_argb.most_common(8))
print("HARD distinct argmax bytes:", len(hard_argb))
print("HARD byte-acc on move positions: %.6f" % (hard_hits / max(ntok, 1)))
print("SOFT argmax byte distribution:", soft_argb.most_common(8))
print("SOFT distinct argmax bytes:", len(soft_argb))
print("SOFT byte-acc on move positions: %.6f" % (soft_hits / max(ntok, 1)))

# corpus frequency of the hard argmax byte among MOVE bytes
top = hard_argb.most_common(1)[0][0]
freq = Counter()
for b in seqs[:2000]:
    arr = np.frombuffer(b, dtype=np.uint8)
    for p in range(0, len(arr) - 1):
        if (p + 1) % 10 == 0:
            freq[int(arr[p + 1])] += 1
tot = sum(freq.values())
print()
print("hard argmax byte = %d (0x%02x)" % (top, top))
print("corpus move-byte frequency of that byte: %.12f"
      % (freq[top] / max(tot, 1)))
print("observed frozen hard_acc              : 0.06955460646735814")
print("argmax byte share among move bytes    : %.12f" % (hard_argb[top] / max(ntok, 1)))
