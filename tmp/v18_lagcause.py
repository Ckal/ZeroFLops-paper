"""Root cause: LAG STRUCTURE vs the v2 byte layout, plus hard-path argmax.

Question A: what byte-distances (lags) does the model actually integrate?
  The move byte of ply p sits at p*PLY_LEN+9 and is predicted from the
  hidden state at p*PLY_LEN+8 (the turn byte). To see the ply's own
  7 column bytes (p*PLY_LEN+0..6) the model needs lags 2..8.
  If the lag set omits some of those distances, the model CANNOT see
  those columns' board state -> it can only emit a static preference.

Question B: what byte does the HARD path always argmax to?
  If hard_acc is frozen, the hard argmax byte is probably constant.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")
PLY_LEN = 10


def rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


raw = torch.load(RUN / "final_float.pt", map_location="cpu", weights_only=False)
meta = raw.get("meta") or {}

from src.gf256_markov_lm_v6 import GFMarkovLM
m = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
               fan_in=meta.get("fan_in", 3), bigram_logp=None,
               seed=meta.get("seed", 32))
m.load_state_dict(raw["state"])
m.eval()
m.set_ste(False)

print("=" * 74)
print("A. LAG STRUCTURE")
print("=" * 74)
print("model.lags        :", list(m.lags))
print("model.lags (set)  :", sorted(set(int(x) for x in m.lags)))
hcpu_raw = m.export_hard()
print("hard lags         :", list(hcpu_raw.get("lags", [])))
print("hard lags (set)   :", sorted(set(int(x) for x in hcpu_raw.get("lags", []))))
print()
print("v2 byte layout at ply p:")
for off, name in [(0, "col0"), (1, "col1"), (2, "col2"), (3, "col3"),
                  (4, "col4"), (5, "col5"), (6, "col6"), (7, "last_col"),
                  (8, "turn"), (9, "move")]:
    print("   p*10+%d  %-9s  <- distance from read pos (p*10+8) = %2d"
          % (off, name, 8 - off))
print()
need = {8 - o for o in range(7)}   # lags required to see each column byte
have = set(int(x) for x in m.lags)
print("lags required to see the ply's 7 column bytes:", sorted(need))
print("lags present in the model                   :", sorted(have))
missing = sorted(need - have)
print("MISSING required lags                       :", missing)
if missing:
    print("   -> columns whose board state is UNREACHABLE from the move position:")
    for o in range(7):
        if (8 - o) in missing:
            print("        col %d (needs lag %d)" % (o, 8 - o))
print()

# --- empirical: perturb each byte of the final ply, measure hidden-state
#     and logit change at the move read position ---------------------------
seqs = rows(SPLIT / "holdout_corpus.txt")
COL_BYTE_OFFSET = 0x41
sens = np.zeros(10)
n = 0
with torch.no_grad():
    for i in range(0, 400):
        seq = bytearray(seqs[i])
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(191, nply)
        start = max(0, nply * PLY_LEN - (ctx + 1))
        pos_move = (nply - 1) * PLY_LEN + 8 - start
        if not (0 <= pos_move < ctx):
            continue

        def lg_of(seqb):
            win = bytes(seqb[start:start + ctx + 1])
            win = win + b"\n" * (ctx + 1 - len(win))
            arr = np.frombuffer(win, dtype=np.uint8)
            x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
            o = m(x)[0]
            return o[pos_move].numpy().copy()

        base = lg_of(seq)
        hstart = (nply - 1) * PLY_LEN
        for off in range(10):
            p = hstart + off
            if p >= len(seq):
                continue
            old = seq[p]
            seq[p] = old ^ 0x01
            pert = lg_of(seq)
            seq[p] = old
            sens[off] += float(np.abs(pert - base).sum())
        n += 1

print("=" * 74)
print("EMPIRICAL BYTE SENSITIVITY at the move read position (n=%d rows)" % n)
print("=" * 74)
if n:
    sens /= n
    for off in range(10):
        nm = ["col0", "col1", "col2", "col3", "col4", "col5", "col6",
              "last_col", "turn", "move"][off]
        print("   byte p*10+%d (%-8s) : L1 logit change = %.6f"
              % (off, nm, sens[off]))
print()

print("=" * 74)
print("B. HARD-PATH ARGMAX BYTE")
print("=" * 74)
from src.c4.agent import hard_to_cpu, hard_min_len_for
hcpu = hard_to_cpu(hcpu_raw)
hl = max(hard_min_len_for(hcpu["lags"]), 1)
print("hard_len:", hl)
hard_argb = Counter()
soft_argb = Counter()
nh = ns = 0
hhit = shit = 0
for i in range(0, 500):
    b = seqs[i]
    arr = np.frombuffer(b, dtype=np.uint8)
    x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        ho = m.hard_forward(x, hcpu).float()[0]
        so = m(x)[0]
    ha = ho[:-1].argmax(-1)
    sa = so[:-1].argmax(-1)
    nxt = arr[1:]
    for p in range(len(nxt)):
        if (p + 1) % PLY_LEN == 0:
            nh += 1
            ns += 1
            hard_argb[int(ha[p])] += 1
            soft_argb[int(sa[p])] += 1
            hhit += (int(ha[p]) == int(nxt[p]))
            shit += (int(sa[p]) == int(nxt[p]))
print("move positions scored:", nh)
print("HARD argmax byte dist:", hard_argb.most_common(8))
print("HARD distinct bytes  :", len(hard_argb))
print("HARD byte-acc        : %.6f" % (hhit / max(nh, 1)))
print("SOFT argmax byte dist:", soft_argb.most_common(8))
print("SOFT distinct bytes  :", len(soft_argb))
print("SOFT byte-acc        : %.6f" % (shit / max(ns, 1)))
if hard_argb:
    top, cnt = hard_argb.most_common(1)[0]
    print()
    print("hard argmax byte = %d (0x%02x) in %.2f%% of move positions"
          % (top, top, 100.0 * cnt / max(nh, 1)))
    freq = Counter()
    for b in seqs[:2000]:
        arr = np.frombuffer(b, dtype=np.uint8)
        for p in range(0, len(arr) - 1):
            if (p + 1) % PLY_LEN == 0:
                freq[int(arr[p + 1])] += 1
    tot = sum(freq.values())
    print("corpus move-byte frequency of that byte: %.14f" % (freq[top] / max(tot, 1)))
    print("observed frozen hard_acc                : 0.06955460646735814")
