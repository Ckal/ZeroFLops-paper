"""Decisive architectural check for V18.

Q1 (blind column): per-column HARD-PATH SENSITIVITY. Perturb the board byte
    for column c, measure the L1 change in the column logits. If sensitivity
    is ~0 for every column, the column head cannot see the board at all and
    the "model" is a constant policy by construction.

Q2 (frozen hard_acc): is the hard-path argmax BYTE constant across positions?
    If yes, hard_acc is pinned to that byte's corpus frequency and cannot
    move -> explains the byte-identical hard_acc across the last evals.

Q3: unreachable fraction (the skill's decisive metric) on unique-optimum rows.

Q4: table_bias / per-column bias parameters in the checkpoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")
MOVE_OFFSET = 9
PLY_LEN = 10
COL_BYTE_OFFSET = 0x41
CTX = 191

OUT = []


def P(s=""):
    OUT.append(str(s))
    print(s)


def rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


raw = torch.load(RUN / "final_float.pt", map_location="cpu", weights_only=False)
meta = raw.get("meta") or {}
P("=" * 74)
P("Q4. CHECKPOINT META")
P("=" * 74)
for k in sorted(meta.keys()):
    v = meta[k]
    if isinstance(v, (int, float, str, bool, type(None), list, tuple)):
        P("  meta.%-24s %r" % (k, v))
    else:
        P("  meta.%-24s <%s>" % (k, type(v).__name__))
P()
sd = raw["state"]
P("  state keys (%d): %s" % (len(sd), sorted(sd.keys())))
P()

from src.gf256_markov_lm_v6 import GFMarkovLM
model = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
                   fan_in=meta.get("fan_in", 3), bigram_logp=None,
                   seed=meta.get("seed", 32))
model.load_state_dict(sd)
model.eval()
model.set_ste(False)

P("=" * 74)
P("Q2. IS THE HARD-PATH ARGMAX BYTE CONSTANT?")
P("=" * 74)
seqs = rows(SPLIT / "holdout_corpus.txt")
probe_rows = list(range(0, len(seqs), max(1, len(seqs) // 600)))[:600]

argmax_bytes = []
argmax_cols = []
col_logits_all = []

with torch.no_grad():
    for i in probe_rows:
        seq = seqs[i]
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(CTX, nply)
        start = max(0, nply - (ctx + 1))
        window = seq[start:start + ctx + 1]
        window = window + b"\n" * (ctx + 1 - len(window))
        arr = np.frombuffer(window, dtype=np.uint8)
        x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
        logits = model(x)[0]
        pos = nply - 1 - start - 1
        if not (0 <= pos < ctx):
            continue
        full = logits[pos]
        col_logits_all.append(full[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].numpy().copy())
        argmax_bytes.append(int(full.argmax()))
        argmax_cols.append(int(full[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].argmax()))

P("  probed %d holdout positions" % len(argmax_bytes))
from collections import Counter
cb = Counter(argmax_bytes)
P("  hard-path argmax BYTE over all 256 values:")
for b, n in cb.most_common(6):
    P("     0x%02x : %5d  (%.2f%%)" % (b, n, 100.0 * n / len(argmax_bytes)))
cc = Counter(argmax_cols)
P("  hard-path argmax COLUMN:")
for k, n in sorted(cc.items()):
    P("     col %d : %5d  (%.2f%%)" % (k, n, 100.0 * n / len(argmax_cols)))
P("  -> distinct argmax bytes: %d ; distinct argmax columns: %d"
  % (len(cb), len(cc)))
L = np.array(col_logits_all)
P()
P("  column-logit matrix over probed positions (mean / std per column):")
P("    mean: %s" % np.round(L.mean(0), 4).tolist())
P("    std : %s" % np.round(L.std(0), 4).tolist())
P("    ptp : %s" % np.round(L.ptp(0), 4).tolist())
P("  -> per-column logit SPREAD across positions (ptp) shows whether the head")
P("     responds to board state at all.")
P()

P("=" * 74)
P("Q1. PER-COLUMN HARD-PATH SENSITIVITY")
P("=" * 74)
P("  perturb the board byte for column c, measure L1 change in the 7 col logits")


def header_cols(seq):
    """Return the 7 header bytes for the final ply (7 columns x 1 byte)."""
    nply = len(seq) // PLY_LEN
    hstart = (nply - 1) * PLY_LEN + MOVE_OFFSET
    return hstart


sens = np.zeros(7)
tried = 0
with torch.no_grad():
    for i in probe_rows[:200]:
        seq = bytearray(seqs[i])
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(CTX, nply)
        start = max(0, nply - (ctx + 1))
        hstart = (nply - 1) * PLY_LEN + MOVE_OFFSET

        def run(seqb):
            win = bytes(seqb[start:start + ctx + 1])
            win = win + b"\n" * (ctx + 1 - len(win))
            arr = np.frombuffer(win, dtype=np.uint8)
            x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
            lg = model(x)[0]
            pos = nply - 1 - start - 1
            return lg[pos][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].numpy().copy()

        base = run(seq)
        ok = True
        for c in range(7):
            off = hstart + c
            if off >= len(seq):
                ok = False
                break
            old = seq[off]
            seq[off] = old ^ 0x01
            pert = run(seq)
            seq[off] = old
            sens[c] += np.abs(pert - base).sum()
        if ok:
            tried += 1
if tried:
    sens /= tried
    for c in range(7):
        P("     col %d : mean L1 logit change = %.6f" % (c, sens[c]))
    P("  -> max sensitivity %.6f (col %d); min %.6f (col %d)"
      % (sens.max(), int(sens.argmax()), sens.min(), int(sens.argmin())))
    P("  -> SENSITIVITY IS %s" % ("~0 FOR ALL COLUMNS (blind head)"
                                 if sens.max() < 1e-3 else "NON-ZERO (head sees the board)"))
P()

P("=" * 74)
P("Q3. UNREACHABLE FRACTION (unique-optimum rows)")
P("=" * 74)
c = np.load("runs/_probe/v18_picks_posdata_split.npz")
idx, pick, legal = c["idx"], c["pick"], c["legal"].astype(bool)
T = np.load(SPLIT / "holdout_targets.npz")["target"]
Tam = T[idx]
opt = np.isclose(Tam, Tam.max(1, keepdims=True), atol=1e-6)
uniq = opt.sum(1) == 1
uq_rows = np.where(uniq)[0]
emitted = set(int(x) for x in np.unique(pick))
P("  rows with a UNIQUE optimum : %d of %d" % (len(uq_rows), len(idx)))
unreach = 0
for r in uq_rows:
    uc = int(np.argmax(opt[r]))
    if uc not in emitted:
        unreach += 1
P("  columns the model ever emits: %s" % sorted(emitted))
P("  unique-optimum rows whose optimal column is NEVER emitted: %d (%.3f%%)"
  % (unreach, 100.0 * unreach / max(len(uq_rows), 1)))
P("  -> hard ceiling on policy score from coverage alone.")
P()
P("  model score on unique-optimum rows : %.3f%%"
  % (100.0 * (opt[uq_rows, pick[uq_rows]] & legal[uq_rows, pick[uq_rows]]).mean()))
P("  chance floor on unique-optimum rows: %.3f%%" % (100.0 / 7))

Path("tmp/v18_sensitivity_report.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_sensitivity_report.txt")
