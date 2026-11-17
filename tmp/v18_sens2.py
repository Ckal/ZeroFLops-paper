"""Decisive architectural checks for V18 (corrected v2 header offsets).

Q1  PER-COLUMN HARD-PATH SENSITIVITY: perturb column c's board byte and
    measure the L1 change in the 7 column logits. (v2 header: 7 column
    bytes at ply_start+0..6, last-col at +7, turn at +8, move byte at +9.)
Q2  RESIDUAL BOARD-DEPENDENT SIGNAL after removing the static per-column
    bias: does the teacher-optimal column still rank above non-optimal ones?
Q3  UNREACHABLE FRACTION on unique-optimum rows (skill's decisive metric).
Q4  Is the hard-path argmax BYTE constant (explains the frozen hard_acc)?
"""
import sys
from collections import Counter
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
P("Q4. CHECKPOINT META (final_float.pt)")
P("=" * 74)
for k in sorted(meta.keys()):
    v = meta[k]
    if isinstance(v, (int, float, str, bool, type(None), list, tuple)):
        P("  meta.%-24s %r" % (k, v))
    else:
        P("  meta.%-24s <%s>" % (k, type(v).__name__))
P()

from src.gf256_markov_lm_v6 import GFMarkovLM
model = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
                   fan_in=meta.get("fan_in", 3), bigram_logp=None,
                   seed=meta.get("seed", 32))
model.load_state_dict(raw["state"])
model.eval()
model.set_ste(False)

seqs = rows(SPLIT / "holdout_corpus.txt")
T_all = np.load(SPLIT / "holdout_targets.npz")["target"]

probe = list(range(0, len(seqs), max(1, len(seqs) // 800)))[:800]

argmax_bytes = []
argmax_cols = []
collog = []
rowidx = []

with torch.no_grad():
    for i in probe:
        seq = seqs[i]
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(CTX, nply)
        start = max(0, nply - (ctx + 1))
        win = seq[start:start + ctx + 1]
        win = win + b"\n" * (ctx + 1 - len(win))
        arr = np.frombuffer(win, dtype=np.uint8)
        x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
        lg = model(x)[0]
        pos = nply - 1 - start - 1
        if not (0 <= pos < ctx):
            continue
        full = lg[pos]
        collog.append(full[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].numpy().copy())
        argmax_bytes.append(int(full.argmax()))
        argmax_cols.append(int(full[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].argmax()))
        rowidx.append(i)

L = np.array(collog)
P("=" * 74)
P("Q4b. HARD-PATH ARGMAX DISTRIBUTION (over %d probed holdout positions)" % len(argmax_bytes))
P("=" * 74)
cb = Counter(argmax_bytes)
for b, n in cb.most_common(6):
    P("   byte 0x%02x : %5d  (%.2f%%)" % (b, n, 100.0 * n / len(argmax_bytes)))
P("   distinct argmax BYTES: %d" % len(cb))
cc = Counter(argmax_cols)
for k, n in sorted(cc.items()):
    P("   col %d : %5d  (%.2f%%)" % (k, n, 100.0 * n / len(cc)))
P("   distinct argmax COLUMNS: %d" % len(cc))
P()
P("   column-logit mean per column : %s" % np.round(L.mean(0), 4).tolist())
P("   column-logit std  per column : %s" % np.round(L.std(0), 4).tolist())
P("   column-logit ptp  per column : %s" % np.round(np.ptp(L, axis=0), 4).tolist())
P()

P("=" * 74)
P("Q2. RESIDUAL BOARD-DEPENDENT SIGNAL (static bias removed)")
P("=" * 74)
static_bias = L.mean(0)
resid = L - static_bias
P("   static bias (mean logit per column): %s" % np.round(static_bias, 4).tolist())
P("   residual argmax distribution       : %s"
  % Counter(int(x) for x in resid.argmax(1)).most_common())
ri = np.array(rowidx)
Tam = T_all[ri]
opt = np.isclose(Tam, Tam.max(1, keepdims=True), atol=1e-6)
pos_scores = []
neg_scores = []
for r in range(len(ri)):
    rs = resid[r]
    for c in range(7):
        (pos_scores if opt[r, c] else neg_scores).append(rs[c])
pos_scores = np.array(pos_scores)
neg_scores = np.array(neg_scores)
P("   teacher-optimal residual logit: mean %.4f  median %.4f  (n=%d)"
  % (pos_scores.mean(), np.median(pos_scores), len(pos_scores)))
P("   non-optimal residual logit   : mean %.4f  median %.4f  (n=%d)"
  % (neg_scores.mean(), np.median(neg_scores), len(neg_scores)))
aucs = []
for r in range(len(ri)):
    p_ = resid[r][opt[r]]
    n_ = resid[r][~opt[r]]
    if len(p_) and len(n_):
        aucs.append((p_[:, None] > n_[None, :]).mean())
aucs = np.array(aucs)
P("   per-position AUC (optimal vs not, residual logits): mean %.4f  median %.4f"
  % (aucs.mean(), np.median(aucs)))
P("   AUC 0.50 = NO signal; > 0.55 = weak but real signal")
P()

P("=" * 74)
P("Q1. PER-COLUMN HARD-PATH SENSITIVITY (perturb board byte for column c)")
P("=" * 74)
sens = np.zeros(7)
cnt = 0
with torch.no_grad():
    for i in probe[:250]:
        seq = bytearray(seqs[i])
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(CTX, nply)
        start = max(0, nply - (ctx + 1))
        hstart = (nply - 1) * PLY_LEN  # v2 header starts at ply start

        def run(seqb):
            win = bytes(seqb[start:start + ctx + 1])
            win = win + b"\n" * (ctx + 1 - len(win))
            arr = np.frombuffer(win, dtype=np.uint8)
            x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
            lg = model(x)[0]
            pos = nply - 1 - start - 1
            return lg[pos][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].numpy().copy()

        base = run(seq)
        for c in range(7):
            off = hstart + c
            if off >= len(seq):
                continue
            old = seq[off]
            seq[off] = old ^ 0x01
            pert = run(seq)
            seq[off] = old
            sens[c] += float(np.abs(pert - base).sum())
        cnt += 1
if cnt:
    sens /= cnt
    for c in range(7):
        P("   col %d : mean L1 column-logit change = %.6f" % (c, sens[c]))
    P("   -> max %.6f (col %d) | min %.6f (col %d)"
      % (sens.max(), int(sens.argmax()), sens.min(), int(sens.argmin())))
    P("   -> verdict: %s"
      % ("BLIND HEAD (all sensitivities ~0)" if sens.max() < 1e-3
         else "head responds to the board"))
P()

P("=" * 74)
P("Q3. UNREACHABLE FRACTION (unique-optimum rows)")
P("=" * 74)
c = np.load("runs/_probe/v18_picks_posdata_split.npz")
idx, pick, legal = c["idx"], c["pick"], c["legal"].astype(bool)
T = np.load(SPLIT / "holdout_targets.npz")["target"]
Tam = T[idx]
opt = np.isclose(Tam, Tam.max(1, keepdims=True), atol=1e-6)
uq = np.where(opt.sum(1) == 1)[0]
emitted = set(int(x) for x in np.unique(pick))
unreach = sum(1 for r in uq if int(np.argmax(opt[r])) not in emitted)
P("   unique-optimum rows: %d of %d" % (len(uq), len(idx)))
P("   columns the model ever emits: %s" % sorted(emitted))
P("   unique-optimum rows whose optimal column is NEVER emitted: %d (%.3f%%)"
  % (unreach, 100.0 * unreach / max(len(uq), 1)))
P("   model score on unique-optimum rows : %.3f%%"
  % (100.0 * (opt[uq, pick[uq]] & legal[uq, pick[uq]]).mean()))
P("   chance floor (1/7)                 : %.3f%%" % (100.0 / 7))
P()

Path("tmp/v18_sensitivity_report.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_sensitivity_report.txt")
