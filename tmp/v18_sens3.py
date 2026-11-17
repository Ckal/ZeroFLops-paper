"""CORRECTED per-column sensitivity for V18.

The previous probe read logits at position (nply-1-start-1), which is the
position predicting a byte deep inside an EARLIER ply. A causal model cannot
see the final ply's header from there, so every sensitivity measured 0.000000
-- a measurement artifact, not a property of the model.

Correct position: logits[pos_move] predicts the final ply's MOVE byte, and
is computed from context that INCLUDES the final ply's 7 header bytes.

  pos_move = (nply-1)*PLY_LEN + 8 - start

Controls:
  + perturb header byte c   -> must change logits[pos_move] if the head sees it
  - perturb the MOVE byte    -> must NOT change logits[pos_move] (causal)
  + perturb an EARLIER ply  -> must change logits[pos_move] (context is used)
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
from src.gf256_markov_lm_v6 import GFMarkovLM
model = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
                   fan_in=meta.get("fan_in", 3), bigram_logp=None,
                   seed=meta.get("seed", 32))
model.load_state_dict(raw["state"])
model.eval()
model.set_ste(False)

seqs = rows(SPLIT / "holdout_corpus.txt")
probe = list(range(0, len(seqs), max(1, len(seqs) // 400)))[:400]

sens = np.zeros(7)
ctrl_move = np.zeros(7)   # should stay 0 (causal)
ctrl_early = 0.0          # should be > 0 (context is used)
n_used = 0
collog = []
rowidx = []


def logits_at(seqb, start, ctx, pos_move):
    win = bytes(seqb[start:start + ctx + 1])
    win = win + b"\n" * (ctx + 1 - len(win))
    arr = np.frombuffer(win, dtype=np.uint8)
    x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
    lg = model(x)[0]
    return lg[pos_move][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].numpy().copy()


with torch.no_grad():
    for i in probe:
        seq = bytearray(seqs[i])
        nply = len(seq) // PLY_LEN
        if nply < 2:
            continue
        ctx = min(CTX, nply)
        start = max(0, nply * PLY_LEN - (ctx + 1))
        winlen = nply * PLY_LEN - start
        if winlen < 2:
            continue
        pos_move = (nply - 1) * PLY_LEN + 8 - start
        if not (0 <= pos_move < ctx):
            continue
        base = logits_at(seq, start, ctx, pos_move)
        collog.append(base.copy())
        rowidx.append(i)

        hstart = (nply - 1) * PLY_LEN
        for c in range(7):
            off = hstart + c
            if off >= len(seq):
                continue
            old = seq[off]
            seq[off] = old ^ 0x01
            pert = logits_at(seq, start, ctx, pos_move)
            seq[off] = old
            sens[c] += float(np.abs(pert - base).sum())

        # control 1: perturb the move byte (after pos_move) -> must not matter
        moff = hstart + 9
        if moff < len(seq):
            old = seq[moff]
            seq[moff] = old ^ 0x01
            pm = logits_at(seq, start, ctx, pos_move)
            seq[moff] = old
            ctrl_move[0] += float(np.abs(pm - base).sum())

        # control 2: perturb a byte in the previous ply -> must matter
        poff = hstart - PLY_LEN + 2
        if 0 <= poff < len(seq):
            old = seq[poff]
            seq[poff] = old ^ 0x01
            pe = logits_at(seq, start, ctx, pos_move)
            seq[poff] = old
            ctrl_early += float(np.abs(pe - base).sum())
        n_used += 1

P("=" * 74)
P("CORRECTED PER-COLUMN SENSITIVITY (final ply header -> move-byte logits)")
P("  rows used: %d ; read position = (nply-1)*PLY_LEN + 8 - start" % n_used)
P("=" * 74)
if n_used:
    sens /= n_used
    for c in range(7):
        P("   header col %d : mean L1 change in the 7 column logits = %.6f" % (c, sens[c]))
    P("   -> max %.6f (col %d) | min %.6f (col %d)"
      % (sens.max(), int(sens.argmax()), sens.min(), int(sens.argmin())))
    P("   -> VERDICT: %s"
      % ("BLIND HEAD (no column header byte changes the move logits)"
         if sens.max() < 1e-4 else "HEAD SEES THE BOARD"))
P()
P("  CONTROL 1  perturb the MOVE byte (after the read position): %.8f  (expect 0)"
  % (ctrl_move[0] / max(n_used, 1)))
P("  CONTROL 2  perturb a byte in the PREVIOUS ply:             %.6f  (expect > 0)"
  % (ctrl_early / max(n_used, 1)))
P()

L = np.array(collog)
P("=" * 74)
P("COLUMN LOGITS AT THE MOVE POSITION")
P("=" * 74)
P("   mean per column : %s" % np.round(L.mean(0), 4).tolist())
P("   std  per column : %s" % np.round(L.std(0), 4).tolist())
P("   ptp  per column : %s" % np.round(np.ptp(L, axis=0), 4).tolist())
P("   argmax column distribution: %s" % sorted(Counter(int(x) for x in L.argmax(1)).items()))
P()

# residual signal after removing static bias
bias = L.mean(0)
resid = L - bias
T_all = np.load(SPLIT / "holdout_targets.npz")["target"]
ri = np.array(rowidx)
Tam = T_all[ri]
opt = np.isclose(Tam, Tam.max(1, keepdims=True), atol=1e-6)
aucs = []
for r in range(len(ri)):
    p_ = resid[r][opt[r]]
    n_ = resid[r][~opt[r]]
    if len(p_) and len(n_):
        aucs.append((p_[:, None] > n_[None, :]).mean())
aucs = np.array(aucs)
P("   static bias (mean logit/column): %s" % np.round(bias, 4).tolist())
P("   residual AUC optimal-vs-not: mean %.4f  median %.4f  (0.50 = no signal)"
  % (aucs.mean(), np.median(aucs)))
P("   residual argmax distribution : %s" % sorted(Counter(int(x) for x in resid.argmax(1)).items()))
P()

Path("tmp/v18_sens_corrected.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_sens_corrected.txt")
