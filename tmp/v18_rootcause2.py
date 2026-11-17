"""Combined root-cause probe for V18, writing every number to a report file.

Answers, in order:
  A. What is the ACTUAL move-column prior in the corpus (move positions only)?
  B. Do the teacher sidecar labels agree with the real move bytes?
  C. Are the exported checkpoints stale (identical weights)?
  D. Is the model exactly a static column-preference policy?
  E. What is the per-teacher-column cross-tab (permutation vs constant)?
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")
MOVE_OFFSET = 9
PLY_LEN = 10
COL_BYTE_OFFSET = 0x41

OUT = []


def P(s=""):
    OUT.append(str(s))
    print(s)


def corpus_rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


def move_cols(seq):
    out = []
    for i in range(MOVE_OFFSET, len(seq), PLY_LEN):
        b = seq[i]
        if COL_BYTE_OFFSET <= b < COL_BYTE_OFFSET + 7:
            out.append(b - COL_BYTE_OFFSET)
    return out


P("=" * 74)
P("A. ACTUAL move-column prior (move positions only, per split)")
P("=" * 74)
for name in ["train", "holdout"]:
    seqs = corpus_rows(SPLIT / f"{name}_corpus.txt")
    allc = np.array([c for s in seqs for c in move_cols(s)])
    cnt = np.bincount(allc, minlength=7)
    P("  %-8s rows=%6d  move-bytes=%8d" % (name, len(seqs), len(allc)))
    P("     raw   : %s" % cnt.tolist())
    P("     pct   : %s" % np.round(100.0 * cnt / len(allc), 3).tolist())
    P("     argmax: col %d @ %.3f%%" % (int(cnt.argmax()), 100.0 * cnt.max() / len(cnt)))
    T = np.load(SPLIT / f"{name}_targets.npz")["target"]
    tam = T.argmax(axis=1)
    tcnt = np.bincount(tam, minlength=7)
    P("     teacher argmax raw: %s" % tcnt.tolist())
    P("     teacher argmax pct: %s" % np.round(100.0 * tcnt / len(tam), 3).tolist())
    P("     teacher argmax mode: col %d" % int(tcnt.argmax()))
    opt = np.isclose(T, T.max(1, keepdims=True), atol=1e-6)
    P("     teacher opt-set mean=%.4f  opt col freq=%s" % (opt.sum(1).mean(), opt.sum(0).tolist()))
    P()

P("=" * 74)
P("B. TEACHER LABELS vs REAL MOVE BYTES (final ply of each row)")
P("=" * 74)
for name in ["train", "holdout"]:
    seqs = corpus_rows(SPLIT / f"{name}_corpus.txt")
    T = np.load(SPLIT / f"{name}_targets.npz")["target"]
    n = min(len(seqs), len(T))
    agree = 0
    argmax_eq = 0
    n_cmp = 0
    probs = []
    for i in range(n):
        seq = seqs[i]
        if len(seq) < PLY_LEN:
            continue
        c = seq[(len(seq) // PLY_LEN - 1) * PLY_LEN + MOVE_OFFSET] - COL_BYTE_OFFSET
        if not (0 <= c < 7):
            continue
        n_cmp += 1
        best = T[i].max()
        if np.isclose(T[i, c], best, atol=1e-6):
            agree += 1
        if int(T[i].argmax()) == c:
            argmax_eq += 1
        probs.append(T[i, c])
    probs = np.array(probs)
    P("  %-8s compared=%d" % (name, n_cmp))
    P("     teacher marks ACTUAL final move optimal : %d/%d = %.3f%%"
      % (agree, n_cmp, 100.0 * agree / max(n_cmp, 1)))
    P("     teacher argmax == ACTUAL final move      : %d/%d = %.3f%%"
      % (argmax_eq, n_cmp, 100.0 * argmax_eq / max(n_cmp, 1)))
    P("     teacher prob on actual move: mean %.5f  median %.5f  frac==0 %.4f"
      % (probs.mean(), np.median(probs), float((probs <= 1e-9).mean())))
    P()

P("=" * 74)
P("C. STALE-EXPORT CHECK (weight hashes across artifacts)")
P("=" * 74)


def fp_hash(obj):
    sd = obj.get("state", obj)
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k]
        if torch.is_tensor(t):
            h.update(k.encode())
            h.update(t.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


hashes = {}
for a in ["checkpoints/best_hard.pt", "checkpoints/best_soft.pt",
          "hard_model_final.pt", "final_float.pt"]:
    f = RUN / a
    if not f.exists():
        P("  %-28s MISSING" % a)
        continue
    o = torch.load(f, map_location="cpu", weights_only=False)
    m = o.get("meta") or {}
    hashes[a] = (fp_hash(o), m.get("step"), m.get("best_hard_acc"), m.get("best_soft_acc"))
    P("  %-28s hash=%s step=%s best_hard_acc=%s best_soft_acc=%s"
      % (a, hashes[a][0], hashes[a][1], hashes[a][2], hashes[a][3]))
P("  distinct weight hashes: %d of %d" % (len(set(v[0] for v in hashes.values())), len(hashes)))
P()

P("=" * 74)
P("D/E. MODEL vs STATIC POLICY, and CROSS-TAB vs TEACHER")
P("=" * 74)
c = np.load("runs/_probe/v18_picks_posdata_split.npz")
idx, pick, legal = c["idx"], c["pick"], c["legal"].astype(bool)
T = np.load(SPLIT / "holdout_targets.npz")["target"]
Tam = T[idx]
opt = np.isclose(Tam, Tam.max(1, keepdims=True), atol=1e-6)
tam = Tam.argmax(axis=1)
n = len(idx)
rows = np.arange(n)

P("  cross-tab teacher-argmax -> model pick:")
P("           " + "".join("%8d" % k for k in range(7)))
for tc in range(7):
    sel = tam == tc
    if sel.sum() == 0:
        continue
    bc = np.bincount(pick[sel], minlength=7)
    P("    tc c%d (%4d) %s" % (tc, sel.sum(), "".join("%8d" % x for x in bc)))
P()
mapping = {}
for tc in range(7):
    sel = tam == tc
    if sel.sum() == 0:
        continue
    bc = np.bincount(pick[sel], minlength=7)
    mapping[tc] = int(bc.argmax())
P("  modal model pick per teacher col: %s" % mapping)
P("  -> identity? %s   constant? %s"
  % (all(k == v for k, v in mapping.items()), len(set(mapping.values())) == 1))
P()

ORDER = [5, 4, 6, 3, 2, 1, 0]
static_pick = np.zeros(n, dtype=int)
for i in range(n):
    for cc in ORDER:
        if legal[i, cc]:
            static_pick[i] = cc
            break
P("  static-policy score (order c5>c4>c6>c3>c2>c1>c0) : %.4f%%"
  % (100.0 * (opt[rows, static_pick] & legal[rows, static_pick]).mean()))
P("  model score                                      : %.4f%%"
  % (100.0 * (opt[rows, pick] & legal[rows, pick]).mean()))
P("  static vs model pick agreement                  : %.3f%%"
  % (100.0 * (static_pick == pick).mean()))
P()
P("  score ladder (same 4000 holdout rows):")
for cc in range(7):
    P("    always-c%-2d : %7.4f%%" % (cc, 100.0 * (opt[:, cc] & legal[:, cc]).mean()))
P("    static-pref  : %7.4f%%" % (100.0 * (opt[rows, static_pick] & legal[rows, static_pick]).mean()))
P("    MODEL        : %7.4f%%" % (100.0 * (opt[rows, pick] & legal[rows, pick]).mean()))
tie = opt.sum(1).mean()
P("    chance floor (random among optimal) : %7.4f%%" % (100.0 * tie / 7))
P("    mean tie-set size                   : %.4f" % tie)
P()
model_hit = opt[rows, pick] & legal[rows, pick]
c3_hit = opt[:, 3] & legal[:, 3]
P("  model right where always-c3 wrong : %d rows (%.3f%%)" % ((model_hit & ~c3_hit).sum(), 100 * (model_hit & ~c3_hit).mean()))
P("  model wrong where always-c3 right : %d rows (%.3f%%)" % ((~model_hit & c3_hit).sum(), 100 * (~model_hit & c3_hit).mean()))
P()

Path("tmp/v18_rootcause_report.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n[written] tmp/v18_rootcause_report.txt")
