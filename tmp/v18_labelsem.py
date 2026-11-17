"""Label-sidecar semantics + why holdout_optimal_pct froze at 30.7%.

1. What exactly does holdout_targets.npz store per row? (shape, value range,
   whether values are probabilities per column or a single target column)
2. Is the teacher-optimal set a function of the BOARD or of a FIXED column
   prior? (i.e. is the label itself board-blind?)
3. What is the theoretical ceiling of a constant-column policy on this
   holdout, and what is the teacher-optimal ceiling?
4. Why does 'holdout_optimal_pct' sit at 30.7% while edge_optimal_pct is
   ~48.8%? (tie-counting semantics)
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT = Path("runs/c4/posdata_split")
SPLIT23 = Path("runs/c4/posdata_v23")


def rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


out = []


def P(s=""):
    out.append(str(s))
    print(s)


P("=" * 74)
P("1. SIDECAR STRUCTURE")
P("=" * 74)
for name in ("train_targets.npz", "holdout_targets.npz"):
    z = np.load(SPLIT / name)
    P("%s keys: %s" % (name, list(z.keys())))
    for k in z.keys():
        a = z[k]
        P("   %-18s shape=%-14s dtype=%-10s min=%s max=%s"
          % (k, str(a.shape), str(a.dtype),
             np.round(float(a.min()), 6), np.round(float(a.max()), 6)))
P()

# The trainer's target: (N,7) soft weights -> hard target column = argmax
tr = np.load(SPLIT / "train_targets.npz")["target"]
ho = np.load(SPLIT / "holdout_targets.npz")["target"]
P("=" * 74)
P("2. TARGET SEMANTICS")
P("=" * 74)
P("train target shape %s ; holdout target shape %s" % (tr.shape, ho.shape))
rs = tr.sum(1)
P("train row sums: min %.6f max %.6f mean %.6f" % (rs.min(), rs.max(), rs.mean()))
hs = ho.sum(1)
P("holdout row sums: min %.6f max %.6f mean %.6f" % (hs.min(), hs.max(), hs.mean()))
if rs.min() > 0.99 and rs.max() < 1.01:
    P("-> each row is a DISTRIBUTION over the 7 columns (soft teacher)")
else:
    P("-> rows are NOT a normalized distribution")
P()

arg_tr = tr.argmax(1)
arg_ho = ho.argmax(1)
P("train argmax column distribution : %s"
  % sorted(Counter(int(x) for x in arg_tr).items()))
P("holdout argmax column distribution: %s"
  % sorted(Counter(int(x) for x in arg_ho).items()))
P()

# optimal set = columns within tolerance of the max
def opt_sets(t, tol=1e-6):
    mx = t.max(1, keepdims=True)
    return np.isclose(t, mx, atol=tol, rtol=0)

otr = opt_sets(tr)
oho = opt_sets(ho)
P("=" * 74)
P("3. OPTIMAL-SET SIZE (teacher tie-set)")
P("=" * 74)
P("train   mean optimal-set size : %.4f" % otr.sum(1).mean())
P("holdout mean optimal-set size : %.4f" % oho.sum(1).mean())
P("holdout size histogram        : %s"
  % sorted(Counter(oho.sum(1).astype(int)).items()))
P()

P("=" * 74)
P("4. CEILINGS ON THE HOLDOUT")
P("=" * 74)
# best constant column by argmax agreement
const = {}
for c in range(7):
    const[c] = 100.0 * oho[:, c].mean()
P("constant-column accuracy (counting any member of the optimal set):")
for c in range(7):
    P("   col %d : %6.3f%%" % (c, const[c]))
P("   best constant col: %d at %.3f%%"
  % (max(const, key=const.get), max(const.values())))
P()

# ceiling: fraction of rows where SOME column is optimal = 100% by construction
P("teacher-optimal ceiling : 100.000% (a perfect teacher scores 100%)")
P()

P("=" * 74)
P("5. IS THE TEACHER BOARD-DEPENDENT?")
P("=" * 74)
# Compare the teacher label against a pure board-independent prior:
# if the teacher labels were random w.r.t. the board, the holdout argmax
# would be independent of board features. Proxy: check the teacher's
# accuracy at predicting ITS OWN argmax on the holdout = 100 by definition,
# so instead test whether the label agrees with the TRAIN distribution.
P("train   argmax dist: %s" % sorted(Counter(int(x) for x in arg_tr).items()))
P("holdout argmax dist: %s" % sorted(Counter(int(x) for x in arg_ho).items()))
# chi-square-ish: max abs deviation of holdout from train prior
train_prior = np.bincount(arg_tr, minlength=7) / len(arg_tr)
hold_prior = np.bincount(arg_ho, minlength=7) / len(arg_ho)
P()
P("train prior per column  : %s" % np.round(train_prior, 5).tolist())
P("holdout prior per column: %s" % np.round(hold_prior, 5).tolist())
P("max abs deviation       : %.5f" % np.abs(train_prior - hold_prior).max())
P("-> if ~0, the holdout teacher prior equals the train teacher prior")
P("   (consistent with a FIXED column prior teacher, not a board-conditional one)")
P()

P("=" * 74)
P("6. SIDE-BY-SIDE: posdata_split vs posdata_v23")
P("=" * 74)
h23 = np.load(SPLIT23 / "holdout_targets.npz")["target"]
o23 = opt_sets(h23)
agree = (o23 & oho).sum(1) / np.maximum((o23 | oho).sum(1), 1)
P("optimal-set Jaccard agreement: mean %.4f" % agree.mean())
P("identical optimal sets        : %.4f%%" % (100.0 * ((o23 == oho).all(1)).mean()))
P("mean optimal-set size v23     : %.4f" % o23.sum(1).mean())
P("mean optimal-set size split   : %.4f" % oho.sum(1).mean())
P()

P("=" * 74)
P("7. EDGE vs HOLDOUT SEMANTICS (why 30.7% vs 48.8%)")
P("=" * 74)
P("holdout_optimal_pct : fraction of holdout positions where the model's")
P("                      chosen column is a member of the teacher's optimal set")
P("edge_optimal_pct    : same question restricted to positions where the")
P("                      teacher has a UNIQUE optimum (ties broken by the")
P("                      teacher's own argmax / edge definition)")
uniq = oho.sum(1) == 1
P("unique-optimum holdout rows: %d / %d (%.2f%%)"
  % (uniq.sum(), len(uniq), 100.0 * uniq.mean()))
P()

Path("tmp/v18_label_semantics.txt").write_text("\n".join(out), encoding="utf-8")
print("\n[written] tmp/v18_label_semantics.txt")
