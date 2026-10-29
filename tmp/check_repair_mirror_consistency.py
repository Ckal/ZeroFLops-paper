"""Mirror consistency of the REPAIRED sidecars.

The mirror corpus is the interleaved file: even row = original game,
odd row = its mirror. So the sidecar rows 2i and 2i+1 must describe the
same game reflected, and their labels must be reflections of each other:
    target[2i+1, s]  ==  mirror(target[2i, s])

For a soft-label target over 7 columns, mirroring means column c -> 6-c.
So target[2i+1] should equal target[2i] reversed along the column axis.

Check three things:
  1. REPAIRED original vs REPAIRED mirror -- does the repaired pair agree?
  2. ORIGINAL original vs ORIGINAL mirror -- the baseline (known: 0
     mismatches over 5,100,175 target checks at build time).
  3. ORIGINAL original vs REPAIRED mirror -- what the mixed pair does.

Also: does the repair introduce any NEW asymmetry? A repaired label that
does not have a mirrored twin in the other row would be a new defect.

The same check on the ply_of axis: the slot layout must be mirrored too.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

ORIG = ROOT / "runs/_probe/perply_v23_full/merged.npz"
REP = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"
MORIG = ROOT / "runs/_probe/perply_v23_full/merged_mirror.npz"
MREP = ROOT / "runs/_probe/perply_v23_full/merged_mirror_repaired.npz"


def mirror_rows(idx):
    """row 2i -> original, 2i+1 -> mirror."""
    return idx % 2 == 0, idx // 2


def compare(a_path, b_path, label_a, label_b, limit=None):
    """Compare sidecar a (original rows) with sidecar b (mirror rows).

    ROW PAIRING: the mirror sidecar is the INTERLEAVED file:
        row 2i   = original game i
        row 2i+1 = mirrored twin of game i
    The plain original sidecar has ONE row per game, so original game i
    must be compared against mirror-sidecar rows 2i and 2i+1 -- NOT
    against row 2i of the plain sidecar (that is a DIFFERENT game).
    """
    A = np.load(a_path)
    B = np.load(b_path)
    ta, tb = A["target"], B["target"]
    pa, pb = A["ply_of"], B["ply_of"]

    n = ta.shape[0]
    if limit:
        n = min(n, limit)

    checked = 0
    matched = 0
    mismatched = 0
    only_in_a = 0        # A has a label, B does not
    only_in_b = 0        # B has a label, A does not
    both = 0
    examples = []

    for i in range(n):
        ia = int(i)
        ib = 2 * i + 1                      # the MIRROR row in the interleaved file
        if ib >= tb.shape[0]:
            break
        for s in range(ta.shape[1]):
            la = pa[ia, s] >= 0
            lb = pb[ib, s] >= 0
            if la and lb:
                both += 1
                checked += 1
                # mirror the A target and compare to B
                ma = ta[ia, s][::-1]
                mb = tb[ib, s]
                if np.abs(ma - mb).max() < 1e-6:
                    matched += 1
                else:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append({"orig_row": ia, "mirror_row": ib,
                                         "slot": s,
                                         "orig": [round(float(x), 3) for x in ta[ia, s]],
                                         "orig_mirrored": [round(float(x), 3) for x in ma],
                                         "mirror": [round(float(x), 3) for x in mb]})
            elif la and not lb:
                only_in_a += 1
            elif lb and not la:
                only_in_b += 1

    return {
        "pair": f"{label_a} vs {label_b}",
        "rows_compared": n,
        "both_labelled": both,
        "matched": matched,
        "mismatched": mismatched,
        "only_in_" + label_a: only_in_a,
        "only_in_" + label_b: only_in_b,
        "match_pct": round(100.0 * matched / max(1, both), 4),
        "examples": examples,
    }


def main() -> int:
    out = {}
    print("=== 1. ORIGINAL original vs ORIGINAL mirror (baseline) ===")
    r = compare(ORIG, MORIG, "orig", "mirror")
    print(f"  both labelled : {r['both_labelled']:,}")
    print(f"  matched       : {r['matched']:,} ({r['match_pct']}%)")
    print(f"  mismatched    : {r['mismatched']:,}")
    print(f"  only in orig  : {r['only_in_orig']:,}")
    print(f"  only in mirror: {r['only_in_mirror']:,}")
    out["baseline_original"] = r

    print("\n=== 2. REPAIRED original vs REPAIRED mirror ===")
    r2 = compare(REP, MREP, "repaired", "mirror")
    print(f"  both labelled : {r2['both_labelled']:,}")
    print(f"  matched       : {r2['matched']:,} ({r2['match_pct']}%)")
    print(f"  mismatched    : {r2['mismatched']:,}")
    print(f"  only in orig  : {r2['only_in_repaired']:,}")
    print(f"  only in mirror: {r2['only_in_mirror']:,}")
    out["repaired_pair"] = r2

    print("\n=== 3. ORIGINAL original vs REPAIRED mirror (mixed) ===")
    r3 = compare(ORIG, MREP, "orig", "mirror")
    print(f"  both labelled : {r3['both_labelled']:,}")
    print(f"  matched       : {r3['matched']:,} ({r3['match_pct']}%)")
    print(f"  mismatched    : {r3['mismatched']:,}")
    print(f"  only in orig  : {r3['only_in_orig']:,}")
    print(f"  only in mirror: {r3['only_in_mirror']:,}")
    out["mixed_pair"] = r3

    print("\n=== VERDICT ===")
    # The repair filled holes. A filled hole in the ORIGINAL that has no
    # mirror twin filled means the repair is not mirror-symmetric -- a new
    # defect. A filled hole in BOTH is fine and expected.
    new_defect = r2["only_in_repaired"] - r["only_in_orig"]
    print(f"  new asymmetric labels introduced by the repair: {new_defect:,}")
    if r2["mismatched"] > r["mismatched"]:
        print(f"  !! mismatches grew {r['mismatched']} -> {r2['mismatched']}")
    else:
        print(f"  mismatches did not grow ({r['mismatched']} -> {r2['mismatched']})")
    out["new_asymmetric_labels"] = new_defect
    out["VERDICT"] = ("REPAIR_MIRROR_SOUND"
                      if new_defect <= 0 and r2["mismatched"] <= r["mismatched"]
                      else "REPAIR_MIRROR_PROBLEM")

    (ROOT / "runs/_probe/repair_mirror_consistency.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nVERDICT: {out['VERDICT']}")
    return 0 if out["VERDICT"] == "REPAIR_MIRROR_SOUND" else 1


if __name__ == "__main__":
    sys.exit(main())
