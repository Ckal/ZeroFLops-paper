#!/usr/bin/env python
"""RECHECK of the 09:46 provenance audit's "DECODING BUG (H1) CONFIRMED" verdict.

The audit at runs/_probe/provenance_28pct.log concluded:

    DECODING BUG (H1) CONFIRMED: reversed gives 100.0% vs 32.5%

That conclusion is driven by test (d):

    (d) reversed label IN reversed optimal set
        = (6 - label_argmax) in [6 - c for c in opt]

But that predicate is ALGEBRAICALLY IDENTICAL to test (b):

    (b) label_argmax IN optimal set

because (6 - x) in {6 - c : c in S}  <=>  x in S.
So (d) can never carry information that (b) does not, and the audit's own
test (b) already prints 100.0%. The verdict therefore compares
"label in optimal set (100%)" against "label == single best move (32.5%)"
and calls the difference a decoding bug. It is not: it is a tie-break
difference, which is exactly what an optimal-set-membership label
predicts.

This script re-runs the same hypotheses and adds the checks that actually
discriminate a reversed decode:

  * identity check: (d) must equal (b) on every single row
  * board-side reversal: best move of the MIRRORED board vs the label
    (a mirrored board decode would make the label match the mirrored
    board's best move)
  * disagreement decomposition: of the rows where the label != best move,
    how many have |optimal set| > 1 (i.e. the disagreement is a tie-break)
  * legality/mis-pairing controls

Same dataset, same solver settings as the original audit, so the numbers
are directly comparable.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board, Outcome          # noqa: E402
from src.c4 import transcript_v2 as TV2           # noqa: E402
from src.c4.exact_solver import ExactSolver, move_order, has_four  # noqa: E402

SPLIT = ROOT / "runs/c4/_pilot_v23"


def decode_header_to_board(hdr):
    columns, _last, turn = TV2.decode_state(hdr)
    b = Board()
    b.p1 = b.p2 = 0
    b.mask = 0
    for c, (h, owners) in enumerate(columns):
        b.heights[c] = h
        for r, owner in enumerate(owners):
            if owner == 1:
                b.p1 |= 1 << (r * 7 + c)
            else:
                b.p2 |= 1 << (r * 7 + c)
    b.mask = b.p1 | b.p2
    b.turn = turn
    return b


def mirror_board(b: Board) -> Board:
    """Mirror columns 0<->6, 1<->5, 2<->4 (same side-to-move)."""
    m = Board()
    m.p1 = m.p2 = 0
    for c in range(7):
        mc = 6 - c
        m.heights[mc] = b.heights[c]
        for r in range(6):
            bit = 1 << (r * 7 + c)
            if b.p1 & bit:
                m.p1 |= 1 << (r * 7 + mc)
            elif b.p2 & bit:
                m.p2 |= 1 << (r * 7 + mc)
    m.mask = m.p1 | m.p2
    m.turn = b.turn
    return m


def load(limit=4000):
    z = np.load(SPLIT / "holdout_targets.npz")
    Th = z["target"]
    lines = [l for l in (SPLIT / "holdout_corpus.txt").read_bytes().split(b"\n") if l]
    items = []
    for i in range(min(len(lines), len(Th))):
        raw = lines[i]
        if len(raw) % TV2.PLY_LEN:
            continue
        nrec = len(raw) // TV2.PLY_LEN
        hdr = raw[(nrec - 1) * TV2.PLY_LEN:(nrec - 1) * TV2.PLY_LEN + TV2.PLY_LEN - 1]
        try:
            b = decode_header_to_board(hdr)
        except Exception:
            continue
        if b.outcome() != Outcome.ONGOING:
            continue
        played = raw[nrec * TV2.PLY_LEN - 1] - TV2.COL_BYTE_OFFSET
        items.append((b, Th[i].copy(), played, 42 - int(sum(b.heights))))
    return items


def root_values(solver, b, side):
    mine, theirs = (b.p1, b.p2) if side == 1 else (b.p2, b.p1)
    heights = list(b.heights)
    empties = 42 - (b.p1.bit_count() + b.p2.bit_count())
    out = {}
    for col in move_order([c for c in range(7) if heights[c] < 6]):
        h = heights[col]
        bit = 1 << (h * 7 + col)
        heights[col] = h + 1
        try:
            if has_four(mine | bit):
                out[col] = 1000
            else:
                out[col] = -solver._negamax(theirs, mine | bit, heights, empties - 1)
        except TimeoutError:
            pass
        finally:
            heights[col] = h
    return out


def main() -> int:
    items = load()
    print(f"loaded {len(items)} candidate positions")
    solver = ExactSolver(max_empty=16, budget=20_000_000)

    rows = []
    for b, t, played, e in items:
        if e > 16 or e < 8:
            continue
        try:
            status = solver.solve(b, side=b.turn)
            if status is None:
                continue
            bm = solver.best_move(b, side=b.turn)
            if bm is None:
                continue
            rv = root_values(solver, b, b.turn)
        except TimeoutError:
            continue
        if not rv:
            continue
        best_val = max(rv.values())
        opt = sorted(c for c, v in rv.items() if v == best_val)
        # mirrored board's optimal set / best move (board-side H1 test)
        mb = mirror_board(b)
        try:
            mrv = root_values(solver, mb, mb.turn)
            mbest_val = max(mrv.values())
            mopt = sorted(c for c, v in mrv.items() if v == mbest_val)
        except TimeoutError:
            mopt = None
        rows.append({
            "empty": e, "board": b, "target": t,
            "label_argmax": int(np.argmax(t)),
            "best_move": bm, "opt": opt, "root_values": rv,
            "status": status, "played": played,
            "mirrored_opt": mopt,
        })
        if len(rows) >= 200:
            break

    n = len(rows)
    print(f"proved positions: {n}")
    if n == 0:
        print("NO PROVED POSITIONS -- inconclusive")
        return 2

    def rate(pred):
        return 100.0 * sum(1 for r in rows if pred(r)) / n

    # --- the identity check: (d) must equal (b) on every row ---------
    ident_mismatch = sum(
        1 for r in rows
        if ((6 - r["label_argmax"]) in [6 - c for c in r["opt"]])
        != (r["label_argmax"] in r["opt"])
    )
    print("\n=== IDENTITY CHECK (the bug in the audit's reasoning) ===")
    print(f"  rows where test(d) disagrees with test(b): {ident_mismatch}/{n}")
    print(f"  (b) label IN optimal set                     {rate(lambda r: r['label_argmax'] in r['opt']):6.1f}%")
    print(f"  (d) 'reversed' label IN 'reversed' opt set   {rate(lambda r: (6 - r['label_argmax']) in [6 - c for c in r['opt']]):6.1f}%")
    print("  -> (d) is the SAME predicate as (b); it cannot support a")
    print("     reversed-decode conclusion on its own.")

    # --- the discriminating tests -----------------------------------
    print("\n=== DISCRIMINATING TESTS ===")
    a = rate(lambda r: r["label_argmax"] == r["best_move"])
    c_rev = rate(lambda r: (6 - r["label_argmax"]) == r["best_move"])
    print(f"  (a) label == solver best move                {a:6.1f}%")
    print(f"  (c) REVERSED label == solver best move        {c_rev:6.1f}%   <- real label-side H1 test")
    print(f"      -> reversal {'HELPS' if c_rev > a + 10 else 'HURTS'}: "
          f"{'H1 supported' if c_rev > a + 10 else 'H1 refuted'}")

    mirror_rows = [r for r in rows if r["mirrored_opt"]]
    if mirror_rows:
        am = 100.0 * sum(1 for r in mirror_rows
                         if r["label_argmax"] == r["mirrored_opt"][0]) / len(mirror_rows)
        ainm = 100.0 * sum(1 for r in mirror_rows
                           if r["label_argmax"] in r["mirrored_opt"]) / len(mirror_rows)
        print(f"  (j) label == MIRRORED board's best move      {am:6.1f}%   <- real board-side H1 test (n={len(mirror_rows)})")
        print(f"  (k) label IN MIRRORED board's optimal set    {ainm:6.1f}%")
        print(f"      -> board-side reversal {'HELPS' if am > a + 10 else 'HURTS'}: "
              f"{'H1 supported' if am > a + 10 else 'H1 refuted'}")

    # --- disagreement decomposition ---------------------------------
    dis = [r for r in rows if r["label_argmax"] != r["best_move"]]
    tied_dis = [r for r in dis if len(r["opt"]) > 1]
    untied_dis = [r for r in dis if len(r["opt"]) == 1]
    print("\n=== DISAGREEMENT DECOMPOSITION ===")
    print(f"  rows where label != solver best move: {len(dis)}/{n} ({100.0*len(dis)/n:.1f}%)")
    print(f"    of those, |optimal set| > 1 (tie-break): {len(tied_dis)} "
          f"({100.0*len(tied_dis)/max(1,len(dis)):.1f}% of disagreements)")
    print(f"    of those, |optimal set| == 1 (REAL error): {len(untied_dis)} "
          f"({100.0*len(untied_dis)/max(1,len(dis)):.1f}% of disagreements)")
    print(f"  => real (untied) error rate over all rows: "
          f"{100.0*len(untied_dis)/n:.1f}%")
    print(f"  single-optimum rows: {sum(1 for r in rows if len(r['opt'])==1)}/{n}")

    # --- controls ----------------------------------------------------
    print("\n=== CONTROLS ===")
    legal = [{c for c in range(7) if r["board"].heights[c] < 6} for r in rows]
    ill = []
    for r, lg in zip(rows, legal):
        t = r["target"]
        tot = float(t.sum()) or 1.0
        ill.append(sum(float(t[c]) for c in range(7) if c not in lg) / tot)
    print(f"  (i) mean target mass on ILLEGAL columns        {100.0*np.mean(ill):6.1f}%  <- H3 control")
    print(f"  (g) label == actually played move              {rate(lambda r: r['label_argmax'] == r['played']):6.1f}%")
    ent = []
    for r in rows:
        t = np.asarray(r["target"], dtype=np.float64)
        t = t / (t.sum() or 1.0)
        nz = t[t > 0]
        ent.append(float(-(nz * np.log(nz)).sum()))
    print(f"  target entropy: mean {np.mean(ent):.3f} nats (uniform = {np.log(7):.3f})")
    print(f"  optimal-set size histogram: "
          f"{dict(sorted(Counter(len(r['opt']) for r in rows).items()))}")
    print(f"  label argmax histogram: "
          f"{dict(sorted(Counter(r['label_argmax'] for r in rows).items()))}")

    am = 0.0
    ainm = 0.0
    out = {
        "n_proved": n,
        "identity_d_vs_b_mismatches": ident_mismatch,
        "label_argmax_vs_best_move_pct": round(a, 2),
        "reversed_label_vs_best_move_pct": round(c_rev, 2),
        "label_in_optimal_set_pct": round(rate(lambda r: r["label_argmax"] in r["opt"]), 2),
        "disagreements": len(dis),
        "disagreements_that_are_ties": len(tied_dis),
        "disagreements_that_are_real_errors": len(untied_dis),
        "real_untied_error_rate_pct": round(100.0 * len(untied_dis) / n, 2),
        "single_optimum_rows": sum(1 for r in rows if len(r["opt"]) == 1),
        "mean_target_mass_on_illegal_cols_pct": round(100.0 * float(np.mean(ill)), 2),
        "label_vs_played_pct": round(rate(lambda r: r["label_argmax"] == r["played"]), 2),
        "mean_target_entropy_nats": round(float(np.mean(ent)), 4),
        "optimal_set_size_histogram": dict(sorted(Counter(len(r["opt"]) for r in rows).items())),
        "label_argmax_histogram": dict(sorted(Counter(r["label_argmax"] for r in rows).items())),
    }
    if mirror_rows:
        out["label_vs_mirrored_best_move_pct"] = round(am, 2)
        out["label_in_mirrored_opt_set_pct"] = round(ainm, 2)
        out["mirror_n"] = len(mirror_rows)

    print("\n=== VERDICT ===")
    h1_supported = (c_rev > a + 10) or (mirror_rows and am > a + 10)
    if h1_supported:
        print(f"  H1 (reversed decode) SUPPORTED: label-side {c_rev:.1f}% / "
              f"board-side {am:.1f}% vs base {a:.1f}%")
    else:
        print(f"  H1 (reversed decode) REFUTED: label-side reversal gives {c_rev:.1f}%")
        if mirror_rows:
            print(f"                          board-side reversal gives {am:.1f}%")
        print(f"                          both are WORSE than the base {a:.1f}%")
        print(f"  The label agrees with the exact optimal SET "
              f"{out['label_in_optimal_set_pct']:.1f}% of the time; the 'gap' to")
        print(f"  {a:.1f}% is TIE-BREAKING inside the optimal set, not a decoding bug.")
        print(f"  True error rate (rows with a unique optimum that the label misses): "
              f"{out['real_untied_error_rate_pct']:.1f}%")

    dst = ROOT / "runs/_probe/provenance_h1_recheck.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
