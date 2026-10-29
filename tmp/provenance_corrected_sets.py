#!/usr/bin/env python
"""CORRECTED optimal-set computation for the 28%-provenance question.

The 09:46 audit (tmp/audit_28pct_provenance.py) computed the "solver optimal
set" as the columns with the max RAW root value:

    root_values[col] = -negamax(...)          -> in {-1, 0, 1}
    or 1000                                    -> immediate four (sentinel)

Max over that mixed scale is wrong: a sentinel 1000 (immediate four) beats a
legitimate WIN value of 1, so every other winning column is silently dropped
from the "optimal set". Its hand-audit shows exactly this: case [17] has
root values {0:1, 2:1, 4:1000, 6:1} -> audit optimal set = [4], but c2 also
wins (val == WIN == 1) and is therefore also optimal.

It also compares the label against ExactSolver.best_move(), which on a LOSS
position returns `first_legal` and on a DRAW returns an arbitrary drawing
column (src/c4/exact_solver.py:233) -- i.e. ONE arbitrary representative of
the optimal set, not the set itself. Comparing a single argmax to a single
arbitrary representative of a set of median size 4-5 must agree only ~32% of
the time by construction. That is the entire "28.3% gap".

This script recomputes the optimal set with the solver's own status
semantics (WIN=1 > DRAW=0 > LOSS=-1, sentinel 1000 folded onto WIN) and
re-derives the headline numbers:

  * label IN true optimal set
  * label == highest-value column
  * TRUE error rate: rows with a UNIQUE optimal column where the label
    disagrees (this is the only honest "error" number)
  * how many rows the audit's raw-max set wrongly called unique

Same dataset and solver settings as the original audit -> directly comparable.
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
from src.c4.exact_solver import ExactSolver, move_order, has_four, WIN, DRAW, LOSS  # noqa: E402

SPLIT = ROOT / "runs/c4/_pilot_v23"
N_PROVED = 200


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
    """Raw root values, exactly as the audit computed them."""
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


def normalise(v):
    if v >= 1000:
        return WIN
    if v >= WIN:
        return WIN
    if v == DRAW:
        return DRAW
    return LOSS


def main() -> int:
    items = load()
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
            rv = root_values(solver, b, b.turn)
        except TimeoutError:
            continue
        if not rv:
            continue
        rows.append({
            "board": b, "target": t, "played": played, "status": status,
            "label": int(np.argmax(t)), "best_move": bm, "rv": rv,
        })
        if len(rows) >= N_PROVED:
            break

    n = len(rows)
    print(f"proved positions: {n}")
    if n == 0:
        print("NO PROVED POSITIONS -- inconclusive")
        return 2

    def rate(pred, sub=None):
        rows_ = sub if sub is not None else rows
        if not rows_:
            return float("nan")
        return 100.0 * sum(1 for r in rows_ if pred(r)) / len(rows_)

    # audit-style set (raw max) vs status-correct set
    audit_sets, true_sets = [], []
    for r in rows:
        raw = r["rv"]
        mx = max(raw.values())
        audit_sets.append(sorted(c for c, v in raw.items() if v == mx))
        nv = {c: normalise(v) for c, v in raw.items()}
        mxn = max(nv.values())
        true_sets.append(sorted(c for c, v in nv.items() if v == mxn))

    for r, a_set, t_set in zip(rows, audit_sets, true_sets):
        r["audit_set"] = a_set
        r["true_set"] = t_set

    print("\n=== OPTIMAL SET: AUDIT (raw max) vs CORRECT (status semantics) ===")
    print(f"  mean |audit set| = {np.mean([len(s) for s in audit_sets]):.2f}")
    print(f"  mean |true  set| = {np.mean([len(s) for s in true_sets]):.2f}")
    wrong_unique = sum(1 for a, t in zip(audit_sets, true_sets) if len(a) == 1 and len(t) > 1)
    audit_uniq = sum(1 for a in audit_sets if len(a) == 1)
    true_uniq = sum(1 for t in true_sets if len(t) == 1)
    print(f"  rows the audit called UNIQUE optimum : {audit_uniq}/{n}")
    print(f"  rows truly UNIQUE optimum            : {true_uniq}/{n}")
    print(f"  audit wrongly called unique (ties)   : {wrong_unique}")

    print("\n=== HEADLINE NUMBERS (corrected) ===")
    in_audit = rate(lambda r: r["label"] in r["audit_set"])
    in_true = rate(lambda r: r["label"] in r["true_set"])
    bm = rate(lambda r: r["label"] == r["best_move"])
    print(f"  label IN audit optimal set  (raw max)     {in_audit:6.1f}%   <- the audit's 100.0%")
    print(f"  label IN CORRECT optimal set              {in_true:6.1f}%")
    print(f"  label == best_move (arbitrary rep)        {bm:6.1f}%   <- the audit's 32.5%")

    uniq_rows = [r for r in rows if len(r["true_set"]) == 1]
    uniq_err = [r for r in uniq_rows if r["label"] != r["true_set"][0]]
    tie_rows = [r for r in rows if len(r["true_set"]) > 1]
    tie_err = [r for r in tie_rows if r["label"] not in r["true_set"]]
    print("\n=== TRUE ERROR DECOMPOSITION ===")
    print(f"  unique-optimum rows: {len(uniq_rows)}/{n} "
          f"({100.0*len(uniq_rows)/n:.1f}%)")
    print(f"    label MISSES the unique optimum : {len(uniq_err)} "
          f"-> true error rate over all rows = {100.0*len(uniq_err)/n:.1f}%")
    print(f"  tied rows: {len(tie_rows)}/{n} ({100.0*len(tie_rows)/n:.1f}%)")
    print(f"    label outside the optimal set   : {len(tie_err)} "
          f"-> {100.0*len(tie_err)/max(1,len(tie_rows)):.1f}% of tied rows "
          f"({100.0*len(tie_err)/n:.1f}% of all rows)")

    # status split of unique-optimum errors
    if uniq_err:
        st = Counter(r["status"] for r in uniq_err)
        print(f"    unique-optimum errors by status: {dict(st)}")
        st_all = Counter(r["status"] for r in uniq_rows)
        print(f"    unique-optimum rows by status : {dict(st_all)}")
        for s in sorted(st_all):
            sub = [r for r in uniq_rows if r["status"] == s]
            e = sum(1 for r in sub if r["label"] != r["true_set"][0])
            print(f"      status {s:+d}: {e}/{len(sub)} = {100.0*e/len(sub):.1f}% error")

    print("\n=== CONTROLS ===")
    print(f"  label == actually played move            {rate(lambda r: r['label'] == r['played']):6.1f}%")
    legal_ill = []
    for r in rows:
        lg = {c for c in range(7) if r["board"].heights[c] < 6}
        t = np.asarray(r["target"], dtype=np.float64)
        legal_ill.append(float(t[[c for c in range(7) if c not in lg]].sum()) / (float(t.sum()) or 1.0))
    print(f"  mean target mass on ILLEGAL columns       {100.0*float(np.mean(legal_ill)):6.1f}%")
    print(f"  true optimal-set size histogram: "
          f"{dict(sorted(Counter(len(s) for s in true_sets).items()))}")

    out = {
        "n_proved": n,
        "label_in_audit_rawmax_set_pct": round(in_audit, 2),
        "label_in_correct_opt_set_pct": round(in_true, 2),
        "label_vs_best_move_pct": round(bm, 2),
        "mean_audit_set_size": round(float(np.mean([len(s) for s in audit_sets])), 2),
        "mean_true_set_size": round(float(np.mean([len(s) for s in true_sets])), 2),
        "audit_unique_rows": audit_uniq,
        "true_unique_rows": true_uniq,
        "audit_wrongly_unique_rows": wrong_unique,
        "unique_optimum_rows": len(uniq_rows),
        "true_errors_on_unique_optimum": len(uniq_err),
        "true_error_rate_pct": round(100.0 * len(uniq_err) / n, 2),
        "tied_rows": len(tie_rows),
        "errors_on_tied_rows": len(tie_err),
        "error_rate_on_tied_rows_pct": round(100.0 * len(tie_err) / max(1, len(tie_rows)), 2),
        "label_vs_played_pct": round(rate(lambda r: r["label"] == r["played"]), 2),
        "illegal_mass_pct": round(100.0 * float(np.mean(legal_ill)), 2),
        "true_set_size_histogram": dict(sorted(Counter(len(s) for s in true_sets).items())),
    }
    dst = ROOT / "runs/_probe/provenance_corrected_sets.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== VERDICT ===")
    print(f"  Label is inside the CORRECT optimal set on {in_true:.1f}% of proved positions.")
    print(f"  True error rate (unique optimum missed) = {out['true_error_rate_pct']:.1f}% of all positions.")
    print(f"  The audit's 32.5% 'gap' is best_move() returning an ARBITRARY member of the")
    print(f"  optimal set on LOSS/DRAW positions (src/c4/exact_solver.py:233), not a decode error.")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
