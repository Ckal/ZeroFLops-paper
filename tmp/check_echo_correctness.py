"""Is the model's "echo" actually CORRECT PLAY?

The echo diagnostic says the model answers column X with column X about half
the time against a one-column opponent. The label echo rate says the labels
only do that 17% of the time overall. Both can be true if the rate depends
on the POSITION:

  * if the opponent has stacked 3+ pieces in column X, then X is a
    VERTICAL THREAT and the correct move is very often X itself (block it);
  * if the opponent has stacked 1-2 pieces, there is no threat and X is
    usually a WASTED move -- a different column is correct.

So the real question is not "does the model copy?" but:
    "in the positions where it copies, is the correct answer also X?"

This measures, in exactly the positions where the echo happens:
  - what the MODEL chose;
  - what the SOLVER says is optimal (model-independent ground truth);
  - the opponent's stack height in the copied column;
  - whether the copy was CORRECT.

A ground-truth solver is used, not the labels, so this does not inherit any
label bias. The exact solver is used where it is conclusive; the canonical
strong solver otherwise; and the source is recorded so a fallback is never
silently presented as exact.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board, Outcome            # noqa: E402
from src.c4.ui_server import _load_agent           # noqa: E402
from src.c4.exact_solver import ExactSolver        # noqa: E402
from src.c4 import strong_solver as STRONG         # noqa: E402
from src.c4.mirror import canonical_board          # noqa: E402


def optimal_set(b, solver, depth=6, budget=0.05, exact_max_empty=12):
    """Ground-truth optimal columns. Returns (set, source)."""
    cb, flipped = canonical_board(b)
    free = 42 - int(sum(cb.heights))
    if free <= exact_max_empty:
        try:
            best, _v = solver.solve(cb)
        except TimeoutError:
            best = None
        if best:
            s = set(best)
            return ({6 - c for c in s} if flipped else s), "exact"
    STRONG.clear_tt()
    c = STRONG.strong_policy(cb, [], depth=depth, budget=budget)
    legal = cb.valid_actions()
    if c is None or c not in legal:
        c = legal[0] if legal else 0
    return ({6 - c for c in {c}} if flipped else {c}), "strong"


def main() -> int:
    ckpt = ROOT / "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"
    agent = _load_agent(str(ckpt), device="cpu")
    solver = ExactSolver(max_empty=12, budget=2_000_000)

    STACK_COL = 6          # the human-visible column 7 (internal 0..6)
    rows = []

    for game in range(30):
        b = Board()
        moves = []
        # the opponent (P1) always plays STACK_COL
        while True:
            legal = b.valid_actions()
            if not legal:
                break
            # --- opponent: stack column 7 ---
            if STACK_COL in legal:
                opp = STACK_COL
            else:
                opp = int(np.random.default_rng(game).choice(legal))
            b.apply_move(opp)
            moves.append(opp)
            if b.is_win() != 0:
                break
            legal = b.valid_actions()
            if not legal:
                break
            # --- model: what does it choose, and is it right? ---
            before = b.heights[STACK_COL]
            with_move = Board()
            for c in moves:
                with_move.apply_move(c)
            lg = agent.column_logits(b, moves)
            mask = agent.legal_mask(b).to(lg.device)
            NEG = torch.tensor(-1e9, dtype=lg.dtype, device=lg.device)
            p = torch.softmax(torch.where(mask, lg, NEG), dim=-1).cpu().numpy()
            choice = int(np.argmax(p))
            best, src = optimal_set(b, solver)
            rows.append({
                "game": game,
                "n_moves": len(moves),
                "opp_stack_height_before": before,
                "model_choice": choice,
                "echoed": choice == STACK_COL,
                "optimal": sorted(best),
                "optimal_src": src,
                "correct": choice in best,
                "p_choice": float(p[choice]),
                "p_stack": float(p[STACK_COL]),
            })
            b.apply_move(choice)
            moves.append(choice)
            if b.is_win() != 0:
                break

    n = len(rows)
    print("=== IS THE COPY CORRECT? (opponent stacks column 7) ===")
    print(f"  model decisions analysed: {n}")
    echo = [r for r in rows if r["echoed"]]
    print(f"  the model copied column 7: {len(echo)} ({100.0*len(echo)/max(1,n):.1f}%)")
    if echo:
        corr = sum(1 for r in echo if r["correct"])
        print(f"  of those copies, CORRECT per the solver: {corr} "
              f"({100.0*corr/max(1,len(echo)):.1f}%)")
        by_h = collections.Counter(r["opp_stack_height_before"] for r in echo)
        by_h_corr = collections.Counter(r["opp_stack_height_before"] for r in echo
                                       if r["correct"])
        print("  copies by opponent stack height (height: copies, of which correct):")
        for h in sorted(by_h):
            print(f"      height {h}: {by_h[h]:>4} copies, "
                  f"{by_h_corr.get(h,0):>4} correct "
                  f"({100.0*by_h_corr.get(h,0)/max(1,by_h[h]):.0f}%)")
    non_echo = [r for r in rows if not r["echoed"]]
    if non_echo:
        corr = sum(1 for r in non_echo if r["correct"])
        print(f"  when it did NOT copy, it was correct {corr}/"
              f"{len(non_echo)} ({100.0*corr/len(non_echo):.1f}%)")
    allc = sum(1 for r in rows if r["correct"])
    print(f"  overall correctness: {allc}/{n} ({100.0*allc/max(1,n):.1f}%)")
    print()
    srcs = collections.Counter(r["optimal_src"] for r in rows)
    print(f"  ground-truth source: {dict(srcs)}")
    print()
    # the decisive split: threat positions (height>=3) vs non-threat
    threat = [r for r in rows if r["opp_stack_height_before"] >= 3]
    nothreat = [r for r in rows if r["opp_stack_height_before"] < 3]
    for name, grp in (("THREAT (stack>=3)", threat), ("NO THREAT (stack<3)", nothreat)):
        if not grp:
            continue
        ec = sum(1 for r in grp if r["echoed"])
        cr = sum(1 for r in grp if r["correct"])
        print(f"  {name}: {len(grp)} decisions, echoed {ec} "
              f"({100.0*ec/len(grp):.0f}%), correct {cr} "
              f"({100.0*cr/len(grp):.0f}%)")
        if ec:
            cc = sum(1 for r in grp if r["echoed"] and r["correct"])
            print(f"      of the copies: {cc}/{ec} correct "
                  f"({100.0*cc/max(1,ec):.0f}%)")

    (ROOT / "runs/_probe/echo_correctness.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote runs/_probe/echo_correctness.json")
    return 0


if __name__ == "__main__":
    import torch   # noqa: E402  (imported late so the import error above is clear)
    sys.exit(main())
