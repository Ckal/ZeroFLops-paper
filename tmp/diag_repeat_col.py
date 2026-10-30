"""Diagnose the user's "repeated column 7" complaint.

Three separate questions, measured separately:
  1. Does the model actually repeat a column when a good move exists?
  2. When it does, is the repeated column a TACTICAL choice (a win/block
     it genuinely had) or a blind centre/edge bias?
  3. Is the repeated column the argmax of the HARD path at the decision
     position -- i.e. did the model play what it thought?

Uses the SAME canonical hard path as the live server (Agent.choose with
use_hard=True) and the SAME exact-solver for ground truth.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

sys.path.insert(0, ".")

import copy

import torch

from src.c4.board import Board, Outcome, P1
from src.c4.exact_solver import ExactSolver
from src.c4.ui_server import _load_agent


def hard_row(agent: Agent, board: Board, moves: list[int]):
    """The masked hard-column distribution the model actually plays from."""
    import torch
    legal = torch.tensor([c < 6 for c in board.heights], dtype=torch.bool)
    lg = agent.hard_column_logits(board, list(moves))
    neg = torch.tensor(-1e9, dtype=lg.dtype)
    masked = torch.where(legal, lg, neg)
    probs = torch.softmax(masked, dim=-1)
    return probs.detach().numpy().tolist(), [bool(x) for x in legal.tolist()]


def solver_optimal_cols(solver, board: Board, side: int):
    """Exact-solver optimal columns, or None when unproved."""
    status = solver.solve(board, side)
    if status is None:
        return None
    vals = {}
    for c in list(board.valid_actions()):
        b2 = copy.deepcopy(board)
        b2.apply_move(c)
        if b2.outcome() != Outcome.ONGOING:
            vals[c] = 1
            continue
        v2 = solver.solve(b2, 3 - side)
        if v2 is None:
            continue
        vals[c] = -v2
    if not vals:
        return []
    mx = max(vals.values())
    return [c for c, v in vals.items() if v == mx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--max-empty", type=int, default=14)
    ap.add_argument("--out", default="runs/_probe/repeat_col_diag.json")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    agent = _load_agent(args.model, device="cpu")
    solver = ExactSolver(max_empty=args.max_empty, budget=2_000_000)

    col_counts = collections.Counter()
    repeats = []          # one record per position where the model repeated
    dec_mismatch = 0      # model played != hard argmax at the decision spot
    total_model_moves = 0
    solver_checked = 0
    repeat_optimal = 0
    repeat_not_optimal = 0

    for g in range(args.games):
        board = Board()
        moves: list[int] = []
        last_model_col = None
        while board.outcome() == Outcome.ONGOING and len(moves) < 42:
            if board.turn == P1:
                # The opponent plays a random legal move (a weak but honest
                # probe; the point is the MODEL's behaviour, not its strength).
                legal = board.valid_actions()
                c = rng.choice(legal)
                board.apply_move(c)
                moves.append(c)
                continue

            # Model's turn.
            probs, legal = hard_row(agent, board, moves)
            legal_cols = [c for c in range(7) if legal[c]]
            best = max(legal_cols, key=lambda c: probs[c])
            # What the live path actually does.
            played = agent.choose(board, list(moves), use_hard=True)
            total_model_moves += 1
            if played != best:
                dec_mismatch += 1

            col_counts[played] += 1

            if last_model_col is not None and played == last_model_col:
                rec = {
                    "game": g,
                    "col": played,
                    "prob": round(probs[played], 4),
                    "second_best_prob": round(max(probs[c] for c in legal_cols if c != played), 4),
                    "moves": list(moves),
                }
                # Was the repeat the exact-solver-optimal move?
                opt = solver_optimal_cols(solver, board, P1)
                solver_checked += 1
                if opt is None:
                    rec["solver"] = "unproved"
                else:
                    rec["solver_optimal"] = played in opt
                    rec["n_optimal"] = len(opt)
                    rec["optimal_cols"] = opt
                    if played in opt:
                        repeat_optimal += 1
                    else:
                        repeat_not_optimal += 1
                repeats.append(rec)

            board.apply_move(played)
            moves.append(played)
            last_model_col = played

    total = max(1, total_model_moves)
    print(f"games: {args.games}   model moves: {total_model_moves}")
    print(f"played != hard argmax: {dec_mismatch}  "
          f"({100.0*dec_mismatch/total:.2f}%)")
    print("\ncolumn share of the model's moves:")
    for c in range(7):
        n = col_counts[c]
        print(f"  {'ABCDEFG'[c]} (c{c}): {n:5d}  {100.0*n/total:5.2f}%")
    print(f"\nrepeats (same column as the previous model move): {len(repeats)}")
    print(f"  exact-solver optimal:     {repeat_optimal}")
    print(f"  not solver-optimal:       {repeat_not_optimal}")

    # How often was the repeat justified by a strong margin?
    strong = [r for r in repeats if r["prob"] > 0.5]
    print(f"  hard prob > 50%:          {len(strong)}")
    nonopt = [r for r in repeats if r.get("solver_optimal") is False]
    if nonopt:
        avg_p = sum(r["prob"] for r in nonopt) / len(nonopt)
        print(f"  non-optimal repeats mean hard prob: {avg_p:.3f}")

    out = {
        "model": args.model,
        "games": args.games,
        "model_moves": total_model_moves,
        "dec_mismatch": dec_mismatch,
        "col_share": {str(c): col_counts[c] for c in range(7)},
        "col_share_pct": {str(c): round(100.0 * col_counts[c] / total, 3) for c in range(7)},
        "repeats": len(repeats),
        "repeats_optimal": repeat_optimal,
        "repeats_not_optimal": repeat_not_optimal,
        "solver_checked": solver_checked,
        "repeat_examples": repeats[:40],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
