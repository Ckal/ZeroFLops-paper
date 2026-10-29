"""GROUND TRUTH: is the one-column-stacker position winnable for the second player?

Why this exists
---------------
The echo diagnostic showed the model loses 100/100 games to an opponent that
always plays column 7 (internal c6) and wins 100/100 against column 1
(internal c0). Two confounds had to be removed before that could be called a
model defect:

1. The earlier "strong reference" row was NOT a control at all. `strong_ref`
   consults the exact solver only when at most 8 cells are empty, so in the
   opening it silently falls back to `agent.choose(...)` -- the same model.
   Both rows were the same policy.

2. A naive "pure exact solver" policy is ALSO not a control. `ExactSolver.solve`
   returns None (UNPROVED) when more than `max_empty` cells are empty, i.e.
   for the whole opening. A policy that cannot solve the position has no
   move to pick, and a naive implementation degrades to "first legal column".
   That is why the first version of this script played [6,0,6,0,6,0,6] and
   lost at ply 7: the solver "policy" was actually playing column 0.

The correct model-independent control is `src.c4.opponents.minimax_policy`:
a depth-3 negamax with alpha-beta that does not consult the model and does
not silently degrade.

This script therefore compares, as the SECOND player:
  - `minimax`  : canonical model-independent depth-3 control
  - `model`    : the shipped checkpoint, greedy
against a stacker that always plays one fixed column.

If minimax also loses to the c6-stacker, the position is simply lost for the
second player and the model is not at fault. If minimax wins and the model
loses, the defect is in the MODEL.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.opponents import minimax_policy, lookahead_policy
from src.c4.ui_server import _load_agent

OUTCOME_NAMES = {Outcome.ONGOING: "ONGOING", Outcome.WIN_P1: "WIN_P1",
                 Outcome.WIN_P2: "WIN_P2", Outcome.DRAW: "DRAW"}


def stacker(x: int):
    """Opponent that always plays column x while legal, else first legal."""
    def f(board, moves):
        legal = board.valid_actions()
        if x in legal:
            return x
        return legal[0]
    return f


def play(p2_pol, p1_pol):
    """P1 = stacker, P2 = policy under test. Returns the game record."""
    b = Board()
    moves, p2_moves = [], []
    while b.outcome() == Outcome.ONGOING and len(moves) < 42:
        c = p2_pol(b, moves) if b.turn == P2 else p1_pol(b, moves)
        if b.turn == P2:
            p2_moves.append(c)
        b.apply_move(c)
        moves.append(c)
    o = b.outcome()
    winner = P1 if o == Outcome.WIN_P1 else (P2 if o == Outcome.WIN_P2 else None)
    return {"winner": winner, "outcome": OUTCOME_NAMES.get(o, str(o)),
            "filled": bin(b.mask).count("1"), "moves": moves,
            "p2_moves": p2_moves}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--out", default="runs/_probe/c6_stacker_ground_truth.json")
    args = ap.parse_args()

    agent = _load_agent(args.model, device="cpu")

    def model_greedy(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    policies = [
        ("minimax d3", minimax_policy),
        ("lookahead d2", lookahead_policy),
        ("model greedy", model_greedy),
    ]

    rows = []
    print("=" * 78)
    print("GROUND TRUTH (P2 = policy under test, P1 = one-column stacker)")
    print("model:", args.model)
    print("=" * 78)
    for x in (6, 5, 3, 0):
        for pname, pol in policies:
            wins = draws = losses = 0
            fills = []
            first = None
            for _ in range(args.games):
                r = play(pol, stacker(x))
                if r["winner"] == P2:
                    wins += 1
                elif r["winner"] is None:
                    draws += 1
                else:
                    losses += 1
                fills.append(r["filled"])
                if first is None:
                    first = r["moves"]
            n = wins + draws + losses
            rows.append({"opp_col": x, "policy": pname, "games": n,
                         "wins": wins, "draws": draws, "losses": losses,
                         "win_pct": round(100.0 * wins / max(1, n), 1),
                         "avg_filled": round(sum(fills) / max(1, len(fills)), 1),
                         "example_moves": first})
            print(f"  stack c{x} (user col {x+1}) | {pname:13s} "
                  f"W{wins:3d} D{draws:3d} L{losses:3d}  "
                  f"P2 win {100.0*wins/max(1,n):5.1f}%  board {sum(fills)/max(1,len(fills)):4.1f}/42")
            if first:
                print(f"       line: {first[:12]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "games": args.games, "rows": rows},
                  f, indent=2)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
