"""DECISIVE: is the c6-stacker position winnable for the second player?

The echo diagnostic showed the model loses 100/100 games to an opponent
that always plays column 7 (internal c6), but wins 100/100 against an
opponent that always plays column 1 (internal c0). The "strong
reference" row was NOT a control: `strong_ref` only consults the exact
solver when at most 8 cells are empty, so in the opening it silently
falls back to `agent.choose(...)` -- the same model. Both rows were the
same policy.

This script uses a PURE exact-solver policy (no model fallback) to
establish the ground truth: can the second player win at all against a
one-column stacker, and is the answer symmetric between c0 and c6?

If the solver wins against the c6-stacker and the model loses, the
defect is in the MODEL, not in the position. If the solver also loses,
the position is simply lost and the model is not at fault.

The result is the direct answer to the user's report:
"me and the model picked column 7 only ... then it was draw".
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.exact_solver import ExactSolver
from src.c4.ui_server import _load_agent


def make_solver_policy(max_empty: int, budget: int):
    """A pure exact-solver policy: never falls back to the model."""
    solver = ExactSolver(max_empty=max_empty, budget=budget)

    def pol(board: Board, moves):
        legal = board.valid_actions()
        if not legal:
            raise ValueError("no legal action")
        best, best_val = None, None
        for c in legal:
            b2 = copy.deepcopy(board)
            b2.apply_move(c)
            if b2.outcome() != Outcome.ONGOING:
                val = 1
            else:
                v = solver.solve(b2, 3 - board.turn)
                if v is None:
                    # unproved: treat as 0 so a provable line always wins
                    val = 0
                else:
                    val = -v
            if best_val is None or val > best_val:
                best_val, best = val, c
        return best if best is not None else legal[0]
    return pol


def opp_repeat_col(x: int):
    def f(board, moves):
        legal = board.valid_actions()
        if x in legal:
            return x
        return legal[0]
    return f


def play(model_pol, opp_pol, model_is_p1: bool):
    """model_is_p1=False -> the model is P2, the stacker is P1."""
    b = Board()
    moves = []
    model_moves = []
    while b.outcome() == Outcome.ONGOING and len(moves) < 42:
        if b.turn == (P1 if model_is_p1 else P2):
            c = model_pol(b, moves)
            model_moves.append(c)
        else:
            c = opp_pol(b, moves)
        b.apply_move(c)
        moves.append(c)
    o = b.outcome()
    winner = None
    if o == Outcome.WIN_P1:
        winner = P1
    elif o == Outcome.WIN_P2:
        winner = P2
    names = {Outcome.ONGOING: "ONGOING", Outcome.WIN_P1: "WIN_P1",
             Outcome.WIN_P2: "WIN_P2", Outcome.DRAW: "DRAW"}
    return {"winner": winner, "outcome": names.get(o, str(o)),
            "filled": bin(b.mask).count("1"), "moves": moves,
            "model_moves": model_moves}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-empty", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--out", default="runs/_probe/c6_stacker_ground_truth.json")
    args = ap.parse_args()

    agent = _load_agent(args.model, device="cpu")

    def model_greedy(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    solver_pol = make_solver_policy(args.max_empty, args.budget)

    rows = []
    print("=" * 78)
    print("GROUND TRUTH: is the one-column-stacker position winnable?")
    print(f"(exact solver, max_empty={args.max_empty}, budget={args.budget:,})")
    print("=" * 78)
    for x in (6, 5, 3, 0):
        for pol_name, pol in (("exact solver", solver_pol),
                              ("model greedy", model_greedy)):
            wins = draws = losses = 0
            fills = []
            first_line = None
            for _ in range(args.games):
                r = play(pol, opp_repeat_col(x), model_is_p1=False)
                if r["winner"] == P2:
                    wins += 1
                elif r["winner"] is None:
                    draws += 1
                else:
                    losses += 1
                fills.append(r["filled"])
                if first_line is None:
                    first_line = r["moves"]
            n = wins + draws + losses
            avg = sum(fills) / max(1, len(fills))
            rows.append({"opp_col": x, "policy": pol_name, "games": n,
                         "wins": wins, "draws": draws, "losses": losses,
                         "win_pct": round(100.0 * wins / max(1, n), 1),
                         "avg_filled": round(avg, 1),
                         "example_moves": first_line})
            print(f"  stack c{x} (col {x+1}) | {pol_name:13s} "
                  f"W{wins:3d} D{draws:3d} L{losses:3d}  "
                  f"win {100.0*wins/max(1,n):5.1f}%  board {avg:4.1f}/42")
            if first_line and pol_name == "exact solver":
                print(f"       solver line: {first_line}")

    out = {"model": args.model, "max_empty": args.max_empty,
           "budget": args.budget, "games": args.games, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
