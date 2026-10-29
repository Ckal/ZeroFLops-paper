"""Reconstruct the user's ACTUAL game and check whether the model had a win.

The user played column 7 (internal c6); the model also played column 7;
the game was a draw with "many open fields". This script:

  1. Replays candidate move sequences that match the user's description
     (both sides playing c6 repeatedly) and reports the result.
  2. For the sequence where the model DID deviate, asks the exact solver:
     was the model's deviation a mistake?
  3. Answers the core question: with the opponent locked to column 7,
     does the MODEL still have a winning plan -- and does it find it?

The key measurement: same opponent, same board, model policy vs a strong
reference policy. If the reference wins and the model draws, the draw is
the model's fault. If both draw, the opponent's policy caused the draw.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.exact_solver import ExactSolver
from src.c4.ui_server import _load_agent

SOLVER = None


def solve_value(solver, board, side):
    return solver.solve(board, side)


def best_exact(solver, board, side):
    """Best column under the exact solver, or None if unproved."""
    status = solver.solve(board, side)
    if status is None:
        return None, None
    vals = {}
    for c in board.valid_actions():
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
        return status, []
    mx = max(vals.values())
    return status, [c for c, v in vals.items() if v == mx]


def describe(board):
    rows = []
    for r in range(5, -1, -1):
        row = ""
        for c in range(7):
            v = board.cell(r, c)
            row += "." if v == 0 else ("Y" if v == 1 else "B")
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/_probe/user_game_replay.json")
    args = ap.parse_args()

    agent = _load_agent(args.model, device="cpu")
    solver = ExactSolver(max_empty=12, budget=2_000_000)

    print("=" * 72)
    print("1. THE MODEL vs AN OPPONENT LOCKED TO COLUMN 7 (internal c6)")
    print("=" * 72)

    def model_pick(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    def opp_c6(board, moves):
        legal = board.valid_actions()
        return 6 if 6 in legal else legal[0]

    # The model plays P2 (second), the opponent P1 always plays c6.
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    fills = []
    move_seqs = []
    for g in range(40):
        board = Board()
        moves = []
        while board.outcome() == Outcome.ONGOING and len(moves) < 42:
            if board.turn == P2:
                c = model_pick(board, moves)
            else:
                c = opp_c6(board, moves)
            board.apply_move(c)
            moves.append(c)
        o = board.outcome()
        if o == Outcome.WIN_P2:
            outcomes["win"] += 1
        elif o == Outcome.DRAW:
            outcomes["draw"] += 1
        else:
            outcomes["loss"] += 1
        fills.append(bin(board.mask).count("1"))
        move_seqs.append((moves, o))
    print(f"  model (P2) vs opponent locked to c6, over 40 games:")
    print(f"    model wins {outcomes['win']}, draws {outcomes['draw']}, "
          f"losses {outcomes['loss']}")
    print(f"    average board fill at end: {sum(fills)/len(fills):.1f}/42 "
          f"({100.0*sum(fills)/len(fills)/42:.1f}%)")

    print()
    print("=" * 72)
    print("2. THE MODEL vs THE SAME OPPONENT, BUT THE MODEL USES THE SOLVER")
    print("=" * 72)
    out2 = {"win": 0, "draw": 0, "loss": 0}
    fills2 = []
    for g in range(40):
        board = Board()
        moves = []
        while board.outcome() == Outcome.ONGOING and len(moves) < 42:
            if board.turn == P2:
                st, opts = best_exact(solver, board, P2)
                c = opts[0] if opts else model_pick(board, moves)
            else:
                c = opp_c6(board, moves)
            board.apply_move(c)
            moves.append(c)
        o = board.outcome()
        if o == Outcome.WIN_P2:
            out2["win"] += 1
        elif o == Outcome.DRAW:
            out2["draw"] += 1
        else:
            out2["loss"] += 1
        fills2.append(bin(board.mask).count("1"))
    print(f"  solver-guided model vs opponent locked to c6, over 40 games:")
    print(f"    model wins {out2['win']}, draws {out2['draw']}, losses {out2['loss']}")
    print(f"    average board fill at end: {sum(fills2)/len(fills2):.1f}/42 "
          f"({100.0*sum(fills2)/len(fills2)/42:.1f}%)")

    print()
    print("=" * 72)
    print("3. IS THE DRAW THE MODEL'S FAULT? (same game, model deviates)")
    print("=" * 72)
    # Take one real model-vs-c6 game and look at the FIRST model decision
    # where the exact solver could prove a better move existed.
    board = Board()
    moves = []
    findings = []
    while board.outcome() == Outcome.ONGOING and len(moves) < 42:
        if board.turn == P2:
            st, opts = best_exact(solver, board, P2)
            played = model_pick(board, moves)
            if st is not None and opts:
                rec = {
                    "ply": len(moves),
                    "played": played,
                    "solver_optimal": played in opts,
                    "n_optimal": len(opts),
                    "optimal": opts,
                    "filled": bin(board.mask).count("1"),
                }
                findings.append(rec)
                if played not in opts:
                    print(f"  ply {len(moves):2d} (board {rec['filled']:2d}/42): "
                          f"model played c{played}, exact optimal = {opts} "
                          f"-> MISTAKE")
            c = played
        else:
            c = opp_c6(board, moves)
        board.apply_move(c)
        moves.append(c)

    proven = [f for f in findings if f["solver_optimal"] is not None]
    n_mistake = sum(1 for f in proven if not f["solver_optimal"])
    print(f"  model decisions where the exact solver could prove the answer: "
          f"{len(proven)}")
    print(f"  of those, the model's move was NOT optimal: {n_mistake}")
    if proven:
        print(f"  -> optimal rate on provable decisions: "
              f"{100.0*(len(proven)-n_mistake)/len(proven):.1f}%")
    print(f"  final board ({bin(board.mask).count('1')}/42 filled):")
    for row in describe(board):
        print(f"    {row}")

    out = {"model": args.model,
           "vs_c6_model": {"outcomes": outcomes,
                           "avg_fill": round(sum(fills)/len(fills), 1)},
           "vs_c6_solver": {"outcomes": out2,
                            "avg_fill": round(sum(fills2)/len(fills2), 1)},
           "provable_decisions": len(proven),
           "mistakes": n_mistake,
           "detail": findings}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
