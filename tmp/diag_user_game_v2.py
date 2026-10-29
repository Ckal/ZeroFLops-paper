"""Reconstruct the user's ACTUAL game and check whether the model had a win.

The user played column 7 (internal c6); the model also played column 7;
the game was declared a draw with "many open fields".  This script
answers three things:

  1. What does the MODEL do against an opponent locked to c6?
  2. What does an INDEPENDENT strong control (canonical
     `src.c4.opponents.minimax_policy`) do against that same opponent?
  3. On the model-vs-c6 line, how many of the model's decisions could the
     exact solver PROVE, and how many of those were mistakes?

IMPORTANT (methodology, learned the hard way):
  * The exact solver cannot prove a position with many empty cells.  When
    it returns None we MUST NOT silently fall back to the model under
    test and then report the result as "solver-guided" -- that silently
    measures the model twice and looks like a confirmation.  The fallback
    count is reported explicitly, and the strong reference is the
    canonical `minimax_policy`, which is model-independent.
  * The hard export path is used everywhere (`use_hard=True`), because
    that is what the UI and the duel harness actually run.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.exact_solver import ExactSolver
from src.c4.opponents import minimax_policy
from src.c4.ui_server import _load_agent


def best_exact(solver, board, side):
    """Best column set under the exact solver, or (None, []) if unproved."""
    status = solver.solve(board, side)
    if status is None:
        return None, []
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


def play_games(policy_p2, opp, n=40, collect_first=False):
    """Play n games; P1 = opponent policy, P2 = policy_p2."""
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    fills = []
    lines = []
    for _ in range(n):
        board = Board()
        moves = []
        while board.outcome() == Outcome.ONGOING and len(moves) < 42:
            c = policy_p2(board, moves) if board.turn == P2 else opp(board, moves)
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
        if collect_first:
            lines.append((moves, o))
    return outcomes, fills, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/_probe/user_game_replay.json")
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()

    agent = _load_agent(args.model, device="cpu")
    solver = ExactSolver(max_empty=12, budget=2_000_000)

    def model_pick(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    def opp_c6(board, moves):
        legal = board.valid_actions()
        return 6 if 6 in legal else legal[0]

    def strong_pick(board, moves):
        """Model-INDEPENDENT control: canonical minimax."""
        return minimax_policy(board, list(moves))

    print("=" * 72)
    print("1. THE MODEL (P2) vs AN OPPONENT LOCKED TO COLUMN 7 (internal c6)")
    print("=" * 72)
    outcomes, fills, model_lines = play_games(model_pick, opp_c6, args.games,
                                             collect_first=True)
    print(f"  over {args.games} games:")
    print(f"    model wins {outcomes['win']}, draws {outcomes['draw']}, "
          f"losses {outcomes['loss']}")
    print(f"    average board fill at end: {sum(fills)/len(fills):.1f}/42 "
          f"({100.0*sum(fills)/len(fills)/42:.1f}%)")

    print()
    print("=" * 72)
    print("2. SAME OPPONENT, MODEL-INDEPENDENT STRONG CONTROL (minimax d3)")
    print("=" * 72)
    out2, fills2, _ = play_games(strong_pick, opp_c6, args.games)
    print(f"  over {args.games} games:")
    print(f"    strong wins {out2['win']}, draws {out2['draw']}, "
          f"losses {out2['loss']}")
    print(f"    average board fill at end: {sum(fills2)/len(fills2):.1f}/42 "
          f"({100.0*sum(fills2)/len(fills2)/42:.1f}%)")
    print()
    if out2["win"] > outcomes["win"]:
        print("  => the c6-locked opponent is DEFEATABLE.  The model's")
        print("     result against it is a MODEL deficiency, not an")
        print("     artefact of the opponent.")
    elif out2["loss"] == out2["win"] == 0:
        print("  => the strong control also cannot win here; the opponent")
        print("     itself is the limiting factor for that line.")

    print()
    print("=" * 72)
    print("3. HOW MANY MODEL DECISIONS COULD THE EXACT SOLVER PROVE?")
    print("=" * 72)
    board = Board()
    moves = []
    findings = []
    n_unproved = 0
    n_proved = 0
    n_mistake = 0
    while board.outcome() == Outcome.ONGOING and len(moves) < 42:
        if board.turn == P2:
            st, opts = best_exact(solver, board, P2)
            played = model_pick(board, moves)
            if st is not None and opts:
                n_proved += 1
                rec = {"ply": len(moves), "played": played,
                       "solver_optimal": played in opts,
                       "n_optimal": len(opts), "optimal": opts,
                       "filled": bin(board.mask).count("1")}
                findings.append(rec)
                if played not in opts:
                    n_mistake += 1
                    print(f"  ply {len(moves):2d} (board {rec['filled']:2d}/42): "
                          f"model played c{played}, exact optimal = {opts} "
                          f"-> MISTAKE")
            else:
                n_unproved += 1
            c = played
        else:
            c = opp_c6(board, moves)
        board.apply_move(c)
        moves.append(c)

    print(f"  model decisions PROVABLE by the exact solver : {n_proved}")
    print(f"  model decisions UNPROVABLE (solver returned None): {n_unproved}")
    print(f"  of the provable ones, NOT optimal              : {n_mistake}")
    if n_proved:
        print(f"  -> optimal rate on provable decisions: "
              f"{100.0*(n_proved-n_mistake)/n_proved:.1f}%")
    print(f"  final board ({bin(board.mask).count('1')}/42 filled):")
    for row in describe(board):
        print(f"    {row}")

    out = {
        "model": args.model,
        "games": args.games,
        "vs_c6_model": {"outcomes": outcomes,
                        "avg_fill": round(sum(fills) / len(fills), 1)},
        "vs_c6_strong_minimax": {"outcomes": out2,
                                 "avg_fill": round(sum(fills2) / len(fills2), 1)},
        "exact_solver_provable_decisions": n_proved,
        "exact_solver_unproved_decisions": n_unproved,
        "provable_mistakes": n_mistake,
        "detail": findings,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
