"""WHY does the model lose to a one-column stacker? Measure the echo rate.

Ground truth (runs/_probe/c6_stacker_ground_truth.json) established that the
position IS winnable for the second player: the model-independent
`minimax_policy` and `lookahead_policy` from src/c4/opponents.py both beat
the c6/c5/c3 stackers 5/5, while the model loses 5/5.

This script isolates the mechanism. Hypothesis: the model ANTAGONISES the
opponent -- it answers column X with column X -- so the stacker never
completes a vertical four and the model never builds its own attack either.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.ui_server import _load_agent

MODEL = "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"


def stacker(x):
    def f(b, _m):
        legal = b.valid_actions()
        return x if x in legal else legal[0]
    return f


def play(p2_pol, p1_pol):
    b = Board()
    moves, p2_moves = [], []
    echo = resp = 0
    while b.outcome() == Outcome.ONGOING and len(moves) < 42:
        if b.turn == P2:
            c = p2_pol(b, moves)
            p2_moves.append(c)
            if moves:
                resp += 1
                if c == moves[-1]:
                    echo += 1
        else:
            c = p1_pol(b, moves)
        b.apply_move(c)
        moves.append(c)
    o = b.outcome()
    w = P1 if o == Outcome.WIN_P1 else (P2 if o == Outcome.WIN_P2 else None)
    return w, moves, p2_moves, echo, resp, o


def main() -> int:
    agent = _load_agent(MODEL, device="cpu")

    def model_greedy(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    rows = []
    print("=" * 78)
    print("ECHO RATE: does the model answer column X with column X?")
    print("=" * 78)
    for x in (6, 5, 3, 0):
        E = R = wins = 0
        first_line = None
        for _ in range(5):
            w, moves, p2, e, r, o = play(model_greedy, stacker(x))
            E += e
            R += r
            if w == P2:
                wins += 1
            if first_line is None:
                first_line = (moves, o)
        rate = 100.0 * E / max(1, R)
        print(f"  vs stack c{x}: echo {E}/{R} = {rate:5.1f}%   P2 wins {wins}/5")
        rows.append({"opp_col": x, "echo": E, "responses": R,
                     "echo_pct": round(rate, 1), "p2_wins": wins,
                     "example_moves": first_line[0]})
        if first_line:
            moves = first_line[0]
            print(f"       P1 (stacker): {moves[0::2]}")
            print(f"       P2 (model)  : {moves[1::2]}")

    out = {"model": MODEL, "rows": rows}
    with open("runs/_probe/model_echo_mechanism.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote runs/_probe/model_echo_mechanism.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
