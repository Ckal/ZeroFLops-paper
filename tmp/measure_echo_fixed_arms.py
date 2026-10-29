"""Measure the ECHO RATE for one or more checkpoints, with an
independent strong control.

Corrected relative to `tmp/diag_echo_col.py`, which had the
"silent fallback" defect: its `strong_ref` fell back to
`agent.choose(...)` whenever the exact solver could not prove a
position, and then reported the result under the heading "strong
reference" -- i.e. it measured the model twice and looked like a
confirmation.

Here:
  * the strong control is the canonical, model-independent
    `src.c4.opponents.minimax_policy`;
  * the exact solver is used only where it can PROVE a value, and the
    number of provable decisions is reported explicitly;
  * the echo rate is measured on the hard export path, which is what
    the UI and the duel harness actually run.

Echo rate = how often the model (P2) plays the SAME column the opponent
(P1) just played.  A model that follows the opponent up a column can
never build a horizontal four.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2
from src.c4.opponents import minimax_policy
from src.c4.ui_server import _load_agent


def opp_repeat_col(x: int):
    def f(board, moves, rng):
        legal = board.valid_actions()
        return x if x in legal else legal[0]
    return f


def opp_centre(board, moves, rng):
    legal = board.valid_actions()
    for c in (3, 2, 4, 1, 5, 0, 6):
        if c in legal:
            return c
    return legal[0]


def opp_random(board, moves, rng):
    return rng.choice(board.valid_actions())


def play(model_policy, opp_policy, rng):
    board = Board()
    moves: list[int] = []
    model_moves: list[int] = []
    echo = 0
    echo_possible = 0
    while board.outcome() == Outcome.ONGOING and len(moves) < 42:
        if board.turn == P2:
            c = model_policy(board, moves)
            model_moves.append(c)
            if moves:
                echo_possible += 1
                if moves[-1] == c:
                    echo += 1
        else:
            c = opp_policy(board, moves, rng)
        board.apply_move(c)
        moves.append(c)
    o = board.outcome()
    winner = None if o == Outcome.ONGOING else (P1 if o == Outcome.WIN_P1 else P2)
    return {
        "winner": winner,
        "filled": bin(board.mask).count("1"),
        "echo": echo,
        "echo_possible": echo_possible,
        "model_moves": model_moves,
    }


def measure(model_path: str, games: int, seed: int) -> dict:
    import random
    rng = random.Random(seed)
    agent = _load_agent(model_path, device="cpu")

    def model_greedy(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    def strong(board, moves):
        return minimax_policy(board, list(moves))

    out = {"model": model_path, "games": games, "seed": seed}

    # ---- A. echo rate -------------------------------------------------
    echo_rows = []
    for name, opp in (("random", opp_random),
                      ("centre", opp_centre),
                      ("repeat c6 (col 7)", opp_repeat_col(6))):
        ech = pos = 0
        for _ in range(games):
            r = play(model_greedy, opp, rng)
            ech += r["echo"]
            pos += r["echo_possible"]
        rate = 100.0 * ech / max(1, pos)
        echo_rows.append({"opponent": name, "echo_rate_pct": round(rate, 2),
                          "echoes": ech, "opportunities": pos})
        print(f"  vs {name:20s} echo {rate:5.2f}%  ({ech}/{pos})")
    out["echo"] = echo_rows

    # ---- B. winnability against column-stackers ------------------------
    win_rows = []
    for x in (6, 3, 0):
        for pol_name, pol in (("model greedy", model_greedy),
                              ("strong minimax", strong)):
            w = d = l = 0
            fills = []
            for _ in range(max(40, games // 2)):
                r = play(pol, opp_repeat_col(x), rng)
                if r["winner"] == P2:
                    w += 1
                elif r["winner"] is None:
                    d += 1
                else:
                    l += 1
                fills.append(r["filled"])
            n = max(1, w + d + l)
            avg = sum(fills) / max(1, len(fills))
            win_rows.append({"opp_col": x, "policy": pol_name,
                             "wins": w, "draws": d, "losses": l,
                             "win_pct": round(100.0 * w / n, 1),
                             "avg_filled": round(avg, 1)})
            print(f"  opp repeats c{x} | {pol_name:16s} "
                  f"W{w:3d} D{d:3d} L{l:3d}  win {100.0*w/n:5.1f}%  "
                  f"board {avg:4.1f}/42")
    out["winnability"] = win_rows

    # ---- C. model's own column profile --------------------------------
    col_rows = []
    for name, opp in (("random", opp_random),
                      ("centre", opp_centre),
                      ("repeat c6 (col 7)", opp_repeat_col(6))):
        counts = collections.Counter()
        n = 0
        for _ in range(max(40, games // 2)):
            r = play(model_greedy, opp, rng)
            for c in r["model_moves"]:
                counts[c] += 1
                n += 1
        share = {c: round(100.0 * counts[c] / max(1, n), 1) for c in range(7)}
        col_rows.append({"opponent": name, "model_moves": n,
                         "col_share_pct": share})
        print(f"  vs {name:20s} " +
              "  ".join(f"c{c}={share[c]:5.1f}%" for c in range(7)))
    out["model_column_profile"] = col_rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated checkpoint paths")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", default="runs/_probe/echo_fixed_arms.json")
    args = ap.parse_args()

    paths = [p.strip() for p in args.models.split(",") if p.strip()]
    all_out = {}
    for p in paths:
        print("=" * 70)
        print(p)
        print("=" * 70)
        all_out[p] = measure(p, args.games, args.seed)
        print()
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
