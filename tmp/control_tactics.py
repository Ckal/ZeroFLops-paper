"""CONTROL for tmp/measure_tactics_v2.py.

A measurement that says "the model takes an immediate win only 26.9% of
the time" is either a real defect or a broken harness. This runs the
SAME measurement code against three model-independent controls:

  minimax  -- src.c4.opponents.minimax_policy (exact-ish, model-free)
  lookahead -- src.c4.opponents.lookahead_policy (model-free)
  random   -- uniform over legal columns

If the controls score ~100% on take-win and block-win, the harness is
sound and the model is genuinely weak. If the controls also score low,
the harness is wrong and the model number is meaningless.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.board import Board  # noqa: E402
from src.c4 import patterns  # noqa: E402


def random_board(rng: random.Random, n_moves: int) -> Board:
    b = Board()
    for _ in range(n_moves):
        legal = b.valid_actions()
        if not legal:
            break
        b.apply_move(rng.choice(legal))
    return b


def wins_for(board: Board, player: int) -> list[int]:
    return list(patterns.immediate_wins(board, player))


def clone(board: Board) -> Board:
    b = Board()
    b.p1 = board.p1
    b.p2 = board.p2
    b.turn = board.turn
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.opponents import minimax_policy, lookahead_policy

    def make_policies():
        # src.c4.opponents.*_policy(board, moves) -- the second argument is
        # the move history list, used only for RNG seeding.
        return {
            "minimax": lambda b, rng, mv: minimax_policy(b, mv),
            "lookahead": lambda b, rng, mv: lookahead_policy(b, mv),
            "random": lambda b, rng, mv: rng.choice(b.valid_actions()),
        }

    results = {}
    for name, pol in make_policies().items():
        rng = random.Random(args.seed)
        history: list[int] = []
        stats = {
            "win_now": {"n": 0, "took": 0},
            "block_now": {"n": 0, "blocked": 0},
            "clean_no_force": {"n": 0, "gave_away": 0},
        }
        for _ in range(args.n):
            b = random_board(rng, rng.randint(4, 30))
            legal = b.valid_actions()
            if not legal:
                continue
            me, opp = b.turn, 3 - b.turn
            my_wins = wins_for(b, me)
            opp_wins = wins_for(b, opp)
            if my_wins:
                stats["win_now"]["n"] += 1
                if pol(b, rng, history) in my_wins:
                    stats["win_now"]["took"] += 1
                continue
            if opp_wins:
                if len(opp_wins) > 1:
                    continue
                stats["block_now"]["n"] += 1
                if pol(b, rng, history) in opp_wins:
                    stats["block_now"]["blocked"] += 1
                continue
            stats["clean_no_force"]["n"] += 1
            b2 = clone(b)
            b2.apply_move(pol(b, rng, history))
            # apply_move() flips the turn, so b2.turn IS the opponent now.
        # The opponent's immediate wins are wins_for(b2, b2.turn).
        if wins_for(b2, b2.turn):
                stats["clean_no_force"]["gave_away"] += 1
        for k, v in stats.items():
            num = (v["took"] if k == "win_now"
                   else v["blocked"] if k == "block_now"
                   else v["n"] - v["gave_away"])
            v["pct"] = round(100 * num / v["n"], 1) if v["n"] else None
        results[name] = stats
        print(f"\n=== control: {name} ===")
        print(f"  take an immediate win : {stats['win_now']['pct']:5.1f}%  "
              f"({stats['win_now']['took']}/{stats['win_now']['n']})")
        print(f"  block opponent win    : {stats['block_now']['pct']:5.1f}%  "
              f"({stats['block_now']['blocked']}/{stats['block_now']['n']})")
        print(f"  dont hand over a win  : {stats['clean_no_force']['pct']:5.1f}%  "
              f"({stats['clean_no_force']['n'] - stats['clean_no_force']['gave_away']}"
              f"/{stats['clean_no_force']['n']})")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
