"""Measure the shipped model's TACTICAL competence directly.

Question: does the model (a) take an immediate win when one exists,
(b) block an opponent's immediate win when it cannot win itself,
(c) avoid handing the opponent an immediate win?

This is a policy-quality measurement, NOT a teacher-agreement one. The
labels here are ground truth computed from the board geometry
(src/c4/patterns.py), independent of the teacher corpus.

All measurements use the SHIPPED hard path because that is what the UI
and the duel harness use.
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
    """Play a random legal game up to n_moves, return the board."""
    b = Board()
    for _ in range(n_moves):
        legal = b.valid_actions()
        if not legal:
            break
        b.apply_move(rng.choice(legal))
    return b


def wins_for(board: Board, player: int) -> list[int]:
    """Columns where `player` would complete a four right now."""
    return list(patterns.immediate_wins(board, player))


def clone(board: Board) -> Board:
    b = Board()
    b.p1 = board.p1
    b.p2 = board.p2
    b.turn = board.turn
    return b


def model_choose(agent, b: Board) -> int:
    """Hard-path argmax over legal columns."""
    legal = b.valid_actions()
    logits = agent.hard_column_logits(b, legal)
    vals = [float(logits[c]) for c in legal]
    return legal[max(range(len(legal)), key=lambda i: vals[i])]


def load_agent(ckpt_path: str):
    """Load exactly the way src/c4/ui_server.py loads it."""
    from src.c4.ui_server import _load_agent
    return _load_agent(ckpt_path, device="cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    agent = load_agent(args.ckpt)

    rng = random.Random(args.seed)
    stats = {
        "win_now": {"n": 0, "took": 0},
        "block_now": {"n": 0, "blocked": 0},
        "clean_no_force": {"n": 0, "gave_away": 0},
    }
    examples = []

    for _ in range(args.n):
        b = random_board(rng, rng.randint(4, 30))
        legal = b.valid_actions()
        if not legal:
            continue

        me = b.turn
        opp = 3 - me
        my_wins = wins_for(b, me)
        opp_wins = wins_for(b, opp)

        if my_wins:
            stats["win_now"]["n"] += 1
            mv = model_choose(agent, b)
            if mv in my_wins:
                stats["win_now"]["took"] += 1
            elif len(examples) < 8:
                examples.append({"kind": "missed_win", "move": mv,
                                 "wins": my_wins})
            continue

        if opp_wins:
            if len(opp_wins) > 1:
                # opponent threatens two squares: one move cannot cover
                # both. Not a fair single-answer test.
                continue
            stats["block_now"]["n"] += 1
            mv = model_choose(agent, b)
            if mv in opp_wins:
                stats["block_now"]["blocked"] += 1
            elif len(examples) < 8:
                examples.append({"kind": "failed_block", "move": mv,
                                 "opp_wins": opp_wins})
            continue

        # nothing forcing: does the model hand over an immediate win?
        stats["clean_no_force"]["n"] += 1
        mv = model_choose(agent, b)
        b2 = clone(b)
        b2.apply_move(mv)
        # apply_move() flips the turn, so b2.turn IS the opponent now.
        # The opponent's immediate wins are wins_for(b2, b2.turn).
        if wins_for(b2, b2.turn):
            stats["clean_no_force"]["gave_away"] += 1
            if len(examples) < 8:
                examples.append({"kind": "gave_away", "move": mv})

    for k, v in stats.items():
        num = (v["took"] if k == "win_now"
               else v["blocked"] if k == "block_now"
               else v["n"] - v["gave_away"])
        v["pct"] = round(100 * num / v["n"], 1) if v["n"] else None

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "stats": stats, "examples": examples}
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
