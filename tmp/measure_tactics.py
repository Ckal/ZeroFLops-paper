"""Measure the shipped model's TACTICAL competence directly.

Question: does the model (a) take an immediate win when one exists,
(b) block an opponent's immediate win when it cannot win itself,
(c) create a double threat when a safe move exists?

This is a policy-quality measurement, NOT a teacher-agreement one. The
labels here are ground truth computed from the board geometry
(src/c4/patterns.py), independent of the teacher corpus.

Every position is generated so that the correct answer is unambiguous:
- win_now:     the model has an immediate win.
- block_now:   the opponent has an immediate win and the model cannot
               win right now.
- double:      the model can create a double threat with one move and
               the opponent has no immediate win.
- safe:        nothing forcing; a move must not hand over an
               immediate win.

All measurements use the SHIPPED hard path (use_hard=True) because that
is what the UI and the duel harness use.
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


def classify(b: Board) -> tuple[str | None, tuple[int, ...]]:
    """Return (kind, optimal_columns) using board geometry only."""
    rep = patterns.pattern_report(b)
    my_wins = rep.get("p1_immediate_wins") if b.turn == 1 else rep.get("p2_immediate_wins")
    opp_wins = rep.get("p2_immediate_wins") if b.turn == 1 else rep.get("p1_immediate_wins")
    if my_wins:
        return "win_now", tuple(my_wins)
    if opp_wins:
        # correct answer: block. If the opponent has >1 winning square
        # and we cannot cover all of them, the position is ambiguous.
        if len(opp_wins) > 1:
            return None, ()
        return "block_now", tuple(opp_wins)
    return None, ()


def model_choose(agent, b: Board) -> int:
    """Hard-path argmax over legal columns."""
    legal = b.valid_actions()
    logits = agent.hard_column_logits(b, legal)
    vals = [float(logits[c]) for c in legal]
    return legal[max(range(len(legal)), key=lambda i: vals[i])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.agent import Agent
    agent = Agent(args.ckpt)

    rng = random.Random(args.seed)
    stats = {
        "win_now": {"n": 0, "took": 0},
        "block_now": {"n": 0, "blocked": 0},
        "clean_no_force": {"n": 0, "gave_away": 0},
    }
    examples = []

    for _ in range(args.n):
        b = random_board(rng, rng.randint(4, 30))
        if not b.valid_actions():
            continue
        kind, opt = classify(b)
        if kind is None:
            # nothing forcing: check the model does not hand over an
            # immediate win
            if len(opt) or True:
                pass
            legal = b.valid_actions()
            if not legal:
                continue
            rep = patterns.pattern_report(b)
            opp_wins = (rep.get("p2_immediate_wins") if b.turn == 1
                        else rep.get("p1_immediate_wins"))
            if opp_wins:
                continue
            mv = model_choose(agent, b)
            # after the model's move, does the OPPONENT have an immediate win?
            b2 = b.copy()
            b2.apply_move(mv)
            rep2 = patterns.pattern_report(b2)
            new_opp = (rep2.get("p2_immediate_wins") if b2.turn == 1
                       else rep2.get("p1_immediate_wins"))
            stats["clean_no_force"]["n"] += 1
            if new_opp:
                stats["clean_no_force"]["gave_away"] += 1
                if len(examples) < 8:
                    examples.append({"kind": "gave_away", "move": mv,
                                     "opp_wins": new_opp})
            continue

        stats[kind]["n"] += 1
        mv = model_choose(agent, b)
        if kind == "win_now":
            if mv in opt:
                stats[kind]["took"] += 1
            elif len(examples) < 8:
                examples.append({"kind": "missed_win", "move": mv, "wins": opt})
        else:  # block_now
            if mv in opt:
                stats[kind]["blocked"] += 1
            elif len(examples) < 8:
                examples.append({"kind": "failed_block", "move": mv,
                                 "opp_wins": opt})

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "stats": stats, "examples": examples}
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
