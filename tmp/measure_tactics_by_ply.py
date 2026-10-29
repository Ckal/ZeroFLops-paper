"""Is the model's tactical weakness concentrated EARLY in the game?

The user asked whether the fix might be "prevent 4, go for 4 -- only in
the first 10 moves". This measures take-win / block-win competence
bucketed by ply, so the hypothesis can be tested rather than assumed.

Buckets: 1-6, 7-12, 13-20, 21-30.
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

BUCKETS = [(1, 6), (7, 12), (13, 20), (21, 30)]


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


def bucket_of(ply: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= ply <= hi:
            return f"{lo}-{hi}"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.ui_server import _load_agent
    agent = _load_agent(args.ckpt, device="cpu")

    rng = random.Random(args.seed)
    per = {f"{lo}-{hi}": {"win_now": [0, 0], "block_now": [0, 0],
                          "clean": [0, 0]} for lo, hi in BUCKETS}

    for _ in range(args.n):
        n_moves = rng.randint(4, 30)
        b = random_board(rng, n_moves)
        legal = b.valid_actions()
        if not legal:
            continue
        ply = sum(1 for _ in range(0))  # placeholder
        # ply = number of stones on the board
        ply = bin(b.p1).count("1") + bin(b.p2).count("1")
        bk = bucket_of(ply)
        if bk not in per:
            continue

        me, opp = b.turn, 3 - b.turn
        my_wins = wins_for(b, me)
        opp_wins = wins_for(b, opp)

        if my_wins:
            per[bk]["win_now"][1] += 1
            legal_cols = legal
            logits = agent.hard_column_logits(b, legal_cols)
            vals = [float(logits[c]) for c in legal_cols]
            mv = legal_cols[max(range(len(legal_cols)),
                                 key=lambda i: vals[i])]
            if mv in my_wins:
                per[bk]["win_now"][0] += 1
            continue
        if opp_wins:
            if len(opp_wins) > 1:
                continue
            per[bk]["block_now"][1] += 1
            logits = agent.hard_column_logits(b, legal)
            vals = [float(logits[c]) for c in legal]
            mv = legal[max(range(len(legal)), key=lambda i: vals[i])]
            if mv in opp_wins:
                per[bk]["block_now"][0] += 1
            continue
        per[bk]["clean"][1] += 1
        logits = agent.hard_column_logits(b, legal)
        vals = [float(logits[c]) for c in legal]
        mv = legal[max(range(len(legal)), key=lambda i: vals[i])]
        b2 = clone(b)
        b2.apply_move(mv)
        # apply_move() flips the turn, so b2.turn IS the opponent now.
        # The opponent's immediate wins are wins_for(b2, b2.turn).
        if wins_for(b2, b2.turn):
            per[bk]["clean"][0] += 1

    out = {}
    print(f"\n=== {args.ckpt} ===")
    print(f"{'ply bucket':>10} | {'take win':>18} | {'block win':>18} | {'no give-away':>18}")
    print("-" * 76)
    for lo, hi in BUCKETS:
        bk = f"{lo}-{hi}"
        row = []
        for key in ("win_now", "block_now", "clean"):
            hit, tot = per[bk][key]
            pct = round(100 * hit / tot, 1) if tot else None
            row.append((pct, hit, tot))
        (w, wh, wn), (bl, bh, bn), (cl, ch, cn) = row
        print(f"{bk:>10} | {str(w)+'%':>9} ({wh}/{wn}) | "
              f"{str(bl)+'%':>9} ({bh}/{bn}) | {str(cl)+'%':>9} ({ch}/{cn})")
        out[bk] = {"win_now_pct": w, "win_now_n": wn,
                   "block_now_pct": bl, "block_now_n": bn,
                   "gave_away_pct": (round(100 * (cn - ch) / cn, 1)
                                     if cn else None),
                   "clean_n": cn}

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
             "buckets": out}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
