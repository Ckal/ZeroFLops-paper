"""CORRECTED ply-bucketed tactical probe.

Tests the user's hypothesis: "does the model only learn the basics in the
first 10 moves?".

The earlier version of this probe (tmp/measure_tactics_by_ply.py) passed
the LEGAL-COLUMN LIST where `hard_column_logits` expects the MOVE
HISTORY, so its buckets measured a garbage prompt. This version fixes
that and re-measures.

Buckets: 1-6, 7-12, 13-20, 21-30 (ply = number of moves already played).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.board import Board  # noqa: E402
from src.c4 import patterns  # noqa: E402


def wins_for(board, player):
    return list(patterns.immediate_wins(board, player))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.ui_server import _load_agent
    agent = _load_agent(args.ckpt, device="cpu")

    buckets = [(1, 6), (7, 12), (13, 20), (21, 30)]
    st = {b: collections.Counter() for b in buckets}

    rng = random.Random(args.seed)
    for _ in range(args.n):
        b = Board()
        hist = []
        target_n = rng.randint(1, 30)
        while len(hist) < target_n:
            legal = b.valid_actions()
            if not legal:
                break
            c = rng.choice(legal)
            b.apply_move(c)
            hist.append(c)
        legal = b.valid_actions()
        if not legal or not hist:
            continue
        ply = len(hist)
        bucket = next((bb for bb in buckets if bb[0] <= ply <= bb[1]), None)
        if bucket is None:
            continue

        logits = agent.hard_column_logits(b, hist)
        pick = max(legal, key=lambda c: float(logits[c]))

        my_wins = wins_for(b, b.turn)
        opp_wins = wins_for(b, 3 - b.turn)
        c = st[bucket]
        c["n"] += 1
        if my_wins:
            c["win_now_n"] += 1
            c["win_now_ok"] += int(pick in my_wins)
        elif len(opp_wins) == 1:
            c["block_now_n"] += 1
            c["block_now_ok"] += int(pick in opp_wins)
        elif len(opp_wins) > 1:
            c["double_threat_n"] += 1
        else:
            c["clean_n"] += 1
            b2 = Board()
            b2.p1, b2.p2, b2.turn = b.p1, b.p2, b.turn
            b2.apply_move(pick)
            c["gave_away"] += int(bool(wins_for(b2, b2.turn)))

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "BUGFIXED": "moves argument is the MOVE HISTORY",
           "buckets": {}}
    print("=== CORRECTED ply-bucketed tactical probe ===")
    print(f"{'ply':>8} | {'n':>5} | {'win_now':>16} | {'block_now':>16} | "
          f"{'gave_away':>12}")
    print("-" * 72)
    for bkt in buckets:
        c = st[bkt]
        n = c["n"]
        if not n:
            continue
        wn, bn = c["win_now_n"], c["block_now_n"]
        wp = round(100 * c["win_now_ok"] / wn, 1) if wn else None
        bp = round(100 * c["block_now_ok"] / bn, 1) if bn else None
        gp = round(100 * c["gave_away"] / c["clean_n"], 1) if c["clean_n"] else None
        print(f"{bkt[0]:>3}-{bkt[1]:<4} | {n:>5} | "
              f"{(str(wp)+'%' if wp is not None else '-'):>8} (n={wn:<4}) | "
              f"{(str(bp)+'%' if bp is not None else '-'):>8} (n={bn:<4}) | "
              f"{(str(gp)+'%' if gp is not None else '-'):>12}")
        out["buckets"][f"{bkt[0]}-{bkt[1]}"] = {
            "n": n, "win_now_n": wn, "win_now_pct": wp,
            "block_now_n": bn, "block_now_pct": bp,
            "clean_n": c["clean_n"], "gave_away_pct": gp,
            "double_threat_n": c["double_threat_n"],
        }
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
