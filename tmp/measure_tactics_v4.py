"""CORRECTED tactical probe (v4) -- authoritative.

Fixes applied vs v3 / by_ply_v2:
  1. `moves` argument = MOVE HISTORY (not the legal-column list).
  2. Board copy bug: `b2.p1, b2.p2, b2.turn = b.p1, b2.p2, b.turn` copied
     b2's OWN (empty) p2 instead of b's p2. That silently dropped every
     opponent disc and made the give-away metric meaningless.
     Here every field is copied: p1, p2, mask, heights, turn, last_move.
  3. Give-away is defined precisely:
        opponent had NO immediate win before our move
        AND opponent HAS an immediate win after our move
        AND our move did not win.
     (v3's version flagged "opponent already had a playable three",
      which is a different and much broader condition.)

Also reports the UI's displayed percentages next to the tactical truth,
which is what the user actually asked about.
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

import torch  # noqa: E402

from src.c4.board import Board  # noqa: E402
from src.c4 import patterns  # noqa: E402

BUCKETS = [(1, 6), (7, 12), (13, 20), (21, 30)]


def wins_for(board: Board, player: int) -> list[int]:
    return list(patterns.immediate_wins(board, player))


def copy_board(b: Board) -> Board:
    """Exact copy of every field that apply_move() touches."""
    c = Board()
    c.p1 = b.p1
    c.p2 = b.p2
    c.mask = b.mask
    c.heights = list(b.heights)
    c.turn = b.turn
    c.last_move = b.last_move
    return c


def model_pick(agent, b: Board, hist: list[int]):
    """Hard-path argmax. `hist` is the MOVE HISTORY."""
    logits = agent.hard_column_logits(b, hist)
    legal = b.valid_actions()
    neg = torch.tensor(-1e9, dtype=logits.dtype)
    mask = torch.tensor([c in legal for c in range(7)], dtype=torch.bool)
    probs = torch.softmax(torch.where(mask, logits, neg), dim=-1).tolist()
    pick = max(legal, key=lambda c: float(logits[c]))
    return pick, probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.ui_server import _load_agent
    agent = _load_agent(args.ckpt, device="cpu")

    rng = random.Random(args.seed)
    st = {b: collections.Counter() for b in BUCKETS}
    overall = collections.Counter()
    disp = {"n": 0, "top_is_win": 0, "pct_on_top": [], "pct_on_win": [],
            "pct_top_over_50": 0, "pct_top_at_100": 0, "n_100": 0}

    for _ in range(args.n):
        b = Board()
        hist = []
        target = rng.randint(1, 30)
        while len(hist) < target:
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
        bucket = next((bb for bb in BUCKETS if bb[0] <= ply <= bb[1]), None)
        if bucket is None:
            continue

        pick, probs = model_pick(agent, b, hist)
        me = b.turn
        opp = 3 - me
        my_wins = wins_for(b, me)
        opp_wins = wins_for(b, opp)

        c = st[bucket]
        c["n"] += 1
        overall["n"] += 1

        if my_wins:
            c["win_n"] += 1
            overall["win_n"] += 1
            c["win_ok"] += int(pick in my_wins)
            overall["win_ok"] += int(pick in my_wins)
            # displayed-% semantics on win-available positions
            disp["n"] += 1
            disp["pct_on_top"].append(probs[pick] * 100)
            disp["pct_on_win"].append(max(probs[w] * 100 for w in my_wins))
            disp["top_is_win"] += int(pick in my_wins)
            if probs[pick] >= 0.5:
                disp["pct_top_over_50"] += 1
            if probs[pick] >= 0.999:
                disp["pct_top_at_100"] += 1
                disp["n_100"] += 1
            continue

        if len(opp_wins) == 1:
            c["block_n"] += 1
            overall["block_n"] += 1
            c["block_ok"] += int(pick in opp_wins)
            overall["block_ok"] += int(pick in opp_wins)
            continue
        if len(opp_wins) > 1:
            c["double_n"] += 1
            overall["double_n"] += 1
            continue

        # clean: no immediate win for anyone
        b2 = copy_board(b)
        b2.apply_move(pick)
        we_won = bool(wins_for(b2, me))
        opp_now = wins_for(b2, opp)
        c["clean_n"] += 1
        overall["clean_n"] += 1
        if not we_won and opp_now:
            c["gave_away"] += 1
            overall["gave_away"] += 1

    def pct(num, den):
        return round(100 * num / den, 1) if den else None

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "FIXES": ["moves=history", "full board copy", "precise give-away"],
           "overall": {}, "buckets": {}, "displayed_pct": {}}
    print("=== CORRECTED tactical probe (v4) ===")
    print(f"{'ply':>8} | {'n':>5} | {'take win':>14} | {'block':>14} | "
          f"{'gave away':>12}")
    print("-" * 66)
    for bkt in BUCKETS:
        c = st[bkt]
        n = c["n"]
        if not n:
            continue
        w = f"{pct(c['win_ok'], c['win_n'])}% (n={c['win_n']})"
        bl = f"{pct(c['block_ok'], c['block_n'])}% (n={c['block_n']})"
        ga = f"{pct(c['gave_away'], c['clean_n'])}% (n={c['clean_n']})"
        print(f"{bkt[0]:>3}-{bkt[1]:<4} | {n:>5} | {w:>14} | {bl:>14} | {ga:>12}")
        out["buckets"][f"{bkt[0]}-{bkt[1]}"] = {
            "n": n, "win_n": c["win_n"], "take_win_pct": pct(c["win_ok"], c["win_n"]),
            "block_n": c["block_n"], "block_pct": pct(c["block_ok"], c["block_n"]),
            "clean_n": c["clean_n"], "gave_away_pct": pct(c["gave_away"], c["clean_n"]),
            "double_threat_n": c["double_n"]}
    o = overall
    out["overall"] = {
        "n": o["n"], "win_n": o["win_n"],
        "take_win_pct": pct(o["win_ok"], o["win_n"]),
        "block_n": o["block_n"], "block_pct": pct(o["block_ok"], o["block_n"]),
        "clean_n": o["clean_n"],
        "gave_away_pct": pct(o["gave_away"], o["clean_n"]),
        "double_threat_n": o["double_n"]}
    print()
    print(f"OVERALL: take-win {out['overall']['take_win_pct']}%  "
          f"block {out['overall']['block_pct']}%  "
          f"gave-away {out['overall']['gave_away_pct']}%")

    d = out["displayed_pct"] = {
        "n_win_positions": disp["n"],
        "model_top1_is_winning_column_pct": pct(disp["top_is_win"], disp["n"]),
        "mean_displayed_pct_on_model_top": round(sum(disp["pct_on_top"]) /
                                                 max(len(disp["pct_on_top"]), 1), 1),
        "mean_displayed_pct_on_best_winning_col": round(
            sum(disp["pct_on_win"]) / max(len(disp["pct_on_win"]), 1), 1),
        "pct_of_win_positions_where_top_is_over_50pct": pct(disp["pct_top_over_50"], disp["n"]),
        "pct_of_win_positions_where_top_is_100pct": pct(disp["pct_top_at_100"], disp["n"]),
        "n_positions_with_top_at_100pct": disp["n_100"],
    }
    print()
    print("=== displayed % vs tactical truth ===")
    for k, v in d.items():
        print(f"   {k:>48}: {v}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
