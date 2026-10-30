"""CONTROL v2 -- corrected tactical control harness.

Runs the MODEL and three model-free controls (minimax, lookahead, random)
on the IDENTICAL set of positions, using the IDENTICAL position generator
as tmp/measure_tactics_v4.py, so every percentage is directly
comparable.

Two defects in tmp/control_tactics.py are fixed here.

DEFECT 1 -- incomplete board copy (vacuous give-away row)
---------------------------------------------------------
The original `clone()` copied only p1, p2 and turn. It did NOT copy
`mask` or `heights`, which are the two fields `Board.apply_move()` uses to
decide WHERE a disc lands:

    row = self.heights[col]                   # always 0 in the broken clone
    self.mask |= bit                          # mask was empty
    assert self.cell(ROWS - 1, col) == EMPTY  # always true on an empty mask

So the control's "move" landed in row 0 of the chosen column, ignoring
gravity. In the usual case that column already held a disc in row 0, so
`self.p1 |= bit` was a NO-OP: the board did not change at all, the
opponent's disc set was unchanged, and the opponent could never gain a
NEW immediate win. Every policy therefore scored 100% on
"dont hand over a win" -- random included. That row measured nothing.

This script reports the broken-copy number next to the fixed-copy number
so the defect and the fix are both visible in one artifact.

DEFECT 2 -- different position populations
------------------------------------------
The original control drew boards with `random_board(rng, rng.randint(4, 30))`
and kept a move history that was never appended to. The model probe
(v4) used a different generator. Percentages from the two scripts were
not directly comparable. Here ONE generator produces the positions and
every policy sees the same board AND the same move history.

FEASIBILITY (vacuous-zero-rate-trap)
-------------------------------------
A give-away rate is only meaningful if giving away was POSSIBLE. For each
clean position the script also counts how many legal moves would have
created an opponent immediate win (`dangerous_moves`). `gave_away_pct` is
reported together with:
  * n_clean_positions_where_giving_away_was_possible
  * pct_of_clean_positions_with_at_least_one_dangerous_move
If that is 0, the give-away rate is VACUOUS and must not be read as
success.
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


def wins_for(board: Board, player: int) -> list[int]:
    return list(patterns.immediate_wins(board, player))


def copy_full(b: Board) -> Board:
    """Exact copy of every field apply_move() touches."""
    c = Board()
    c.p1 = b.p1
    c.p2 = b.p2
    c.mask = b.mask
    c.heights = list(b.heights)
    c.turn = b.turn
    c.last_move = b.last_move
    return c


def copy_broken(b: Board) -> Board:
    """The ORIGINAL control clone: p1, p2, turn only (no mask, no heights)."""
    c = Board()
    c.p1 = b.p1
    c.p2 = b.p2
    c.turn = b.turn
    return c


def gave_away(b: Board, move: int, me: int, opp: int, copier) -> bool:
    """True iff `move` hands the opponent a win it did not have before.

    copier is copy_full (correct) or copy_broken (the original defect).
    """
    b2 = copier(b)
    b2.apply_move(move)
    we_won = bool(wins_for(b2, me))
    opp_now = wins_for(b2, opp)
    return (not we_won) and bool(opp_now)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.opponents import minimax_policy, lookahead_policy

    model_agent = None
    if args.ckpt:
        from src.c4.ui_server import _load_agent
        model_agent = _load_agent(args.ckpt, device="cpu")

    def model_pick(b: Board, hist: list[int]) -> int:
        logits = model_agent.hard_column_logits(b, hist)
        legal = b.valid_actions()
        return max(legal, key=lambda c: float(logits[c]))

    # ONE generator -> every policy sees the same board and the same history.
    rng = random.Random(args.seed)
    stats = {name: {"win_now": {"n": 0, "took": 0},
                    "block_now": {"n": 0, "blocked": 0},
                    "clean": {"n": 0, "gave_away": 0,
                              "gave_away_broken": 0,
                              "gave_away_when_possible": 0,
                              "dangerous_positions": 0,
                              "dangerous_moves": 0,
                              "broken_board_unchanged": 0}}
              for name in ("model", "minimax", "lookahead", "random")}

    n_positions = 0
    for _ in range(args.n):
        b = Board()
        hist: list[int] = []
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
        n_positions += 1

        me, opp = b.turn, 3 - b.turn
        my_wins = wins_for(b, me)
        opp_wins = wins_for(b, opp)

        picks = {
            "minimax": minimax_policy(b, hist),
            "lookahead": lookahead_policy(b, hist),
            "random": rng.choice(legal),
        }
        if model_agent is not None:
            picks["model"] = model_pick(b, hist)

        if my_wins:
            for name, p in picks.items():
                s = stats[name]["win_now"]
                s["n"] += 1
                s["took"] += int(p in my_wins)
            continue
        if len(opp_wins) == 1:
            for name, p in picks.items():
                s = stats[name]["block_now"]
                s["n"] += 1
                s["blocked"] += int(p in opp_wins)
            continue
        if len(opp_wins) > 1:
            continue  # double threat: both sides have a win, not comparable

        # clean position: no immediate win for anyone
        dangerous_moves = set()
        for m in legal:
            if gave_away(b, m, me, opp, copy_full):
                dangerous_moves.add(m)
        n_dangerous = len(dangerous_moves)

        for name, p in picks.items():
            s = stats[name]["clean"]
            s["n"] += 1
            if n_dangerous > 0:
                s["dangerous_positions"] += 1
                # DECISIVE denominator: of the positions where giving away
                # was possible, how many did THIS policy actually give away?
                if p in dangerous_moves:
                    s["gave_away_when_possible"] += 1
            s["dangerous_moves"] += n_dangerous
            if gave_away(b, p, me, opp, copy_full):
                s["gave_away"] += 1
            if gave_away(b, p, me, opp, copy_broken):
                s["gave_away_broken"] += 1
            # did the broken clone leave the board untouched?
            bb = copy_broken(b)
            bb.apply_move(p)
            if (bb.p1, bb.p2) == (b.p1, b.p2):
                s["broken_board_unchanged"] += 1

    def pct(num, den):
        return round(100 * num / den, 1) if den else None

    out = {"n_positions": n_positions, "seed": args.seed,
           "ckpt": args.ckpt, "generator": "same as measure_tactics_v4",
           "policies": {}}
    print("=" * 74)
    print(f"CORRECTED CONTROL -- {n_positions} identical positions")
    print("=" * 74)
    for name, s in stats.items():
        if s["win_now"]["n"] == 0 and s["block_now"]["n"] == 0:
            continue
        w, bl, cl = s["win_now"], s["block_now"], s["clean"]
        row = {
            "take_win_pct": pct(w["took"], w["n"]),
            "take_win_n": w["n"],
            "block_pct": pct(bl["blocked"], bl["n"]),
            "block_n": bl["n"],
            "gave_away_pct": pct(cl["gave_away"], cl["n"]),
            "gave_away_n": cl["gave_away"],
            "gave_away_pct_BROKEN_COPY": pct(cl["gave_away_broken"], cl["n"]),
            "clean_n": cl["n"],
            # THE decisive number: conditional on giving away being possible
            "gave_away_when_possible_pct": pct(cl["gave_away_when_possible"],
                                               cl["dangerous_positions"]),
            "gave_away_when_possible_n": cl["gave_away_when_possible"],
            "gave_away_when_possible_denominator": cl["dangerous_positions"],
            "clean_positions_where_giving_away_was_possible": cl["dangerous_positions"],
            "pct_clean_positions_with_a_dangerous_move": pct(cl["dangerous_positions"], cl["n"]),
            "mean_dangerous_moves_per_clean_position": round(
                cl["dangerous_moves"] / cl["n"], 3) if cl["n"] else None,
            "pct_broken_copy_left_board_unchanged": pct(cl["broken_board_unchanged"], cl["n"]),
        }
        out["policies"][name] = row
        print(f"\n--- {name} ---")
        print(f"  take an immediate win              : {row['take_win_pct']:5.1f}%  "
              f"({w['took']}/{w['n']})")
        print(f"  block opponent win                 : {row['block_pct']:5.1f}%  "
              f"({bl['blocked']}/{bl['n']})")
        print(f"  dont hand over a win (FIXED copy)  : {row['gave_away_pct']:5.1f}%  "
              f"({cl['gave_away']}/{cl['n']})")
        print(f"  dont hand over a win (BROKEN copy) : {row['gave_away_pct_BROKEN_COPY']:5.1f}%  "
              f"({cl['gave_away_broken']}/{cl['n']})   <- the old control's row")
        print(f"  gave away WHEN POSSIBLE           : "
              f"{row['gave_away_when_possible_pct']:5.1f}%  "
              f"({cl['gave_away_when_possible']}/{cl['dangerous_positions']})")
        print(f"  FEASIBILITY: giving away possible in "
              f"{row['clean_positions_where_giving_away_was_possible']}/{cl['n']} clean "
              f"positions ({row['pct_clean_positions_with_a_dangerous_move']}%)")
        print(f"  FEASIBILITY: mean dangerous moves/clean position = "
              f"{row['mean_dangerous_moves_per_clean_position']}")
        print(f"  broken copy left the board unchanged in "
              f"{row['pct_broken_copy_left_board_unchanged']}% of clean positions")

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    for name, row in out["policies"].items():
        feas = row["pct_clean_positions_with_a_dangerous_move"]
        label = "VACUOUS" if feas == 0 else "feasible"
        print(f"  {name:>10}: give-away {row['gave_away_pct']}% "
              f"(broken-copy reading {row['gave_away_pct_BROKEN_COPY']}%)  "
              f"feasibility={label} ({feas}% of clean positions)")
        print(f"  {name:>10}: gave away WHEN POSSIBLE "
              f"{row['gave_away_when_possible_pct']}%  "
              f"({row['gave_away_when_possible_n']}/"
              f"{row['gave_away_when_possible_denominator']})")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
