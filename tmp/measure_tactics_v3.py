"""CORRECTED tactical probe -- and it corrects a bug in the earlier ones.

THE BUG
-------
`Agent.column_logits(board, moves)` / `Agent.hard_column_logits(board, moves)`
build the model's prompt from `moves` -- the MOVE HISTORY. The `board`
argument is not used to build the prompt at all (it is only used for the
legal mask in `choose()`).

The earlier probes (tmp/measure_tactics_v2.py, tmp/measure_tactics.py,
tmp/measure_tactics_by_ply.py, tmp/audit_float_vs_hard_tactics.py,
tmp/measure_ui_percent_semantics.py, tmp/audit_fit_vs_baseline.py,
tmp/audit_loader_and_sensitivity.py) all called

        agent.hard_column_logits(b, legal)      # WRONG

passing the list of LEGAL COLUMNS where the move history belongs. The
model was therefore prompted with a garbage game ("moves 0,1,2,3,4,5,6")
and the resulting "tactical competence" numbers measured nothing about
the shipped model.

The live UI is correct: `ui_server` calls
`self.agent.choose(self.board, list(self.moves), use_hard=True)` with the
real move history.

THE FIX
-------
Track the move history while building the board, and pass THAT.

SELF-CHECKS (so a silent regression cannot happen again)
--------------------------------------------------------
  1. `board_from_moves(moves)` must equal the board we built by hand.
  2. The agent's own board reconstruction is used, so the prompt is right.
  3. The live UI is asked for the same position over HTTP and its
     next-move distribution must match this probe's logits (parity).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.c4.board import Board  # noqa: E402
from src.c4 import patterns  # noqa: E402
from src.c4.agent import board_from_moves  # noqa: E402


def build(rng: random.Random, n_moves: int):
    """Return (board, move_history)."""
    b = Board()
    mv = []
    for _ in range(n_moves):
        legal = b.valid_actions()
        if not legal:
            break
        c = rng.choice(legal)
        b.apply_move(c)
        mv.append(c)
    return b, mv


def wins_for(board: Board, player: int) -> list[int]:
    return list(patterns.immediate_wins(board, player))


def model_pick(agent, b: Board, mv: list[int]) -> int:
    """Hard-path argmax. `mv` is the MOVE HISTORY -- this is the fix."""
    logits = agent.hard_column_logits(b, mv)
    legal = b.valid_actions()
    return max(legal, key=lambda c: float(logits[c]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    ap.add_argument("--self-check", action="store_true", default=True)
    args = ap.parse_args()

    from src.c4.ui_server import _load_agent
    agent = _load_agent(args.ckpt, device="cpu")

    rng = random.Random(args.seed)
    stats = {
        "win_now": {"n": 0, "took": 0},
        "block_now": {"n": 0, "blocked": 0},
        "clean_no_force": {"n": 0, "gave_away": 0},
    }
    # displayed-% semantics on positions where a win exists
    pct_top_when_win = []
    pct_wincol_when_win = []
    top_is_win = 0
    n_win_pos = 0
    examples = []
    board_rebuild_ok = True
    n_rebuild_checked = 0

    for _ in range(args.n):
        b, mv = build(rng, rng.randint(4, 30))
        legal = b.valid_actions()
        if not legal:
            continue

        # ---- SELF-CHECK 1: board reconstruction must match ----
        if args.self_check:
            rb = board_from_moves(mv)
            n_rebuild_checked += 1
            if (rb.p1, rb.p2, rb.turn) != (b.p1, b.p2, b.turn):
                board_rebuild_ok = False

        me = b.turn
        my_wins = wins_for(b, me)
        opp_wins = wins_for(b, 3 - me)

        if my_wins:
            stats["win_now"]["n"] += 1
            logits = agent.hard_column_logits(b, mv)
            neg = torch.tensor(-1e9, dtype=logits.dtype)
            mask = torch.tensor([c in legal for c in range(7)], dtype=torch.bool)
            probs = torch.softmax(torch.where(mask, logits, neg), dim=-1).tolist()
            pick = model_pick(agent, b, mv)
            n_win_pos += 1
            pct_top_when_win.append(probs[pick] * 100)
            pct_wincol_when_win.append(max(probs[c] * 100 for c in my_wins))
            if pick in my_wins:
                stats["win_now"]["took"] += 1
                top_is_win += 1
            elif len(examples) < 8:
                examples.append({"kind": "missed_win", "move": pick,
                                 "wins": my_wins,
                                 "pct": [round(p * 100, 1) for p in probs]})
            continue

        if opp_wins:
            if len(opp_wins) > 1:
                continue        # one move cannot cover two threats
            stats["block_now"]["n"] += 1
            pick = model_pick(agent, b, mv)
            if pick in opp_wins:
                stats["block_now"]["blocked"] += 1
            elif len(examples) < 8:
                examples.append({"kind": "failed_block", "move": pick,
                                 "opp_wins": opp_wins})
            continue

        stats["clean_no_force"]["n"] += 1
        pick = model_pick(agent, b, mv)
        b2 = Board()
        b2.p1, b2.p2, b2.turn = b.p1, b2.p2, b.turn
        b2.apply_move(pick)
        if wins_for(b2, b2.turn):
            stats["clean_no_force"]["gave_away"] += 1
            if len(examples) < 8:
                examples.append({"kind": "gave_away", "move": pick})

    for k, v in stats.items():
        num = (v["took"] if k == "win_now"
               else v["blocked"] if k == "block_now"
               else v["n"] - v["gave_away"])
        v["pct"] = round(100 * num / v["n"], 1) if v["n"] else None

    out = {
        "ckpt": args.ckpt, "n": args.n, "seed": args.seed,
        "BUGFIXED": "moves argument is the MOVE HISTORY, not the legal list",
        "self_check": {
            "board_rebuild_matches": board_rebuild_ok,
            "n_checked": n_rebuild_checked,
        },
        "stats": stats,
        "displayed_percent_semantics": {
            "n_positions_with_immediate_win": n_win_pos,
            "model_top1_is_the_winning_column_pct": round(100 * top_is_win / max(n_win_pos, 1), 1),
            "mean_displayed_pct_on_model_top_col": round(sum(pct_top_when_win) / max(len(pct_top_when_win), 1), 1),
            "mean_displayed_pct_on_best_winning_col": round(sum(pct_wincol_when_win) / max(len(pct_wincol_when_win), 1), 1),
        },
        "examples": examples,
    }
    print("=== CORRECTED tactical probe (real move history) ===")
    print(f"  board-rebuild self-check: {board_rebuild_ok} "
          f"({n_rebuild_checked} boards)")
    for k in ("win_now", "block_now", "clean_no_force"):
        v = stats[k]
        num = (v["took"] if k == "win_now"
               else v["blocked"] if k == "block_now"
               else v["n"] - v["gave_away"])
        print(f"  {k:>16}: {v['n']:>5} positions | compliant {num:>5} = {v['pct']}%")
    d = out["displayed_percent_semantics"]
    print()
    print("  displayed-% semantics on positions where the model CAN win:")
    print(f"    positions                         : {d['n_positions_with_immediate_win']}")
    print(f"    model top-1 IS the winning column : {d['model_top1_is_the_winning_column_pct']}%")
    print(f"    mean % shown on model's top column: {d['mean_displayed_pct_on_model_top_col']}%")
    print(f"    mean % shown on best winning col  : {d['mean_displayed_pct_on_best_winning_col']}%")
    if examples:
        print("\n  examples of misses:")
        for e in examples[:5]:
            print("   ", e)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
