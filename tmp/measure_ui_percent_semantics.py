"""What SHOULD the UI's percentage mean, and what does it mean NOW?

The UI shows a softmax over the model's column logits. The user asks:
"should it say 100% if there is a chance of winning?"

There are two different questions hiding here:

  (Q1) When the model HAS an immediate win, does the panel show ~100% on
       the winning column?  If the model does not know it is winning, the
       percentage will be low -- but the board truth says "you can win
       right now". These are different facts and the UI must not conflate
       them.

  (Q2) Is the percentage a WIN PROBABILITY? It is not. It is
       softmax(model logits), i.e. the model's preference over columns.
       A 90% column means "the model strongly prefers this column", NOT
       "90% chance of winning the game".

This script measures (Q1) directly on the live hard path, and (Q2) by
showing the actual numbers on forced positions, so the semantics can be
stated precisely instead of assumed.
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


def random_board(rng: random.Random, n_moves: int) -> Board:
    b = Board()
    for _ in range(n_moves):
        legal = b.valid_actions()
        if not legal:
            break
        b.apply_move(rng.choice(legal))
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from src.c4.ui_server import _load_agent
    agent = _load_agent(args.ckpt, device="cpu")

    rng = random.Random(args.seed)
    n_win_pos = 0
    top1_is_win = 0
    top1_pct_when_win = []
    win_col_pct_when_win = []
    examples = []

    for _ in range(args.n):
        b = random_board(rng, rng.randint(6, 32))
        legal = b.valid_actions()
        if not legal:
            continue
        my_wins = list(patterns.immediate_wins(b, b.turn))
        if not my_wins:
            continue
        n_win_pos += 1
        logits = agent.hard_column_logits(b, legal)
        NEG = torch.tensor(-1e9, dtype=logits.dtype)
        mask = torch.tensor([c < 6 for c in b.heights], dtype=torch.bool)
        masked = torch.where(mask, logits, NEG)
        probs = torch.softmax(masked, dim=-1).tolist()
        top = max(range(7), key=lambda i: probs[i])
        if top in my_wins:
            top1_is_win += 1
        top1_pct_when_win.append(probs[top] * 100)
        win_col_pct_when_win.append(max(probs[c] * 100 for c in my_wins))
        if len(examples) < 6 and top not in my_wins:
            examples.append({
                "winning_cols": my_wins,
                "model_top_col": top,
                "model_top_pct": round(probs[top] * 100, 1),
                "winning_col_pct": round(max(probs[c] * 100 for c in my_wins), 1),
                "all_pct": [round(p * 100, 1) for p in probs],
            })

    out = {
        "n_positions_with_immediate_win": n_win_pos,
        "model_top1_is_the_winning_column_pct": round(100 * top1_is_win / max(n_win_pos, 1), 1),
        "mean_top1_pct_when_win_available": round(sum(top1_pct_when_win) / max(len(top1_pct_when_win), 1), 1),
        "mean_best_winning_col_pct": round(sum(win_col_pct_when_win) / max(len(win_col_pct_when_win), 1), 1),
        "examples_where_model_missed_the_win": examples,
    }
    print("\n=== What the UI percentage actually says on a forced win ===")
    print(f"  positions where the model CAN win right now : {n_win_pos}")
    print(f"  model's top-1 column IS the winning column  : "
          f"{out['model_top1_is_the_winning_column_pct']}%")
    print(f"  mean displayed % on the model's top column  : "
          f"{out['mean_top1_pct_when_win_available']}%")
    print(f"  mean displayed % on the BEST winning column : "
          f"{out['mean_best_winning_col_pct']}%")
    print()
    print("  => the % is a softmax over the model's own column preferences.")
    print("     It is NOT a win probability. On a position where the model")
    print("     has not learned the win, the board truth ('you can win right")
    print("     now') and the model's % disagree -- and they must both be")
    print("     shown, each labelled for what it is.")
    if examples:
        print("\n  examples where the model missed an available win:")
        for e in examples:
            print(f"    winning col {e['winning_cols']} | model picked "
                  f"{e['model_top_col']} at {e['model_top_pct']}% | "
                  f"winning col showed {e['winning_col_pct']}% | all={e['all_pct']}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
