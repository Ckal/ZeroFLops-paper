"""Quantify the blast radius of the is_full() bug for the MODEL policy.

The bug terminated a game as a DRAW as soon as the top cell of the last
column (internal c6) was filled -- with up to 36 of 42 cells still empty.

This measures, with the REAL shipped checkpoint:
  * how often the model's own play triggers the bug
  * how long the games really are vs. how long the buggy code recorded
  * the model's column usage (the user saw column 7 repeatedly)
"""
from __future__ import annotations

import collections
import statistics
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, ROWS, COLS
from src.c4.ui_server import _load_agent


def old_is_full(mask: int) -> bool:
    """The buggy one-bit implementation."""
    return bool((mask >> (ROWS * COLS - 1)) & 1)


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else (
        "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    agent = _load_agent(model, device="cpu")

    lens_old: list[int] = []
    lens_new: list[int] = []
    trunc_draws = 0
    colcount: collections.Counter = collections.Counter()
    early_examples: list[tuple[list[int], int]] = []

    for g in range(n):
        b = Board()
        m: list[int] = []
        while True:
            legal = b.valid_actions()
            if not legal:
                break
            c = agent.choose(b, list(m), use_hard=True)
            b.apply_move(c)
            m.append(c)
            colcount[c] += 1
            if b.is_win():
                break
        lens_new.append(len(m))

        # Replay under the OLD buggy terminal test.
        b2 = Board()
        m2: list[int] = []
        for c in m:
            b2.apply_move(c)
            m2.append(c)
            if b2.is_win() or old_is_full(b2.mask):
                break
        lens_old.append(len(m2))
        if old_is_full(b2.mask) and not b2.is_win():
            trunc_draws += 1
            if len(early_examples) < 5:
                early_examples.append((list(m2), bin(b2.mask).count("1")))

    print("=" * 72)
    print(f"MODEL-vs-MODEL GAMES  (n={n})   model={model}")
    print("=" * 72)
    print(f"  mean length OLD (buggy): {statistics.mean(lens_old):.1f} plies")
    print(f"  mean length NEW (fixed): {statistics.mean(lens_new):.1f} plies")
    print(f"  games cut short by the bug (fake DRAW, no four): "
          f"{trunc_draws} / {n} = {100.0*trunc_draws/n:.1f}%")
    print()
    tot = sum(colcount.values())
    print("  model's column usage:")
    for c in range(7):
        print(f"    c{c} (display column {c+1}): {colcount[c]:5d}  "
              f"{100.0*colcount[c]/tot:5.1f}%")
    print()
    print(f"  -> c6 (display column 7) usage: {100.0*colcount[6]/tot:.1f}%")
    print()
    if early_examples:
        print("  examples of games the bug truncated:")
        for mv, filled in early_examples:
            print(f"    {filled:2d}/42 cells filled, moves={mv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
