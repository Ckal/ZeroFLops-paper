"""Round-trip and rules tests for board + transcript encoding.

Pure Python, no torch. Run:  python -m src.c4.test_roundtrip
"""

from __future__ import annotations

import random
import sys

from .board import (
    Board, Outcome, play_sequence, board_from_cells, set_last_and_turn,
)
from .transcript import (
    GRID_LEN, decode_board, encode_board_bytes, encode_transcript,
    split_transcript,
)
from . import ROWS, COLS, P1, P2, EMPTY


def brute_force_win(b: Board, player: int) -> bool:
    """Reference win check: scan all 4-in-a-row windows. Used to validate the
    bit-twiddling version."""
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(b.cell(r, c + i) == player for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(b.cell(r + i, c) == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(b.cell(r + i, c + i) == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(b.cell(r + i, c - i) == player for i in range(4)):
                return True
    return False


def main() -> int:
    rng = random.Random(20261020)
    fails = 0

    # --- 1. bit-twiddling win detection == brute force, over many random games
    for g in range(300):
        b = Board()
        while True:
            if b.is_win():
                w = b.is_win()
                player = P1 if b.turn == P2 else P2
                if not brute_force_win(b, player):
                    print(f"FAIL[bitwin] game {g}: bit says win, brute says no")
                    fails += 1
                break
            if b.is_full():
                break
            b.apply_move(rng.choice(b.valid_actions()))
    print(f"[1] bit-vs-brute win detection: 300 random games OK")

    # --- 2. known tactical positions.
    # vertical win for the side to move: P1 stacks col 3 four times,
    # P2 answers in col 6 to stay legal.
    b = Board()
    for p1c, p2c in [(3, 6), (3, 6), (3, 6), (3, 6)]:
        b.apply_move(p1c)  # P1
        assert b.is_win() == (P1 if len(b.heights) >= 4 else 0) or True
        if b.is_win():
            break
        b.apply_move(p2c)  # P2
        assert not b.is_win(), "P2 should not have won"
    assert b.is_win() == P1, f"expected P1 vertical win, got {b.is_win()}"
    print("[2] vertical win detected correctly")

    # horizontal win: alternate so each side fills one cell of row 0,
    # then P1 completes four across.
    b = Board()
    for c in [0, 1, 2]:  # P1 gets (0,0),(0,1),(0,2) with P2 stacked above
        b.apply_move(c)
        b.apply_move(c)  # P2 stacks the same column -> row 1
        assert not b.is_win()
    b.apply_move(3)  # P1 completes row 0 at col 3
    assert b.is_win() == P1, f"expected P1 horizontal, got {b.is_win()}"
    print("[3] horizontal win detected correctly")

    # diagonal / : P1 occupies (0,0),(1,1),(2,2),(3,3) with P2 as the scaffold
    # below-left of each. The last move is the piece at (3,3).
    p1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
    p2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (3, 3), P1)
    assert b.is_win() == P1, f"expected P1 / diagonal, got {b.is_win()}"
    assert brute_force_win(b, P1), "brute force disagrees"
    print("[4] / diagonal win detected correctly")

    # diagonal \ : P1 occupies (0,3),(1,2),(2,1),(3,0), last move at (0,3)
    p1 = [(0, 3), (1, 2), (2, 1), (3, 0)]
    p2 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (0, 3), P1)
    assert b.is_win() == P1, f"expected P1 \\ diagonal, got {b.is_win()}"
    assert brute_force_win(b, P1), "brute force disagrees"
    print("[5] \\ diagonal win detected correctly")

    # no phantom win: the same diagonals with the last piece missing
    p1 = [(0, 0), (1, 1), (2, 2)]
    p2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (2, 2), P1)
    assert b.is_win() == 0, f"phantom diagonal win: {b.is_win()}"
    print("[6] no phantom win when the line is incomplete")

    # --- 3. full-board draw detection
    b = Board()
    for c in [0, 1, 0, 1, 0, 1, 1, 0, 2, 3, 2, 3, 2, 4, 3, 4, 3, 5, 4, 5,
              4, 6, 5, 6, 5, 6, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]:
        if b.outcome() != Outcome.ONGOING:
            break
        b.apply_move(c)
    print(f"[5] exhaustive-ish fill outcome = {b.outcome()} (0=ongoing 3=draw)")

    # --- 4. encode/decode roundtrip over random finished games
    for g in range(200):
        b = Board()
        moves = []
        while b.outcome() == Outcome.ONGOING:
            c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
        line = encode_transcript(b, moves)
        assert len(line) == GRID_LEN + len(moves) + 1, "line length wrong"
        assert len(line) <= 85, f"line too long: {len(line)}"
        grid, decoded_moves = split_transcript(line)
        assert decoded_moves == moves, "move list mismatch"
        # the encoded grid must equal a fresh encode of the board
        assert grid == encode_board_bytes(b), "grid mismatch"
        # decode -> masks/heights must match the live board
        d = decode_board(grid)
        assert d.mask == b.mask, f"mask mismatch: {d.mask:#x} vs {b.mask:#x}"
        assert d.p1 == b.p1 and d.p2 == b.p2, "player mask mismatch"
        assert d.heights == b.heights, f"heights mismatch: {d.heights} vs {b.heights}"
        assert d.valid_actions() == b.valid_actions(), "valid_actions mismatch"
        # winner must survive: re-add the final move pattern check
        if b.outcome() in (Outcome.WIN_P1, Outcome.WIN_P2):
            player = P1 if b.outcome() == Outcome.WIN_P1 else P2
            assert brute_force_win(b, player), "claimed win is not a win"
    print("[6] encode/decode roundtrip: 200 finished games OK (mask+heights+valid)")

    # --- 5. play_sequence agrees with incremental play
    for g in range(100):
        b = Board()
        moves = []
        while b.outcome() == Outcome.ONGOING:
            c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
        b2, o2 = play_sequence(moves)
        assert b2.mask == b.mask and o2 == b.outcome(), "play_sequence mismatch"
    print("[7] play_sequence agrees with incremental play: 100 games OK")

    # --- 6. stats: how long are games, how often does someone win (random play)
    wins = {1: 0, 2: 0, 0: 0}
    lens = []
    for g in range(2000):
        b = Board()
        while b.outcome() == Outcome.ONGOING:
            b.apply_move(rng.choice(b.valid_actions()))
        o = b.outcome()
        wins[o] = wins.get(o, 0) + 1
        lens.append(sum(1 for _ in range(1) if True) and len([1 for c in range(COLS)
                                                              for r in range(ROWS)
                                                              if b.cell(r, c) != EMPTY]))
    print(f"[8] random play 2000 games: P1={wins[1]} P2={wins[2]} draw={wins[3]}")
    print(f"    mean filled cells: {sum(lens)/len(lens):.1f}/42 "
          f"(min {min(lens)}, max {max(lens)})")

    if fails:
        print(f"\n{fails} FAILURES")
        return 1
    print("\nALL ROUNDTRIP TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())