"""Byte encoding for Connect-Four transcripts.

A transcript is one game on one line. Format:

  <42 bytes: 6 rows x 7 cells, top-to-bottom, left-to-right>
  <move bytes, one per move: 0x41..0x47 = 'A'..'G' for columns 0..6>
  <0x0A '\\n'>

Max length: 42 + 42 + 1 = 85 bytes. The model context window is set to 96
(one game fits comfortably; any window that crosses a \\n is two games, which
the model learns to handle naturally as the loss mask only counts move bytes).

The grid phase is constant-byte ~68% of all bytes (empty cells dominate
opening). Move bytes carry the signal we care about.
"""

from __future__ import annotations

from . import ROWS, COLS, EMPTY, P1, P2, TOK_EMPTY, TOK_X, TOK_O
from .board import Board, _BoardBuilder, _set_cell


COL_BYTES = b"ABCDEFG"  # 0x41..0x47
COL_BYTE_OFFSET = 0x41  # ord('A')


def encode_board_bytes(b: Board) -> bytes:
    out = bytearray(ROWS * COLS)
    for r in range(ROWS):
        for c in range(COLS):
            cell = b.cell(r, c)
            if cell == EMPTY:
                out[r * COLS + c] = TOK_EMPTY
            elif cell == P1:
                out[r * COLS + c] = TOK_X
            else:
                out[r * COLS + c] = TOK_O
    return bytes(out)


def encode_moves(moves: list[int]) -> bytes:
    return bytes(COL_BYTE_OFFSET + m for m in moves)


def encode_transcript(b: Board, moves: list[int]) -> bytes:
    return encode_board_bytes(b) + encode_moves(moves) + b"\n"


GRID_LEN = ROWS * COLS  # 42


def decode_board(grid: bytes, dst: Board | None = None) -> Board:
    """Rebuild a Board's masks/heights from ROWS*COLS grid bytes.

    Placement order is not recovered (only the final masks and column heights),
    which is exactly what forward play needs. test_roundtrip.py verifies masks,
    heights and win detection all agree with the original board.
    """
    if len(grid) != GRID_LEN:
        raise ValueError(f"grid must be {GRID_LEN} bytes, got {len(grid)}")
    if dst is None:
        dst = Board()
    b = _BoardBuilder()
    for r in range(ROWS):
        for c in range(COLS):
            by = grid[r * COLS + c]
            if by == TOK_EMPTY:
                continue
            if by not in (TOK_X, TOK_O):
                raise ValueError(f"bad cell byte {by:#x} at ({r},{c})")
            _set_cell(b, r, c, P1 if by == TOK_X else P2)
    dst.mask = b.mask
    dst.p1 = b.p1
    dst.p2 = b.p2
    dst.heights = list(b.heights)
    return dst


def split_transcript(line: bytes) -> tuple[bytes, list[int]]:
    """Read one transcript line -> (final_grid_bytes, moves). Tests / play.py only."""
    if not line.endswith(b"\n"):
        raise ValueError("transcript must end with newline")
    payload = line[:-1]
    if len(payload) < GRID_LEN:
        raise ValueError(f"transcript too short: {len(payload)} bytes")
    grid = payload[:GRID_LEN]
    moves = []
    for by in payload[GRID_LEN:]:
        off = by - COL_BYTE_OFFSET
        if not (0 <= off <= 6):
            raise ValueError(f"bad move byte {by:#x}")
        moves.append(off)
    return grid, moves
