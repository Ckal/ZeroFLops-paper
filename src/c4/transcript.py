"""Byte encoding for Connect-Four transcripts.

TRAINING FORMAT (what the model sees) -- see the block comment below
-------------------------------------------------------------------
One line per game, one fixed-width 10-byte record per ply:

    [h0 h1 h2 h3 h4 h5 h6] [last] [turn] [move]
     0x30..0x36            0x2E|A..G  X|O    A..G

Column heights + side to move reconstruct the whole 42-cell board exactly
(Connect Four columns fill strictly alternately from the bottom). Max line
length 421 bytes (42 plies x 10 + newline).

VISUAL FORMAT (rendering + round-trip tests)
--------------------------------------------
    <42 grid bytes: 6 rows x 7 cells, top-to-bottom, left-to-right>
    <move bytes 0x41..0x47>  <0x0A>
"""

from __future__ import annotations

from . import ROWS, COLS, EMPTY, P1, P2, TOK_EMPTY, TOK_X, TOK_O
from .board import Board, _BoardBuilder, _set_cell


COL_BYTES = b"ABCDEFG"  # 0x41..0x47
COL_BYTE_OFFSET = 0x41  # ord('A')
HEIGHT_OFFSET = 0x30    # ord('0')
NO_LAST = 0x2E          # ord('.')
GRID_LEN = ROWS * COLS  # 42
PLY_LEN = 10            # 7 heights + last + turn + move
MAX_PLIES = GRID_LEN
MAX_LINE = PLY_LEN * MAX_PLIES + 1  # 421 bytes including the newline


# --------------------------------------------------------------------------
# TRAINING FORMAT -- this is what the model actually sees
# --------------------------------------------------------------------------
#
# One line per game, one fixed-width record per ply:
#
#     [h0 h1 h2 h3 h4 h5 h6] [last] [turn] [move]     = 10 bytes per ply
#      0x30..0x36            0x2E|A..G  X|O          A..G
#
# hN is the height of column N (0..6, stored as ASCII '0'..'6' so the corpus
# stays printable). `last` is the column of the previous move ('.' if none).
# `turn` is 'X' for P1 or 'O' for P2. `move` is the column just played.
#
# WHY HEIGHTS ARE A COMPLETE STATE: in Connect Four a column fills strictly
# alternately from the bottom, so a column of height 3 is always X,O,X bottom-up.
# Seven heights plus the side to move therefore reconstruct the entire 42-cell
# board exactly. That is an 8-byte state, not a 42-byte grid -- an 80x smaller
# alphabet footprint for the same information.
#
# The move byte at offset 10*i+9 is the supervised target: predict which column
# the side to move plays, given the position plus recent history in the
# left-context. Every ply is a real decision, so one 421-byte line yields 42
# training targets.


def encode_ply(b: Board, move: int) -> bytes:
    """One 10-byte ply record for `move` played on board `b` (BEFORE the move)."""
    if not (0 <= move < COLS):
        raise ValueError(f"move out of range: {move}")
    hs = bytes(HEIGHT_OFFSET + h for h in b.heights)
    last = NO_LAST if b.last_move is None else COL_BYTE_OFFSET + b.last_move
    turn = TOK_X if b.turn == P1 else TOK_O
    return hs + bytes((last, turn, COL_BYTE_OFFSET + move))


def encode_game(moves: list[int]) -> bytes:
    """Encode a full move list into one training line (ends with newline)."""
    b = Board()
    out = bytearray()
    for m in moves:
        out += encode_ply(b, m)
        b.apply_move(m)
    out.append(0x0A)
    return bytes(out)


def target_offsets() -> list[int]:
    """Byte offsets inside one line that carry a supervised move target."""
    return list(range(PLY_LEN - 1, MAX_LINE - 1, PLY_LEN))


def plies_of(line: bytes) -> int:
    return len(line.rstrip(b"\n")) // PLY_LEN


def decode_game(line: bytes) -> list[int]:
    """Inverse of encode_game: recover the move list from a training line."""
    if not line.endswith(b"\n"):
        raise ValueError("training line must end with newline")
    payload = line[:-1]
    if len(payload) % PLY_LEN != 0:
        raise ValueError(f"line length {len(payload)} is not a multiple of {PLY_LEN}")
    return [payload[i + PLY_LEN - 1] - COL_BYTE_OFFSET
            for i in range(0, len(payload), PLY_LEN)]


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
