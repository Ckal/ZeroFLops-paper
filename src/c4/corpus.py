"""Self-play corpus construction for Connect Four.

Produces winners-only training lines in the fixed-width ply format the V6
byte-LM consumes (see src/c4/transcript.py for the byte layout):

    <10 bytes per ply><newline>    max 421 bytes

Each ply record carries the full position (7 column heights + side to move),
so every move byte is a well-posed prediction target: given the position and
the preceding plies in its left-context, name the column played.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

from .board import Board, Outcome
from .transcript import encode_game


Game = Tuple[List[int], int]   # moves list + outcome value
Winners = List[Tuple[List[int], int]]  # filtered to wins: P1 -> +1, P2 -> -1

# A policy gets the board AND the move history: the byte-LM is a sequence
# model, so its context is the whole game, not just the current position.
Policy = Callable[[Board, List[int]], int]


def play_random_game(rng: random.Random,
                     policy: Policy | None = None,
                     ) -> Game:
    """Play one game under ``policy`` (default: uniform random).

    Returns (moves, outcome_value) where outcome_value is +1 if P1 won,
    -1 if P2 won, 0 on a draw. Empty boards that win on move 0 are impossible
    in Connect Four, so the loop is safe.
    """
    b = Board()
    moves: List[int] = []
    while b.outcome() == Outcome.ONGOING:
        valid = b.valid_actions()
        if not valid:
            break
        if policy is None:
            c = rng.choice(valid)
        else:
            c = policy(b, moves)
        moves.append(c)
        b.apply_move(c)
    o = b.outcome()
    val = 0
    if o == Outcome.WIN_P1:
        val = +1
    elif o == Outcome.WIN_P2:
        val = -1
    return (moves, val)


def play_games(n: int,
               rng: random.Random,
               policy: Policy | None = None,
               ) -> List[Game]:
    """Play ``n`` games under ``policy`` (default uniform random)."""
    return [play_random_game(rng, policy) for _ in range(n)]


def winners_only(games: List[Game]) -> Winners:
    """Keep only decided games; drop draws. Each winner is (moves, sign)."""
    return [(m, v) for m, v in games if v != 0]


def games_to_lines(decided: List[Tuple[List[int], int]]) -> List[bytes]:
    """Encode each decided game to a one-line training corpus row.

    Line = fixed-width 10-byte records, one per ply, plus a trailing newline.
    Each record carries the full position (7 column heights + side to move),
    so the move byte at offset 9 is a well-posed prediction target.
    """
    return [encode_game(moves) for moves, _ in decided]


def append_corpus(path, decided: List[Tuple[List[int], int]]) -> int:
    """Append ``decided`` games to ``path``, one per line. Returns row count."""
    rows = 0
    with open(path, "ab") as f:
        for line in games_to_lines(decided):
            f.write(line)  # encode_game already ends with \n
            rows += 1
    return rows


def stats(lines: List[bytes]) -> dict:
    if not lines:
        return {"count": 0}
    lens = [len(l) for l in lines]
    return {
        "count": len(lines),
        "min_len": min(lens),
        "max_len": max(lens),
        "mean_len": round(sum(lens) / len(lens), 1),
        "byte_total": sum(lens),
    }
