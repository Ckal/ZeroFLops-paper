#!/usr/bin/env python3
"""Mirror training data left-right to enforce spatial invariance.

Connect Four is perfectly symmetric: flipping columns 0↔6, 1↔5, 2↔4
creates an equivalent game with identical win/loss outcomes but different
board patterns. This forces the model to learn spatial invariance rather
than memorizing "column X is good".

Addresses "outer column weakness" mentioned in RESULTS.md and provides
free 2x data augmentation.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.transcript_v2 import (
    decode_game, encode_game, COL_BYTE_OFFSET, PLY_LEN, HEADER_LEN,
    COLUMN_FLAG, encode_column, decode_column
)


def mirror_column(c: int) -> int:
    """Flip column index left-right."""
    return 6 - c


def mirror_moves(moves: list[int]) -> list[int]:
    """Mirror a full game's moves left-right."""
    return [mirror_column(c) for c in moves]


def mirror_game_bytes(game_bytes: bytes) -> bytes:
    """Mirror v2-encoded game by re-encoding from decoded moves.
    
    The v2 format encodes each ply as:
      - 7 column bytes (0x81..0xFF, one per column with ownership info)
      - last_move byte 
      - turn byte
      - move byte (the column played)
    Total: PLY_LEN=10 bytes per ply
    
    To mirror: decode moves, flip them, re-encode.
    """
    # Decode to get moves
    moves = decode_game(game_bytes)  # list of column indices
    
    # Mirror the moves
    mirrored_moves = mirror_moves(moves)
    
    # Re-encode the mirrored game
    return encode_game(mirrored_moves)


def augment_corpus(input_file: Path, output_file: Path):
    """Read corpus, mirror each game, concatenate originals + mirrors."""
    print(f"Reading corpus from: {input_file}")
    
    # Read as binary (v2 transcript is raw bytes)
    content = input_file.read_bytes()
    
    # Split by newlines - each line is one game
    games = [g for g in content.split(b'\n') if len(g) >= PLY_LEN]
    
    print(f"Original games: {len(games)}")
    
    mirrored = []
    errors = 0
    
    for i, game_bytes in enumerate(games):
        try:
            # Mirror the game
            mirrored_game = mirror_game_bytes(game_bytes)
            mirrored.append(mirrored_game)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error mirroring game {i}: {e}")
    
    print(f"Mirrored games: {len(mirrored)}")
    print(f"Errors: {errors}")
    
    # Concatenate: originals first, then mirrors
    # Write with newlines between games
    output_lines = list(games) + mirrored
    
    total_bytes = sum(len(g) for g in output_lines) + len(output_lines)  # add newlines
    print(f"Output size: {len(games)} + {len(mirrored)} = {len(output_lines)} games, {total_bytes:,} bytes")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        for i, game in enumerate(output_lines):
            f.write(game)
            f.write(b'\n')
    
    print(f"Saved to: {output_file}")
    return len(games), len(mirrored), errors


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Mirror Connect Four training data")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="Input corpus.txt (v2 transcript format)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output augmented corpus")
    args = ap.parse_args()
    
    # Default: use the current training corpus if not specified on C:/Users/Chris4K/projects/ZeroFLops-paper/runs/_probe/perply_v23_full/corpus_320k_fixed.txt
    if args.corpus is None:
        args.corpus = ROOT / "runs" / "_probe" / "perply_v23_full" / "corpus_320k_fixed.txt"
    
    if args.output is None:
        output_name = args.corpus.stem + "_augmented.txt"
        args.output = args.corpus.parent / output_name
    
    if not args.corpus.exists():
        print(f"Error: corpus file not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)
    
    n_orig, n_mirror, errors = augment_corpus(args.corpus, args.output)
    
    print(f"\n=== SUMMARY ===")
    print(f"Original:     {n_orig:,} games")
    print(f"Augmented:  {n_mirror:,} mirrored games")  
    print(f"Total:      {n_orig + n_mirror:,} games (2x)")
    print(f"Errors:     {errors:,}")
