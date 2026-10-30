#!/usr/bin/env python3
"""Flip target arrays left-right for mirror-augmented corpus.

For mirrored games, the column indices are flipped (0↔6, 1↔5, 2↔4).
The optimal move in the original game maps to the mirrored column in the mirrored game.
So we flip each row's target distribution accordingly.

Input: perply_targets.npz with shape (N, 19, 7) or similar
Output: new .npz with flipped targets
"""

import sys
from pathlib import Path
import numpy as np


def flip_columns(arr):
    """Flip column indices in target array: col 0↔6, 1↔5, 2↔4, 3 stays.
    
    Works on shapes like:
      (N,)          -> single moves as column index
      (N, T)        -> moves per ply  
      (N, T, 7)     -> probability distributions over 7 columns
    """
    if arr.ndim == 2:
        # Moves as indices: just flip them in-place
        return 6 - arr
    
    elif arr.ndim == 3 and arr.shape[2] == 7:
        # Probability distribution: permute the last axis
        perm = np.array([6, 5, 4, 3, 2, 1, 0])  # mirror permutation
        return np.take(arr, perm, axis=2)
    
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Flip targets for mirrored corpus")
    ap.add_argument("--targets", type=Path, required=True,
                    help="Input perply_targets.npz or similar")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output flipped targets (default: ..._flipped.npz)")
    
    args = ap.parse_args()
    
    if not args.targets.exists():
        print(f"Error: file not found: {args.targets}", file=sys.stderr)
        sys.exit(1)
    
    if args.output is None:
        args.output = args.targets.parent / (args.targets.stem + "_flipped.npz")
    
    print(f"Loading targets from: {args.targets}")
    data = np.load(args.targets, allow_pickle=False)
    
    print("Keys found:")
    for k in data.files:
        arr = data[k]
        shape_str = f"{arr.shape} {arr.dtype}"
        print(f"  {k}: {shape_str}")
    
    flipped_data = {}
    
    # 'target' is the main array to flip (per-ply probability distribution)
    if 'target' in data.files:
        original = data['target']
        print(f"\nFlipping target array...")
        flipped = flip_columns(original)
        assert flipped.shape == original.shape, "Shape mismatch!"
        flipped_data['target'] = flipped
    
    # Other arrays (ply_of, teacher, n_plies) can stay unchanged
    # These are scalar counts/indices, not column-dependent
    for k in data.files:
        if k != 'target':
            flipped_data[k] = data[k].copy()
    
    # Save
    print(f"Saving to: {args.output}")
    np.savez(args.output, **flipped_data)
    
    # Verify
    print("\nVerification:")
    for k in flipped_data:
        f_shape = flipped_data[k].shape
        o_shape = data[k].shape if k in data.files else "N/A"
        match = "✓" if f_shape == o_shape else "✗"
        print(f"  {match} {k}: shape={f_shape}")


if __name__ == "__main__":
    main()
