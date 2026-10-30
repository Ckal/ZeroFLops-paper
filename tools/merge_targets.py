#!/usr/bin/env python3
"""Merge original + augmented targets sidecars for full 640k training."""

import sys
from pathlib import Path
import numpy as np


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Concatenate original + flipped targets")
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--flipped", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    
    args = ap.parse_args()
    
    if args.output is None:
        output_name = args.original.stem + "_augmented.npz"
        args.output = args.original.parent / output_name
    
    print(f"Loading original: {args.original}")
    orig = np.load(args.original, allow_pickle=False)
    
    print(f"Loading flipped: {args.flipped}")
    flip = np.load(args.flipped, allow_pickle=False)
    
    # Verify keys match
    assert set(orig.files) == set(flip.files), "Keys mismatch!"
    
    # Stack vertically
    merged = {}
    for k in orig.files:
        orig_arr = orig[k]
        flip_arr = flip[k]
        
        print(f"  {k}: {orig_arr.shape} + {flip_arr.shape}")
        
        merged[k] = np.concatenate([orig_arr, flip_arr], axis=0)
        print(f"    → {merged[k].shape}")
    
    # Save combined
    print(f"\nSaving merged to: {args.output}")
    np.savez(args.output, **merged)
    
    # Final stats
    print("\n=== FINAL STATS ===")
    if 'target' in merged:
        n_games = merged['target'].shape[0]
        print(f"Total games: {n_games:,}")
        print(f"Target shape: {merged['target'].shape}")


if __name__ == "__main__":
    main()
