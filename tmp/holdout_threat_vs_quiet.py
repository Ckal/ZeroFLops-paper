#!/usr/bin/env python3
"""Split holdout agreement by threat-on-board vs quiet positions."""

import sys
from pathlib import Path
import numpy as np

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.transcript_v2 import decode_game
from src.c4.board import Board, is_win_state

HOLDOUT_DIR = ROOT / "runs" / "_probe" / "disjoint_holdout"
# Or use the published HF dataset if available locally:
# HOLDOUT_DIR = Path(r"C:\Users\<user>\AppData\Local\Temp\hfdataset\holdout")

def has_immediate_threat(board: Board) -> bool:
    """Check if current position threatens immediate win or requires blocking."""
    # Check for any column with 3 consecutive same-color pieces touching empty space
    # This is a simplified threat detector
    
    grid = board.grid
    N_ROWS, N_COLS = len(grid), len(grid[0]) if grid else (6, 7)
    
    for col in range(N_COLS):
        run_player = 0
        run_len = 0
        
        for row in range(N_ROWS):
            piece = grid[row][col] if row < N_ROWS and col < N_COLS else 0
            
            if piece == 0:
                # Empty above a run could be exploitable
                if run_len >= 3:
                    return True
                run_player = 0
                run_len = 0
            elif piece != run_player:
                if run_len >= 3 and run_player != 0:
                    return True
                run_player = piece
                run_len = 1
            else:
                run_len += 1
        
        # Check final run
        if run_len >= 3 and run_player != 0:
            return True
    
    return False

def analyze_holdout():
    corpus_path = HOLDOUT_DIR / "corpus.txt"
    targets_path = HOLDOUT_DIR / "targets.npz"
    
    if not corpus_path.exists():
        # Try HF dataset path
        HF_HOLDOUT = Path(r"C:\Users\<user>\AppData\Local\Temp\hfdataset\holdout")
        if (HF_HOLDOUT / "corpus.txt").exists():
            corpus_path = HF_HOLDOUT / "corpus.txt"
            targets_path = HF_HOLDOUT / "perply_targets.npz"
    
    print(f"Loading: {corpus_path}")
    print(f"Targets: {targets_path}")
    
    content = corpus_path.read_text(encoding="utf-8")
    lines = [l for l in content.strip().split(chr(10)) if l]
    targets = np.load(targets_path, allow_pickle=True)
    
    print(f"GAMES: {len(lines)} | TARGETS SHAPE: {targets['target'].shape}")
    
    threat_correct = 0
    threat_total = 0
    quiet_correct = 0
    quiet_total = 0
    
    for game_idx in range(min(len(lines), targets['target'].shape[0])):
        line = lines[game_idx]
        try:
            moves = decode_game(line)
            n_plies = targets['n_plies'][game_idx]
            
            board = Board()
            
            for ply_idx, col in enumerate(moves[:min(n_plies, len(moves))]):
                teacher = targets['teacher'][game_idx, ply_idx]
                
                if teacher == 0:  # unlabeled
                    board.place(col)
                    continue
                
                # Check threat status BEFORE move
                is_threat = has_immediate_threat(board)
                
                # Get predicted column (argmax of target distribution)
                pred_col = int(np.argmax(targets['target'][game_idx, ply_idx, 1:]))  # skip col 0 padding
                
                correct = (pred_col == col)
                
                if is_threat:
                    threat_correct += correct
                    threat_total += 1
                else:
                    quiet_correct += correct
                    quiet_total += 1
                
                board.place(col)
                
        except Exception as e:
            # Skip malformed
            pass
    
    print(f"\n=== RESULT ===")
    print(f"THREAT AGREE: {threat_correct}/{threat_total} = {threat_correct/threat_total*100:.1f}%" if threat_total > 0 else "No threats found")
    print(f"QUIET AGREE:  {quiet_correct}/{quiet_total} = {quiet_correct/quiet_total*100:.1f}%" if quiet_total > 0 else "No quiet positions found")
    
    if threat_total > 0 and quiet_total > 0:
        diff = (threat_correct/threat_total) - (quiet_correct/quiet_total)
        print(f"\nWHICH WINS? {'THREAT' if diff > 0 else 'QUIET'} by {abs(diff)*100:.1f}pp")

if __name__ == "__main__":
    analyze_holdout()
