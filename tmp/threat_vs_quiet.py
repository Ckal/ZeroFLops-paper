#!/usr/bin/env python3
"""Measure holdout agreement split by threat-on-board vs quiet positions."""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.transcript_v2 import decode_game
from src.c4.board import Board

def has_immediate_threat(board: Board) -> bool:
    """Check if position has any 3-in-a-row that could become a winning threat.
    
    A "threat" position is one where some player has 3 consecutive pieces
    in any direction, meaning they're on the verge of winning (need only 1 more).
    These are the positions where "win now / block now" tactics dominate.
    """
    for target_mask in [board.p1, board.p2]:
        if target_mask == 0:
            continue
            
        for r in range(6):
            for c in range(7):
                # Skip empty cells
                bit = r * 7 + c
                if not (target_mask & (1 << bit)):
                    continue
                
                player_bit = target_mask & (1 << bit)
                
                # Count consecutive pieces in all 4 directions through this cell
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                    count = 0
                    
                    # Positive direction
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 6 and 0 <= nc < 7:
                        if target_mask & (1 << (nr * 7 + nc)):
                            count += 1
                        else:
                            break
                        nr += dr
                        nc += dc
                    
                    # Negative direction
                    nr, nc = r - dr, c - dc
                    while 0 <= nr < 6 and 0 <= nc < 7:
                        if target_mask & (1 << (nr * 7 + nc)):
                            count += 1
                        else:
                            break
                        nr -= dr
                        nc -= dc
                    
                    # Total including current cell
                    if count + 1 >= 3:
                        return True
    
    return False


def main():
    # Load holdout from HF dataset (already verified LFS-tracked)
    holdout_dir = Path(r"C:\Users\Chris4K\AppData\Local\Temp\hfdataset\holdout")
    
    print(f"Loading holdout from: {holdout_dir}")
    
    # Read as binary - the file is NOT text, it's v2-encoded byte sequences
    corpus_bytes = (holdout_dir / "corpus.txt").read_bytes()
    targets = np.load(holdout_dir / "perply_targets.npz", allow_pickle=False)
    
    # Split by newlines in binary
    games_binary = [g for g in corpus_bytes.split(b'\n') if g]
    n_games = min(len(games_binary), targets['target'].shape[0])
    
    print(f"HOLDOUT: {len(games_binary)} games")
    print(f"Targets shape: {targets['target'].shape}")
    print(f"Processing first {n_games} games")
    
    threat_correct = 0
    threat_total = 0
    quiet_correct = 0
    quiet_total = 0
    decode_errors = 0
    
    for game_idx in range(n_games):
        game_bytes = games_binary[game_idx]
        n_plies_limit = targets['n_plies'][game_idx]
        
        try:
            # Decode game - passes bytes directly (not decoded string)
            # transcript_v2.decode_game accepts bytes or str
            moves = decode_game(game_bytes)
            
            n_plies = min(n_plies_limit, len(moves), 19)  # cap at 19 (target array size)
            board = Board()
            
            for ply_idx in range(n_plies):
                col = moves[ply_idx]
                teacher = targets['teacher'][game_idx, ply_idx]
                
                # Skip unlabeled plies
                if teacher == 0:
                    board.apply_move(col)
                    continue
                
                # Check threat status BEFORE this move is played
                is_threat = has_immediate_threat(board)
                
                # Get model's predicted column
                # target[game, ply] shape is (19, 7) - prob distribution over columns 0-6 for each of up to 19 plies
                # We want the argmax over the current ply only
                pred_col = int(np.argmax(targets['target'][game_idx, ply_idx]))
                
                correct = int(pred_col == col)
                
                if is_threat:
                    threat_correct += correct
                    threat_total += 1
                else:
                    quiet_correct += correct
                    quiet_total += 1
                
                board.apply_move(col)
            
        except Exception as e:
            # Silently skip malformed games
            decode_errors += 1
            if decode_errors <= 5:
                print(f"  Skipped game {game_idx}: {e}")
    
    # Report results
    print("\n" + "="*60)
    print("HOLDOUT AGREEMENT BY POSITION TYPE")
    print("="*60)
    
    if threat_total > 0:
        threat_rate = threat_correct / threat_total * 100
        print(f"🔥 Threat positions: {threat_correct:5d}/{threat_total:5d} = {threat_rate:6.1f}%")
    else:
        print("🔥 Threat positions: No threats detected (detector may be too strict)")
    
    if quiet_total > 0:
        quiet_rate = quiet_correct / quiet_total * 100
        print(f"✖️ Quiet positions:  {quiet_correct:5d}/{quiet_total:5d} = {quiet_rate:6.1f}%")
    else:
        print("✖️ Quiet positions: No quiet positions found")
        
    if decode_errors > 0:
        print(f"⚠️ Decode errors: {decode_errors} games skipped")
    
    # The verdict
    if threat_total > 0 and quiet_total > 0:
        diff = threat_rate - quiet_rate
        winner = "THREAT 🔥" if diff > 0 else "QUIET ✖️"
        
        print("\n" + "="*60)
        print(f"RESULT: {winner} by {abs(diff):.1f}pp")
        print("="*60)
        
        if diff > 5:
            print("\n📊 This CONFIRMS the commenter's hypothesis:")
            print("   The AI excels at tactical win/block moves (memorized)")
            print("   but is WEAKER at positional quiet play.")
            print(f"\n   • Threat agreement ({threat_rate:.1f}%) >> Quiet agreement ({quiet_rate:.1f}%)")
            print("   • Gap of {diff:.1f}% means the model learned patterns, not strategy")
        else:
            print("\n📊 Surprising result - both position types similar:")
            print("   The model may have genuine generalization ability.")

if __name__ == "__main__":
    main()
