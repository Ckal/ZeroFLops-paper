# tools/test_hybrid_override.py
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
from src.c4.board import Board, P1, P2
from src.c4.opponents import get as get_opponent, _threat_after
from src.c4.duel import _load_hard_agent, _model_policy, duel, _explore

def hybrid_policy(agent, hard=True):
    """Overrides the neural net with win/block if available."""
    def policy(board: Board, moves):
        side = board.turn
        opp = P2 if side == P1 else P1
        legal = board.valid_actions()
        
        # 1. Win in 1 (Override)
        for c in legal:
            if _threat_after(board, c, side):
                return c
        # 2. Block opp in 1 (Override)
        for c in legal:
            if _threat_after(board, c, opp):
                return c
                
        # 3. Quiet position: let the neural net decide
        return int(agent.choose(board, moves, use_hard=hard))
        
    return policy

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tools.test_hybrid_override <checkpoint.pt>")
        sys.exit(1)
        
    ckpt_path = sys.argv[1]
    agent = _load_hard_agent(ckpt_path)
    
    rng = random.Random(42)
    heuristic = get_opponent("heuristic")
    # Match your RESULTS.md evaluation setup: opponent explores at 15%
    noisy_heuristic = _explore(heuristic, 0.15, rng) 
    
    print("\n--- Test 1: Standard Model vs 1-ply Heuristic ---")
    pol_standard = _model_policy(agent, hard=True, eps=0.0, rng=rng)
    res_standard = duel(pol_standard, noisy_heuristic, games=500, seed=42)
    print(f"Standard Win Rate: {res_standard['A_win_rate']*100:.1f}%")
    
    print("\n--- Test 2: HYBRID Model (Override) vs 1-ply Heuristic ---")
    pol_hybrid = hybrid_policy(agent, hard=True)
    res_hybrid = duel(pol_hybrid, noisy_heuristic, games=500, seed=42)
    print(f"Hybrid Win Rate: {res_hybrid['A_win_rate']*100:.1f}%")
    
    print("\n" + "="*50)
    if res_hybrid['A_score_pct'] > 60:
        print("SUCCESS: The Hybrid model beats 1-ply!")
        print("Your network DID learn positional play. The 'failure' was just greedy blunders.")
    else:
        print("FAILURE: The Hybrid model still loses.")
        print("Your network has NOT learned positional play in quiet moments.")
        print("Its 41% holdout agreement was just memorizing tactical patterns.")
    print("="*50)

if __name__ == "__main__":
    main()
