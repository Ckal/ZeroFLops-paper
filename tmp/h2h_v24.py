import random, sys, json
sys.path.insert(0, '.')
from src.c4.duel import _load_hard_agent, _model_policy, duel

SEED = 20261029
GAMES = 200

random.seed(SEED)
a = _load_hard_agent('runs/_probe/v24/checkpoints/best_hard.pt')       # step 18000 (promoted)
b = _load_hard_agent('runs/_probe/compare/step20000_hard.pt')          # step 20000

a_pol = _model_policy(a, hard=True, eps=0.0)
b_pol = _model_policy(b, hard=True, eps=0.0)

r = duel(a_pol, b_pol, GAMES, seed=SEED)
print(json.dumps(r, indent=2))
print()
print(f"step18000 (A) vs step20000 (B): A_score {r['A_score_pct']:.1f}%  "
      f"W{r['A_wins']} L{r['B_wins']} D{r['draws']}")
from math import comb
n = r['A_wins'] + r['B_wins']
p = sum(comb(n,k) for k in range(r['A_wins'], n+1))/2**n*2 if n else 1.0
print(f"one-sided binomial p = {min(p,1.0):.4f}")
