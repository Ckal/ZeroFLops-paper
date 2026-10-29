import random, sys, json
sys.path.insert(0, '.')
from src.c4.duel import _load_hard_agent, _model_policy, duel
from src.c4.board import Board

SEED = 20261029
random.seed(SEED)
a = _load_hard_agent('runs/_probe/v24/checkpoints/best_hard.pt')
b = _load_hard_agent('runs/_probe/compare/step20000_hard.pt')
a_pol = _model_policy(a, hard=True, eps=0.0)
b_pol = _model_policy(b, hard=True, eps=0.0)

same = 0; tot = 0
b1 = Board(); seq = []
for ply in range(40):
    if not b1.valid_actions(): break
    pa = a_pol(b1, list(seq)); pb = b_pol(b1, list(seq))
    tot += 1
    if pa == pb: same += 1
    seq.append(pa); b1 = b1.apply_move(pa)
print(f"same column on identical positions: {same}/{tot} = {100*same/tot:.1f}%")
print(f"  first divergence plies: ", end="")
b1 = Board(); seq = []
for ply in range(40):
    if not b1.valid_actions(): break
    pa = a_pol(b1, list(seq)); pb = b_pol(b1, list(seq))
    if pa != pb:
        print(f"ply {ply} (A={pa}, B={pb})"); break
    seq.append(pa); b1 = b1.apply_move(pa)
else:
    print("none")

# Model greedy, OPPONENT explores -> the canonical strength protocol
rng_a = random.Random(SEED)
for g in (400,):
    random.seed(SEED)
    a_pol_e = _model_policy(a, hard=True, eps=0.10, rng=rng_a)
    b_pol_e = _model_policy(b, hard=True, eps=0.10, rng=random.Random(SEED+1))
    r = duel(a_pol_e, b_pol_e, g, seed=SEED)
    print(json.dumps(r, indent=2))
    from math import comb
    n = r['A_wins'] + r['B_wins']
    if n:
        p = sum(comb(n,k) for k in range(r['A_wins'], n+1))/2**n*2
        print(f"  step18000 score {r['A_score_pct']:.1f}%  p={min(p,1.0):.4f}")
