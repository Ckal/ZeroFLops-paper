"""Smoke test: bottleneck model soft/hard forward + training step."""
import sys
sys.path.insert(0, r"C:\Users\Chris4K\projects\ZeroFLops-paper")

import torch
torch.manual_seed(32)

from src.gf256_markov_lm_v6_bottleneck import GFMarkovLM

# Small config to move fast (d=8, 2 layers, fan_in=2)
model = GFMarkovLM(d=8, layers=2, fan_in=2, seed=32)
model.eval()
model.set_ste(False)

n_par = sum(p.numel() for p in model.parameters())
print(f"param count (d=8 test cfg): {n_par:,}")

B, T = 2, 96  # T must be >= max lag (64) for the hard path (see agent.py:48-57)
tokens = torch.randint(0, 256, (B, T))
targets = torch.randint(0, 256, (B, T))

# 1) Soft forward
soft_logits = model(tokens)
ok_soft = (soft_logits.shape == (B, T, 256)
           and not bool(torch.allclose(soft_logits, torch.zeros_like(soft_logits))))
print(f"[1] soft forward: {tuple(soft_logits.shape)}, nonzero={ok_soft}")

# 2) Export
hard = model.export_hard()
print(f"[2] export keys: {sorted(hard.keys())}")

# 3) Hard forward (deploy path) — must NOT return 0.0 / crash
hard_logits = model.hard_forward(tokens, hard)
ok_hard = (hard_logits.shape == (B, T, 256)
           and not bool(torch.allclose(hard_logits, torch.zeros_like(hard_logits))))
print(f"[3] hard forward: {tuple(hard_logits.shape)}, nonzero={ok_hard}")

# 4) Argmax valid
am = soft_logits.argmax(-1)
ham = hard_logits.argmax(-1)
ok_range = (am.min() >= 0 and am.max() <= 255 and ham.min() >= 0 and ham.max() <= 255)
print(f"[4] argmax ranges valid: {ok_range} (soft {int(am.min())}-{int(am.max())}, hard {int(ham.min())}-{int(ham.max())})")

# 5) Training step: loss is finite, backward works, params change
import torch.nn.functional as F
model.train()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
model.set_ste(False)
logits = model(tokens)
loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
ok_loss = bool(loss.isfinite())
p0 = model.readout_mlp[0].weight.detach().clone()
opt.zero_grad()
loss.backward()
opt.step()
p1 = model.readout_mlp[0].weight.detach().clone()
ok_update = not bool((p0 == p1).all())
print(f"[5] training step: loss={loss.item():.4f} finite={ok_loss}, mlp updated={ok_update}")

print()
print("SMOKE TEST PASSED" if (ok_soft and ok_hard and ok_range and ok_loss and ok_update)
      else "SMOKE TEST FAILED")
