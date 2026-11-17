"""Pin the root cause of V18's constant-column-5 policy.

Three hypotheses to separate:
  H1  STALE MODEL   -- eval/training is looking at a frozen copy of the
                       parameters (weights stop changing after some step).
  H2  MAPPING BUG   -- the model's column output is mis-indexed relative to
                       the teacher label columns (a permutation bug).
  H3  TRUE COLLAPSE -- the model genuinely learned "always column 5".
"""
import hashlib
import itertools
from pathlib import Path

import numpy as np
import torch

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")
COL_BYTE_OFFSET = 0x41


def fp_hash(obj):
    h = hashlib.sha256()
    sd = obj["state"] if "state" in obj else obj
    for k in sorted(sd.keys()):
        t = sd[k]
        if torch.is_tensor(t):
            h.update(k.encode())
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


print("=" * 72)
print("H1: STALE MODEL -- do the exported weights differ across steps?")
arts = ["checkpoints/best_hard.pt", "checkpoints/best_soft.pt",
        "hard_model_final.pt", "final_float.pt"]
hashes = {}
for a in arts:
    f = RUN / a
    if not f.exists():
        print("  %-28s MISSING" % a)
        continue
    o = torch.load(f, map_location="cpu", weights_only=False)
    h = fp_hash(o)
    step = (o.get("meta") or {}).get("step", o.get("step"))
    hashes[a] = h
    print("  %-28s step=%-6s hash=%s" % (a, step, h))
uniq = len(set(hashes.values()))
print("  -> distinct weight hashes: %d of %d artifacts" % (uniq, len(hashes)))
if uniq == 1:
    print("  -> ALL ARTIFACTS IDENTICAL: the model never changed after the"
          " first save (STALE / FROZEN EXPORT).")
elif uniq < len(hashes):
    print("  -> SOME artifacts are byte-identical: check step vs hash pairs above.")

print()
print("=" * 72)
print("H2/H3: MODEL PICK vs TEACHER ARGMAX -- is it a permutation or constant?")
c = np.load("runs/_probe/v18_picks_posdata_split.npz")
idx, pick, legal = c["idx"], c["pick"], c["legal"].astype(bool)
T = np.load(SPLIT / "holdout_targets.npz")["target"]
tam = T[idx].argmax(axis=1)
opt = np.isclose(T[idx], T[idx].max(1, keepdims=True), atol=1e-6)

print("  cross-tab: teacher argmax column -> model pick column")
print("        model->   " + "".join("%7d" % k for k in range(7)))
for tc in range(7):
    sel = tam == tc
    if sel.sum() == 0:
        continue
    bc = np.bincount(pick[sel], minlength=7)
    print("  teacher c%d (%4d): %s" % (tc, sel.sum(),
                                      "".join("%7d" % x for x in bc)))

# Is there a consistent MAP teacher-col -> model-col?
print("\n  per teacher column, the modal model pick:")
mapping = {}
for tc in range(7):
    sel = tam == tc
    if sel.sum() == 0:
        continue
    bc = np.bincount(pick[sel], minlength=7)
    m = int(bc.argmax())
    mapping[tc] = m
    print("    teacher c%d -> model c%d   (%.1f%% of its rows; %d/%d rows)"
          % (tc, m, 100.0 * bc[m] / sel.sum(), bc[m], sel.sum()))
print("  mapping:", mapping)
print("  -> identity?" , all(k == v for k, v in mapping.items()))
print("  -> constant?" , len(set(mapping.values())) == 1)

# Score the mapping-induced permutation against the labels
print("\n  scoring the induced map as a column permutation:")
for perm_name, perm in [("induced", [mapping.get(i, i) for i in range(7)]),
                         ("identity", list(range(7)))]:
    pp = np.array([perm[p] for p in pick])
    s = 100.0 * (opt[np.arange(len(idx)), pp] & legal[np.arange(len(idx)), pp]).mean()
    print("    %-8s %s -> %.3f%%" % (perm_name, perm, s))

# Constant column baselines for reference
print("\n  constant-column baselines on the same rows:")
for cc in range(7):
    s = 100.0 * (opt[:, cc] & legal[:, cc]).mean()
    print("    always-c%d -> %.3f%%" % (cc, s))
tie = opt.sum(1).mean()
print("  random-among-optimal oracle -> %.3f%% (mean tie-set %.4f)"
      % (100.0 * tie / 7, tie))

print()
print("=" * 72)
print("Bias inspection of the exported model:")
raw = torch.load(RUN / "final_float.pt", map_location="cpu", weights_only=False)
sd = raw["state"]
for k in sorted(sd.keys()):
    t = sd[k]
    if torch.is_tensor(t) and t.ndim == 1 and t.numel() >= 7:
        seg = t[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].detach().numpy()
        if np.any(np.abs(seg) > 1e-9) and np.ptp(seg) > 1e-9:
            print("  %-22s slice[0x41:0x48] = %s" % (k, np.round(seg, 4).tolist()))
print("  (a static per-column bias with a clear peak = the collapse is in the")
print("   parameters themselves, not in the export plumbing.)")
