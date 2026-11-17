"""Final confirmation: is the model EXACTLY equivalent to a fixed static
column-preference policy (pick the highest-preference legal column)?

If yes, the model learned NO board-dependent column signal, and the 30.73%
holdout number is a property of a constant policy, not of learned play.
"""
import numpy as np

c = np.load("runs/_probe/v18_picks_posdata_split.npz")
idx, pick, legal = c["idx"], c["pick"], c["legal"].astype(bool)
T = np.load("runs/c4/posdata_split/holdout_targets.npz")["target"]
opt = np.isclose(T[idx], T[idx].max(1, keepdims=True), atol=1e-6)
n = len(idx)
rows = np.arange(n)

# Static preference order recovered from the model's mean column logits:
# col5 > col4 > col6 > col3 > col2 > col1 > col0
ORDER = [5, 4, 6, 3, 2, 1, 0]

print("=== static-preference policy: pick highest-preference LEGAL column ===")
static_pick = np.zeros(n, dtype=int)
for i in range(n):
    for cc in ORDER:
        if legal[i, cc]:
            static_pick[i] = cc
            break

agree = (static_pick == pick).mean()
s_static = 100.0 * (opt[rows, static_pick] & legal[rows, static_pick]).mean()
s_model = 100.0 * (opt[rows, pick] & legal[rows, pick]).mean()
print("  static-policy score          : %.3f%%" % s_static)
print("  model score                  : %.3f%%" % s_model)
print("  static vs model pick agreement: %.2f%%" % (100 * agree))
print("  -> model is %s a fixed static column policy"
      % ("ESSENTIALLY" if agree > 0.95 else "NOT"))

print()
print("=== score ladder on the same 4000 rows ===")
ladder = []
for cc in range(7):
    s = 100.0 * (opt[:, cc] & legal[:, cc]).mean()
    ladder.append(("always-c%d" % cc, s))
ladder.append(("static pref order c5>c4>c6>c3>c2>c1>c0", s_static))
ladder.append(("MODEL (final_float.pt)", s_model))
tie = opt.sum(1).mean()
ladder.append(("random-among-optimal oracle", 100.0 * tie / 7))
for name, s in ladder:
    mark = ""
    if name.startswith("MODEL"):
        mark = "   <-- THE REPORTED NUMBER"
    elif name.startswith("random"):
        mark = "   <-- chance floor"
    print("  %-40s %7.3f%%%s" % (name, s, mark))

print()
print("=== how far above chance is the model, really? ===")
floor = 100.0 * tie / 7
print("  model %.3f%%  -  chance %.3f%%  =  %+.3f pp" % (s_model, floor, s_model - floor))
best_const = max(s for _, s in ladder if s.startswith("always"))
print("  best constant (always-c3) %.3f%%  -> model is %+.3f pp BELOW it"
      % (best_const, s_model - best_const))

print()
print("=== per-position: does the model ever beat the best constant column? ===")
model_hit = opt[rows, pick] & legal[rows, pick]
const3_hit = opt[:, 3] & legal[:, 3]
better = model_hit & ~const3_hit
worse = ~model_hit & const3_hit
print("  model right where c3 wrong: %d (%.2f%% of rows)" % (better.sum(), 100 * better.mean()))
print("  model wrong where c3 right: %d (%.2f%% of rows)" % (worse.sum(), 100 * worse.mean()))
print("  net: %+d rows" % (better.sum() - worse.sum()))
