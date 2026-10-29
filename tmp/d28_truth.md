### 5r. THE V24 PRE-FLIGHT: what changed, what was measured, what is still open

**Date:** 2026-10-28 evening, before the midnight V24 launch.

#### 5r.1 The step rate was wrong by a factor of two

The earlier "1.30 s/step" figure came from the `fx_a/fx_b/fx_c` arms.
Their logs show the three arms **started 12 seconds apart** — they ran
**concurrently on one GPU**, so 1.3164 s/step was the *shared* rate.

A single V24 run gets the whole GPU. Measured on the real V24 path
(`src.c4.train`, per-ply sidecars, dense 12 lags, ctx 192, batch 16):

| quantity | value |
|---|---|
| train-only | **0.646 s/step** |
| one evaluation | 7.9 s |
| amortised eval (every 2,000 steps) | +0.004 s/step |
| **effective** | **0.650 s/step** |
| one epoch (20,000 steps) | **3.6 h** |
| 100,000 steps (5 epochs) | **18.1 h** |

#### 5r.2 Batch 32 is 1.22×, not 2×

Claude's throughput table came from `tg_scale4096.py`, a 4096-row V22
architecture with a different loop. On the real path: batch 16 →
24.77 rows/s, batch 32 → 30.10 rows/s. **Batch 16 stays** — the paper
constraint stands, and 1.22× does not justify changing the protocol.

#### 5r.3 The trainer now saves and restores the run state

`opt_state` and `rng_state` (Python `random`, NumPy `default_rng`, Torch)
are written at the **top level** of every float checkpoint and restored on
resume. `meta` is untouched, because `meta` is reused by the hard export,
which must stay a lean integer-table payload.

Measured: a 12-step uninterrupted run and a 6-then-resume-to-12 run
produce **bit-identical** weights and Adam moments
(`max |Δweight| = 0.000e+00`, `max |ΔAdam moment| = 0.000e+00`).

#### 5r.4 The holdout path had never run, and was broken twice

`run_v24.sh` never passed `--holdout-corpus` / `--holdout-targets`, so
the whole generalisation path was **never executed**. Wiring it up found:

- `Holdout.__init__` assumed the old `(N, 7)` sidecar; the production
  per-ply sidecar is `(N, K, 7)`, so `bool(opt[pick])` raised
  `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
  Fixed: the schema is detected explicitly, and the final-ply slot is
  resolved through `ply_of`.
- `PerPlyAgreement` assumed torch tensors; the holdout caller passes
  numpy arrays, so `.to(device)` raised
  `AttributeError: 'numpy.ndarray' object has no attribute 'to'` — and
  because the failure was only *logged*, the run continued **silently
  without the generalisation metric and without the abort rules**.

#### 5r.5 The abort rules killed the run at step 0

Rule 2 ("a column optimal in the labels was NEVER picked") fired
immediately on a **randomly initialised** model. Fixed: both rules now
require `--abort-warmup` (default 20,000 = one epoch), and `--abort-on`
defaults to **`warn`** — the finding is recorded and the run continues.
Stopping an 18-hour run must be explicit.

#### 5r.6 Disk

Measured: float snapshot **203 MB**, hard export **34 MB**. For 50
snapshots: unbounded = **11.9 GB** (does not fit in 9.6 GB free);
`--keep-last 3` = **0.9 GB**; `--keep-hard-all` = **2.6 GB**. Pruning was
validated on a real run: exactly three float and three hard snapshots
survive and `best_*` is never touched.

#### 5r.7 Open items

- The 100,000-step LABEL value is **unknown**. Log-linear extrapolation
  of the fx_a curve gives **≈ 63 %**, not the 70–78 % previously quoted.
- Cosine LR and EMA are implemented but **not numerically piloted**; they
  ship **disabled**.
- No corrected per-ply-versus-single-label A/B has been run.
- The full release battery must pass on the converged model.

---
