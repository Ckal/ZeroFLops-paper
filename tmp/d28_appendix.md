
---

## Appendix D28 — Claude's review, applied: the pre-launch package

**Date:** 2026-10-28 (evening, before the midnight V24 launch)
**Status:** V24 pre-flight complete. Not yet launched.

### D28.1 What Claude got right, and what I checked rather than believed

Claude sent a review of the trainer with five substantive claims. Each was
checked against the **shipped** code, not against a description of it.

| # | Claude's claim | Verdict | Evidence |
|---|---|---|---|
| 1 | The historical per-ply path had a **second** indexing error (`// (W+1)` over a `B×W` grid), so batch row *k* received the early positions *t < k* from row *k−1* | **CONFIRMED** | `tmp/verify_second_index_error.py` reproduces the exact failure pattern |
| 2 | The *current* fix is correct | **CONFIRMED** | decoded-ply invariant: **6504/6504** production sampled positions pass, including small-`t`, `k>0` cases |
| 3 | The historical "per-ply supervision LOST", "mirror is a net negative", "symmetry not worth it" decisions must be **withdrawn** | **CONFIRMED** | those decisions predate the sidecar-row fix; they measured the bug, not the model |
| 4 | The trainer saves **no optimizer state and no RNG state**, and has **no LR schedule and no EMA** | **CONFIRMED** | all four were absent before this session's edits |
| 5 | Batch 32 gives **2×** the throughput | **REFUTED** | real V24 path measures **1.22×** (see D28.4) |

Claude also claimed `src/gf256_markov_lm.py` "already has that checkpoint
logic and can be adopted". **That specific pointer is inaccurate** —
that file contains no optimizer-state or RNG-state checkpoint
implementation to reuse. The logic was written from scratch here.

### D28.2 What was wrong with the earlier conclusions

Because only the **per-ply** path read the wrong rows, the historical
A/B compared *correct* single-label supervision against *garbage* per-ply
supervision. The conclusion "dense per-ply supervision LOST" therefore
compared a working path against a broken one. The corrected
`fx_a/fx_b/fx_c` arms are the mirror/symmetry comparison — they are
**not** a per-ply-versus-single-label test, and no such test has been run.

### D28.3 The trainer changes (all opt-in, historical runs reproduce exactly)

| Change | Flag | Default | Why |
|---|---|---|---|
| Optimizer state saved/restored | — | **always on** | Adam restarting with empty moments spikes the loss after every restart |
| Python/NumPy/Torch RNG state saved/restored | — | **always on** | without it a resume samples different rows and drifts off the original trajectory |
| Cosine LR schedule | `--lr-schedule` | `none` | constant LR starts oscillating once the curve bends |
| EMA weights | `--ema-decay`, `--ema-warmup`, `--ema-export` | `0.0` (off) | flattens the late-training oscillation |
| Disjoint-holdout per-ply agreement | `--holdout-corpus` / `--holdout-targets` | off | the promotion metric samples the **training** corpus; that measures *fit*, not generalisation |
| Per-column pick/label histograms | — | always on | separates "the data lacks c1" from "the model ranks c1 last" |
| Checkpoint pruning | `--keep-last`, `--keep-hard-all` | `3` | 50 snapshots × 237 MB = 11.9 GB does not fit |
| Abort rules | `--abort-warmup`, `--abort-flat-evals`, `--abort-flat-below`, `--abort-on` | `20000` / `4` / `55.0` / `warn` | a wrong abort on an 18 h run is more expensive than a run that continues past a plateau |

Optimizer and RNG state are stored at the **top level** of the float
checkpoint (`opt_state`, `rng_state`), not inside `meta` — `meta` is
reused by the hard export, which must stay a lean integer-table payload.

### D28.4 Batch size: the measured answer, not the claimed one

Claude's table came from `tg_scale4096.py`, a **4096-row V22**
architecture with a different loop. The real V24 path
(`src.c4.train`, per-ply sidecars, dense 12 lags, ctx 192) measures:

| batch | s/step (train only) | rows/s | gain vs 16 |
|---|---|---|---|
| 16 | **0.646** | 24.77 | 1.00× |
| 32 | 1.063 | 30.10 | **1.22×** |
| 64 | 1.27 | ~50 (different loop) | — |

At the V24 evaluation cadence (eval every 2,000 steps, one eval ≈ 7.9 s)
the amortised cost is **+0.004 s/step**, so the effective rate is
**0.650 s/step** and:

- one epoch (20,000 steps) ≈ **3.6 h**
- 100,000 steps (5 epochs) ≈ **18.1 h**

Claude's "5 epochs in ~9 h" is not supported by the real path.
**Batch 16 remains the documented configuration** — the paper constraint
stands, and 1.22× is not worth changing the experimental protocol for.

### D28.5 The concurrency correction: the real step rate is 0.65 s, not 1.30 s

The earlier 1.30 s/step figure came from `fx_a/fx_b/fx_c`, whose logs show
the three arms **started 12 seconds apart** — they ran **concurrently on
one GPU**. Each was therefore running at roughly a third of the
throughput, and 1.3164 s/step was the *shared* rate, not the single-run
rate. A single V24 run gets the whole GPU: **0.650 s/step**.

This changes the timeline by a factor of two and is the single most
important number in this appendix.

### D28.6 The holdout path had never been executed — and was broken twice

`run_v24.sh` never passed `--holdout-corpus` / `--holdout-targets`, so
the holdout scoring path had **never run**. Wiring it up exposed two real
defects:

1. **`Holdout.__init__` assumed the old `(N, 7)` sidecar.** The
   production per-ply sidecar is `(N, K, 7)` (one target row per labelled
   ply), so `self.optimal[i]` was a `(K, 7)` tensor and
   `bool(opt[pick])` raised
   `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
   Fixed: the loader now detects the schema explicitly and, for a
   per-ply sidecar, resolves the final-ply slot through `ply_of` — the
   same slot/ply distinction the training path was fixed for.
2. **`PerPlyAgreement` assumed torch tensors.** The holdout caller
   passes numpy arrays straight out of `np.load`, so `.to(device)` raised
   `AttributeError: 'numpy.ndarray' object has no attribute 'to'`.
   Because that failure was **caught and only logged**, the run continued
   *silently without the generalisation metric and without the abort
   rules*. Fixed: the scorer normalises its inputs.

Both were found only because the flag was finally wired up — an
untested path is not a working path.

### D28.7 The abort rules fired at step 0 and killed the run

With the holdout working, the new abort rule-2 ("a column that is optimal
in the labels was NEVER picked") fired **immediately at step 0** on a
*randomly initialised* model — of course it never picks c4 — and raised
`SystemExit` before a single training step ran.

Two fixes:

- both rules now require `--abort-warmup` (default 20,000 = one epoch);
- `--abort-on` defaults to **`warn`**: the finding is recorded in
  `ABORT.txt` and the run **continues**. Stopping an 18-hour run must be
  an explicit decision, because a false positive costs far more than a
  run that continues past a recoverable plateau.

### D28.8 Resume is now provably lossless

`tmp/test_resume_state.py` — **19 PASS, 0 FAIL**:
`final_float.pt`, `best_soft.pt` and every `step*_soft.pt` carry
`opt_state` + `rng_state`; the hard export carries neither (lean
payload) but keeps its transcript stamp; a hard-export resume explains
itself and does not claim to restore Adam moments.

`tmp/test_resume_equivalence.py` — **12 PASS, 0 FAIL**: an
uninterrupted 12-step run and a 6-step-then-resume-to-12 run produce
**bit-identical** results:

```text
max |weight difference|      = 0.000e+00
max |Adam moment difference|  = 0.000e+00
```

This is the property that matters for an 18-hour run: the final artifact
no longer depends on where the interruptions happened.

### D28.9 Disk: the 5 GB question, answered with measured numbers

Measured artifact sizes:

| artifact | size |
|---|---|
| float snapshot (weights + Adam moments + RNG) | **203 MB** |
| hard export | **34 MB** |
| corpus copy + sidecar | 55 MB |

For a 100,000-step run with a snapshot every 2,000 steps (50 snapshots):

| policy | footprint | fits in 9.6 GB? |
|---|---|---|
| unbounded | 50 × 237 MB = **11.9 GB** | **no** |
| `--keep-last 3` (default) | 3 × 237 MB + best ≈ **0.9 GB** | **yes** |
| `--keep-last 3 --keep-hard-all` | ≈ **2.6 GB** | yes |

The volume was at **5.5 GB free (100 % full)** when this work started;
only scratch/probe directories were deleted to recover **9.6 GB**. No
historical corpus or useful checkpoint was removed. The pruning code was
validated on a real run: with `--keep-last 3` exactly three float and
three hard snapshots survive, and `best_*` is never touched.

### D28.10 V24 pre-flight: what was actually executed

A 60-step smoke test of the **real launcher** (`run_v24.sh`, ctx 192,
batch 16, transcript v2, dense lags, seed 32, real holdout) completed
end to end with exit 0:

```text
step  0 | LABEL 16.3% (n=4000) | H-LABEL 11.9% (n=512)
step 30 | LABEL 32.4% (n=4000) | H-LABEL 30.5% (n=512)
step 60 | LABEL 36.8% (n=4000) | H-LABEL 31.2% (n=512)
[snapshot saved -> step0/30/60]
saved -> hard_model_final.pt
saved -> final_float.pt
```

The holdout agreement **tracks** the training agreement from the very
first evaluation — which is what makes it usable as a generalisation
signal and as an abort input.

### D28.11 What is still open

- The 100,000-step LABEL value is **unknown**; the log-linear
  extrapolation of the fx_a curve gives ≈ 63 %, not the 70–78 % earlier
  suggested.
- The LR/EMA options are implemented but **not numerically piloted**; they
  ship disabled.
- No corrected per-ply-versus-single-label A/B has been run.
- `runs/_probe/disjoint_holdout.*` is the previously established
  canonical-board-disjoint artifact; a fresh exact overlap count has not
  been recomputed.
- The full release battery (hard evaluator, mirror/c1–c4, in-game,
  trajectory, seat, artifact, restoration) must pass on the converged
  model before any release claim.
