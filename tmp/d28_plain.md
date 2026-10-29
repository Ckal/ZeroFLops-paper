
## 9.14 The midnight run: what we know, what we fixed, what we do not know

### The good news first

Three things that were broken and would have quietly ruined an 18-hour run
are now fixed and **proven**, not just patched:

**1. Resuming is now lossless.** If the run crashes at 3 a.m., restarting
it continues on **exactly** the same trajectory. We proved this by
training twice — once straight through, once with a deliberate
interruption in the middle — and comparing the results:

```text
max |weight difference|      = 0.000e+00
max |Adam moment difference|  = 0.000e+00
```

Before this fix, a restart silently threw away the optimiser's memory
(Adam's momentum estimates) and re-rolled the dice on which data rows got
sampled. The "final" model would have depended on *when* the computer
happened to hiccup.

**2. The generalisation check actually works now.** The number we use to
decide "is this checkpoint the best one?" was being measured on the
**training** data. That tells you how well the model memorised, not
whether it learned anything general. We now also score a separate set of
games the model has never seen. And the code path for it had **never
once been executed** before tonight, because the launcher never turned it
on. Turning it on revealed two separate crashes, both of which were
being *swallowed* — the run would have carried on for 18 hours without
the check it was supposed to have.

**3. The disk will not fill.** A resumable checkpoint is 203 MB. Sixty of
them is 12 GB, which does not fit. The new pruning keeps the three most
recent resumable checkpoints plus the best one — about 0.9 GB.

### The number that changed by half

The timeline you were shown earlier was based on **1.30 seconds per step**.
That was wrong, and in the good direction.

The three earlier test runs (fx_a, fx_b, fx_c) **all started 12 seconds
apart and ran at the same time on the same graphics card**. They were
sharing it three ways, so each was crawling at roughly a third speed.
The 1.30 s/step was the *shared* rate.

V24 runs alone, so it gets the whole card:

| | |
|---|---|
| Real rate | **0.65 s/step** |
| One full pass over the data (20,000 steps) | **3.6 hours** |
| Five passes (100,000 steps) | **18.1 hours** |

So the run finishes around **18:00 tomorrow evening**, not the following
morning.

### What it will probably score

Being straight with you: the honest answer is "about 63 %, and it could
be lower."

Earlier I quoted 70–78 %. That came from assuming the learning curve
would keep its shape indefinitely. It probably won't — curves flatten.
When I fit the real curve from the corrected 6,000-step run and
extrapolate, I get **≈ 63 %** at 100,000 steps. The 500-row overfitting
run reached 91 %, but that was memorising half a thousand games, so it is
an upper bound on the *method*, not a prediction for this run.

What I will not do is promise a number. Watch the two columns in the log:

- `LABEL` — how well it fits the training labels (the thing being
  optimised)
- `H-LABEL` — how well it does on games it has never seen (the thing that
  actually tells you if it learned)

If `LABEL` climbs and `H-LABEL` stays flat, the model is memorising and
we should stop. That is exactly what the new abort rule watches for.

### The hour-by-hour, honestly

Assuming a start at 00:00, at 0.65 s/step:

| Time | Step | Passes over data | `LABEL` (rough) |
|---|---|---|---|
| 00:30 | ~2,800 | 0.14 | ~41 % |
| 01:00 | ~5,500 | 0.28 | ~45 % |
| 02:00 | ~11,100 | 0.55 | ~50 % |
| 03:00 | ~16,600 | 0.83 | ~53 % |
| 04:00 | ~22,200 | 1.11 | ~55 % |
| 06:00 | ~33,300 | 1.66 | ~58 % |
| 12:00 | ~66,500 | 3.32 | ~61 % |
| ~18:05 | 100,000 | 5.00 | ~63 % |

Read the `LABEL` column as "roughly where the curve points", not as a
promise. The step counts are solid; the percentages are a projection.

### Playing it while it trains

You do not have to wait 18 hours to play. Every ~2,000 steps (about 22
minutes) a new checkpoint is published, and the game page can swap it in
**without losing the game you are in the middle of**. We tested that
specifically: 14 out of 14 checks pass.

### What we are still not claiming

- We do not yet know the final score. Nobody can know that before the run.
- The learning-rate schedule and the weight-averaging options exist and
  work, but they have **not** been test-flown, so they ship switched off.
  A long run with a constant learning rate can start wobbling near the
  end; if the curve looks like it is wobbling, that is the switch to
  flip on the next run.
- We still owe a clean test of "per-ply supervision" versus "single
  label" supervision. The old answer to that question was based on broken
  code and has been thrown out.
- Nothing is called "released" until the converged model passes the full
  battery: hard evaluator, mirror and column checks, in-game strength,
  seat fairness, trajectory validity, and artifact restoration.
