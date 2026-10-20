# 04 - Parameter reference

This document explains the arguments in the script in a compact, practical way.

The user-facing CLI is a training interface. It controls data, model size, training behavior, evaluation, and checkpointing.

---

## 1) Core setup

### `--data`

Default: `tinyshakespeare.txt`

The text corpus used for training.

- larger, higher quality data usually helps more than magic tuning knobs
- a small dataset can make a large model look artificially good

Good use:

- standard low-cost experiments
- resume from the same corpus
- compare across seeds or architecture choices

---

### `--run-dir`

Directory used for checkpoints, logs, manifests, and artifacts.

Why it matters:

- it keeps all outputs from a run together
- it makes resuming and comparing runs easier

---

### `--resume`

Resume training from a checkpoint or a run directory.

Good for:

- continuing an interrupted experiment
- continuing a high-quality run
- testing a different schedule on a good model

---

### `--resume-step`

Choose which checkpoint to resume from within a run directory.

Examples:

- `latest`
- `best`
- `best-soft`
- `best-hard`
- integer step number such as `6000`

This is extremely useful when you want to resume from the strongest checkpoint instead of the most recent one.

---

### `--stream-data`

Load the corpus lazily instead of keeping it all in RAM.

Good when:

- the dataset is large
- memory is limited
- you want to use big corpus files without storing them in memory

---

## 2) Model shape and capacity

### `--d`

Hidden size / width of the model.

- larger means more capacity
- more expensive and slower

Typical values:

- 32: small debug
- 64: good baseline
- 128: medium

---

### `--seed`

Random seed.

Controls reproducibility.

Use the same seed when comparing architecture or schedule changes.

---

### `--layers`

Number of transformer layers.

- more layers = deeper model
- more compute, often more expressiveness

Typical values:

- 2–4 = tiny
- 6 = small baseline
- 8–12 = deeper, heavier experiments

---

### `--fan_in` / `--fan-in`

Controls the fan-in / interaction width of the model structure.

What to expect:

- bigger fan-in often adds capacity
- but it usually slows steps significantly
- it often needs more training time before the gain is visible

This is a workload and capacity search knob, not a “first 500 steps” metric knob.

---

## 3) Training loop

### `--steps`

Total optimizer steps.

This is the main length of the run.

Good rule:

- quick smoke tests: 5k–20k
- architecture sweeps: 20k–50k
- serious training: 50k–600k depending on task and compute

---

### `--batch`

Batch size per optimization update.

Tradeoff:

- more batch = faster wall-clock throughput and more stable updates
- less batch = more stochastic but often more exploratory learning

Common values:

- 2, 4, 8 are common for small GPU training

---

### `--ctx`

Context window length.

Controls how many tokens the model can attend to at once.

- longer context = better long-dependency modeling but more compute
- shorter context = easier and cheaper

Typical values:

- 128 or 256: small/medium runs
- 512–1024: longer-range experiments

---

### `--lr`

Learning rate.

This controls how large each update is.

- too low = slow learning
- too high = instability

Typical starting range depends on the task, but small baseline values often start around `3e-3` to `5e-4` depending on scale and objective.

---

### `--clip`

Gradient clipping limit.

Prevents giant gradient updates.

Common default:

- `1.0`

If training becomes erratic, this is one of the first things to revisit.

---

## 4) Entropy and confidence shaping

### `--ent_max`

Maximum entropy target.

This sets how much uncertainty the model is allowed to maintain.

- larger value = more uncertainty / softer distribution
- smaller value = more confident / sharper distribution

This is one of the main regularization knobs.

---

### `--ent_warmup`

How long the entropy control is warmed up.

This delays the full entropy pressure so the model has time to learn some useful structure first.

- larger warmup = smoother early training
- smaller warmup = more aggressive entropy control sooner

---

### `--perm_w`

Permutation / structure weight.

This is a shaping term and not usually the first knob to change.

It influences how strongly ordering or structural constraints contribute.

---

## 5) Hard objective schedule

### `--hard_start`

When the hard objective begins to take effect.

This is usually set after an early stabilization period.

- later start = smoother early training
- earlier start = more difficult learning sooner

Typical values:

- `1500` or similar in a small run

---

### `--hard_ramp`

How long the hard objective is increased over time.

- shorter ramp = faster hard transition
- longer ramp = smoother transition

This is a stability knob.

---

### `--hard_w`

Hard objective weight.

This controls how strongly the hard objective contributes to learning.

- low = smoother, easier learning
- high = more difficult optimization

This is another knob where a small change can create large changes in the training dynamics.

---

## 6) Evaluation frequency and size

### `--eval_every`

How often to run validation.

Common value:

- `500`

Lower values create more noise and more overhead. Higher values are cheaper but less frequent.

---

### `--eval_seqs`

How many evaluation sequences are used.

More sequences = more stable result, more compute.

Typical values:

- `64` or `128`

---

### `--eval_chunk_size`

How many eval sequences to process per chunk.

Useful for controlling memory use during evaluation.

---

### `--final-eval-seqs`

Validation sequences for a final evaluation pass.

If it is `0`, the script often falls back to its default behavior.

---

## 7) Checkpointing and persistence

### `--checkpoint-every`

Number of optimizer updates between durable checkpoints.

- lower = more checkpoints, better recovery
- higher = less disk overhead

Typical choices:

- `6000` or similar

---

### `--no_ckpt`

Disable checkpointing.

Useful for:

- short debug runs
- exploration runs where you do not need recovery

Not recommended for serious training.

---

## 8) Device and experiment tracking

### `--device`

Training device.

- `cuda` if available
- otherwise `cpu`

---

### `--trackio`

Enable Trackio experiment tracking.

This is optional and helps log metrics over time.

---

### `--trackio-project`

Project name in the tracking backend.

---

### `--trackio-run`

Stable run name.

---

### `--trackio-space-id`

Optional Space ID for remote tracking.

---

### `--trackio-server-url`

Optional custom server endpoint.

---

### `--trackio-log-every`

How often to log training metrics.

---

### `--trackio-artifacts`

Whether to upload artifacts from checkpoints and run files.

---

## 9) Short summary of the main knobs

### Capacity knobs

- `--d`
- `--layers`
- `--fan_in`
- `--ctx`

### Optimization knobs

- `--lr`
- `--clip`
- `--batch`

### Stability and shaping knobs

- `--ent_max`
- `--ent_warmup`
- `--hard_start`
- `--hard_ramp`
- `--hard_w`
- `--perm_w`

### Measurement knobs

- `--eval_every`
- `--eval_seqs`
- `--eval_chunk_size`

### Persistence knobs

- `--checkpoint-every`
- `--resume`
- `--resume-step`

---

## 10) Best practice

If you are changing one thing, do it intentionally:

- change only one major variable at a time
- keep the seed fixed when comparing settings
- judge based on sustained trends, not single eval spikes
- save checkpoints often

---

## Next document

See [05 - Tuning playbook](05-tuning-playbook.md) for a practical recipe set: what to do when training is unstable, when to increase `fan_in`, when to lower `hard_w`, and how long to run each experiment.
