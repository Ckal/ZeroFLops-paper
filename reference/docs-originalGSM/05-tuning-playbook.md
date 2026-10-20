# 05 - Tuning playbook

This document is the practical guide. It tells you what to do when the model is unstable, when to raise `fan_in`, how long to run a test, and how to interpret the logs without overreacting to noisy metrics.

---

## 1) The basic philosophy

Training is not a single knob game.

There are three main classes of knobs:

1. capacity knobs: model structure and size
2. optimizer knobs: learning rate, clipping, batch size
3. schedule knobs: entropy, hard objective timing and weight

Good tuning means changing one category at a time and judging the result over a meaningful window.

---

## 2) How long should you run before deciding a setting works?

This is an underrated question.

### Good rule of thumb

- 1k–3k steps: only early signals
- 10k steps: rough sense of whether the run is trending
- 20k–30k: strong early decision point
- 50k: good choice for architecture or fan-in comparisons
- 100k: good for slow but promising improvements

### Concrete examples

If you change `--fan_in` from 4 to 8:

- at 500 steps, you may see soft BPC drop but hard BPC spike badly
- at 1500–3000 steps, the model may still look worse or unstable
- at 20k–50k steps, the real question is whether best-so-far soft BPC and hard BPC eventually improve enough to justify the slowdown

That is why a bad-looking early run is not enough to conclude the architecture is wrong.

### Important caveat

If the new setting is very different from the baseline, it may need more time.

Higher `fan_in` often needs longer before it is judged.

### If you change only one thing

You usually need at least 20k–50k steps to know whether a change is actually helping.

---

## 3) Fan-in experiment rule

This is one of the main questions in this project.

### What to do

When testing a new `fan_in` value:

- keep the same seed
- keep the same data
- keep the same context length and batch
- run a controlled training length such as 20k–50k steps
- compare best-so-far soft BPC and best-so-far hard BPC

### What not to do

Do not decide based on:

- a single hard-BPC spike
- the first 1000 steps
- one lucky eval point

### Decision rule

A higher fan-in is worth keeping only if it materially improves the best-so-far metrics after a sustained window.

If it makes training much slower but does not show meaningful improvement, it is probably not worth it.

---

## 4) When the model is too unstable

### Signs

- loss spikes and never recovers
- soft BPC jumps upward repeatedly
- hard BPC remains extremely high
- entropy becomes erratic
- the model becomes too confident too early

### First things to try

1. lower `--lr`
2. lower `--hard_w`
3. increase `--hard_start`
4. increase `--hard_ramp`
5. increase `--ent_warmup`
6. keep `--clip` at a sensible value like `1.0`

### Concrete example

Suppose the run does this:

- loss: 3.9 -> 3.1 -> 2.9 -> 3.4 -> 3.6
- soft BPC: 3.6 -> 3.1 -> 2.9 -> 3.0 -> 3.2
- hard BPC: 8.0 -> 7.2 -> 6.8 -> 6.7 -> 6.9
- entropy: 0.8 -> 0.2 -> 0.1 -> 0.3 -> 0.4

That pattern is unstable. A good first response is to lower `--lr`, reduce `--hard_w`, and delay `--hard_start`.

### Why this works

These changes reduce the optimization pressure and let the model learn in a calmer order.

---

## 5) When training is too slow

### Signs

- soft BPC barely moves over many thousands of steps
- training loss declines very slowly
- entropy stays too high or too low in a stuck pattern

### First things to try

1. slightly raise `--lr` if training is stable
2. lower `--hard_start` if the hard objective is not coming online soon enough
3. increase `--hard_w` a bit if harder learning is helpful
4. check whether the model is simply too small for the task

### Concrete example

If loss is moving like:

- 2.8 -> 2.5 -> 2.3 -> 2.25 -> 2.21 -> 2.19

and soft BPC is slowly improving, then training is likely healthy. If the improvements are too slow, a modest `--lr` bump or a slightly stronger `--hard_w` is reasonable.

### Important caution

Do not raise `--lr` blindly in a run that is already noisy. That usually makes things worse.

---

## 6) When the model is too confident too early

### Signs

- entropy falls too fast
- model seems to lock in early and stop exploring
- loss falls quickly at first but then stalls

### First things to try

1. raise `--ent_max`
2. increase `--ent_warmup`
3. lower `--hard_w` if the hard objective is too aggressive
4. give the model more time before switching on hard pressure

### Interpretation

This means the model is becoming overconfident very quickly. That often hurts long-term learning and can flatten the run early.

---

## 7) When hard BPC is noisy but soft BPC is improving

This is usually not a failure.

### Typical pattern

- soft BPC keeps dropping steadily
- hard BPC spikes or stays weird for a while
- later it settles down

### What this means

The model is learning the main distribution well, but the harder objective is still adapting and the metric is noisy.

### What to do

- wait for the schedule to settle
- do not panic after a single spike
- compare the best-so-far values over a longer window

---

## 8) When you should stop a run

A run is often worth continuing if:

- loss still trends down
- soft BPC is improving over time
- entropy remains healthy
- hard BPC is noisy but gradually improving or stabilizing

A run is usually not worth continuing if:

- loss flatlines for long periods
- soft BPC no longer improves over many checkpoints
- hard BPC stays broken after the schedule has already started
- entropy is clearly broken or oscillating without recovery

### Best way to judge

Judge by long-window trends, not by a single checkpoint.

---

## 9) Good practical experiment recipes

### Recipe A: quick debug run

Purpose: verify the training loop works.

```bash
python originalGSM.py \
  --seed 64 \
  --data smollm_subset_1024MB.txt \
  --stream-data \
  --run-dir runs/debug \
  --steps 10000 \
  --device cuda \
  --ctx 256 \
  --batch 4 \
  --d 64 \
  --layers 6 \
  --fan_in 4 \
  --lr 3e-3 \
  --clip 1.0 \
  --ent_max 2.0 \
  --hard_start 1500 \
  --hard_ramp 1000 \
  --hard_w 0.5
```

### Recipe B: fan-in comparison

Use same seed and same overall training config, only change `--fan_in`.

Run 20k–50k steps and compare:

- best soft BPC
- best hard BPC
- runtime and throughput
- quality of generated samples

### Recipe C: stable long run

Use a more conservative schedule and let it train longer.

```bash
python originalGSM.py \
  --seed 64 \
  --data smollm_subset_1024MB.txt \
  --stream-data \
  --run-dir runs/stable \
  --steps 600000 \
  --device cuda \
  --ctx 256 \
  --batch 4 \
  --d 64 \
  --layers 6 \
  --fan_in 6 \
  --lr 5e-4 \
  --clip 1.0 \
  --ent_max 5.0 \
  --hard_start 1500 \
  --hard_ramp 1000 \
  --hard_w 1.0 \
  --eval_every 500
```

---

## 10) Very short cheat sheet

### Increase capacity

- `--d`
- `--layers`
- `--fan_in`
- `--ctx`

### Stabilize training

- decrease `--lr`
- increase `--clip`
- increase `--hard_start`
- increase `--hard_ramp`
- lower `--hard_w`
- increase `--ent_warmup`

### Push harder learning

- increase `--hard_w`
- lower `--hard_start`
- decrease `--hard_ramp`
- lower `--ent_max` slowly if overconfidence is not a problem

### Improve general quality

- lower soft BPC over time
- keep hard BPC trending downward after initial noise
- maintain healthy entropy behavior

---

## 11) Final practical advice

The way to become good at this script is simple:

- treat loss as the first signal
- treat soft BPC as the main quality signal
- treat hard BPC as a tougher, noisier signal
- treat entropy as a stability indicator
- treat capacity changes as a long-term hypothesis

Do not decide based on a single weird metric point. Decide based on the trend.

---

## Summary

The right tuning workflow is:

1. start small
2. keep one variable changing at a time
3. use 20k–50k steps for meaningful comparisons
4. judge best-so-far metrics, not one noisy eval point
5. give higher fan-in time to prove itself
6. always track entropy and loss together with BPC

If you do that, the training behavior becomes much easier to read and much easier to improve.

---

Back to [README](README.md) for the full doc index.
