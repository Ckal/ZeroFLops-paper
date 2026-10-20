# 02 - Model capacity and architecture

This document explains the actual architecture knobs: how large the model is, how much capacity it has, and what those settings usually mean in practice.

The main controls here are:

- `--d`
- `--layers`
- `--fan_in` / `--fan-in`
- `--ctx`
- `--batch`
- `--seed`

These are the model and training-shape knobs.

---

## 1) Model capacity: what is it?

Capacity means how much information the model can represent.

A model with more capacity can in principle:

- learn more complex patterns
- memorize more structure from the text
- gain more from longer training

But more capacity also means:

- more parameters
- more memory
- more compute
- slower training
- more risk of overfitting or optimization difficulty

### In simple terms

If the model is too small, it may plateau early. If it is too large, it may need more data or more steps to use its capacity well.

---

## 2) `--d` : hidden dimension / width

`--d` is the model width.

### What it changes

This controls how large each hidden representation is.

- larger `d` = wider model
- wider model = more capacity
- wider model = more memory and compute

### Typical effect

If you increase `d`:

- the model can represent more information per token
- training may improve, but slower per step
- generation may become more coherent and expressive

### When it is a good idea

Use larger `d` when:

- you are training longer runs
- you want more capacity for the corpus
- you want to test whether the model is bottlenecked by width

### When it is a bad idea

Avoid raising it too much if:

- your data is small
- your compute is limited
- you are doing quick experiments

### Rule of thumb

For small research runs, `d=32` or `d=64` is often a reasonable start. Larger values can be useful once the run is already stable.

---

## 3) `--layers` : model depth

`--layers` sets how deep the model is.

### What it changes

More layers mean more sequential transformations.

This often enables:

- more abstract reasoning
- deeper feature composition
- more expressive modeling of long-range patterns

### Tradeoffs

More layers usually mean:

- slower per-step compute
- more memory use
- harder optimization

### Good starting point

Many small experiments use around 4–8 layers.

### Rule of thumb

- 2–4 layers: tiny or very small experiments
- 6 layers: good small baseline
- 8–12 layers: more capable but much heavier

### Typical failure mode

A model with too many layers for too little data or too much learning rate can become unstable or underfit in confusing ways.

---

## 4) `--fan_in` / `--fan-in`

This is one of the most important tuning knobs, especially when the architecture is being adjusted.

### What it does

Fan-in controls the degree of incoming interaction or mixing in the model’s internal structure.

In simple terms:

- more fan-in means each feature has more opportunities to combine with more inputs
- the model can represent richer interactions
- but it also costs more compute and tends to learn more slowly in the early stages

### Why it can be confusing

A bigger fan-in often looks bad at first.

This is normal because:

- the model is simply doing more work per step
- the optimization problem gets harder
- the run may need more steps before the benefits appear

### The main practical lesson

Higher fan-in is not automatically better. It is only better if:

- the model eventually reaches a better best-so-far metric
- the improvement is sustained over a meaningful training window
- the added compute is worth the result

### Why it can slow training

A higher fan-in often reduces throughput because each token requires more operations.

So even if the model later improves, it may do so more slowly and with higher cost.

### When to raise it

Raise `--fan_in` when:

- the model seems underpowered for the task
- you want to test if richer mixing helps
- you have enough compute and long enough training runs

### When not to raise it

Do not raise it if:

- you are only doing short debug runs
- you want fast iteration
- you are already seeing good quality at lower fan-in

### Decision rule

A higher fan-in is worthwhile only if it produces clearly better best-so-far quality after a sustained window such as 20k–50k steps.

---

## 5) `--ctx` : context length

`--ctx` sets how many tokens the model can attend to in one pass.

### What it changes

Longer context lets the model use more surrounding text when predicting the next token.

This matters for:

- long dependency modeling
- coherent paragraphs
- long-range references

### Tradeoff

Longer context means:

- much higher memory usage
- more compute per example
- slower training and evaluation

### Typical choices

- small experiments: 128 or 256
- moderate runs: 256 or 512
- longer context experiments: 1024+ when the task is designed for it

### Rule of thumb

If you are unsure, start with 256. It is a practical default for small scale experiments.

---

## 6) `--batch` : batch size

Batch size controls how many examples are processed before an optimizer step.

### What it changes

- larger batch: more examples per update, often more stable and faster wall-clock throughput
- smaller batch: more stochastic, can help in some learning patterns, but slower and noisier

### Why it matters

Large batch sizes are often better for throughput, but sometimes worse for optimization dynamics.

Small batches can produce noisier but more exploratory updates.

### Recommended starting point

For small GPU experiments, `--batch 2`, `--batch 4`, or `--batch 8` are common starting points.

### Very large batch sizes

Very large batches can sometimes reduce training quality if the learning rate is not adjusted accordingly.

### You usually tune this together with `--lr`

If you increase batch size, consider whether the learning rate still makes sense.

---

## 7) `--seed` : randomness and reproducibility

The seed controls the random initialization and the randomness in the data ordering and training dynamics.

### Why it matters

This is important because small changes in initialization can lead to visibly different early training curves.

### Why seeds are important for comparison

If you compare fan-in 4 vs fan-in 6, you want to hold the seed constant when possible.

Otherwise you might be observing randomness rather than a real architecture difference.

### “Golden seed” concept

A “golden seed” is a seed that gives a strong, stable, repeatable run for your task.

In practice, people often:

- test several seeds
- keep the one that gives the best steady baseline
- then compare architecture changes against that seed

### Common practice

- use a fixed seed for controlled experiments
- if testing exploration, try 2–5 seeds and compare the best or mean performance

### Rule of thumb

For a fair comparison, do not change both the seed and the architecture at the same time unless you really want to test the combined effect.

---

## 8) How large or small should the model be?

This is one of the most common questions.

There is no single perfect size, but the usual logic is:

### Tiny models

- `d=16` to `d=32`
- `layers=2` to `4`
- `ctx=128` to `256`
- often used for quick debug and iteration

These are useful for:

- testing the code path
- seeing if the training loop works
- probing whether the data and objective are behaving

### Small models

- `d=32` to `d=64`
- `layers=4` to `8`
- `ctx=128` to `256`
- good for low-cost experiments

This is often the sweet spot for fast iteration.

### Medium models

- `d=64` to `d=128`
- `layers=6` to `12`
- `ctx=256` to `512`
- often stronger and more stable

This is a common range for model research.

### Large models

- `d=128+`
- `layers=12+`
- `ctx=512+`
- expensive and slower

Only use this when the task and compute budget justify it.

### General rule

If you are not sure where to start, begin with:

- `d=64`
- `layers=6`
- `ctx=256`
- `batch=4`

This is a standard small baseline for this kind of training problem.

---

## 9) Capacity vs compute

This is the most important practical relationship.

Increasing capacity is not free. Bigger models can do better, but they also:

- train slower
- use more VRAM
- require more careful tuning
- need more data or more training steps to pay off

### The real question

Is the capacity increase helping enough to justify the compute cost?

### Decision rule

If a model is larger but does not improve best-so-far metrics meaningfully over a sustained window, it is probably not a good trade.

---

## 10) A quick capacity guide

| Range | Typical use |
|---|---|
| tiny | quick debugging, verifying script works |
| small | fast experiments, architecture sweeps |
| medium | realistic research runs |
| large | expensive experiments, long training |

For this script, the sweet spot for many runs is small to medium, not huge.

---

## 11) Summary

The architecture knobs define the model’s potential.

- `d` = width / representation size
- `layers` = depth / abstraction depth
- `fan_in` = interaction richness
- `ctx` = context window length
- `batch` = example count per optimization step
- `seed` = reproducibility and variation control

More capacity can help, but only if:

- the model is trained long enough
- the optimizer settings are stable
- the learning schedule is appropriate
- the added compute is worth the gain

---

## Next document

See [03 - Stability, entropy, and schedule control](03-stability-entropy-and-schedule.md) to learn how the training schedule controls learning, confidence, and when the hard objective turns on.
