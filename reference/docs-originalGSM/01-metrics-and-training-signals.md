# 01 - Metrics and training signals

This is the first document to read if you want to understand what the logs are really telling you.

Training is not only about raw loss. The script is designed to simultaneously observe multiple signals:

- loss
- soft BPC
- hard BPC
- entropy
- checkpoint quality
- sample behavior in generation

Those metrics each tell a slightly different story.

---

## 1) Loss

Loss is the training objective. It is the value the optimizer is trying to minimize.

### What it means

If the model predicts the next token poorly, loss is high. As it learns, loss goes down.

### What low vs high means

- lower loss = model is fitting the data better
- higher loss = model is struggling or undertrained

### Concrete examples

For a small transformer trained on English text, a rough ballpark is:

- loss around 3.5–4.5: very early, mostly random or barely learning
- loss around 2.5–3.5: definitely learning, but still rough
- loss around 1.8–2.5: good progress for a small model
- loss around 1.2–1.8: strong for a small-to-medium run
- loss below 1.0: very strong, but this depends heavily on data, context, and model size

These numbers are not universal. A small dataset, tiny model, or very short run can have a different baseline. But they are useful sanity checks.

### Important caveat

Loss is not always the best single judge of final quality. A model can have low loss but still produce weird or unnatural text. That is why additional evaluation metrics matter.

### What to watch

A run is usually healthy when loss is clearly trending downward over many steps.

A failed run often shows:

- loss oscillating wildly
- loss spiking repeatedly without recovery
- loss flatlining for long periods

A simple example:

- step 0: loss ~3.8
- step 500: loss ~2.9
- step 2000: loss ~2.4
- step 10000: loss ~1.9

This is a healthy progression. It is not magic, but it is exactly the kind of pattern you want to see.

---

## 2) Soft BPC

Soft BPC stands for soft bits-per-character.

### Why it matters

This is usually the clearest general indicator of model quality.

It answers roughly: "How well is the model predicting text on this evaluation set?"

### Interpreting it

- lower is better
- early training values are often much worse
- later values should move downward

### Concrete examples

For a small model on English text, rough soft-BPC ranges are:

- 3.6–4.0: near-random or very early model
- 3.0–3.5: clearly learning, still rough
- 2.5–3.0: good early-to-mid progress
- 2.2–2.5: strong for a small baseline
- below 2.0: very strong, but only if the model and data are large enough and the run is already mature

For example:

- step 0: 3.67
- step 500: 3.12
- step 1000: 2.99
- step 2000: 2.77
- step 5000: maybe 2.5–2.7

That is a healthy pattern. A run is improving if it keeps trending lower over time.

### Good pattern

A good run often looks like:

- step 0: high BPC
- step 500: lower
- step 1000: lower still
- continuing reduction over many thousands of steps

### Bad pattern

- soft BPC flat for a long time
- no sustained improvement over 10k+ steps
- high variance with no long-term downward trend

This usually means the run is not learning much, or the setup is not good for the amount of compute used.

---

## 3) Hard BPC

Hard BPC is a stricter signal than soft BPC.

### Why it exists

Soft BPC is smoother and more stable. Hard BPC tries to emphasize harder, more difficult prediction cases.

This is useful because the model may get the easy cases right but still struggle with the difficult or rarer ones.

### What makes it harder

Hard BPC is often:

- more sensitive to distribution mismatch
- noisier from run to run
- more volatile around training schedule changes

### Typical early behavior

It often spikes at the start of training or when the hard objective is introduced.

Example behavior:

- step 500: hard BPC very high
- step 1500: still high
- after hard schedule starts: it may suddenly improve or settle down

This can look alarming, but it is not necessarily a bad sign.

### How to interpret it

Look for the shape of the curve, not just one point.

A healthy hard-BPC curve usually has:

- a noisy early phase
- a later reduction
- eventual stabilization at a lower level

Concrete examples:

- step 0: 3.67
- step 500: 8.0 or even worse during early instability
- step 1500: still elevated
- after the hard schedule turns on: 3.6–4.0 is often a much healthier regime

This is not necessarily a failure. It often means the model is still calibrating the hard objective and the metric is very noisy. The important thing is whether it eventually settles down.

A bad hard-BPC curve often has:

- prolonged high values with no recovery
- repeated spikes that do not fade
- no meaningful downward trend even after long training

Example of a bad early signal:

- 8.0 at step 500
- 7.0 at step 1000
- 6.5 at step 1500
- still 6.0 at step 5000

That is likely a bad or poorly tuned hard objective setup.

---

## 4) Entropy

Entropy measures the spread of the model’s output distribution.

### In plain English

It tells you how uncertain the model is.

- high entropy = model is unsure, diffuse, spread-out predictions
- low entropy = model is very confident, concentrated predictions

### Why it matters here

This script explicitly uses entropy control.

That means entropy is not just a passive metric; it is part of the optimization strategy.

### Healthy entropy pattern

A good run often shows:

- entropy drops as the model learns
- entropy does not collapse too early
- it settles into a healthy range rather than hitting extremes

Concrete examples:

- entropy around 0.8–1.0 at step 0: very early, not yet trained
- entropy around 0.2–0.4 after a few hundred or a few thousand steps: common for a learning model
- entropy around 0.05–0.15: quite confident, but still in a reasonable range for some small models
- entropy below 0.02: usually extremely sharp and can be brittle or overconfident

A good trajectory often looks like:

- 0.83 -> 0.23 -> 0.12 -> 0.09 -> 0.07

That is a healthy tightening pattern.

### Unhealthy entropy behavior

- entropy stays too high for too long: underfitting or weak learning
- entropy drops to near zero too early: overconfident, brittle model
- entropy oscillates violently: unstable dynamics

A rough rule of thumb:

- if entropy is bouncing around 0.1–0.9 with no consistent motion, the model is probably unstable or poorly scheduled
- if entropy goes from 0.8 to 0.05 in a couple hundred steps, that can be too fast for some setups

### Simple rule

Entropy should usually move downward over time, but not collapse instantly and not stay wild forever.

---

## 5) Channel

The script uses multiple channels or objective pathways.

You can think of them as different ways the model is being evaluated or influenced.

### The two main channels

- soft channel: smoother learning signal, often more stable
- hard channel: tougher signal, more selective, often noisier

### Why the model has both

One channel helps the model learn broad structure.
The other channel pushes it to handle harder examples or sharper distinctions.

### What this means in logs

You may see:

- soft BPC dropping cleanly
- hard BPC delayed, noisy, or spiky

That does not mean the run is bad.

It means the model is learning in stages.

The model may not optimize both channels equally at the same time.

---

## 6) What “stable” vs “unstable” training means

This is one of the most important concepts.

### Stable training

A stable run shows:

- loss trending down
- soft BPC improving over time
- entropy not exploding or collapsing suddenly
- hard BPC eventually settling

### Unstable training

A training run becomes unstable when the optimization behavior starts to break down.

Examples:

- loss spikes dramatically and then does not recover
- soft BPC jumps upward repeatedly
- entropy oscillates wildly
- hard BPC keeps exploding with no long-term trend
- model quality gets worse while training continues

### Common causes of instability

- learning rate too high
- hard objective weight too large too soon
- entropy target too aggressive
- gradient clipping too weak
- model capacity too high for the current schedule
- run with bad initialization seed or bad data assumptions

### A good signal of recovery

Sometimes a run goes unstable for a few evaluations, but the model recovers. That can be okay if:

- it improves again later
- loss trends downward after the spike
- soft BPC eventually moves lower

That is different from a persistent failure mode.

---

## 7) What a healthy run looks like

A healthy run usually has a pattern like this:

- step 0 to ~500: fast gain
- step 500 to 2000: large improvements, some noise
- 2000+: continued but slower gains
- entropy decreases gradually
- soft BPC continues getting better
- hard BPC may still be noisy but should trend toward lower values

This is exactly the pattern you want during long training.

---

## 8) What a weak run looks like

A weak run often shows:

- little movement in loss after a while
- soft BPC stuck around the same value for many thousands of steps
- hard BPC remains poor or unstable
- entropy either refuses to settle or collapses too early

At that point, the problem is not necessarily the model being bad. It may be:

- wrong data settings
- wrong learning rate
- too much hard weighting
- too much fan-in for the setup
- not enough steps yet

---

## 9) The most useful interpretation rule

Do not judge a run from one number at one step.

Judge the run by:

- trend over multiple checkpoints
- best-so-far values
- whether the metrics improve over a meaningful window

A good rule is:

- compare 5–10 evaluations before making a conclusion
- or compare the best value so far, not the current single eval

This is especially important with hard BPC, which is often noisy.

---

## 10) Summary

The most important signals are:

- loss = direct training signal
- soft BPC = general quality signal
- hard BPC = harder and more selective signal
- entropy = confidence / uncertainty signal
- channel = soft vs hard training behavior

The model is usually healthy if:

- loss goes down
- soft BPC improves over time
- entropy eventually settles
- hard BPC may be noisy but trends in the right direction

The model is usually unstable if:

- loss and eval metrics oscillate badly
- entropy behaves erratically
- hard BPC stays inflated without recovery
- quality gets worse as training continues

---

## Next document

See [02 - Model capacity and architecture](02-model-capacity-and-architecture.md) for a deeper explanation of `d`, `layers`, `fan_in`, `ctx`, and how model size affects training behavior.
