# 03 - Stability, entropy, and schedule control

This document explains what it means for training to become unstable and how the schedule controls the model’s behavior over time.

The most important knobs here are:

- `--lr`
- `--clip`
- `--ent_max`
- `--ent_warmup`
- `--perm_w`
- `--hard_start`
- `--hard_ramp`
- `--hard_w`

These are the knobs that decide whether the model learns smoothly or becomes fragile.

---

## 1) What “training becomes unstable” means

A model is unstable if the optimization behavior becomes erratic.

### Common symptoms

- loss suddenly spikes upward
- soft BPC jumps up instead of down
- hard BPC explodes or remains extremely high for too long
- entropy oscillates wildly
- the model becomes overconfident too quickly
- the model fails to recover from early mistakes

### In plain English

The model is trying to optimize too aggressively, or the objective is too hard at the wrong time.

### A quick example

If `--lr` is too high, the optimizer takes giant steps. The model may bounce around the optimum instead of converging.

If `--hard_w` is too large too early, the model is pushed into hard objectives before it understands the easy structure of the data.

That often produces exactly the kind of hard-BPC spikes you see in the logs.

---

## 2) `--lr` : learning rate

This is the most important knob for optimization.

### What it does

It controls the size of parameter updates.

### If it is too low

- learning is slow
- model may take too long to improve
- training can appear stuck even though it is not broken

### If it is too high

- updates become too large
- optimization may oscillate
- training becomes unstable
- loss and eval metrics may jump up and down

### Simple mental model

Think of learning rate like the step size in a hill-climbing problem.

- too small: you crawl slowly
- too large: you overshoot and bounce around

### Typical best practice

Start modestly. If the run is stable and improving, you can then try a slightly higher value only if needed.

### Very important pairing

`--lr` should be considered together with:

- `--hard_w`
- `--clip`
- `--batch`
- `--fan_in`

A bigger model or more aggressive objective often needs a smaller learning rate or more careful clipping.

---

## 3) `--clip` : gradient clipping

Gradient clipping limits how large a gradient update can be.

### What it does

It protects the model from exploding updates.

### Why it helps

If gradients become too large, optimization can become unstable or chaotic.

### When to use it

Use clipping whenever:

- training is noisy
- loss spikes seem out of control
- hard objective is being ramped aggressively

### Interpreting it

- higher clipping = more protection
- lower clipping = more risk of instability

### Typical values

`1.0` is common and often a good default for small research training.

---

## 4) Entropy control

Entropy is the uncertainty in the probability distribution.

### Why it matters

If the model becomes too confident too early, it can overfit poorly or collapse into brittle predictions.

The script attempts to regulate that behavior.

---

## 5) `--ent_max` : maximum entropy target

This sets the highest entropy the model is allowed to maintain.

### What it does

It prevents the distribution from becoming too spread out or too flat.

### When it is high

- the model is allowed to stay uncertain longer
- more smoothing
- slower or more cautious learning

### When it is low

- the model is pushed toward sharper confidence earlier
- can learn faster in the short term
- can become overconfident or brittle if too low

### Simple interpretation

`--ent_max` is like a thermostat for uncertainty.

- too low: model becomes overconfident too early
- too high: model stays fuzzy and underfits

### Typical use

This is often tuned after observing whether the entropy is collapsing too aggressively.

---

## 6) `--ent_warmup` : entropy warmup

This controls how long the entropy mechanism is gradually introduced.

### What it does

It avoids suddenly enforcing entropy constraints before the model has learned a usable signal.

### Why it helps

If you turn entropy control on too early, the model can get confused or oscillate.

### If you increase this

- entropy regularization starts more gently
- training may be smoother
- learning can be slower at first

### If you decrease this

- entropy control kicks in sooner
- model may become more controlled and less noisy earlier
- but it can also become overly constrained for the wrong stage of training

---

## 7) `--perm_w` : permutation / structure weighting

This is a structural weighting term.

### What it does

It changes how strongly a permutation-like or ordering constraint contributes to learning.

### In plain English

It is another shaping term that keeps the model from becoming too random or too overconfident in its internal structure.

### Usually not the first knob to change

This is more of a secondary shaping term. It is useful, but it is not usually the first thing to tune.

### When it matters

It matters when training dynamics are otherwise stable but the model is not capturing a useful internal ordering or structure.

---

## 8) `--hard_start` : when the hard objective begins

This is one of the most meaningful schedule features.

### What it does

The hard objective is not activated immediately. It starts later, after the model has had some time to learn the easier structure.

### Why this is useful

If you force the hard objective too early, the model is trying to solve the hardest cases before it understands the basics.

That often causes:

- poor early performance
- large hard-BPC spikes
- unstable entropy behavior
- noisy or washed-out learning

### When to increase it

Increase `--hard_start` if:

- hard BPC is exploding early
- the model looks unstable in the first few thousand steps
- training seems chaotic before the model has settled

### When to lower it

Lower it if you want the model to start learning the harder cases earlier, but only if the model is stable enough to handle it.

---

## 9) `--hard_ramp` : how quickly the hard objective turns on

This is the ramp duration for the hard component.

### What it does

Instead of switching the hard objective on all at once, it is increased gradually.

### Why ramping helps

A sudden jump in objective difficulty can destabilize training.

A gradual ramp gives the model time to adapt.

### If you increase this

- the transition becomes smoother
- fewer sudden spikes
- slower onset of the harder learning phase

### If you decrease this

- the hard objective comes online faster
- training can get more difficult earlier
- more chance of unstable dynamics

### Good practical interpretation

This is a smoothing parameter for the hard learning schedule.

---

## 10) `--hard_w` : hard objective weight

This is the weight applied to the hard objective.

### What it does

It tells the model how strongly to care about the harder learning signal.

### If it is low

- the model focuses more on easy structure
- training is smoother
- harder cases may be ignored too long

### If it is high

- the model is pushed harder on difficult examples
- stronger learning pressure
- risk of instability and noisy metrics

### Two very important practical facts

1. High `hard_w` does not necessarily mean better quality.
2. Early hard pressure can produce weird hard-BPC spikes even when the model is improving overall.

### Use it carefully

If hard BPC is terrible early on, do not immediately assume the model is bad. It may simply be that the hard objective is too strong at the wrong time.

---

## 11) Why the schedule matters so much

The model is trained with a controlled progression, not a static objective.

The training schedule helps the model do this in stages:

1. learn easy structure
2. gradually introduce harder objectives
3. keep confidence under control with entropy shaping
4. let the optimizer stabilize as the model gets stronger

This is better than throwing all of the pressure at once, because it reduces the chance of unstable training.

---

## 12) Common unstable patterns and how to fix them

### Pattern A: loss spikes hard early

Possible causes:

- `--lr` too high
- `--hard_w` too high
- `--hard_start` too early
- `--hard_ramp` too short

Fixes:

- lower `--lr`
- lower `--hard_w`
- increase `--hard_start`
- increase `--hard_ramp`
- add more clipping

### Pattern B: entropy collapses immediately

Possible causes:

- `--ent_max` too low
- entropy control starts too aggressively
- model is overconfident too soon

Fixes:

- raise `--ent_max`
- increase `--ent_warmup`
- reduce hard pressure

### Pattern C: soft BPC improves, hard BPC stays bad forever

Possible causes:

- too little hard supervision
- hard objective not turned on soon enough
- hard metric is still in the noisy phase

Fixes:

- lower `--hard_start` a bit
- increase `--hard_w`
- allow more training time before deciding the setting is bad

### Pattern D: training flatlines but no obvious crash

Possible causes:

- learning rate too low
- model too small or too large for the task
- fan-in not helping
- schedule is not aggressive enough for the objective

Fixes:

- try small `--lr` increase
- revisit `--fan_in`
- increase training length if you suspect slow progress

---

## 13) Very practical rule: change one knob at a time

When tuning a run, do not change five things at once.

That makes it impossible to know what caused the change.

Good experiment discipline:

- keep the seed fixed
- change one main knob
- run enough steps to see a real trend
- compare with same schedule and same data

---

## 14) Summary

Training stability is controlled mostly by:

- learning rate
- gradient clipping
- entropy shaping
- hard objective timing and strength

If the schedule is well tuned:

- loss improves smoothly
- soft BPC moves down
- entropy falls in a controlled manner
- hard BPC may be noisy but eventually stabilizes

If the schedule is badly tuned:

- the run looks chaotic
- metrics explode or flatline
- the model never recovers

---

## Next document

See [04 - Parameter reference](04-parameter-reference.md) for a full list of all command-line arguments and what each one is meant to do.
