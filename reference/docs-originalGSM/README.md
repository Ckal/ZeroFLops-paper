# originalGSM docs index

This folder contains a more complete reference for the training script and for understanding the knobs that control learning, stability, and capacity.

## Start here

- [01 - Metrics and training signals](01-metrics-and-training-signals.md)
- [02 - Model capacity and architecture](02-model-capacity-and-architecture.md)
- [03 - Stability, entropy, and schedule control](03-stability-entropy-and-schedule.md)
- [04 - Parameter reference](04-parameter-reference.md)
- [05 - Tuning playbook](05-tuning-playbook.md)

## Quick summary

If you only want the short version:

- loss = direct training signal
- soft BPC = general quality metric
- hard BPC = stricter, noisier metric
- entropy = uncertainty / confidence level
- `layer`, `d`, `fan_in`, `ctx`, `batch` control capacity and compute
- `lr`, `clip`, `hard_start`, `hard_ramp`, `hard_w`, `ent_max` control stability and optimization behavior

## Recommended reading order

1. [01 - Metrics and training signals](01-metrics-and-training-signals.md)
2. [02 - Model capacity and architecture](02-model-capacity-and-architecture.md)
3. [03 - Stability, entropy, and schedule control](03-stability-entropy-and-schedule.md)
4. [04 - Parameter reference](04-parameter-reference.md)
5. [05 - Tuning playbook](05-tuning-playbook.md)

## Good rule of thumb

- start with small runs
- change one thing at a time
- judge based on best-so-far metrics over a long enough window
- do not judge a run from a single hard-BPC spike
- compare at least 10k–30k steps before deciding a setting is bad
- when fan-in changes, expect more compute and slower training before you trust the output

---

## Great questions to ask while training

- Is loss still moving down?
- Is soft BPC still improving over a long window?
- Is hard BPC noisy but trending down, or is it flat and bad?
- Is entropy healthy, or is the model collapsing to overconfidence?
- Did the model become unstable only around the hard objective turn-on?
- Is the increase in compute worth the change in quality?

---

## Very short glossary

- instability: the model starts oscillating, exploding, crashing, or diverging in metrics
- entropy: how spread out the probability distribution is
- capacity: how much information a model can represent
- fan-in: how much mixing / interaction each unit has with its inputs
- BPC: bits-per-character, lower is better

---

## Quick example run

```bash
python originalGSM.py \
  --seed 64 \
  --data smollm_subset_1024MB.txt \
  --stream-data \
  --run-dir runs/gf64_1gb_fan_6 \
  --steps 600000 \
  --device cuda \
  --ctx 256 \
  --batch 4 \
  --eval-every 500 \
  --eval-seqs 64 \
  --eval-chunk-size 4 \
  --fan-in 6
```

This is a reasonable small-to-medium experiment. It is large enough to learn something meaningful, but still fast enough to iterate quickly.
