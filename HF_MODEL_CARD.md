---
license: mit
tags:
  - language-modeling
  - character-level
  - enwik8
  - count-tables
  - document-memory
  - dynamic-evaluation
---

# ZeroFLops enwik8 language models

Checkpoints from the hard-residual LM research line of the ZeroFLops project.
The code that produced and evaluates these checkpoints lives in the
companion GitHub repository: **[Ckal/ZeroFLops-paper](https://github.com/Ckal/ZeroFLops-paper)**.

## What this model is

A character-level language model over enwik8. Two parts:

1. **A dense residual transformer** (86.3 M parameters for `e8b_cache`) that
   produces the output distribution in the normal way.
2. **A hard table** — a count table over byte contexts, words and the current
   document, holding **81,785,402 integer entries**. It is built by counting,
   not by floating-point arithmetic.

The table is *not* the output. These checkpoints were trained with
`--hard_feature`, which means each expert's byte-prediction distribution and
its evidence counters are fed to the transformer as **extra input dimensions**:
256 logit values for each of the 5 experts, plus 23 statistics — 1303 extra
dimensions added alongside the token embedding. The transformer then learns how
much to trust each expert.

> **Naming caveat.** The table mechanism is the "zero FLOPs" idea *for the
> lookup side*: the memory is integer counting, so no matrix multiply is spent
> on it. The published checkpoints are **not** zero-FLOP models overall — they
> still run a full float transformer. A separate table-only mode (`--mode hard`)
> exists for the pure-counting configuration.

### The 5 experts

| expert | what it counts |
|---|---|
| `ngram_o2` | 2-byte contexts |
| `ngram` | the main expert — mid-length byte contexts |
| `ngram_long` | long contexts (8, 12, 16 bytes) |
| `word` | lowercase word tokens |
| `cache` | bytes seen so far in the current document |

## Files

| file | what it is | size | sha256 |
|---|---|---|---|
| `enwik8_e8b_cache_best.pt` | e8b_cache — the strongest dynamic-evaluation model (val 1.0463, dynamic **0.9756**) | 331 MB | `f916c445e448f07478de8d31e3bc707898a0675dfec44f947a17127ae50f594c` |
| `enwik8_e8l_cache_best.pt` | e8l_cache — the longest-trained model (400k steps, val 1.0350) | 152 MB | `8f51f30dbdf5a43f82e7d2d686edd4127434ec878b984381a2662d76ab0cc636` |
| `enwik8_e8b_alone_best.pt` | e8b_alone — the **no-memory** baseline (85.3M, test 1.0750, dynamic 0.9990). Needs no table file, so it stands alone | 334 MB | `ea92db1d8554e0534daf24998dcf162aec1aa37227fa8f8771f5d44b59bccf54` |
| `p_e8_tab_hard.pt` | the hard table (5 experts, 81,785,402 entries) used by **both** enwik8 checkpoints | 1.13 GB | `6ef63a411e96646766fb1bf813053ad698438aa7787ded99f9d7fefa183fd45d` |
| `gf256_hard_model_v7.pt` | the GF(256) Connect Four V7 hard checkpoint — a *different* research line, published here so the artifact is not only on GitHub | 113 MB | `e196957c349be328b50edd975e6bba9a7a47115c49139279ebb85ab882d2d827` |

**`p_e8_tab_hard.pt` is required.** Both enwik8 checkpoints store the table
path in their `hard_path` field (`p_e8_tab_hard.pt`) and resolve it relative to
the checkpoint's own directory, so download all three files **into the same
folder**. Without the table, the checkpoints cannot be loaded.

## Measured results (enwik8, standard 5 MB test split)

Bits per byte, lower is better. 1.20 BPC means the text can be stored in 15%
of its size.

### Static evaluation

`params` counts the transformer's parameters only. The hard table adds
81.8 M integer entries on top — as the paper notes, the table size has to be
reported next to the parameter count.

| model | params | context | steps | val | test | probe | table |
|---|---|---|---|---|---|---|---|
| e8b_cache | 86.3 M | 1024 | 250k | 1.0463 | 1.0270 | 56.9% | yes |
| e8l_cache | 39.5 M | 512 | 400k | 1.0350 | 1.0176 | — | yes |
| e8b_alone | 85.3 M | 1024 | 250k | 1.0917 | 1.0750 | 69.2% | no |
| e8f_alone | 38.8 M | 512 | 200k | 1.1193 | 1.1050 | 63.1% | no |

`e8b_alone` and `e8f_alone` are the same architecture **without** the table
inputs, so the table's contribution is the gap between the `*_cache` and
`*_alone` rows.

### Dynamic evaluation ("learning while reading")

After scoring each window the model takes one small training step on it, so
it only ever learns from text it has already been scored.

| model | test | test, learning while reading | gain | step size |
|---|---|---|---|---|
| e8b_cache | 1.0270 | **0.9756** | 0.0513 | rms 5e-06 |
| e8b_alone | 1.0750 | 0.9990 | 0.0760 | rms 1e-05 |
| e8f_alone | 1.1050 | 1.0481 | 0.0569 | rms 5e-06 |
| e8l_cache | 1.0176 | 0.9804 | 0.0371 | rms 5e-06 |
| e8l_cache | 1.0176 | 0.9795 | 0.0380 | rms 1e-05 |

The two `e8l_cache` rows are the **same checkpoint** evaluated at two learning
rates (5e-06 and 1e-05); the lower step size wins.

Published reference: Transformer-XL 277 M with dynamic evaluation reaches 0.94
on the same test file (Dai et al. 2019).

### What the memory is worth

The ablation in the paper (`paper/analysis_e8l/analysis.json`) isolates each
component. It was run on the `e8l_cache` checkpoint in a separate harness, so
its baseline ("memory as trained", 1.0630) sits above that run's training-time
validation (1.0350) — the two are measured on different window sets. Read the
rows as deltas against their own baseline, not against the table above.

| configuration | val BPC |
|---|---|
| memory as trained | **1.0630** |
| no long contexts (8+ bytes) | 1.1051 |
| no word expert | 1.0763 |
| no document cache | 1.1612 |
| byte contexts only up to 2 bytes | 1.1665 |
| evidence counts all set to 0 | 1.1135 |
| all tables empty (unigram only) | 1.2928 |
| no memory at all (uniform, no counts) | 6.5112 |
| memory of a *different* window | 12.8174 |

The last row is the control that matters: feeding the model memory from a
different window is catastrophic, so the mechanism genuinely reads the
current document rather than acting as a generic prior.

## Reproducing

```bash
# data
python download_hf_corpus.py            # enwik8 + web-text into data/

# train (the CFG block at the top of run_all.py holds every setting)
python run_all.py --stages gold
python run_all.py --stages e8final

# static evaluation -> paper/results.md
python run_all.py --report

# dynamic evaluation -> runs/p_dyn__*.json
python run_all.py --stages dyn
```

`--report` does **not** re-run dynamic evaluation, and the component ablation
is a separate artifact (`paper/analysis_e8l/analysis.json`), not part of
`--report`.

The corpora and the trained checkpoints are not in git — they are
reproducible from the commands above. The full parameter surface and the
download commands are documented in `RESEARCH_STATUS.md` in the GitHub repo.

### Loading a published checkpoint

```python
import argparse, torch
from hard_residual_lm import HardModel, build_model   # from the GitHub repo

# weights_only=False is required here: the checkpoint also stores the argparse
# namespace of the training run, which torch cannot load under weights_only=True.
ck = torch.load("enwik8_e8b_cache_best.pt", map_location="cpu", weights_only=False)
args = argparse.Namespace(**ck["args"]); args.device = "cpu"

hm = HardModel.load("p_e8_tab_hard.pt", "cpu").set_runtime(args)
model = build_model(args, hm)
model.load_state_dict(ck["model"])
model.eval()
```

## Caveats

- The probe is a **multiple-choice knowledge probe**: 65 fixed questions, each
  with 4 candidate answers, scored by which answer costs the fewest bits. The
  first answer is the correct one. It is a diagnostic, not a win rate — and it
  is *not* a game-playing benchmark.
- `e8l_cache` has no `alone` counterpart checkpoint, so its memory
  contribution is inferred from the ablation rather than measured directly on
  that checkpoint.
- The cross-domain transfer claim (code / wiki_de / wiki_en) was never
  measured. The domain stage is prepared but has not been run.
- The dynamic-evaluation rows depend on the learning rate used during
  evaluation; the numbers above are not reproducible from the checkpoint alone
  without the step size.

## Citation

TBD — the paper is in preparation.

## Further reading

- Discussion #1 — the release write-up (ablation, mechanism, file list).
- Discussion #2 — "0.9756 BPC on enwik8 — trained in 10 hours on one GPU",
  including the wrong-window control that shows the model reads the document
  rather than acting as a generic prior.
