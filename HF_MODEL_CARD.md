---
license: mit
tags:
  - language-modeling
  - character-level
  - enwik8
  - hard-expert
  - dynamic-evaluation
---

# ZeroFLops enwik8 language models

Checkpoints from the hard-residual LM research line of the ZeroFLops project.
The code that produced and evaluates these checkpoints lives in the
companion GitHub repository: **[Ckal/ZeroFLops-paper](https://github.com/Ckal/ZeroFLops-paper)**.

## What this model is

A character-level language model over enwik8 with a **hard-expert** forward
pass. Instead of a dense softmax over the vocabulary, the model routes each
position through a set of discrete *experts*:

- **n-gram experts** over byte contexts of increasing length
- a **word expert**
- a **document cache** — counts of the bytes seen in the current document

Only one expert fires per position, and each expert's output is an exact
integer count. That is what "zero FLOPs" refers to on this side of the
project: the expensive arithmetic happens in the *bookkeeping* (counting),
not in a floating-point matmul.

## Files

| file | what it is | sha256 |
|---|---|---|
| `enwik8_e8b_cache_best.pt` | e8b_cache — the strongest dynamic-evaluation model (val 1.0463, dynamic **0.9756**) | `f916c445e448f07478de8d31e3bc707898a0675dfec44f947a17127ae50f594c` |
| `enwik8_e8l_cache_best.pt` | e8l_cache — the longest-trained model (400k steps, val 1.0350, dynamic 0.9795) | `8f51f30dbdf5a43f82e7d2d686edd4127434ec878b984381a2662d76ab0cc636` |
| `gf256_hard_model_v7.pt` | the GF(256) Connect Four V7 hard checkpoint — a *different* research line, published here so the artifact is not only on GitHub | `e196957c349be328b50edd975e6bba9a7a47115c49139279ebb85ab882d2d827` |

All three were re-downloaded from this repo and hash-verified against the
local originals.

## Measured results (enwik8, standard 5 MB test split)

Bits per byte, lower is better. 1.20 BPC means the text can be stored in 15%
of its size.

### Static evaluation

| model | params | context | steps | val | test | probe |
|---|---|---|---|---|---|---|
| e8b_cache | 86.3 M | 1024 | 250k | 1.0463 | 1.0270 | 56.9% |
| e8l_cache | 39.5 M | 512 | 400k | 1.0350 | 1.0176 | — |
| e8b_alone | 85.3 M | 1024 | 250k | 1.0917 | 1.0750 | 69.2% |
| e8f_alone | 38.8 M | 512 | 200k | 1.1193 | 1.1050 | 63.1% |

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

Published reference: Transformer-XL 277 M with dynamic evaluation reaches 0.94
on the same test file.

### What the memory is worth

The ablation in the paper (`paper/analysis_e8l/analysis.json`) isolates each
component:

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
python run_all.py --stage gold
python run_all.py --stage e8final

# evaluate
python run_all.py --report

# dynamic evaluation
python run_all.py --stage dyn
```

The corpora and the trained checkpoints are not in git — they are
reproducible from the commands above. The full parameter surface and the
download commands are documented in `RESEARCH_STATUS.md` in the GitHub repo.

## Caveats

- The probe column is the fraction of positions where the model ranks the
  legally playable move first. It is a diagnostic, not a win rate.
- The `e8l_cache` run has no `alone` counterpart, so its memory contribution
  is inferred from the ablation rather than measured directly on that
  checkpoint.
- The cross-domain transfer claim (code / wiki_de / wiki_en) was never
  measured. The domain stage is prepared but has not been run.

## Citation

TBD — the paper is in preparation.
