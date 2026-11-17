# ZeroFLops — Two Language-Model Lines From One Repository

This repository holds two research lines. They share the same "counting beats
multiplying" idea and the same infrastructure, but they are separate papers with
separate artifacts.

| Line | What it is | Headline | Artifacts |
|---|---|---|---|
| **B — Hard-residual LM** | character-level LM on enwik8; a count table (byte contexts, words, document cache) is fed to the transformer as extra input dimensions | **0.9756 bits/byte** on enwik8 (below 1.0 BPC) with dynamic evaluation | [`Chris4K/zeroflops-enwik8-lm`](https://huggingface.co/Chris4K/zeroflops-enwik8-lm) |
| **A — GF(256) integer program** | character-level LM whose trained form compiles to pure integer ops (XOR, table lookups, adds) | **Hard BPC = 3.071** on Tiny Shakespeare, beating the trigram baseline (3.087) | `release/artifacts/`, `release/GF256_LANGUAGE_MODEL_PAPER.md` |

---

## Line B — hard-residual LM (enwik8)

### Result

| model | params | ctx | test BPC | test BPC, learning while reading |
|---|---|---|---|---|
| `e8b_cache` | 86.3 M | 1024 | 1.0270 | **0.9756** |
| `e8l_cache` | 39.5 M | 512 | 1.0176 | 0.9795 |

Published comparison on the same 5 MB test file: Transformer-XL 24 layers /
277 M reaches 0.99 static, **0.94 with dynamic evaluation** (Dai et al. 2019).
Our best is below 1.0 BPC while reading.

"Learning while reading" (dynamic evaluation): after each window is scored the
model takes one small training step on it, so it only ever learns from text it
has already been scored on.

### How it works

1. A dense residual transformer produces the output distribution normally.
2. A **hard table** — a count table over byte contexts, words and the current
   document — holds **81,785,402 integer entries**. It is built by counting, not
   by floating-point arithmetic.

The table is *not* the output. These checkpoints were trained with
`--hard_feature`, which feeds each expert's byte-prediction distribution and
its evidence counters to the transformer as **extra input dimensions**: 256
logit values for each of the 5 experts, plus 23 statistics — 1303 extra
dimensions alongside the token embedding. The transformer then learns how much
to trust each expert.

The table mechanism is the "zero FLOPs" idea *for the lookup side*: the memory
is integer counting, so no matrix multiply is spent on it. The published
checkpoints are **not** zero-FLOP models overall — they still run a full float
transformer. A separate table-only mode (`--mode hard`) exists for the pure
counting configuration.

### Ablation (val BPC)

Measured on the **`e8l_cache`** checkpoint (39.5 M params, 512 ctx) in a
separate harness — see [`paper/analysis_e8l/analysis.json`](paper/analysis_e8l/analysis.json).
Its baseline ("memory as trained", 1.0630) sits above that run's training-time
validation (1.0350) because the two are measured on different window sets. Read
the rows as deltas against their own baseline.

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
| memory of a **different window** | 12.8174 |

The last row is the control that matters: feeding counts from the wrong window
collapses the model from 1.0630 to 12.8174 BPC. The mechanism genuinely reads
the current document; it is not a generic prior.

The sibling file `paper/analysis_e8/analysis.json` holds the same ablation for
the shorter `e8_cache` run (38.8 M params, 200k steps), with its own baseline
(1.0901 → 14.0315).

### Checkpoints and model card

<https://huggingface.co/Chris4K/zeroflops-enwik8-lm> — discussion #1 has the
full write-up.

| file | what it is | size |
|---|---|---|
| `enwik8_e8b_cache_best.pt` | e8b_cache — the strongest dynamic-evaluation model | 331 MB |
| `enwik8_e8l_cache_best.pt` | e8l_cache — the longest-trained model (400k steps) | 152 MB |
| `p_e8_tab_hard.pt` | the hard table (5 experts, 81,785,402 entries) used by **both** checkpoints | 1.13 GB |

**`p_e8_tab_hard.pt` is required.** Both checkpoints store the table path in
their `hard_path` field (`p_e8_tab_hard.pt`) and resolve it relative to the
checkpoint's own directory, so download all three files **into the same folder**.

### Where the numbers live

- [`paper/results.md`](paper/results.md) — the full results table, the dynamic
  evaluation table, the leave-one-out check and sample texts
- [`paper/analysis_e8l/analysis.json`](paper/analysis_e8l/analysis.json) — the
  ablation numbers above, as measured
- [`HF_MODEL_CARD.md`](HF_MODEL_CARD.md) — the model card, mirrored on Hugging Face
- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — the research log, the ablation
  table and the open questions

---

## Line A — GF(256) integer program

A character-level language model whose final trained form compiles to a pure
integer program: XOR, table lookups, integer additions. **Zero FLOPs at
inference.**

### Result

**Hard BPC = 3.071** on Tiny Shakespeare validation, beating the trigram
baseline (3.087 BPC).

### Contents

```
ZeroFLops-paper/
├── README.md                         (this file)
├── paper.md                          The paper (sections 1–10)
├── src/
│   ├── gf256_markov_lm.py            V6 training script (reproduces the result)
│   ├── generate.py                   Standalone text-generation driver
│   └── INFERENCE-WOW.ipynb           Inference + text-generation demo
├── data/
│   └── tinyshakespeare.txt           Validation corpus (1.1 MB)
├── release/
│   ├── artifacts/                    Trained integer programs
│   ├── GF256_LANGUAGE_MODEL_PAPER.md
│   └── CONNECT_FOUR_PAPER.md
└── reference/
    └── docs-originalGSM/             V1–V5 lineage + tuning playbook
```

### Reproducing the result (≤ 30 minutes on a single GPU)

```bash
# 1. Train V6 (~12k steps)
python src/gf256_markov_lm.py \
    --seed 32 \
    --steps 20000 \
    --batch 4 \
    --ctx 128 \
    --d 32 \
    --layers 3 \
    --ent_max 2.0 \
    --ent_warmup 1000 \
    --hard_start 12000 \
    --hard_ramp 4000 \
    --hard_w 0.5 \
    --lr 3e-3 \
    --eval_every 500

# 2. Load + run the artifact
jupyter nbconvert --to notebook --execute src/INFERENCE-WOW.ipynb
```

The notebook looks for `gf256_hard_model.pt` in the working directory. The
trained `.pt` (118 MB, sha256 `e196957c349be328b50edd975e6bba9a7a47115c49139279ebb85ab882d2d827`)
is published on Hugging Face as
[`gf256_hard_model_v7.pt`](https://huggingface.co/Chris4K/zeroflops-enwik8-lm).

### Hardware footprint of the trained model

| Component | Size |
|-----------|------|
| Embedding | 8,192 bytes |
| Coefficients | 288 bytes |
| Wiring | 576 bytes |
| S-boxes | 24,576 bytes |
| Readouts | ~33 MB |
| **Total program size** | **~33 MB** |

**Inference per character:** ~30,000 lookups + ~3,000 XORs. **No FLOPs.**

---

## Reproducing Line B

```bash
pip install -r requirements.txt
curl -L -o data/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
unzip data/enwik8.zip -d data/enwik8     # 100,000,000 bytes
python hard_residual_lm.py --preset enwik8 --hard_feature --cache --words \
    --steps 250000 --out runs
```

`RESEARCH_STATUS.md` §7 has the full data and reproduction commands.

## License

Code: see `src/` headers. Paper text: © 2026 Christof Kaller.
