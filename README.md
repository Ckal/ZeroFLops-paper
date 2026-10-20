# GF(256) Lagrange Field Transformer — Paper Package

A character-level language model whose final trained form compiles to a
pure integer program: XOR, table lookups, integer additions. **Zero FLOPs
at inference.**

## Result

**Hard BPC = 3.071** on Tiny Shakespeare validation, beating the
trigram baseline (3.087 BPC).

## Contents

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
├── artifacts/
│   └── gf256_hard_model.pt           Trained V6 integer program (118 MB)
└── reference/
    └── docs-originalGSM/             V1–V5 lineage + tuning playbook
        ├── README.md
        ├── 01-metrics-and-training-signals.md
        ├── 02-model-capacity-and-architecture.md
        ├── 03-stability-entropy-and-schedule.md
        ├── 04-parameter-reference.md
        └── 05-tuning-playbook.md
```

## Reproducing the result (≤ 30 minutes on a single GPU)

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

The notebook looks for `gf256_hard_model.pt` in the working directory;
the trained `.pt` shipped under `artifacts/` is byte-identical to the
one produced by the run above (see commit notes).

## Hardware footprint of the trained model

| Component | Size |
|-----------|------|
| Embedding | 8,192 bytes |
| Coefficients | 288 bytes |
| Wiring | 576 bytes |
| S-boxes | 24,576 bytes |
| Readouts | ~33 MB |
| **Total program size** | **~33 MB** |

**Inference per character:** ~30,000 lookups + ~3,000 XORs. **No FLOPs.**

## License

Code: see `src/` headers. Paper text: © 2026 Christof Kaller.
