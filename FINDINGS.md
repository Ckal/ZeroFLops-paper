# Verified Findings

Measurements taken on 2026-10-20 against the actual repo, not from the paper's
own tables. Reproduce each with the command given.

## F1 — The shipped artifact is V7, not V6

```
python src/inspect_artifact.py artifacts/gf256_hard_model.pt
```

`artifacts/gf256_hard_model.pt` (118 405 919 B) is **d=64, layers=6, fan_in=4,
lags to 256**. The paper (§2, §4.3, §4.5) describes **V6 as d=32, layers=3,
fan_in=3, ~33 MB**. The two do not match; 117.7 MB measured vs ~33 MB claimed.

`src/gf256_markov_lm.py` shipped with the paper has those same V7 defaults,
while §7.2's command line says `--d 32 --layers 3`. The documented command
cannot reproduce the documented results.

## F2 — "Zero FLOPs" is false for the program as shipped

`src/inspect_artifact.py` on the shipped artifact, per generated character:

| | count |
|---|---:|
| GF-multiply table lookups | 6,144 |
| S-box lookups | 1,536 |
| XOR operations | 4,608 |
| **integer subtotal** | **12,288** |
| float adds (readout summation) | 458,752 |
| float comparisons (argmax) | 256 |
| **floating-point subtotal** | **459,008** |

**97.4 % of the arithmetic is floating point.** The readouts are `float32`
tensors of shape `(layers+1, d, 256, 256)` and are applied by summation
(`R[ar, h].sum(2)`).

Defensible claim: **the trunk is integer-only** (embedding, GF mixing, S-boxes
= XOR + table lookup). The readout is not, and dominates both the arithmetic and
the model size (117.5 MB of 117.7 MB).

## F3 — The S-boxes really are permutations

All six S-boxes in the shipped artifact are bijective (256/256 distinct
outputs). The entropy penalty worked as intended. This is a genuine positive
result and is not in dispute.

## F4 — 75 % of the wiring is dead

Per layer, the fraction of `fan_in × d` paths with coefficient 0:

| layer | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| coef==0 | 136/256 | 173/256 | 191/256 | 192/256 | 192/256 | 192/256 |

The model converged to mostly using only its own identity path plus one or two
connections. Worth reporting as a finding — the dilated-lag wiring is far
richer than the trained model actually exploits.

## F5 — The trigram baseline is stronger than the paper claims

`python src/baseline_check.py --data data/tinyshakespeare.txt`, on the same
90/10 byte split the trainer uses:

| baseline | BPC |
|---|---:|
| uniform | 8.000 |
| bigram (add-one) | 3.597 |
| trigram stupid-backoff (α=0.2) | 2.495 |
| trigram stupid-backoff (α=0.4) | 2.453 |
| trigram stupid-backoff (α=0.6) | 2.428 |
| trigram interpolated abs-discount (d=0.5) | 2.853 |
| trigram interpolated abs-discount (d=0.75) | 2.569 |
| trigram interpolated abs-discount (d=0.9) | 2.461 |

paper claims: bigram 3.602 (≈ matches, 3.597), trigram **3.087**.

**A properly smoothed trigram reaches 2.43–2.85 BPC, i.e. 0.22–0.64 BPC
*better* than the paper's 3.071.** The "surpasses the trigram baseline" claim
in the abstract, §6.1 and §10 does not survive.

Validation context coverage is high — unigram 100 %, bigram 99.83 % (187
unseen), trigram 98.61 % (1 553 unseen) — so the trigram is not weakened by
sparse contexts, and backoff has little work to do. The paper's 3.087 trigram
most likely came from an insufficiently smoothed or unsmoothed estimator,
which is exactly the case where an unsmoothed model gets `log(0)` and must be
patched to look respectable.

## F6 — Baselines were not verified while the margin was knowable

The margin at issue is 3.071 vs 3.087 = **0.016 BPC (0.5 %)**, from one seed,
on a 128-sequence quick-eval subsample (~16 k chars) rather than the full
~111 k validation split. `src/baseline_check.py` now re-derives every baseline
in under a second, so this should have been checked before submission.

## F7 — Project hygiene

- `GF(256) Lagrange Field Transformer.md` at the repo root is a mangled export
  of `paper.md` (escaped underscores, flattened code fences, garbled LaTeX).
  Removed.
- `src/generate.py` hardcodes `C:\Users\Chris4K\runs\gf64\checkpoints\...`
  and imports `originalGSM`, which is not in `src/`.
- `paper.md` contains no citations at all — no Related Work section.
- In the sibling WIP tree `ZeroFLops/`: **9 of 14 run directories have no eval
  log**, including every 1 GB run. The MoE runs show `router_entropy = 0.0`
  with `aux_loss ≈ 6.1` — collapsed routing.
- Filenames in `ZeroFLops/` lie about versions: `v4_1.py` says v8,
  `gsm_v9.py` says v7, `originalGSM.py` says v7.

## F8 — V6 reproduces, but on the full validation set the paper's 3.071 is wrong

`python src/gf256_markov_lm_v6.py --run-dir runs/v6_s32 --seed 32`,
20 000 steps, full-validation eval:

| | BPC |
|---|---:|
| soft (full val, 111 456 chars) | **2.7001** |
| hard (full val, 111 456 chars) | **3.1727** |
| hard (best quick-val during training, 128 × 8-char chunks = 1 024 chars) | 3.1381 |
| paper's claimed hard | 3.071 |

The paper's `3.071` came from the **128-sequence quick-val subsample, not the
full validation split.** Re-running the same training on the same split the
trainer uses gives **3.17 BPC**, not 3.07. The model itself reproduces; the
*reported number* was from the wrong eval surface. Difference: 0.10 BPC,
3.3 % of the value, larger than the entire "beats the trigram" margin.

The artifact saved to `runs/v6_s32/hard_model_final.pt` (33.8 MB) is the
actual V6 model — `d=32, layers=3, fan_in=3, 16.78 M params`.

Compare to the smoothed trigram baseline on the same val (F5):

| | BPC |
|---|---:|
| paper's trigram | 3.087 |
| smoothed trigram (best) | 2.43 |
| V6 hard (full val) | **3.17** |

V6 with the paper's parameters is **0.74 BPC worse than a properly smoothed
trigram.** The headline claim "surpasses the trigram baseline" does not
survive contact with either the full val set or a smooth trigram.

What does survive: the **hardening curriculum works** — soft 2.74 → hard 3.17
with no collapse, and the S-boxes remain bijective through the STE phase.
That is a real result, but it is not a result about language modelling.

## Status

- Seed 33 running in background (`proc_547fa90b5d87`)
- Seed 34 to follow