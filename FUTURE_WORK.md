# Future Work

ZeroFLops V6 is the verified baseline. Everything below starts **from a
reproduced V6 run**, not from the drifted v9/v10 forks.

## 0. Prerequisite: verified V6

The headline number (Hard BPC 3.071) currently comes from a single seed
evaluated on a 128-sequence subsample. Before any scaling work:

- reproduce V6 (d=32, layers=3, fan_in=3) on the **full** validation split
- across ≥3 seeds, report mean ± std for soft BPC and hard BPC
- re-derive the bigram/trigram baselines on the same split, with the fraction
  of validation contexts actually observed in training

If the 0.016 BPC margin over trigram does not survive multiple seeds, the
paper's claim changes shape. Find out before scaling.

## 1. Eliminating the readout (the real "zero FLOPs")

The integer trunk is done: embedding, GF mixing and S-boxes are XOR + table
lookup only. The **readout is not** — it is a learned `float32` tensor of
shape `(layers+1, d, 256, 256)` applied by summation, and it accounts for
~97 % of the arithmetic per character (see `src/inspect_artifact.py`).

Options, in increasing order of ambition:

1. **Quantize the readout to int8/int16** with a per-channel scale. Trivial to
   implement; measure the BPC cost. The V9/V10 code already claims an int16
   readout export path — verify whether it was ever measured.
2. **Factorize the readout** as `(layers+1, d, 256)` × `(256, 256)`-free
   mixture: replace the sum over `d` row-fetch-and-add steps with a smaller
   number of composite lookups.
3. **Move the decision into the trunk**: have the last layer's S-box directly
   emit a byte distribution, so no float readout is needed at all. This is
   the version that would make "zero floating-point at inference" literally
   true.

## 2. Capacity: d=64 and beyond

V7 (d=64, layers=6, fan_in=4, lags to 256) exists as code and as a 118 MB
artifact, but no run of it has ever produced a logged metric. Its program size
is dominated by the readout, which grows as `O(d · layers · 256²)`.

- d=64, layers=6: 117.7 MB (measured)
- d=128, layers=6: ~235 MB — the readout, not the trunk, is the scaling wall

Any capacity increase must be paired with readout work from §1, or the model
just gets bigger without getting cheaper.

## 3. Corpus: 1 MB → 200 MB → 1 GB

Corpora exist at `../ZeroFLops/smollm_subset_{20kb,50MB,200MB,1024MB}.txt`.
Train at fixed architecture and vary only the corpus, then measure against
baselines computed on the *same* split.

Byte-level BPC falls sharply with corpus size, so a 1 GB result is not
comparable to the Tiny Shakespeare number. Report the corpus with every BPC.

## 4. Mixture of Experts — redo from scratch

The earlier MoE attempts (`../ZeroFLops/runs/gf64_moe_test_*`) **collapsed**:
`router_entropy` reached exactly 0.0 while `aux_loss` sat at ~6.1. No expert
diversity. Treat those runs as a negative result worth recording, not a
starting point.

If MoE is retried:

- start from a verified V6 config; routing is the only change
- log the per-expert usage histogram **from step 0**, so collapse is visible
  immediately rather than at the end of a long run
- treat `router_entropy ≈ 0` as a failed run, not a tuning detail
- the load-balancing auxiliary loss needs a loss-biasing coefficient to
  actually move `aux_loss`; check the gradient reaches the router

## 5. Engineering

- `runs/` naming convention: `<arch>_<corpus>_d<d>L<layers>_fan<n>_s<seed>_<date>/`,
  each with `config.json`, `data.json` (incl. corpus sha256), `logs/*.csv`,
  `NOTES.md`
- give `ZeroFLops/` version control — it is 13 GB with no git and no backup
- decide LFS vs raw for the 118 MB artifact
- `generate.py` hardcodes `C:\Users\Chris4K\runs\...` and imports
  `originalGSM`, which is not in `src/` — fix before shipping
- reproduce the MoE hardening insight (if any) on the integer path: routing
  must also compile to a table lookup, or the MoE version is not an integer
  program at all