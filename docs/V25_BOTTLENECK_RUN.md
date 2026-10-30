# V25b — bottleneck readout on the 640k mirror corpus

Branch: `feat/threat-vs-quiet-measure`. Run dir: `runs/_probe/v25b`, log `runs/_probe/v25b.log`.

## Question

Does removing the per-layer lookup readout force the GF(256) trunk to compute
rather than memorise? Measured by **holdout** label agreement (disjoint boards),
not training label agreement.

## What changed vs V6 (`src/gf256_markov_lm_v6.py`, untouched)

| | V6 | V6b (`src/gf256_markov_lm_v6_bottleneck.py`, `--arch v6b`) |
|---|---|---|
| readout | `(layers+1)` float tables `(d,256,256)`, one per trunk depth incl. the embedding, summed | one MLP on the LAST layer only: `d*256 -> 512 -> 256`, ReLU, dropout 0.1 |
| readout params (d32, L3) | 8.39M | 4.3M |
| total params | ~16.8M | 12.72M |
| trunk | unchanged | unchanged |

Notes, to keep claims honest:

- The old readout was **never integer**. It indexes float tables by integer states
  and sums floats (`FINDINGS.md`: ~97% of arithmetic is float readout). The
  zero-FLOP claim applies to the trunk only; V6b keeps that status.
- The first MLP layer on a one-hot input is itself a sum of `d` lookup tables
  (256 x 512 each), so V6b is a *factorised, rank-512, last-layer-only* readout,
  not a fundamentally different mechanism. What it really removes is the
  early-layer shortcut (embedding -> logits). Whether that forces "reasoning"
  is the hypothesis this run tests; do not assume it.
- Deploy path: `hard_forward` runs the integer trunk, then one-hot -> float MLP.

## Data (the valid pair)

- corpus `runs/_probe/perply_v23_full/corpus_320k_mirror_v3.txt` — 640000 rows,
  interleaved: even = original, odd = left-right mirror.
- targets `runs/_probe/perply_v23_full/merged_mirror_v2.npz` — verified:
  even rows == `merged_repaired.npz`, odd rows == columns flipped.

**Do not use** `corpus_320k_fixed_augmented.txt` (Hermes, 2026-10-30): it has
5,740,175 lines instead of 640,000, because `tools/mirror_corpus.py` re-encodes
games incorrectly. Its targets (`merged_repaired_augmented.npz`) are fine but are
ordered [originals, mirrors], not interleaved, so they cannot pair with the
valid corpus. The trainer refuses the mismatch (row-count guard).

## Prior result to compare against

`docs/CURRENT_TRUTH.md` §6: mirror augmentation on V6 left solver-optimality
tied at 71.8% (n=560). Expect mirror to help only invariance/seat symmetry, not
strength; the bottleneck is the new variable.

## How to run / resume

```bash
V24_ARCH=v6b V24_TAG=v25b V24_STEPS=80000 \
V24_CORPUS=runs/_probe/perply_v23_full/corpus_320k_mirror_v3.txt \
V24_SIDECAR=runs/_probe/perply_v23_full/merged_mirror_v2.npz \
bash run_v24.sh          # Git Bash on Windows: C:\Program Files\Git\bin\bash.exe
```

Resume: add `V24_RESUME=runs/_probe/v25b/checkpoints/stepNNNN_soft.pt` (float
snapshot, not a hard export). 80000 steps = 2 epochs of 640k rows at batch 16.

Score any run dir with `tmp/measure_promotion_scorer.py <run_dir>`. It now picks
the bottleneck class when the export has `readout_mlp`. UI server and `duel.py`
do the same.

## Verification done

- `tmp/test_bottleneck_smoke.py` passes (soft, export, hard, train step).
- 40-step end-to-end run: trainer LABEL 37.0% == independent scorer 37.02%.
- Not verified: a numeric soft-vs-hard parity check; soft/float serving of V6b via
  `soft_loader.py` (not wired; use hard exports).

## Success criteria

Compare against the V24 (V6) run at equal steps on HOLDOUT label agreement and
`LABEL`. If V6b holdout is not above V6, the memorisation hypothesis is not
supported by this architecture.

## Next (from the handover)

1. Longer run if the curve is still climbing.
2. Model + rule wrapper ("win if you can, block if you must") vs rule alone.
