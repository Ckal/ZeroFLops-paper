# V6 Config Ledger — canonical reproduction spec

**Status:** live run in progress (`runs/v6_s32`, seed 32)
**Authority order:** notebook `src/INFERENCE-WOW.ipynb` cell 5 > `src/gf256_markov_lm_v6.py` defaults > `paper.md`

The notebook is ground truth. The paper's §7 text is a *description* of that
run and is wrong in four places. Where they disagree, the notebook wins and the
paper gets corrected (see FINDINGS.md F9, F10, F13).

---

## 1. The exact command that produced the paper's headline

Notebook cell 5, `run_main_with_args([...])`:

```
--steps 120000  --batch 4  --seed 32  --ctx 128
--eval_every 400  --eval_seqs 64  --eval_chunk_size 4  --no_ckpt
```

Everything not on that list came from cell 4's `build_parser` defaults.

## 2. Full resolved hyperparameter table

| # | hyperparam | notebook (AUTHORITATIVE) | paper.md says | script default | match |
|---|---|---|---|---|---|
| 1 | `d` | 32 | 32 | 32 | ✅ |
| 2 | `layers` | 3 | 3 | 3 | ✅ |
| 3 | `fan_in` | **3** | **4** (§5.2) | **3** | ✅ fixed |
| 4 | `seed` | 32 | 32 | 32 | ✅ |
| 5 | `steps` | **120000** | **82400** | **120000** | ✅ fixed |
| 6 | `batch` | 4 | 4 | 4 | ✅ |
| 7 | `ctx` | 128 | 128 | 128 | ✅ |
| 8 | `lr` | 3e-3 | 3e-3 | 3e-3 | ✅ |
| 9 | `clip` | 1.0 | 1.0 | 1.0 | ✅ |
| 10 | `ent_max` | 2.0 | 2.0 | 2.0 | ✅ |
| 11 | `ent_warmup` | 1000 | 1000 | 1000 | ✅ |
| 12 | `perm_w` | 1.0 | 1.0 | 1.0 | ✅ |
| 13 | `hard_start` | **10000** | **12000** (§7.2) | **10000** | ✅ fixed |
| 14 | `hard_ramp` | **2000** | **4000** | **2000** | ✅ fixed |
| 15 | `hard_w` | 0.5 | 0.5 | 0.5 | ✅ |
| 16 | `lr_drop` | 0.2 (hardcoded at `hard_start`) | not stated | 0.2 | ✅ |
| 17 | `lags` | `(0,1,2,4,8,16,32,64)` | same | same | ✅ (F12) |
| 18 | `n_coef` | 8 | 8 | 8 | ✅ |
| 19 | `eval_every` | 400 | not stated | 500 | ⚠️ log-only |
| 20 | `eval_seqs` | 64 | not stated | 128 | ⚠️ log-only |
| 21 | `eval_chunk_size` | 4 | not stated | 8 | ⚠️ log-only |

Rows 19–21 affect only the *logged quick-val* curve, never the headline metric:
the final number comes from a separate pass over the **full** val split
(`evaluate(model, all_val, ...)`), which does not use `eval_seqs`.

## 3. What the notebook's own log shows

From cell 5's captured output (301 logged eval points):

```
step  82400 | soft 2.915 | HARD 3.081      ← the step the paper *names*
...
step 120000 | soft 2.964 | HARD 3.154      ← last logged eval
FINAL on full val (110592 chars): soft 2.902 | HARD 3.071   ← the number the paper *reports*
```

Best logged hard BPC anywhere in the run: **3.081** at step 82,400.
The reported **3.071** is the post-training full-validation pass.

## 4. Reproduction target

| quantity | target | source |
|---|---|---|
| FINAL full-val soft BPC | 2.902 ± ~0.02 | notebook cell 5 |
| FINAL full-val HARD BPC | **3.071 ± ~0.05** | notebook cell 5 |

Tolerance reflects that the notebook's own quick-val curve oscillates ±0.03
between adjacent evals; seed-level run-to-run variance is not yet characterised.

## 5. Baselines (independently re-derived, `src/baseline_check.py`)

| baseline | BPC | note |
|---|---|---|
| uniform | 8.000 | |
| bigram add-one | 3.597 | paper says 3.602 |
| trigram stupid-backoff α=0.6 | **2.428** | best proper trigram |
| trigram interpolated abs-discount d=0.9 | 2.461 | |
| trigram interpolated abs-discount d=0.5 | 2.853 | closest to paper's 3.087 |

**The paper's trigram figure of 3.087 is a poorly-tuned interpolated trigram.**
Properly tuned trigram reaches 2.43–2.46 BPC. The paper compares 3.071 against
3.087 to claim a 0.016 BPC win; the honest comparison is 3.071 vs 2.43, which is
a **loss of ~0.64 BPC**.

This is the single most important correction to the paper. It does not affect
whether the code reproduces — it determines whether the result is worth
publishing as an improvement.

## 6. Open items for the paper

- [ ] Confirm run lands at 3.071 ± 0.05 (in progress)
- [ ] Run seeds 31/33 for variance (3-seed requirement)
- [ ] Rewrite §5.2/§7 to the notebook config (F13 table)
- [ ] Restate the contribution honestly: not a win over trigram
- [ ] Related Work section (F11) — currently zero citations
- [ ] Decide: reframe as "integer-trunk LM with matched soft/hard parity"
      or pivot the claim