# §0h — ROOT CAUSE ISOLATED 2026-10-27: the READOUT collapses, not the trunk

## The question this section answers

> "Should we stop and start a new project, or do you see a way to solve and
> get a working model — one that does not only pick centre and can prevent
> a win, i.e. pick the place that stops the other one winning?"

**Answer: keep the project. The defect is in the READOUT, not the
architecture, and it is now measured.**

This section supersedes the earlier "the GF trunk cannot represent the
policy" hypothesis in §0g/§0f. That hypothesis is **refuted** — see the
board-disjoint probe below.

---

## 1. The measurement that isolates the defect

`tg_readout_decomp.py` decomposes the shipped hard-export column logits
by path. The shipped logit is

```
logits[v] = readout0[ar, h_emb].sum(2)[v]
          + sum_L readoutL[ar, h_L].sum(2)[v]
          + bias[v]
```

where `readout` is a `(d, 256, 256)` table and each `h_L` holds `d`
GF(256) **values** per slot. Run on
`runs/_probe/lm_b_dense_full/checkpoints/step3000_hard.pt` over the
1,500-position disjoint holdout:

| path | offset gap | board std | ratio | argmax = c3 |
|---|---|---|---|---|
| readout0 (embedding) | 0.0163 | 0.0000 | 0.00 | 0 / 1500 |
| layer1 | 0.2337 | 0.2718 | 1.16 | — |
| **layer2** | **2.5759** | **0.4683** | **0.18** | **1275 / 1500** |
| layer3 | 0.1404 | 0.0401 | 0.29 | — |

**layer2 alone picks c3 on 1275/1500 positions.** Its
board-INDEPENDENT per-column offset (2.58) is **5.5× larger** than its
board-dependent spread (0.47). The readout learned the teacher's marginal
column distribution ("c3 is usually right"), and that constant alone
wins the argmax on ~91% of positions — so the network never needs to
read the board.

The explicit `readout_bias` is **innocent** (±0.005). The shortcut lives
inside the trunk readouts, not in the bias term.

Corroborating measurements:

* soft path shows the same collapse (~91% c3, 0% for c0/c1/c6) — the hard
  exporter is not the culprit.
* train strict 24.2% vs holdout strict 25.4% vs ALWAYS-c3 baseline 25.7%
  → the model **underfits to the constant-c3 shortcut**; it is not
  overfitting.

---

## 2. The decisive probe — and the leaky-split trap

A linear probe on the same representation is decisive, because the
shipped logit is *exactly* a linear map of the one-hot encoding of the
hidden slots. A probe on those features is the same model class as the
shipped readout, with a freely-fitted readout instead of the trained one.

### The trap

`lm_full` has **18,783 labels over only 3,856 distinct canonical
boards**. A random **row** split puts near-duplicate boards (and their
mirrors) on both sides, so "val" measures memorisation of board twins.
That leaky split produced a misleading result:

| split | layer1 | layer2 | layer3 | ALL |
|---|---|---|---|---|
| random-row holdout | 0.2467 | 0.2313 | 0.2120 | 0.2533 |
| ALWAYS-c3 baseline | 0.2567 | 0.2567 | 0.2567 | 0.2567 |

Every number *below* baseline — which reads as "the trunk cannot
represent policy".

### The correct split — by canonical board

`tg_probe_boarddisjoint.py` splits by **canonical board**: no board, and
no mirror of a board, appears on both sides. Fit on 15,377 jobs, validate
on 3,406 jobs from 3,856 distinct boards (baseline 0.2334):

| layer | fit | board-disjoint val | gain vs ALWAYS-3 |
|---|---|---|---|
| layer1 | 0.5490 | 0.2939 | **+0.0605** |
| layer2 | 0.6830 | 0.3142 | **+0.0807** |
| layer3 | 0.7541 | 0.2613 | +0.0279 |
| **ALL (sum)** | **0.8309** | **0.3582** | **+0.1248** |
| ALL, class-balanced | 0.7992 | 0.3306 | +0.0972 |

**The GF trunk contains generalisable board policy.** The sum of all
layers beats the ALWAYS-c3 baseline by **+12.5 points** on boards it has
never seen, and layer2 alone by +8.1. The architecture is sound.

**Diagnosis: the trained readout (and the loss that fitted it) is at
fault, not the trunk and not the data.**

---

## 3. The two implemented fixes

### (a) `--col-prior tau` — logit adjustment in the loss

`src/c4/train.py` now subtracts `tau * log p_teacher(c)` from column c's
logit inside `_weighted_move_loss`. With `tau=1` a **uniform** predicted
distribution becomes optimal, so the only way to reduce the loss is to
learn the board-dependent part. The constant offset stops paying off.

The prior is measured from the training sidecar only (never the holdout):

```
c0=-2.639 c1=-2.344 c2=-1.899 c3=-1.011 c4=-1.908 c5=-2.340 c6=-2.599
```

This is a **training-side** change; the hard export is untouched and
inherits the corrected weights. It is not a post-hoc calibration — a
holdout-fitted calibration would be an invalid release result.

### (b) `tg_readout_refit.py` — serve the readout that holds the policy

Refits the served logit to the path that actually carries the signal,
with the constant per-column component removed (`mu` computed on the FIT
split only). It saves a **normal hard checkpoint**, so the shipped loader
serves it with no code change.

---

## 4. What the model can already do (honest numbers)

Paired tactical audit — 307 **identical**, model-independent threat
positions, so the arms are directly comparable:

| arm | blocks the win | constant-c3 | random |
|---|---|---|---|
| dense lags + full-game | **47.2%** | 26.7% | 18.9% |
| sparse lags + full-game | 40.1% | 26.7% | 18.9% |
| dense lags + curriculum | 36.5% | 26.7% | 18.9% |
| sparse lags + curriculum | 26.1% | 26.7% | 18.9% |

The best arm blocks roughly **twice** as often as always-centre and ~2.5×
random. That is "sometimes prevents a loss" — real, but not yet reliable.

Strength (`src/c4.duel`, greedy model, 100 games, seed 2468):

| model | vs centre | vs heuristic |
|---|---|---|
| lm_a_sparse_full | 47.0% | 23.5% |
| lm_b_dense_full | 45.0% | 14.5% |

Baseline map for reference: `centre` beats random 90% but loses 0% to
heuristic, lookahead and minimax; `heuristic` and `lookahead` each beat
`centre` 100%.

---

## 5. Release status — NO V24

Not one hard gate passes yet. Mirror consistency is 6.2%, off-centre
accuracy 6.2%, strength vs heuristic 14.5%. Required order for any
candidate:

1. `tg_ab_eval.py --hard` (canonical hard evaluator)
2. ply-bucket breakdown
3. `tg_mirror_audit.py` (c1–c4 / non-centre, both seats)
4. `tg_block_test.py` (paired tactical audit)
5. `src.c4.duel` (canonical strength harness)
6. `src/generate.py` end-to-end smoke test

---

## 6. The c3-count discrepancy — RESOLVED

The layer decomposition's **total** picked c3 on **1367/1500** holdout
positions, while the canonical hard evaluator reported **1292/1500**.
Same artifact, same positions. **Resolved — there was never a model
bug.** Three different numbers came from three different questions:

| number | source | question it answers |
|---|---|---|
| **1250** | canonical hard evaluator, `shipped_eval.json` | c3 chosen **where c3 is legal** |
| **1367** | `tg_readout_decomp.py` total | c3 chosen by **unmasked** argmax over all 7 columns |
| **1500** | custom soft probe | probe artefact, not a model metric |

`tg_c3_reconcile.py` computes the same 7 logits both ways on the same
1,500 positions and explains the gap exactly:

```
UNMASKED argmax (decomp style): {0:0, 1:3, 2:54, 3:1367, 4:64, 5:8, 6:4}
MASKED   argmax (canonical)   : {0:2, 1:4, 2:92, 3:1250, 4:105, 5:24, 6:23}

positions where c3 is ILLEGAL:                 131
positions where the two argmaxes disagree:     134
raw picked c3 WHILE c3 was illegal:            125
```

**The masked count is 1250 — byte-identical to the canonical
evaluator's `model_col_dist['3'] = 1250`.** The decomposition script
takes `argmax` over all 7 columns *without* masking illegal ones, so on
the 131 positions where c3 is already full it still lets the large c3
constant win. Those 125 illegal-c3 picks are the entire discrepancy.

The "1292" figure was a **different model** — `lm_a_sparse_full`
(sparse lag arm, 28.0% strict), not the shipped dense arm. Comparing
numbers across arms is what created the apparent contradiction.

**Rules that follow:**

- Always mask illegal columns before taking `argmax`. A c3 count
  computed without the mask is not a model metric.
- `tg_ab_eval.py` is the single source of truth for column
  distributions. Do not compare its `model_col_dist` to a number
  produced by a different script.
- When two tools disagree, write a reconciliation harness that computes
  the quantity **both ways on the same positions** and reports the
  per-position difference — do not argue from totals.

---

## 7. The shipped baseline, exactly (for apples-to-apples comparison)

`runs/_probe/shipped_eval.json` — `lm_b_dense_full`, `step3000_hard.pt`,
1,500-position canonical-board-disjoint holdout, hard export:

| metric | value |
|---|---|
| strict accuracy | 28.467 % |
| off-centre strict | 6.188 % |
| model column distribution | c0:2 c1:4 c2:92 **c3:1250** c4:105 c5:24 c6:23 |
| c3 teacher-label accuracy | 92.99 % (n=385) |
| c1 teacher-label accuracy | 0.571 % (n=175) |
| c2 teacher-label accuracy | 9.66 % (n=238) |
| mirror consistency | 6.2 % |
| tactical blocking (307 paired) | 47.2 % |
| strength vs heuristic | ~14.5 % |

**This is the number to beat.** Any candidate that does not reach
≈28.5 % strict while raising non-centre use has not won; a candidate
that raises non-centre use but drops below ≈47 % blocking has failed the
primary tactical requirement.

---

## 8. What the symmetry fixes cost (measured, not assumed)

Two readout refits were completed and gated. Both improved symmetry and
non-centre behaviour; **both lost tactics**.

| metric | shipped | refit2 | refit3 |
|---|---|---|---|
| strict accuracy | 28.47 % | 25.20 % | 26.40 % |
| off-centre strict | 6.19 % | — | 16.50 % |
| mirror consistency | 6.2 % | 28.2 % | 29.6 % |
| c1 teacher accuracy | 0.6 % | 18.3 % | — |
| non-centre selections | 16.7 % | 51.3 % | — |
| c3 selected share | 83 % | 48 % | — |
| **tactical blocking** | **47.2 %** | **36.8 %** | **39.4 %** |
| strength vs heuristic | 14.5 % | 12.5 % | 16.5 % |
| strength vs lookahead | 10.5 % | — | 8.5 % |

The column-prior A/B (`--col-prior tau`) is the better lever, because it
acts during training rather than after:

| arm | strict | off-centre | mirror | blocking |
|---|---|---|---|---|
| tau 0.0 (control) | 28.47 % | 6.19 % | 6.2 % | 47.2 % |
| **tau 0.5** | 28.07 % | 5.38 % | **21.0 %** | **46.3 %** |
| tau 1.0 | 27.20 % | 4.66 % | 4.7 % | 39.1 % |

tau 0.0 reproduces the baseline **exactly** (28.467 %, c3 = 1250),
which proves the harness is sound. tau 0.5 triples mirror consistency
for a 0.9-point blocking cost. tau 1.0 over-corrects and loses.

**Negative result — diversity is not the missing ingredient.** A
controlled 2×2 (sparse/dense lags × diverse/narrow boards, labels and
ply profile matched within 0.6 %) gave:

```
sparse + diverse: 24.47 % strict
dense  + diverse: 24.40 % strict
dense  + narrow : 27.47 % strict
```

More distinct boards per label made it **worse**, not better.

---

## 9. Release status — NO V24

Not one hard gate passes yet. Required order for any candidate:

1. `tg_ab_eval.py --hard` (canonical hard evaluator)
2. ply-bucket breakdown
3. `tg_mirror_audit.py` (c1–c4 / non-centre, both seats)
4. `tg_block_test.py` (paired tactical audit)
5. `src.c4.duel` (canonical strength harness)
6. `src/generate.py` end-to-end smoke test — **PASSED 2026-10-27**
   (model loads, generates, exit 0)

---

## 10. Files

```
tg_readout_decomp.py      layer-by-layer column-logit decomposition
tg_probe_layers.py        per-layer linear probe (leaky row split; superseded)
tg_probe_boarddisjoint.py per-layer probe, canonical-board-disjoint split
tg_readout_refit.py       readout refit + optional checkpoint export
tg_readout_refit2.py      table-level readout refit (function class preserved)
tg_readout_refit3.py      same + mirror-consistency loss (target detached)
tg_c3_reconcile.py        explains 1250 vs 1367 vs 1500 — RESOLVED
run_colprior_ab.sh        tau = 0.0 / 0.5 / 1.0 A/B on dense full-game
run_fullcorpus_ab.sh      full 320k-corpus tau 0.0 vs tau 0.5 A/B
run_gate_fc.sh            canonical release gate for the full-corpus arms
runs/_probe/decomp_lm_b.json
runs/_probe/probe_bd_lm_b.json
runs/_probe/probe_layers_lm_b.json
runs/_probe/shipped_eval.json
runs/_probe/c3_reconcile.json
```
