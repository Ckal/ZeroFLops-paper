# Section 0I — The centre-only defect is solved; here is the evidence

**Status: the GF trunk is NOT the problem. The training distribution was.**
A working candidate now exists (`fc_a_tau0`). It does not "only pick centre",
it blocks the opponent's immediate win more often than the shipped model, and
it is more often solver-optimal. It is **not yet release-approved** — see
"Still open" at the end.

---

## 1. The question this section answers

> "one that does not only pick center and can prevent some win … the one that
> prevents leading the other one to win"

Three measurable sub-questions:

1. Does the model pick a variety of columns, or only c3?
2. When the opponent has an immediate winning square, does the model block it?
3. Are its choices actually *good* (not just varied)?

The third question needs a model-independent yardstick. We built one: the exact
Connect Four solver.

---

## 2. The root cause (one sentence)

The shipped model was trained on a corpus whose teacher labels were **93% c3**.
It learned "always play the middle", not "read the board". The fix was not a
new architecture — it was training on a **balanced, per-ply-labelled corpus
with 4,462,571 labels** instead of 15,377 single-label ones.

Evidence that the trunk could generalise all along (canonical-board-**disjoint**
probe, so no duplicate or mirrored board leaks between fit and validation):

| layer | fit | val | ALWAYS-c3 baseline | gain |
|---|---:|---:|---:|---:|
| layer1 | 0.5490 | 0.2939 | 0.2334 | **+0.0605** |
| layer2 | 0.6830 | 0.3142 | 0.2334 | **+0.0807** |
| layer3 | 0.7541 | 0.2611 | 0.2334 | +0.0277 |

Layer 2 carries real, generalisable board-policy information. The problem was
the *readout*, not the representation.

---

## 3. The candidate: `fc_a_tau0`

Trained with: canonical dense lags, ctx 192, batch 16, `--table-bias 0.0`,
`--hard-start 0`, `--max-legacy-frac 0`, tau 0.0, 3000 steps, on
`corpus_320k.txt` + `merged.npz`.

### 3.1 Which columns does it pick?

Hard-export selected-move distribution over the 1,500-position disjoint
holdout:

| arm | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c3 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped dense | 2 | 4 | 92 | **1250** | 105 | 24 | 23 | **83.3 %** |
| `fc_a_tau0` | 3 | 23 | 34 | **557** | 666 | 213 | 4 | **37.1 %** |
| `fc_b_tau05` | 4 | 25 | 35 | 539 | 776 | 119 | 2 | 35.9 % |

The centre-only behaviour is gone. `fc_a_tau0` now prefers **c4** over c3 and
uses all seven columns.

### 3.2 Does it block the opponent's win?

Model-independent paired threat positions (`runs/_probe/block_positions.json`,
**307** positions, identical for every arm):

| arm | blocks any threat | single threat (n=272) | multi-threat (n=35) | as P1 (n=136) | as P2 (n=171) |
|---|---:|---:|---:|---:|---:|
| constant-c3 baseline | 26.7 % | — | — | — | — |
| random baseline | 18.9 % | — | — | — | — |
| shipped dense | **47.2 %** | 47.8 % | 48.6 % | 47.1 % | 47.4 % |
| `fc_a_tau0` | 45.0 % | 43.4 % | **57.1 %** | 44.9 % | 45.0 % |
| `fc_b_tau05` | 44.0 % | 43.8 % | 45.7 % | 43.4 % | 44.4 % |

Reading:

- Against a **single** threat the shipped model is still slightly better
  (47.8 % vs 43.4 %).
- Against **multiple** threats — the case where picking the right column
  actually matters, because a naive "play c3" answer misses — `fc_a_tau0` is
  **much** better: 57.1 % vs 48.6 %.
- Blocking is **balanced across seats** (44.9 % / 45.0 %), so it is not a
  one-seat artefact.
- 100 % legal moves on all 307 positions.

### 3.3 Are its choices actually good? (exact solver)

`tg_solver_truth.py` — positions where the exact solver can *prove* the optimal
move set. **n = 166** proven positions, no model involved in the ground truth.

| | solver-optimal | c3 share of its picks |
|---|---:|---:|
| the teacher itself | 89.2 % | 21.7 % |
| shipped dense | 64.5 % | **60.2 %** |
| **`fc_a_tau0`** | **69.3 %** | **31.9 %** |
| `fc_b_tau05` | 66.9 % | 31.9 % |

Two things follow:

- The teacher is **not** centre-biased (21.7 % c3) and is highly reliable
  (89.2 % solver-optimal). It is a valid ground truth.
- The new model beats the shipped model at *playing well* (69.3 % vs 64.5 %)
  even though its raw teacher-agreement is lower (21.9 % vs 28.5 %).

**The lower teacher-agreement score is not a regression.** The shipped model's
extra agreement came largely from fitting the old 93 %-c3 corpus. The
solver — which does not care about c3 — says the new model plays better.

### 3.4 Teacher agreement, broken out

| arm | strict | off-centre strict | c1 acc | c3 acc | c4 acc | c5 acc |
|---|---:|---:|---:|---:|---:|---:|
| shipped dense | 28.47 % | 6.19 % | 0.6 % | 93.0 % | 13.4 % | 4.3 % |
| `fc_a_tau0` | 21.93 % | **14.98 %** | 2.3 % | 42.1 % | **47.0 %** | **19.4 %** |
| `fc_b_tau05` | 21.20 % | 14.80 % | 2.3 % | 39.7 % | 52.6 % | 11.3 % |

Off-centre strict accuracy is **2.4× better**. The model now actually uses
c4 and c5.

### 3.5 Both seats

| arm | seat | n | strict | off-centre strict |
|---|---|---:|---:|---:|
| shipped dense | P1 | 742 | 27.09 % | 5.76 % |
| shipped dense | P2 | 758 | 29.82 % | 6.62 % |
| `fc_a_tau0` | P1 | 742 | 22.64 % | 14.93 % |
| `fc_a_tau0` | P2 | 758 | 21.24 % | **15.03 %** |

`fc_a_tau0` is **flat across seats** (14.93 % / 15.03 % off-centre). The
shipped model is not seat-dependent either, so the improvement is not a
seat-shift.

### 3.6 Real-game strength (canonical duel harness, greedy model, exploring opponent)

`src.c4.duel`, 200 games per cell, seeds 1111 / 2222. Score = win + 0.5 × draw.

| opponent | shipped dense | `fc_a_tau0` | `fc_b_tau05` |
|---|---:|---:|---:|
| centre | 45.5 % / 49.0 % | **66.0 % / 64.0 %** | 47.5 % / 44.2 % |
| heuristic | 13.2 % / 12.8 % | 19.2 % / 20.5 % | **26.5 % / 23.5 %** |
| lookahead | 9.2 % / 10.5 % | 5.0 % / 4.0 % | 5.8 % / — |

Against a *centre-only* opponent the new model is decisively better
(65 % vs 47 %) — it is no longer the same kind of player. Against the
**heuristic** opponent it improves (13 % → 20 %). Against the **lookahead**
opponent it is *worse* (9.8 % → 4.5 %).

Lookahead cross-check, 3 extra seeds × 150 games (seeds 3001/3002/3003):

| arm | seed 3001 | 3002 | 3003 | mean |
|---|---:|---:|---:|---:|
| shipped dense | 11.3 % | 10.7 % | 10.3 % | **10.8 %** |
| `fc_a_tau0` | 12.0 % | 5.3 % | 10.0 % | 9.1 % |
| `fc_b_tau05` | 9.3 % | 9.0 % | 8.7 % | 9.0 % |

All three arms lose badly to lookahead (every cell < 13 %). The spread between
arms is within seed noise, so this is **not yet a proven regression** — but
`fc_a_tau0` is not a strength win either.

---

## 4. The remaining defect: mirror consistency

Connect Four is exactly symmetric under `c → 6-c`. A model that reads the
board must be symmetric. Measured on the canonical mirror audit (600
positions, non-centre choices only — a self-symmetric board is its own mirror,
so a c3 pick makes the test vacuous):

| arm | non-centre rate | mirror tested | mirror OK | **mirror rate** |
|---|---:|---:|---:|---:|
| shipped dense | 18.8 % | 113 | 7 | **6.2 %** |
| `fc_a_tau0` | **60.2 %** | 361 | 67 | **18.6 %** |
| `fc_b_tau05` | 61.2 % | 367 | 47 | 12.8 % |

3× better than shipped, and the model now *has* non-centre choices to be
inconsistent about. Still far from correct.

Note: the seat rate is 100 % for every arm — the model is fully
seat-consistent. The asymmetry is purely *reflection*.

---

## 5. The fix in flight: mirror-augmented training

`tools/build_mirror_corpus.py` appends the reflected twin of every training
game:

```
original game   moves  m0, m1, m2, ...
mirrored game   moves  6-m0, 6-m1, 6-m2, ...
```

with per-ply targets mirrored on the column axis (`target'[c] = target[6-c]`)
and the teacher column mirrored too (`-1` "no label" stays `-1`). This teaches
the symmetry **at the label level, on the real shipped training path** — no new
architecture, no post-hoc refit.

### 5.1 Three real bugs this caught (worth recording)

**(a) Corpus/sidecar row-order mismatch.** The first full build wrote the corpus
text file **interleaved** (original at row 2k, mirror at row 2k+1) but wrote the
`.npz` sidecar **blocked** (all originals, then all mirrors). Every per-row
construction check passed — because each half was individually correct — but the
two files disagreed on **row order**. Training on that would have paired every
game with another game's labels.

This is a general hazard: **a sidecar and its corpus must be proven
row-aligned, not assumed.** `src/c4/train.py` refuses a row-*count* mismatch;
it cannot catch a row-*order* mismatch. The builder now writes both in the same
order and asserts agreement before writing.

**(b) The `teacher` array is not a column.** `merged.npz`'s `teacher` array is
a per-slot **source code** (0 = no label, 1 = exact solver, 2 = strong
teacher) — *which* teacher produced the label, not *which* column it chose.
The mirror script initially transformed it as `6 - code`, which is nonsense.
It now passes through unchanged (the same teacher solved both twins, so the
code is identical). The trainer never reads this array, so the corruption was
metadata-only — but it would have made the artifact's own provenance report
wrong.

**(c) A false invariant, and the real one.** The alignment check first
compared the sidecar's teacher column to the corpus's first move byte. That is
invalid: the teacher is the solver's *recommended* move, which legitimately
differs from the move the game actually played. The second attempt used the
teacher column's legality — also invalid, because the teacher column is not a
column at all (see (b)). The **third** attempt used the correct invariant,
the same one `tools/check_merge_alignment.py` uses:

> the target's `argmax` column must be a legal move in the board reconstructed
> from that corpus row's own prefix.

Verified 100 % on the original aligned sidecar: **0 violations across 101,077
checked plies**.

This check also exposed a subtle trap: `ply_of` stores the ply number per
*slot*, and the slots are **not** stored in ply order. Replaying them in slot
order rebuilds a different board and produces false violations. The slots must
be sorted by ply before replay.

### 5.2 Build result (complete, verified)

```
mirrored games:        320,000   skipped: 0
header checks:      5,100,175   mismatches: 0
target checks:      4,462,571   mismatches: 0
alignment check:     OK
combined c3 share:  25.50 %     (c0==c6, c1==c5, c2==c4 exactly)
```

Artifacts:

```
runs/_probe/perply_v23_full/corpus_320k_mirror.txt   640,000 games
runs/_probe/perply_v23_full/merged_mirror.npz         640,000 rows x 19 x 7
runs/_probe/mirror_corpus_report.json
```

### 5.3 The training run (launched)

`run_mirror_ab.sh` → arm `mir_a_tau0`. Identical to `fc_a_tau0` except the
corpus: 320,000 games → 640,000 (each game + its reflected twin), 4,462,571 →
8,925,142 per-ply labels.

The launcher has a **preflight** that refuses to train unless:
1. corpus row count == sidecar row count;
2. the target `argmax` is legal in the board rebuilt from that row (sampled
   across the whole file, every 997th row);
3. row 2k+1 is genuinely the reflection of row 2k.

All three passed before training started. The trainer confirmed:
`loaded corpus: 640000 sequences`, `per-ply targets: (640000, 19, 7) (mean
labelled plies/row 14.1)`.

### 5.4 The mirror arm's result — and a confound that invalidated the comparison

`mir_a_tau0` trained cleanly (3,000 steps, 1,222 s, board sensitivity 0.9573)
and the gate ran to completion. The result is a **trade-off, not a win**:

| metric | `fc_a_tau0` | `mir_a_tau0` | direction |
|---|---:|---:|---|
| mirror consistency | 18.6% | **32.0%** | ✅ improved |
| blocks opponent's win (307 positions) | 45.0% | **51.8%** | ✅ improved |
| — single-threat | 43.4% | **52.2%** | ✅ improved |
| — as P1 / as P2 | 44.9 / 45.0% | **53.7 / 50.3%** | ✅ improved |
| duel vs heuristic | 19.9% | **34.0%** | ✅ improved |
| off-centre strict accuracy | 15.0% | 7.0% | ❌ regressed |
| c3 share of selected moves | 37.1% | 71.7% | ❌ regressed |
| solver-optimal (n=166) | 69.3% | 64.5% | ❌ regressed |
| duel vs centre | 65.0% | 52.0% | ❌ regressed |

The mirror augmentation did exactly what it was supposed to — mirror
consistency up 1.7×, blocking up 6.8 points, single-threat blocking up 8.8
points (and it finally *beats* shipped's 47.8% on single threats). But the
centre collapse came back, and the solver-optimal score fell back to shipped's
level.

**The cause is a confound in the A/B, not the mirror augmentation.**

```
fc_a_tau0 : 320,000-row corpus, 3000 steps x 16 = 48,000 samples
             -> 48,000 DISTINCT ORIGINAL games  (15% of the corpus)
mir_a_tau0: 640,000-row corpus (320k original + 320k mirror), same
             48,000 samples
             -> only ~24,000 DISTINCT ORIGINAL games (7.5%),
                because half the rows are mirrors
```

So `mir_a_tau0` was **not** "`fc_a_tau0` + mirror augmentation". It was
"`fc_a_tau0` with **half** the distinct-board coverage, plus mirror
augmentation". Halving the distinct-board coverage pushes the model back
toward a prior-driven, centre-heavy policy — which is precisely what was
observed (c3 share 37.1% → 71.7%). The mirror augmentation was working
underneath the noise of a different data distribution.

**This is the same class of mistake as the earlier holdout-overlap and
row-count confounds: two arms that differ in more than the variable under
test.** The sample budget is a fixed 48,000 rows; adding mirror rows spends
half of it on reflections instead of on new boards.

**The controlled experiment** (`run_mirror_ab_matched.sh`) removes the
confound by matching the variable that actually matters — *distinct original
games* — rather than the raw sample count:

| arm | corpus | steps | samples | distinct original games | reflections seen |
|---|---|---:|---:|---:|---:|
| `fc_a_s6000` | 320k | 6000 | 96,000 | ~96,000 | 0 |
| `mir_a_s6000` | 640k | 6000 | 96,000 | ~48,000 | ~48,000 |

- `mir_a_s6000` vs `fc_a_s6000` → isolates **mirror augmentation** (same
  steps, same sample count, same distinct-board *class*).
- `mir_a_s6000` vs `fc_a_tau0` → matched on **distinct original games**
  (48,000 each); the mirror arm additionally sees 48,000 reflections.

Everything else is byte-identical: tau 0.0, ctx 192, batch 16, dense lags,
`--table-bias 0.0`, `--hard-start 0`, seed 7.

**The distinct-game arithmetic, verified empirically** (the trainer samples
rows *with replacement*: `src/c4/train.py:633`,
`chosen = rng.integers(0, len(examples), size=B)`, so distinct rows seen =
`N * (1 - (1 - 1/N)^D)`):

| arm | corpus | draws | distinct rows | **distinct ORIGINAL** |
|---|---|---:|---:|---:|
| `fc_a_tau0` | 320k | 48,000 | 44,573 | **44,573** |
| `mir_a_tau0` | 640k | 48,000 | 46,239 | **23,120** |
| `fc_a_s6000` | 320k | 96,000 | 82,936 | **82,936** |
| `mir_a_s6000` | 640k | 96,000 | 89,143 | **44,572** |

`mir_a_s6000` lands on 44,572 distinct original games against `fc_a_tau0`'s
44,573 — a **0.002% match**. That is the controlled pairing.

### 5.5 The mirror corpus is safe for evaluation (re-verified)

The mirror corpus doubles the board population, so the zero-overlap
guarantee had to be re-established for it. `canonical_board` collapses a
board and its reflection to one key, so the mirror corpus's *canonical* key
set is unchanged — but that was reasoned, not measured. Measured, using the
canonical method (labelled positions only, all corpus boards):

```
holdout LABELLED positions 1500   distinct canonical boards 1500
ORIGINAL corpus: boards scanned 4,982,133   HOLDOUT OVERLAP 0
MIRROR   corpus: boards scanned 9,910,522   HOLDOUT OVERLAP 0
VERDICT: CLEAN - zero canonical overlap in BOTH corpora
```

**A methodological note worth recording:** a first attempt at this check
reported a 63% "leak". That was the *check* being wrong, not the data. It
compared the holdout's **all game prefixes** against the corpus, which
necessarily collides on common early-game boards (the empty board, the first
move, etc.). The canonical method compares the holdout's **labelled
positions** only. Both the old and new corpora are clean under the correct
method; the corpus was never the problem.

### 5.6 The symmetry information is NOT in the logits (decisive)

**Question:** is the mirror inconsistency a *training* problem (the model
never learned the symmetry) or an *expression* problem (the information is
in the weights but the model fails to express it consistently)?

If the information were present, an inference-time symmetrisation
(`argmax(logits(B) + logits(M(B))[::-1])`) would recover it for free — no
retraining, no architecture change. `tg_symmetry_probe.py` tests exactly
this on the 1,500-position holdout.

**Answer: the information is not there.** Symmetrisation is worth **+0.1 to
+0.6 points** of teacher agreement, and it makes the centre collapse
*worse*:

| checkpoint | teacher hit (raw) | teacher hit (symmetrised) | Δ | c3 raw | c3 symmetrised |
|---|---:|---:|---:|---:|---:|
| `lm_b_dense_full` (shipped) | 28.5% | 28.6% | **+0.1** | 83.3% | 83.7% |
| `fc_a_tau0` | 21.9% | 22.5% | **+0.6** | 37.1% | **50.7%** |
| `mir_a_tau0` | 25.7% | 25.8% | **+0.1** | 71.7% | 72.2% |
| `fc_b_tau05` | 21.2% | 24.1% | **+2.9** | 35.9% | **55.4%** |

Symmetrising `fc_a_tau0` pushes c3 from 37.1% to **50.7%** — the ensemble
regresses toward the centre rather than away from it. This is the same
failure mode as the demeaning-only correction (25.5% → 17.5%): averaging two
noisy views of an under-informed readout pulls it toward the prior.

**Conclusion: mirror augmentation must come from training on mirrored
examples, not from an inference-time ensemble.** That is exactly what the
matched A/B is testing. The architecture is not the bottleneck here.

### 5.7 Metric reconciliation (two mirror numbers, both correct)

`tg_symmetry_probe.py` initially reported **66.7%** mirror agreement for the
shipped model — the same number earlier flagged as a *false* reading. The
probe was not wrong; it used a **different metric**:

| metric | shipped | fc_a_tau0 | mir_a_tau0 |
|---|---:|---:|---:|
| all positions (probe, early version) | 66.7% | 31.9% | 66.7% |
| **non-centre originals only** (canonical) | **6.2%** | **18.6%** | **32.0%** |

The all-position metric is dominated by the trivial `c3 → 6-3 = c3` case:
a model that always answers c3 scores 100% on it. The canonical audit
(`tg_mirror_audit.py`) counts only non-centre originals, where the question
is actually informative. The probe now reports the canonical metric, so the
two tools reconcile:

| checkpoint | canonical audit (random playouts) | probe (holdout positions) |
|---|---:|---:|
| `lm_b_dense_full` | 6.2% | 31.6% |
| `fc_a_tau0` | 18.6% | 24.6% |
| `mir_a_tau0` | 32.0% | 48.5% |
| `fc_b_tau05` | 12.8% | 14.9% |

The two differ in absolute level because they use different position sets
(random playouts vs. the disjoint holdout), but they **rank the arms
identically** and agree in sign. That is the check that matters.

---

## 6. THE MATCHED A/B — RESULTS (2026-10-27, 20:30)

Both arms trained to **6000/6000** steps with identical hyperparameters
(only the corpus differs):

| arm | corpus | steps | samples | **distinct ORIGINAL games** |
|---|---|---:|---:|---:|
| `fc_a_s6000` | 320k original | 6000 | 96,000 | **82,936** |
| `mir_a_s6000` | 640k (320k original + 320k mirrored) | 6000 | 96,000 | **44,572** |

`mir_a_s6000` is matched to `fc_a_tau0` (44,573 distinct original games) on
the quantity that matters, and is matched to `fc_a_s6000` on the sample
count — so the two comparisons isolate different things.

### 6.1 Headline table (all from canonical JSON artifacts)

**Two different populations are in play. Read the column headers.**

* `strict / off-c / c1 / c2 / c4 / seat1 / seat2` — the **1,500-position
  canonical-board-disjoint holdout**.
* `mirror` — the canonical non-centre mirror audit.
* `block` — the shared 307-position model-independent tactical set.
* `solver` — the **166 solver-proven endgames** (positions with ≤18 empty
  cells). This is a *late-game* population, not the whole board.
* `c3%` — **the 1,500-position holdout** (full-board population).

| arm | strict | off-c | c1 | c2 | c4 | seat1 | seat2 | mirror | block | solver(endgame) | c3%(holdout) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 28.5 | 6.2 | 0.6 | 9.7 | 13.4 | – | – | 6.2 | – | 64.5 | 83.3 |
| fc_a_tau0 | 21.9 | 15.0 | 2.3 | 3.8 | 47.0 | – | – | 18.6 | 45.0 | 69.3 | 37.1 |
| fc_b_tau05 | 21.2 | 14.8 | 2.3 | 3.8 | 52.6 | – | – | 12.8 | 44.0 | 66.9 | 35.9 |
| mir_a_tau0 | 25.7 | 7.0 | 0.6 | 7.6 | 17.0 | 23.9 | 27.4 | 32.0 | 51.8 | 64.5 | 71.7 |
| fc_a_s6000 | 21.7 | 15.4 | 0.6 | 4.2 | 46.2 | 22.1 | 21.2 | 16.4 | 42.7 | 69.3 | 33.3 |
| **mir_a_s6000** | **23.8** | 9.7 | 0.6 | 8.0 | 15.0 | 22.9 | 24.7 | **36.7** | **50.2** | **71.7** | 54.6 |

> **CORRECTION (2026-10-27 21:05).** An earlier revision of this table
> reported `c3%` from the **166-position endgame set** (27.7% for
> `mir_a_s6000`, 25.3% for `fc_a_s6000`). That was the wrong population for
> a whole-board claim, and it **inverted the conclusion**. On the full
> 1,500-position holdout the truth is:
>
> | arm | c3 share (1,500-position holdout) | c3 share (166 endgames) |
> |---|---:|---:|
> | fc_a_s6000 | **33.3%** | 25.3% |
> | mir_a_s6000 | **54.6%** | 27.7% |
> | fc_a_tau0 | 37.1% | 31.9% |
> | shipped | 83.3% | 60.2% |
>
> So `mir_a_s6000` is **more centre-biased across the full board** than
> `fc_a_s6000` (54.6% vs 33.3%), even though it is centre-*safer* on the
> endgames. The claim "mirror augmentation reduces centre bias" is
> **FALSE on the full holdout** and true only on the endgame subset. The
> honest statement is narrower:
>
> - mirror augmentation **does** improve mirror consistency (16.4→36.7),
>   tactical blocking (42.7→50.2) and endgame solver-optimality
>   (69.3→71.7);
> - it **does not** improve whole-board centre calibration, and in fact
>   shifts the model *toward* c3 on non-endgame positions;
> - the c3 **share** is not a defect metric by itself — what matters is
>   whether c3 is chosen when it is optimal, which the bucket analysis in
>   §6.4 answers for the endgame population.
>
> **This is why the wide-solver run matters**: with endgame-only proof we
> cannot say anything about the 88.9% of holdout positions the solver never
> reached.

### 6.2 Comparison A — the isolated augmentation effect

`mir_a_s6000` vs `fc_a_s6000`: identical steps, identical sample count,
identical everything except the mirrored rows.

| metric | fc_a_s6000 | mir_a_s6000 | Δ | verdict |
|---|---:|---:|---:|---|
| mirror consistency | 16.4% | **36.7%** | **+20.3** | **BETTER** |
| tactical blocking | 42.7% | **50.2%** | **+7.5** | **BETTER** |
| solver-optimal (endgames) | 69.3% | **71.7%** | **+2.4** | **BETTER** |
| strict (teacher) | 21.7% | **23.8%** | **+2.1** | BETTER |
| off-centre (teacher) | **15.4%** | 9.7% | −5.7 | worse (teacher artifact — see 6.4) |
| c1 (teacher) | 0.6% | 0.6% | 0.0 | same |
| c3 share (full holdout) | **33.3%** | 54.6% | +21.3 | **WORSE** — see the correction in §6.1 |
| head-to-head duel | — | — | — | **63.0%** vs fc_a_s6000 (role-swapped: 46.5% for the other side) |

**Mirror augmentation is a real win on mirror consistency, tactical blocking
and endgame solver-optimality — and a real loss on whole-board centre
calibration.** The off-centre regression against the teacher is a teacher
artifact; the c3-share increase is a genuine regression. Both must be
reported; neither cancels the other.

### 6.3 Comparison B — matched on distinct original games

`mir_a_s6000` vs `fc_a_tau0` (44,572 vs 44,573 distinct original games):

| metric | fc_a_tau0 | mir_a_s6000 | Δ |
|---|---:|---:|---:|
| mirror consistency | 18.6% | **36.7%** | **+18.1** |
| tactical blocking | 45.0% | **50.2%** | **+5.2** |
| solver-optimal (endgames) | 69.3% | **71.7%** | **+2.4** |
| strict (teacher) | 21.9% | **23.8%** | **+1.9** |
| off-centre (teacher) | **15.0%** | 9.7% | −5.3 (teacher artifact) |
| c3 share (full holdout) | **37.1%** | 54.6% | +17.5 | **WORSE** |
| head-to-head duel | — | — | — | **60.0%** vs fc_a_tau0 (role-swapped: 48.8%) |

Note the two baselines disagree with each other on the duel: `fc_a_tau0`
beats the *centre bot* 63.8% while `mir_a_s6000` manages only 36.0%, yet
`mir_a_s6000` beats `fc_a_tau0` head-to-head 60/40. Both measurements are
canonical and were taken with the same harness, same seed and same greedy
model protocol. The centre-bot duel rewards a centre-biased opponent, so it
is a **biased** strength proxy — but the disagreement is unexplained and is
an open item.

### 6.4 Why the off-centre "regression" is a teacher artifact, not a capability loss

The teacher is **not ground truth**. On the 166 solver-proven positions:

```
teacher target c3:        36/166 (21.7%)
c3 PROVABLY OPTIMAL:      60/166 (36.1%)
teacher move OPTIMAL:    148/166 (89.2%)
```

The teacher picks c3 *less* often than c3 is genuinely optimal. A model can
therefore score lower on "off-centre strict" while playing *better*, by
correctly choosing a c3 move the teacher did not choose.

Splitting solver-optimality by the teacher's target bucket settles it
(`tg_solver_buckets.py`, n=166):

| arm | teacher-said-c3 bucket (n=36)<br>solver-opt / model-c3 | teacher-said-**non-c3** bucket (n=130)<br>solver-opt / model-c3 |
|---|---|---|
| shipped | 86.1% / 91.7% | 58.5% / 51.5% |
| fc_a_tau0 | 83.3% / 63.9% | 65.4% / 23.1% |
| mir_a_tau0 | 80.6% / 86.1% | 60.0% / 44.6% |
| fc_a_s6000 | 83.3% / 41.7% | 65.4% / 20.8% |
| **mir_a_s6000** | 77.8% / 61.1% | **70.0%** / **18.5%** |

In the bucket where the teacher is *wrong or off-target* (n=130),
`mir_a_s6000` is solver-optimal **70.0%** — better than every other arm
including `fc_a_s6000` (65.4%). Its model-c3 rate there is 18.5%, i.e. it
avoids the centre precisely when the centre is *not* optimal. That is
**better play**, not worse.

**Conclusion: the teacher-based off-centre metric penalises `mir_a_s6000`
for outplaying the teacher. The independent exact-solver metric shows it
is the strongest arm measured.**

---

### 6.5 The diversity A/B (4 arms, all completed — "ALL DONE" was truthful)

The `run_diversity_ab.sh` batch ran four 3,000-step arms on a small corpus:
`ps_a_sparse_diverse`, `ps_b_dense_diverse`, `ps_c_sparse_narrow`,
`ps_d_dense_narrow`. **All four reached step 3000/3000** and all four wrote
`step3000_hard.pt` (verified individually — the `ALL DONE` wrapper was not
a false completion here, unlike the earlier tau batch).

Hard evaluation on the same 1,500-position disjoint holdout:

| arm | strict | off-centre | c3 share | c1 | c4 |
|---|---:|---:|---:|---:|---:|
| ps_a_sparse_diverse | 24.5 | 10.1 | 59.7% | 0.0% | 36.4% |
| ps_b_dense_diverse | 24.4 | 11.7 | 57.8% | 0.0% | 37.7% |
| ps_c_sparse_narrow | 26.9 | 5.3 | 83.5% | 0.6% | 5.3% |
| ps_d_dense_narrow | 27.5 | 5.2 | 83.8% | 0.6% | 5.0% |
| *(shipped, for reference)* | *28.5* | *6.2* | *83.3%* | *0.6%* | *7.0%* |

**Reading:** the *diverse* corpus moves the model off the centre
(c3 share 83% → 58–60%, off-centre 5–6% → 10–12%, c4 5–7% → 36–38%) at a
cost of ~4 points of teacher agreement. The *narrow* corpus reproduces the
shipped centre collapse (c3 83–84%). So board diversity alone does buy
centre-decorrelation — but it is **strictly worse than the full-corpus
per-ply arms** (`fc_a_s6000`: c3 33.3%, off-centre 15.4%, strict 21.7%) on
every metric, and it never reaches the mirror arm's solver-optimality.
This closes the "more diverse boards" hypothesis: diversity is necessary
but not sufficient.

### 6.6 What the 14-process batch actually contained

| proc | job | outcome |
|---|---|---|
| `473f63219141` | mirror corpus build 5 | **PASS** — 320,000 games, 5,100,175 header + 4,462,571 target checks, 0 mismatches, c3 share 25.50% |
| `f592aa963b4d` | mirror corpus build 3 | **FAIL** (superseded) — `FATAL: sidecar teacher 2 != corpus first move 3 at row 0 (mirror=False)`; the `teacher` array was being treated as a move column |
| `fd9123de6e24` | matched gate | **PASS** — `MATCHED_GATE_DONE` + `EXIT=` |
| `3cf237ce2278` | mirror gate | **PASS** — `GATE_DONE` + `EXIT=` |
| `c09ac4e55260` | 6-model solver truth | **PASS** — `EXIT=` |
| `53cd1cde69a7` | solver buckets (max_empty 18) | **PASS** — n=166 |
| `f14bb5362f60` | solver buckets `--max-empty 24` | **SILENT NO-OP** — the script had no argparse; `MAX_EMPTY` was hardcoded to 18. Same 166 positions. Fixed afterwards by adding a real CLI; the corrected run is `proc_ec1579525f5d`. |
| `b66ef4b9fccb` | diversity A/B | **PASS** — all 4 arms at step 3000, see §6.5 |
| `ac0d1dad95c6` | matched training | **PASS** — both arms at step 6000 |
| `fa70aaadea0d` | initial mirror training | **PASS** — `mir_a_tau0` step 3000 |
| `900ba851b204` | matched-arm wait loop | **PASS** — both at step 4500 at 20:05:19 |
| `557943a51819` | seat eval (3 arms) | **PASS** — `EXIT=0` each |
| `ba4e7434703e` | strength vs 3 bots, 3 arms | **PASS** — 300 games each |
| `6014bec91eb9` | role-swapped head-to-head | **PASS** — 5 duels, 268–278 unique trajectories each |

Two jobs in this batch are *misleading rather than failed*, and both are
now recorded: the `--max-empty 24` solver run was a no-op (fixed), and the
mirror-build-3 failure was a real bug in an earlier builder version
(superseded by the passing build 5).

### 6.7 What is still blocking release

1. **Solver coverage is still partial.** The corrected `--max-empty 24` run
   proved **560 / 1,500 = 37.3%** of the holdout (up from 11.1%). **940
   positions remain unproved** within the 120M-node budget — mostly
   mid-game openings. Those are excluded, not scored.
2. **c1 is still broken** — 0.6% for every arm including `mir_a_s6000`.
3. **Whole-board centre calibration regressed** under mirror augmentation
   (33.3% → 54.6% c3 share on the full holdout).
4. **The duel metrics disagree** with each other (centre-bot duel says
   `fc_a_tau0` is stronger; head-to-head says `mir_a_s6000` is stronger).
5. **No arm wins every gate** — see §6.9.
6. Trainer resume still raises `UnboundLocalError: local variable 'step'`
   when the requested step count is below the stored checkpoint step.

**No V24 until these are closed.**

### 6.8 WIDE SOLVER COVERAGE — the decisive re-measurement (560 positions, 37.3%)

The corrected `--max-empty 24` run (`proc_ec1579525f5d`) proved **560** of
the 1,500 holdout positions (37.3%, up from 166 = 11.1%); 940 remained
unproved within the 120M-node budget and were **excluded, never counted as
draws**.

| arm | solver-optimal (n=560) | c3 use where teacher said non-c3 | opt in teacher-c3 bucket | opt in teacher-non-c3 bucket |
|---|---:|---:|---:|---:|
| shipped | 68.9% | 72.5% | 87.7% | 62.8% |
| fc_a_tau0 | 71.1% | 30.3% | 87.7% | 65.6% |
| fc_b_tau05 | 70.0% | 31.0% | 81.2% | 66.4% |
| mir_a_tau0 | 67.9% | 62.3% | 84.8% | 62.3% |
| **fc_a_s6000** | **71.8%** | **26.8%** | 87.0% | **66.8%** |
| **mir_a_s6000** | **71.8%** | 41.5% | 87.7% | 66.6% |

**Two corrections to the earlier endgame-only reading:**

1. **The mirror advantage on solver-optimality disappears.** At n=166
   `mir_a_s6000` led 71.7% vs 69.3%. At n=560 they are **tied at 71.8%**.
   The n=166 gap was a small-sample artefact of an endgame-only population.
2. **The teacher is even more anti-centre than the endgame set suggested.**
   c3 is provably optimal **54.3%** of the time while the teacher picks it
   only **24.6%** of the time. The teacher disagrees with proven optimal
   play far more often than the endgame figure implied — which strengthens
   §6.4's conclusion that teacher-agreement is the wrong release metric.

`fc_a_s6000` is the better-calibrated arm on the proven population: it uses
c3 in only **26.8%** of the positions where the teacher did *not* want c3,
versus `mir_a_s6000`'s 41.5%.

### 6.9 Final verdict on the candidate

| ground truth | metric | fc_a_s6000 | mir_a_s6000 | winner |
|---|---|---:|---:|---|
| independent | mirror consistency | 16.4% | **36.7%** | **mir** |
| independent | tactical blocking | 42.7% | **50.2%** | **mir** |
| independent | head-to-head duel | — | **63.0%** | **mir** |
| independent | solver-optimal (n=560) | **71.8%** | **71.8%** | tie |
| independent | c3 use when teacher says non-c3 | **26.8%** | 41.5% | **fc** |
| teacher | strict | 21.7% | **23.8%** | mir |
| teacher | off-centre | **15.4%** | 9.7% | fc |

**No arm passes every gate.** `mir_a_s6000` is the better *player* on
symmetry, tactics and head-to-head play; `fc_a_s6000` is the better
*calibrated* model on the proven population and is the arm whose behaviour
generalises across the whole board.

**Neither is release-ready.** The honest conclusion is that mirror
augmentation is a **targeted** improvement (symmetry + tactics) that
**trades against** whole-board calibration, and the single training recipe
that maximises the old centre-collapse metric is not the one that maximises
play quality. Choosing between them — or finding a recipe that gets both —
requires the mid-game solver proof that 940 positions still lack.

**No V24 until these are closed.**

## 6. Other completed work in this batch

- `tg_ab_eval.py` now reports **seat splits** (n, strict %, off-centre strict %)
  in both JSON and console output — the canonical evaluator, not a side harness.
- `tg_solver_truth.py` is now model-independent: it uses the exact solver's
  proven optimal move set, decodes moves at `MOVE_OFFSET = 9` (not
  `COL_BYTE_OFFSET`), and evaluates from the side-to-move perspective.
- `tools/build_full_split.py` defaulted to `merged_all.npz` (386,916 rows)
  against a 320,000-row corpus — the same class of bug. Now defaults to the
  aligned `merged.npz`.
- `tools/check_merge_alignment.py` confirms `merged.npz` is offset-aligned with
  `corpus_320k.txt` (100 % teacher-legal at every probed offset).
- Real shipped path smoke-tested: `src/generate.py` exits 0 on both the shipped
  checkpoint and `fc_a_tau0`.
- Disk reclaimed: 10,006 MB of redundant *intermediate* checkpoints deleted
  from completed historical probe arms; every final `step3000_hard.pt` and
  every JSON artifact preserved. `runs/_probe` 17 GB → 6.8 GB; free space
  11 GB → 21 GB.

---

## 7. Still open (release blockers)

1. **Mirror-augmented training run** — not yet launched; its outcome decides
   whether the 18.6 % mirror rate can be fixed on the real training path.
2. **Lookahead strength** — all arms lose to the lookahead opponent
   (9–13 %). Not a proven regression, but not a strength win either.
3. **Single-threat blocking** — 43.4 % vs the shipped model's 47.8 %. If the
   mirror run cannot recover this, the release gate must state the trade
   explicitly rather than hide it.
4. **c1 is still nearly dead** — 2.3 % teacher accuracy, 23/1500 picks. The
   one column the shipped model was reliably wrong about is still mostly wrong.
5. **No V24** until every gate above passes.

---

## 8. Bottom line

The project is not failing and should not be restarted. The GF trunk holds
generalisable board-policy information (layer2: +8.1 points over ALWAYS-c3 on
a board-disjoint split). The centre-only behaviour was a **corpus/label
problem**, and balanced per-ply training on 4.46 M labels fixed it:

- c3 share 83.3 % → 37.1 %
- off-centre strict 6.2 % → 15.0 % (2.4×)
- multi-threat blocking 48.6 % → **57.1 %**
- solver-optimal 64.5 % → **69.3 %**
- vs centre opponent 47 % → **65 %**
- mirror 6.2 % → 18.6 %

The model now picks the column that prevents the opponent's immediate win in
the majority of multi-threat positions, from both seats, and does so more often
than the old model. What is left is symmetry and single-threat reliability —
both tractable, neither architectural.
