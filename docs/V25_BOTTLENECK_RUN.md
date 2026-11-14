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
2. Model + rule wrapper ("win if you can, block if you must") vs rule alone:
   see the paired-opening experiment below.


## Results (run finished, 80,000 steps, ~23.9 h)

| step | soft bpb | hard bpb | HOLDOUT | LABEL | H-LABEL |
|---|---|---|---|---|---|
| 0 | 8.03 | 8.02 | 26.1% | 29.4% | 28.5% |
| 2000 | 6.77 | 7.26 | 28.6% | 49.8% | 29.1% |
| 10000 | 7.68 | 10.21 | 28.5% | 55.5% | 25.8% |
| 18000 | 9.05 | 10.53 | 28.1% | 60.6% | 26.4% |
| 32000 (best LABEL) | 13.10 | 11.45 | 29.6% | 63.3% | 25.4% |
| 40000 | 13.94 | 11.41 | 29.2% | 61.2% | 24.0% |
| 80000 | 14.16 | 13.20 | 29.8% | 58.0% | 27.5% |

Independent scorer: `best_hard.pt` (step 32000) LABEL 63.30%; final 58.02%.

V6 baseline (`runs/_probe/v24.log`, different logging, check corpus parity before strict comparison):
HOLDOUT 36.2% at 20k, 34.7% at 40k; LABEL ~59% plateau.

### Verdict
- Bottleneck readout did not show better generalisation in these runs. HOLDOUT
  stays 28-30% (V6: 35-36%). Train LABEL rises 50% -> 63% while H-LABEL
  is flat/noisy ~25-28%; this gap suggests overfitting but does not prove
  exact-board memorisation. The V6 and V6b training sets differ.
- bpb diverges upward after ~6k steps (soft 6.76 -> 14+): the byte-level loss gets worse
  while label accuracy improves; the MLP readout is trading LM quality for label fit.
- Training LABEL continued to improve until step 32000; holdout stayed nearly
  flat. Later training did not improve the final training LABEL.
- Conclusion: the "force reasoning via bottleneck" hypothesis is not supported. V6b is
  worse than V6 on holdout. Keep V6 as the reference.


## Game-play comparison (V6 `v24/best_hard` vs V6b `v25b/best_hard`)

200 games each, hard (integer) policy, opponent eps=0.15, model eps=0.05, both colours.
Score = (wins + 0.5 draws) / games for the model.

| opponent | V6 (v24) | V6b (v25b) |
|---|---|---|
| random | 86.0% | 72.8% |
| centre | 87.5% | 33.5% |
| heuristic | 29.5% | 15.0% |
| lookahead (win/block 1-ply+) | 15.5% | 4.5% |
| minimax | 22.0% | 16.5% |
| **head-to-head** (V6 as A, eps 0.1) | **78.2%** (154-41) | 21.8% |

### Reading
- V6b is clearly weaker than V6 on every opponent and loses head-to-head 41-154.
- V6b loses 27% of games against a random mover and 95% against the simple
  lookahead opponent. These game results alone do not measure win-or-block
  accuracy; see the board-disjoint tactical test below.
- Train LABEL 63% vs holdout 29.6% (V6: ~59% vs ~36%): the extra train fit does not transfer.
  This gap is consistent with overfitting, but parameter count versus position
  count does not establish whether the model memorised individual boards.
- These games did not evaluate the model+rule wrapper; see the experiment below.

## Board-disjoint tactical test

`tmp/test_gf_generalization.py` evaluates the original V24 V6 and V25b V6b
best hard checkpoints on the existing V26 surgery holdout. This is not a
fresh final test set: the V26 work already used it for evaluation and
selection. The script canonicalises each board with its left-right mirror
and checks its seven ownership/height bytes and side to move against every
board in both actual V24 and V25b training corpora. It does not merely
compare move histories or last-move fields.

The original 3306 holdout records contain 3183 distinct board/mirror
classes. Of these, 813 occur in each training corpus, despite the surgery
holdout being disjoint from the *surgery* training positions. Removing
all such overlaps leaves 2370 classes; 85 positions with multiple opposing
immediate threats and no immediate win are excluded from the three categories.
The score therefore covers 2285 distinct, training-board-disjoint positions.

| Policy | Immediate win (n=569) | Single forced block (n=425) | Quiet teacher agreement (n=1291) |
|---|---:|---:|---:|
| Random legal column, expected | 19.87% | 16.14% | 24.36% |
| Centre-first legal column | 25.48% | 25.65% | 43.14% |
| V24 V6 | 41.83% (238) | 50.35% (214) | 35.71% (461) |
| V25b V6b | 23.37% (133) | 22.59% (96) | 34.00% (439) |

Winning and blocking columns are checked by applying each possible move and
calling the board's win detector, independently cross-checking the search
helper. Quiet scores use the existing time-limited teacher labels, not
proof of optimal play; a quiet position can have other tactical properties.

V6 performs substantially above the centre baseline on unseen immediate
wins and blocks, evidence of some tactical generalisation, though it still
misses many. V6b does not beat the centre baseline on those categories in
this sample. This does not isolate the effect of training from initialisation,
identify the role of the GF trunk versus the readout, or quantify exact
memorisation. The distributions, training data and checkpoint selection
differ; use a fresh held-out exam and matched runs for stronger conclusions.
The machine-readable result is `runs/_probe/gf_generalization_tactics.json`.

## Hybrid versus one-ply rule from fresh openings

The earlier hybrid experiment already existed: commit `6d07483` introduced
`tools/test_hybrid_override.py` and recorded 30.8% standard-model wins and
54.6% hybrid wins over 500 games each. The recorded checkpoint and full game
outcomes are not in the commit. The script applied 15% random exploration to
the opponent, including on immediate tactics; 54.6% is a win rate above 50%,
not a loss merely because it is below the script's arbitrary 60% threshold.
The commit's claim that this proved memorisation is not justified by that test.

`tmp/eval_hybrid_fresh.py` repeats the requested comparison using the V24 V6
best hard checkpoint (SHA-256
`134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2`).
Both players deterministically win immediately if possible, otherwise block
an immediate opponent win if possible, using the same ordered tie-break.
On other positions the hybrid uses the model's greedy move and the opponent
uses the one-ply heuristic's centre-first move. No per-move noise is applied.

Seed 872361 produced 600 distinct mirror-canonical, non-terminal, 12-ply
random openings without an immediate win for either side. Each opening
was checked against the entire V24 and V25b training corpora and the V26
surgery train and holdout positions, including mirrored boards; 14 overlapped
the training corpora and none overlapped the surgery set. The first 200
remaining boards were each played twice, with the hybrid as each colour.
The starting boards are disjoint, but subsequent positions need not be.

- Hybrid vs one-ply: **171 wins, 209 losses, 20 draws**, 400 games;
  score **45.25%** (wins + half draws), 393 distinct full continuations.
- Hybrid as player 1: 92 wins, 101 losses, 7 draws; as player 2:
  79 wins, 108 losses, 13 draws.
- Paired-opening bootstrap 95% interval: **41.0% to 49.5%**. The openings
  are generated from the specified random policy, not a representative sample
  of all legal boards. Two games from the same opening are not independent.

On this distribution and checkpoint the win/block-wrapped model did not beat
the rule-only baseline. This is evidence against an advantage from the model's
quiet-position choices in this matchup, not proof that it learned no
positional knowledge. It also does not isolate trunk from readout, test V25b,
or demonstrate strength from ordinary empty-board play. The complete opening
and continuation records are in `runs/_probe/hybrid_fresh_games.json`.

Reproduce: `.venv-c4\Scripts\python.exe tmp\eval_hybrid_fresh.py`. Three
focused checks cover deterministic quiet openings, win/block precedence and
paired-seat scoring; the standard test runner did not discover these scripts
and pytest was unavailable in the environment.
