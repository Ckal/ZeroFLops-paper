# V26 — editing the weights directly: readout surgery, compiled tactics, soft fine-tune

Branch: `test/weight-surgery-reasoning` (branched from `feat/threat-vs-quiet-measure`,
not merged). Starting model: V6 `runs/_probe/v24/checkpoints/best_hard.pt`
(d=32, L3, fan_in 3, step 18000, transcript v2). All checkpoints below live under
`runs/_probe/` (gitignored); the tools that make them are committed.

## Question

A hard checkpoint is a readable file: integer tables for the trunk (embedding,
GF(256) coefficients, S-boxes, wiring) plus float readout tables. Can we change
those numbers directly to make the model play better, instead of retraining?
And can the result be turned back into a soft model and trained further?

## The model chain (what was derived from what)

```
V6 best_hard (original, learned)
   |-- A. readout surgery     -> v24_surgery/best_hard.pt   (float readout edit only)
   |-- B. compiled tactics    -> v24_tactics/best_hard.pt   (new hand-written trunk channels)
              |-- C. soft import (margin 2) -> same behaviour, verified
                        |-- D. STE fine-tune, 800 steps -> v24_soft/best_hard.pt
```

B is built from the original. C is a lossless re-representation of B. D is
trained from C, so D changed relative to B. D also keeps all 606/606 and 569/569
tactics (below), so the fine-tune did not erase the circuit.

## Test data and metrics (same for every row)

- `runs/_probe/v24_surgery/positions.pt`: positions from mixed games (model, random,
  centre, heuristic, lookahead opponents), each labelled by `strong_policy_set`
  (depth 7, 0.05 s budget, so the label is time-budgeted and slightly noisy).
  Train 14,796 / holdout 3,306 positions; holdout games are disjoint from train games
  and holdout positions were removed from train.
- Holdout classes: win-in-1 (n=606), forced block (n=569), strict (<=2 optimal
  moves, n=1964), tie (n=167). "Agreement" = model's argmax is in the solver's
  optimal set.
- **Rule check** (independent of the solver label): does the model play a winning
  column when one exists / a blocking column when the opponent threatens one,
  using the repo's own `_imminent_win_columns`. This is exact, whereas solver
  agreement on win/block positions is capped (~92%) because of the single-label
  artifact in the label set.
- Games: `python -m src.c4.duel`, hard integer policy, both colours. Opponent
  eps 0.15 / model eps 0.05 for fixed opponents; eps 0.1 for head-to-head.
  n=200-300, so differences under ~7 points are noise.
  Win-in-1 / block counts are exact over the 3,306 holdout positions.

## A. Readout surgery (`tools/readout_surgery.py`) — did NOT help

Edit only the 7 move-column slices of the float readout tables, fitted to solver
labels. Trunk untouched. Baseline before edit: 38.0% holdout agreement (matches the
recorded V6 value, ~36%).

| variant | holdout agreement | game result |
|---|---|---|
| plain tie-set NLL, 60 epochs, weak L2 | 39% (train 100%: memorised) | not played |
| plain NLL, lr 0.01, L2 10 | 38.0 -> ~42% | **lost** head-to-head vs original, 36.0% (72-128, n=200) |
| tactic-weighted NLL, lr 0.01, L2 10 | win1 36.6 -> ~45%, block 49 -> ~56% | **tie**, 54.5% (109-91, n=200); vs random/heuristic/lookahead 81 / 27.8 / 15.0 vs original 85 / 30 / 15.5 |

Lesson: label agreement can rise while play gets no better (dominated by tie
positions). Cause: the learned trunk's integer states carry almost no threat
information (original finds a win-in-1 only 36.6% of the time), so a readout of
per-channel lookups cannot recover it.

## B. Compiled tactics (`tools/compile_tactics.py`) — large gain

**How the content was chosen.** Nothing was read off the learned weights; they
are not human-interpretable. The circuit was designed from (1) the documented v2
input (each column byte = height + owner of every stone; board cells sit at fixed
lags from the TURN byte) and (2) the fact that a channel is
`SBOX[ XOR_f coef_f (x) h[ch_f, lag_f] ]`: with coefficients `1<<s` the XOR
concatenates input bits into an index, so the S-box is an arbitrary truth table
over up to 8 bits. Every new table is the output of a short Python function.

Circuit (32 learned channels kept bit-for-bit; 87 layer-1 + 48 layer-2 channels added, d 32 -> 119):

1. layer 0: per-row cell state of a column byte (empty / P1 / P2 / playable /
   empty above playable), turn flag, vertical-3 flag.
2. layer 1: pair nodes (two cells + turn -> mover-relative summary) and vertical nodes.
3. layer 2: one node per 4-cell line (horizontal and both diagonals, 48 lines):
   says which cell wins for me, blocks the opponent, or is unsafe (opponent's
   winning cell directly above the playable one).
4. readout: adds +1000 win, +300 block, -40 unsafe to the move-column logits.

Learned preferences still break ties. The readout stays float (it is in every variant).

| measure | original V6 | B compiled |
|---|---|---|
| win-in-1 played, exact rule (n=606) | ~37% (solver agreement) | **606/606** |
| forced block made, exact rule (n=569) | ~49% (solver agreement) | **569/569** |
| strict-class solver agreement | 32.0% | 32.6% |
| head-to-head vs original | | **97.5%** (195-5, n=200) |
| vs random / centre | 85-86 / 87.5 | 99.5 / 94.0 |
| vs heuristic | 29.5-30 | 59.5 |
| vs lookahead | 15.5 | 57.2 |
| vs minimax | 22 | 60.8 |

## C. Soft import (`tools/soft_finetune.py`, `build_soft`)

Each integer table entry becomes a logit bonus of `margin` (default 2) on the
chosen value. The soft model's coefficient set is the union of every constant in
the payload (12 values) because `import_hard` rejects constants outside the model's
set. Trained with the straight-through estimator, so the forward pass is the
integer program. Soft model: d=119, 62M parameters.

Verified before any training: soft (STE) vs `hard_forward` max logit difference
3.05e-05, argmax agreement 1.0000 on 64 positions, identical holdout numbers to B.
So C is B exactly; it only differs in being trainable.

(Margin 10, used by `import_hard`, was not tested; the ~7.7-logit margin problem
is documented in the `GFLayer` comment: Adam cannot cross it in a realistic step budget.)

## D. STE fine-tune — modest, mixed gain

Loss: tie-set NLL on move columns, weights win/block 1, strict 3, tie 1.
Windows of 64 bytes (receptive field 3 x lag<=20). All tables trainable.

Runs (all from C):

| run | settings | outcome |
|---|---|---|
| a | 300 steps, bs 32, lr 0.01 | stable, holdout ~0.59 |
| b | 1500 steps, bs 32, lr 0.004 | **collapsed**: step 1250 win1 94% -> 78%, step 1500 win1 49% / block 62% (loss spiked to 66). Discarded. |
| c (kept) | 800 steps, bs 32, lr 0.002, grad clip 1, best-checkpoint selection | selected step 800 |

An earlier attempt with bs 64 ran >2 h without finishing (GPU memory ~full) and
was stopped.

Final model D (`v24_soft/best_hard.pt`) vs its starting point B:

| measure | B compiled | D fine-tuned |
|---|---|---|
| win-in-1 played, exact rule | 606/606 | 606/606 |
| forced block made, exact rule | 569/569 | 569/569 |
| strict-class solver agreement (n=1964) | 32.6% | 38.8% (chosen on this holdout) |
| head-to-head vs B | | **81.3%** (234-46, n=300) |
| head-to-head vs ORIGINAL V6 | B: 97.5% (n=200) | **98.3%** (295-5, n=300) |
| vs heuristic | 59.7 | 49.7 |
| vs lookahead | 55.5 | 55.3 |
| vs minimax | 59.8 | 69.2 |

## What can and cannot be claimed

- B is far better than the original (exact rule, 200-game duel, five opponents).
- D is still far better than the original (98.3% head-to-head), and keeps the circuit.
- D vs B is **mixed**: beats B head-to-head and on minimax, worse on heuristic,
  flat on lookahead. Opponent scores at n=300 have ~+-6 points of noise. The
  head-to-head against its own start is the weakest comparison (D was shaped by
  the same position distribution).
- The strict-agreement gain (+6 points) was picked by best-checkpoint selection on
  the same holdout; the games are the independent check and are inconclusive.
- **No evidence the model learned new tactics** (forks, two-move threats). The extra
  strict agreement may be tie-breaking among already-safe moves.
- Training is fragile: lr 0.004 / 1500 steps destroyed the circuit.
- B/D are not "learned reasoning": B's logic is hand-compiled from the rules and
  looks exactly one move ahead (win, block, do not give away a win).
- Not measured: byte-level LM quality (bpb) of B or D; whether margins other than 2
  matter; outcome-based training; any run longer than 800 steps that stays stable.
- Not solved: ~55-60% vs lookahead/minimax means it still loses many games.

## Code changes outside the new tools

- `src/c4/ui_server.py` `_load_agent`: skips the float `import_hard` for payloads with
  `tactics_compiled` (they contain coefficients outside the float model's coef_set);
  the hard path is unaffected. `duel.py` goes through the same loader.
- Not wired: `measure_promotion_scorer.py` was not run on B/D (its corpus-sidecar
  scorer needs a run dir); `tools/eval_tactics.py` is the scorer used here.

## Reproduce

```bash
PY=.venv-c4/Scripts/python.exe
$PY tools/readout_surgery.py --src runs/_probe/v24/checkpoints/best_hard.pt \
    --out runs/_probe/v24_surgery/best_hard.pt --epochs 31 --lr 0.01 --l2 10 \
    --cache runs/_probe/v24_surgery/positions.pt        # first call generates positions (~10 min)
$PY tools/compile_tactics.py --src runs/_probe/v24/checkpoints/best_hard.pt \
    --out runs/_probe/v24_tactics/best_hard.pt
$PY tools/eval_tactics.py runs/_probe/v24_surgery/positions.pt \
    runs/_probe/v24/checkpoints/best_hard.pt runs/_probe/v24_tactics/best_hard.pt
$PY tools/soft_finetune.py --src runs/_probe/v24_tactics/best_hard.pt \
    --positions runs/_probe/v24_surgery/positions.pt \
    --out runs/_probe/v24_soft/best_hard.pt --steps 800 --eval-every 100 --lr 0.002 --margin 2 --parity 64
$PY -m src.c4.duel --a runs/_probe/v24_soft/best_hard.pt --b runs/_probe/v24/checkpoints/best_hard.pt \
    --games 300 --eps 0.1 --seed 7 --device cpu
$PY -m src.c4.ui_server --model runs/_probe/v24_soft/best_hard.pt --port 7861 --no-rate-limit
```

Note: `readout_surgery.py` was last run with its tactic-weighted objective; the
numbers in section A for the unweighted variants came from the earlier version of
that script (same file, before the class weighting was added).

## Next

1. Outcome-based fine-tune (self-play / games vs the solver) instead of solver labels.
2. Compile deeper tactics (forks, open threes) as extra layers and re-measure.
3. Measure bpb of B and D; test other margins and longer, lower-lr runs.
