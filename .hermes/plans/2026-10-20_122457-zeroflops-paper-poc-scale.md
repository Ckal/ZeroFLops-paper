# ZeroFLops-paper: Make the paper true, then prove it, then scale it

**Created:** 2026-10-20 12:24
**Repo under work:** `C:\Users\Chris4K\projects\ZeroFLops-paper\` (git, 1 commit)
**Sibling WIP:** `C:\Users\Chris4K\projects\ZeroFLops\` (13 GB, no git)

---

## What this plan is for

Three stages, in this order, because each one gates the next:

1. **Paper** — the claims in `paper.md` have to match the code and the artifact that actually exist.
2. **First PoC** — reproduce the V6 number (Hard BPC 3.071) end-to-end from a clean run, so the headline claim has evidence behind it.
3. **First scale** — only then push to the 1 GB corpus / MoE, starting from a verified V6.

Nothing in stage 3 starts before stage 2 produces a number that matches the paper.

### Interpretation of your instruction

"start with the paper and first poc and the first scale" is read as:
**paper** = fix the claims and the artifact mismatch → **first PoC** = reproduce V6 from scratch and prove 3.071 is real → **first scale** = the 1 GB / MoE work, rebuilt from the verified V6 rather than from the drifted v9/v10 forks.

If you meant a different "PoC", say so and stage 2 changes; the ordering holds either way.

---

## Stage 0 — Verified facts this plan is built on

Checked directly, not assumed. Re-verify with the commands given before acting.

| # | Fact | How verified |
|---|------|---------------|
| F1 | The shipped 118 MB artifact is **d=64, layers=6, fan_in=4, lags up to 256** — i.e. the *V7* architecture, not V6. | `torch.load` on `artifacts/gf256_hard_model.pt`, see `emb (256,64)`, `readout[0] (64,256,256)`, `lags [0,1,2,4,8,16,32,64,128,256]` |
| F2 | The paper describes V6 as **d=32, layers=3, fan_in=3**, readouts `4 × 32 × 256 × 256`, **total ~33 MB** (§2, §4.3, §4.5). | `paper.md` lines 500, 564–573 |
| F3 | Therefore the paper and the artifact are **different models**. 33 MB vs 118 MB. | F1 + F2 |
| F4 | `src/gf256_markov_lm.py` defaults are `d=64, layers=6, fan_in=4` — the V7 config — while §7.2's command says `--d 32 --layers 3`. The file under the heading **"7.1 The Complete Code"** cannot reproduce §6's numbers. | read of `src/gf256_markov_lm.py:118`, `paper.md:708`, `paper.md:1116` |
| F5 | The readouts are `torch.float32`, and inference sums `d × (layers+1)` of them per character. The "zero FLOPs" claim is false for the program as shipped. | `readout[0] … torch.float32`; `hard_forward` does `R[ar, h].sum(2)` per layer |
| F6 | `GF(256) Lagrange Field Transformer.md` is a **mangled export** of `paper.md` — escaped underscores (`ste\_softmax`), flattened code fences, garbled LaTeX. It is untracked in git and should not be kept. | `diff paper.md` vs the file; `.gitignore` / `git status` |
| F7 | **9 of 14 runs in `ZeroFLops/runs/` have no eval log at all** — including every 1 GB run. There is currently no evidence any scale-up run ever produced a metric. | directory listing of `runs/*/logs/` |
| F8 | The MoE runs show `aux_loss 6.14` and `router_entropy 0.0` — the router collapsed; no expert diversity. | `runs/gf64_moe_test_v3/logs/train_metrics.csv` |
| F9 | Filenames lie about versions: `v4_1.py` docstring says v8, `gsm_v9.py` says v7, `originalGSM.py` says v7. | heads of each file |
| F10 | Bigram baseline re-derives to **3.597 BPC** (add-one smoothed, same 90/10 split) vs the paper's 3.602 — close, so the baseline *method* is roughly sound. Trigram is **not yet verified** (my first script had a counting bug). | `src/baseline_check.py` partial run |

---

## Stage 1 — Paper (make the text match reality)

**Exit criterion:** every number in `paper.md` is either re-derived by a script in `src/`, or explicitly labelled as a target rather than a result.

### T1.1 Decide what the paper is about — V6 or V7 (10 min, needs your call)

The paper currently claims V6 results and ships V7 code. One of them moves.

- **Option A — the paper is about V6 (d=32, layers=3).** Correct per §7.2. Then stage 2 must *produce* a V6 artifact, and `artifacts/gf256_hard_model.pt` is replaced by the reproduced one. The ~33 MB size table in §4.5 stays.
- **Option B — the paper is about V7 (d=64, layers=6).** The artifact already matches. Then §7.1/§7.2/§4.3/§4.5/§6 must be rewritten for d=64, the program size becomes ~118 MB, and the 3.071 number must be re-measured on V7 (unverified today — the V7 logs in `gf256_markov_lm.py:386–453` show hard BPC still 3.09–3.21 at 8k–12k steps).

**My recommendation: Option A.** The numbers in the paper (3.071, step 82,400, seed 32, d=32) are internally consistent with a V6 run, the 33 MB program is the more interesting claim, and it keeps "118 MB" out of the abstract. Option B means rewriting the results section against numbers nobody has measured.

**Decision needed before T1.2.** Everything below assumes Option A.

### T1.2 Fix the code/artifact mismatch (blocked on T1.1)

- Under Option A: `src/gf256_markov_lm.py` gets its defaults changed to the V6 config that §7.2 specifies (`d=32, layers=3, fan_in=3, lags=(0,1,2,4,8,16,32,64), n_coef=8`), and the stale V7 docstring + the trailing shell-session log at the bottom of the file (lines 385–453) are deleted. The V7 config moves to `src/gf256_markov_lm_v7.py`.
- Verify: `python src/gf256_markov_lm.py --help` shows `--d 32` default; the file has no `###` log block at the end.

### T1.3 Replace "zero FLOPs" with the true claim

The honest framing, which is still a real and unusual result:

> The **trunk** — embedding, GF-mixing, S-boxes — is integer-only: XOR, table lookup, no floating point. The **readout** is a learned float32 table, ~448 float adds + a 256-way argmax per character. Removing the readout's arithmetic is open work (`FUTURE_WORK.md`).

- Amend the abstract, §4.5, §6.1 and §10 to say "integer trunk" not "zero FLOPs".
- Add the real per-character op count, split into integer vs float, from `src/inspect_artifact.py`.
- Keep the C++ port as supporting evidence that the trunk really is integer.

### T1.4 Delete the mangled copy, keep one paper (5 min)

- `rm "GF(256) Lagrange Field Transformer.md"` (it is untracked; `paper.md` is the good one).
- Verify: `git status --short` is empty; only `paper.md` remains at the root.

### T1.5 Re-derive every baseline (30 min)

- Finish `src/baseline_check.py` — the conditional-probability denominator is now fixed; it must be re-run end to end and the output pasted into §6.1.
- Report bigram *and* trigram at several smoothing/discount settings, plus the validation-context coverage fraction. **The coverage number is the point**: if e.g. 40 % of trigram contexts are unseen, a 3.087 trigram is a weak bar and the comparison must say so.
- The margin is 3.071 vs 3.087 = **0.016 BPC, ~0.5 %**. §6.3 must state this as a 0.5 % margin over a *specific* baseline, and §8 must acknowledge it is within single-seed noise until stage 2 gives error bars.
- Verify: `python src/baseline_check.py --data data/tinyshakespeare.txt` runs clean and its table matches §6.1.

### T1.6 Structural and honesty edits

- §1 "The Apple Moment" and §10 "The apple didn't fall. It was XORed." — keep at most one, and not in a paper.
- Add a **Related Work** section that does not exist at all right now. Minimum: L0/byte-level n-gram literature, straight-through estimators (Bengio 2013, additive/soft-hard quantisers), binary/ternary neural networks, finite-field and lattice-based ML, Walsh-Hadamard convolutions, FPGA/LUT inference work. **A paper with no citations will be desk-rejected.**
- Add author, affiliation, date, and a "how to reproduce" line to `paper.md`.
- Verify: `grep -c "^|" paper.md` sections present; a Related Work heading exists.

### T1.7 Rewrite the README to describe what is actually in the box (15 min)

Current README claims `gf256_markov_lm.py` "reproduces the result" and that the shipped artifact is "byte-identical" to what the documented run produces. Both are false today. Correct them after T1.2/T1.3.

### T1.8 Commit the paper stage

One commit: `fix(paper): align claims with shipped code and artifact; real baselines; honest float accounting`.

---

## Stage 2 — First PoC: reproduce V6 (the decisive experiment)

**Exit criterion:** a clean, seeded, logged V6 run either reproduces **Hard BPC ≈ 3.071** on the full validation split, or it doesn't — and the paper's number is corrected to whatever it actually is.

This stage exists because the headline claim currently rests on a number nothing reproduces.

### T2.1 Freeze the V6 configuration (15 min)

Write `configs/v6_tinyshakespeare.json` capturing every hyperparameter the run uses, so the run is reproducible without reading the paper.

### T2.2 Run it with full logging

```bash
python src/gf256_markov_lm.py \
  --seed 32 --steps 20000 --batch 4 --ctx 128 \
  --d 32 --layers 3 --fan_in 3 \
  --ent_max 2.0 --ent_warmup 1000 \
  --hard_start 12000 --hard_ramp 4000 --hard_w 0.5 \
  --lr 3e-3 --eval_every 500 --data data/tinyshakespeare.txt
```

Run it in the background; it is long. Log stdout to `runs/v6_poc_20261020/log.txt`.

**Note — two known defects that will corrupt this run.** Fix both *before* launching, or the numbers are untrustworthy:

- The script saves `gf256_hard_model_best.pt` to the **current working directory**, not `artifacts/`. Every run overwrites the same file. Add a `--run-dir` argument.
- `evaluate()` restores `model.train()` unconditionally, so a final full-val evaluation leaves the model in training mode. Harmless for a metric, but it should be `model.train(was_training)`.

### T2.3 Evaluate on the **full** validation split, not a 128-sequence sample

The paper's 3.071 comes from a 128-sequence quick-eval subset of ~16 k characters. Full val is ~111 k characters. Report both, and report the full-val number as the headline. If they differ, the paper must say which one it quotes and why.

### T2.4 Multiple seeds

Run seeds 32, 33, 34 (or at minimum 3 seeds). Report mean ± std for soft BPC, hard BPC, and the gap to trigram. **A 0.016 BPC margin with one seed is not a result.** If the margin survives 3 seeds, the claim is real and the paper can say so precisely. If it doesn't, that is a finding, not a failure.

### T2.5 Record the outcome honestly

- Run `src/inspect_artifact.py` on the new artifact; paste the real size table into §4.5.
- Update §6.1/§6.2 with measured numbers. If 3.071 does not reproduce, change the abstract, the README, and the conclusion to the real number.
- Commit: `exp(v6): reproduce V6 on full val, N seeds — results in paper.md §6`.

---

## Stage 3 — First scale (only after stage 2 passes)

**Exit criterion:** the 1 GB run has a real, logged, evaluated result — which per F7 no scale-up run currently does.

### T3.1 Clean run tracking (the runs/ mess)

`ZeroFLops/runs/` has 14 dirs, 9 with no metrics, 3 that are `_bak`/`_temp` copies of each other. Adopt one convention and apply it going forward:

```
runs/<arch>_<corpus>_<d>L<layers>_fan<fan_in>_s<seed>_<YYYYMMDD>/
    config.json        full resolved hyperparameters
    data.json          corpus path + sha256 + train/val byte counts
    logs/train_metrics.csv
    logs/eval_metrics.csv
    NOTES.md           what was tried, what happened, what to change
```

The existing v9/v10 scripts already write `run.json`/`data.json`/`logs/` in roughly this shape — reuse their writer rather than inventing a second format.

### T3.2 Archive the WIP tree (mechanical, 15 min)

Move `ZeroFLops/` exploration into `ZeroFLops/archive/` per the existing reorg plan: `scripts/`, `checkpoints/`, `data/`, `runs/`, plus an `ARCHIVE_README.md` explaining the lineage. Use `mv`, never `rm` — that directory has no git and no backup.

Also fix the misleading filenames while moving (`v4_1.py` says v8, `gsm_v9.py` says v7) — see F9.

### T3.3 Baselines on the 1 GB corpus

Run `src/baseline_check.py --data <1GB corpus>` so trigram/bigram are measured on the same split. Without this the scale-up has nothing to be compared against.

### T3.4 V7 at scale, from verified V6 settings

Take the stage-2 verified config, change one thing at a time, one run dir each:

- baseline: d=64, layers=6 on 1 GB
- then capacity: d=128 or layers=12
- then corpus: compare 1 MB vs 200 MB vs 1 GB at fixed model

Do **not** carry over the v9/v10 code as a starting point. Those are drifted forks (F9) whose runs never produced a metric (F7).

### T3.5 MoE, done cleanly

The previous MoE attempts collapsed (`aux_loss 6.14`, `router_entropy 0.0` — F8). If MoE is still wanted:

- start from the verified V6 config, add routing as the *only* change
- instrument expert-usage histogram from step 0 so collapse is visible immediately
- treat router entropy ≈ 0 as a failed run, not a tuning detail

---

## Also worth doing (cheap, do alongside)

- **Batch launcher** — `run_v6.sh` / `run_v6.bat` with `train | generate | inspect | baselines | notebook`. The existing reorg plan already has the content drafted; the scripts it references (`--out` flag on the trainer, `--ckpt` on generate) **do not exist in the current code**, so the launcher must be written against the real CLI, not the planned one.
- **`FUTURE_WORK.md`** — one page: d=128, 1 GB done, MoE, and the readout-elimination problem (the honest version of "zero FLOPs").
- **git for `ZeroFLops/`** — 13 GB with no version control is the actual risk in this project.
- **`.gitignore` for the 118 MB artifact** — decide explicitly: LFS or not. Currently it is committed raw.

---

## Risks

1. **The 118 MB artifact may be unreproducible.** If stage 2 shows V6 never hit 3.071, the honest outcome is a corrected paper, not a reworded one. Do not let the abstract keep a number the runs do not support.
2. **Stage 2 costs GPU hours.** Three seeds × 20 k steps. Run them in the background and do stage 1 while they train.
3. **`ZeroFLops/` has no git.** Treat `archive/` moves as irreversible until `git init` lands there.
4. **The 0.016 BPC margin may not survive multiple seeds.** That is the single most likely way this paper changes shape. Find out in stage 2, not after submission.

---

## Deliverables

| Stage | Artifact |
|-------|----------|
| 1 | `paper.md` with every number traceable to a script in `src/`; Related Work; honest integer/float split; `README.md` corrected |
| 2 | `configs/v6_tinyshakespeare.json`, `runs/v6_poc_*/` with full logs, measured soft/hard BPC over 3 seeds on full val, `artifacts/gf256_hard_model.pt` regenerated |
| 3 | `ZeroFLops/archive/` with lineage README, `runs/` under one naming convention, first 1 GB result with baselines, MoE attempt with router instrumentation |