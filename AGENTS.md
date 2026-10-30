# AGENTS.md

Guidance for AI agents (Hermes, Copilot, others) working in this repo.

## What this is

ZeroFLops: a character-level GF(256) "Lagrange field" transformer whose trained
form is an integer program (XOR, GF multiply via tables, S-box lookups). Result:
3.071 BPC on Tiny Shakespeare. A second track trains the same model to play
Connect Four (`src/c4/`). Read `README.md`, `docs/CURRENT_TRUTH.md`
(authoritative, long) and `docs/V25_BOTTLENECK_RUN.md` (latest work) first.

## Layout

- `src/gf256_markov_lm.py` V7 (d64/L6). `_v6.py` V6 (d32/L3, the reference).
  `_v6_1.py`, `_v6_bottleneck.py` (`--arch v6b`).
- `src/c4/` Connect Four: `train.py` (trainer), `ui_server.py`, `duel.py`,
  `transcript_v2.py`, `mirror.py` (canonical left-right symmetry).
- `run_v24.sh` the one training launcher; configure with `V24_*` env vars.
- `tools/` data tools. `tmp/` curated tests + scorers. `runs/` outputs (gitignored).
- Repo root `tg_*.py`, `_*.py`, `run_*.sh`: throwaway experiment scripts, mostly
  gitignored, historical. Do not add more; do not treat them as source of truth.
  Put new experiments in `tmp/` or `tools/` and promote only what is tested.

## Rules

1. **Never edit `gf256_markov_lm_v6.py` to try an idea.** Checkpoints and loaders
   depend on its `export_hard`/`hard_forward`. Make a new file and an `--arch` flag
   (this is how V6b was done). A past in-place edit broke export and had to be reverted.
2. **Reuse before writing.** Mirroring already exists in `src/c4/mirror.py`;
   a second tool (`tools/mirror_corpus.py`) produced a corrupt corpus.
3. **Verify data before training.** Row counts of corpus and sidecar must match
   (corpus is one game per `\n`-terminated line; the trainer enforces this).
   Check content, not just shape: confirm mirror rows equal flipped targets.
4. **Verify a change end to end**, not by a smoke test only: short real run,
   then the independent scorer (`tmp/measure_promotion_scorer.py <run_dir>`)
   must agree with the trainer's LABEL.
5. Every loader that rebuilds a model from an export (`ui_server._load_agent`,
   `duel`, `measure_promotion_scorer`) selects the class from the payload
   (`readout_mlp` => V6b, `d>=64` => V7, else V6). New archs must be added to all.
6. Do not claim "zero FLOPs" for the readout: it is float in every variant.
   Only the trunk is integer.
7. Do not overstate results. Cite holdout (disjoint) label agreement, with n.
   Prior hypotheses and their verdicts are in `docs/CURRENT_TRUTH.md` §6; check
   before re-testing (mirror augmentation was already tied at 71.8%).
8. Keep comments free of absolute local paths and dates (they were scrubbed once).
9. Line endings: files are LF in git; the working tree showed large CRLF-only
   diffs. Use `git diff -w --ignore-cr-at-eol` to see real changes.

## Running (Windows)

- Python: `.venv-c4\Scripts\python.exe`; bash scripts need Git Bash
  (`C:\Program Files\Git\bin\bash.exe`; the default `bash` is WSL and fails).
- Launch/resume: see `docs/V25_BOTTLENECK_RUN.md`. A lock dir `runs/_probe/.v24.lock`
  prevents concurrent chains.
- Long outputs: keep reasoning effort low when running as an agent; a past run
  died with "output-token limit reached" from reasoning alone.

## Branch state

Work happens on `feat/threat-vs-quiet-measure` (ahead of `main`, unmerged).
Commit small and describe what was *verified*, not what was intended.
