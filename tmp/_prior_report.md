# Cron Job: v16-status

**Job ID:** 8248abbfac44
**Run Time:** 2026-10-28 18:40:21
**Schedule:** every 25m

**Prompt Characters:** 1254
## Prompt

[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be automatically delivered to the user — do NOT use send_message or try to deliver the output yourself. Just produce your report/output as your final response and the system handles the rest. SILENT: If there is genuinely nothing new to report, respond with exactly "[SILENT]" (nothing else) to suppress delivery. [SILENT] is a literal ASCII control token — never translate or rephrase it, whatever language the rest of your answer uses. Never combine [SILENT] with content — either report your findings normally, or say [SILENT] and nothing more. FAILURE: If a delegated child fails and this cron run must be recorded as failed, put [CRON_FAILURE] on the first line by itself, then explain the child failure on following lines. RECURSION: This is a run of an EXISTING scheduled job — execute the task now. NEVER create or update a cron job because of recurring or future-schedule language in the task prompt below; treat phrasing like "each Monday" or "every day at 9" as context for this run, not as a request to schedule another job.]

Run: cd /c/Users/Chris4K/projects/ZeroFLops-paper && ./.venv-c4/Scripts/python.exe tg_status.py runs/c4/v7mixed_v18_seed32 1500

**Response Characters:** 3463
## Response

**ZeroFLops-paper status — `runs/c4/v7mixed_v18_seed32`**

Requested command ran successfully (exit 0):

```
v7mixed_v18_seed32  runs\c4\v7mixed_v18_seed32
  step          1500 / 1500    (116.1 steps/min)
  eta         ?
  accuracy    hard 6.96% (+0.00)   soft 5.74%
  LEGAL acc   16.17%   <- 7-way, the trained objective
  HOLDOUT opt 30.73%   <- held-out, ties count correct   THE metric
  bpb         hard 15.709   soft 13.746
  vs heur     W237 L563 D0  / 800
  vs random   W656 L133 D11  / 800
  vs ref      W186 L214 D0  / 400
  corpus      106966 rows, 106966 examples
```

The run itself finished at step 1500 with artifacts written (`hard_model_final.pt`, `final_float.pt`, 1112 s elapsed). Holdout plateaued at ~30.7 % by step 300 while hard BPB degraded 8.0 → 15.709 over the run — the decision metric did not improve.

**Corrected full-scale probes (all three now complete)**

| Arm | Games | Label agreement | Hard BPB |
|---|---|---|---|
| `fx_a_s6000` | 320 000 | 50.925 % | 6.5617 |
| `fx_b_s6000` | 640 000 | 50.325 % | 5.9054 |
| `fx_c_s6000` | 640 000 | 51.025 % | 6.4145 |

- The 90 % acceptance gate is **not met** on the full-scale corpora; all three sit at ~51 %.
- This is genuine signal, not chance: corpus-specific chance baseline is **15.14 %** (acceptance corpus,5 901 positions) and **15.22 %** (fx_b/fx_c, 6 086 positions) — ~51 % is +36 pp over chance.
- It is **not** a corpus-difficulty artifact: label distributions are nearly identical across corpora (99.3–99.5 % one-hot slots; mean tie-set size 1.006–1.007).
- The 91.29 % figure belongs to the separate 500-row `overfit_fixed` acceptance run (+76 pp over its15.14 % chance baseline), not to these probes.
- All three learning curves were still rising at step 6000 (e.g. `fx_c`: 11.53 → 47.10 % at step 5000 → 51.02 % at 6000), so none had converged.

**Two genuine defects found (temporary diagnostics only; no production source changed)**

1. **Mislabeled metadata.** `final.json["best_ladder_net"]` is written as `best_sel[0]`, which is now the *label-agreement percentage*, not a ladder-net score. Hence the values 50.925 / 50.325 / 51.025 appear under a ladder-net key.
2. **Resume-seed scale mismatch.** `src/c4/train.py:2314–2332` seeds the primary slot from `meta["ladder_net"]` (a signed net-wins count), while the primary slot is compared against a 0–100 percentage when a per-ply sidecar is loaded. Seed and comparison are on different scales.

`tg_gate_check.py` reports **FAIL**, but its first check is a **stale regex**: it requires the literal `sel = (float(_ladder_net), ...)`, while current source uses `sel = (_primary, ho_pct, -hard_bpb, float(best_distinct))`. The remaining 23 checks pass. So that FAIL is a false alarm, not evidence of a model regression.

**Bottom line:** `v7mixed_v18_seed32` is complete but flat at 30.7 % holdout; the corrected full-scale probes reach ~51 % label agreement, well below the 90 % gate, and were still improving when their 6 000-step budget ended. The shipped gate script needs its `sel()` regex updated, and `final.json` should stop reporting label agreement under `best_ladder_net`.

**Left open:** the exact magnitude of the resume-seed mismatch was not fully enumerated — the final checkpoint-meta scan timed out / hit the Hermes shell-backend hiccup, so the `ladder_net` values across all arms' checkpoints were not captured in a durable file. The source-level scale mismatch itself is confirmed by the code above.
