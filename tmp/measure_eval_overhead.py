"""MEASURE the real V24 throughput, including the eval overhead.

Why this exists
---------------
`tmp/measure_batch_scaling.py` ran the trainer with `--eval-every 1000000`,
so it measured the TRAIN cost only:

    batch 16 -> 0.735 s/step -> 21.8 rows/s

But the corrected A/B (`run_fixed_ab_corrected.sh`) ran `--eval-every 500`
and measured 1.315 s/step -> 12.2 rows/s.  The 0.58 s/step gap is the
per-eval cost amortised over 500 steps.  V24 uses `--eval-every 2000`, so
the amortised overhead is 4x smaller -- and the batch decision has to be
made with that number, not the train-only number.

Claude's review claims batch 32 gives 2x throughput, which would turn a
~37 h run into ~9 h.  Measured on the real path the train-only gain is
1.27x.  This script measures the V24-realistic number (eval cadence
included) so the batch decision rests on the configuration we will
actually run.

Arms:
    (batch, eval_every) in {(16, 1000000), (32, 1000000),
                             (16, 50),          (32, 50)}
The two `eval_every=50` arms have evals at steps 0/50/100; the difference
against the matching no-eval arm is the cost of ONE eval, which is what
V24 amortises over `eval_every` steps.

Run: ./.venv-c4/Scripts/python.exe tmp/measure_eval_overhead.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv-c4/Scripts/python.exe")
CORPUS = "runs/_probe/perply_v23_full/corpus_320k_fixed.txt"
SIDECAR = "runs/_probe/perply_v23_full/merged_repaired.npz"
DENSE = "0,1,2,3,4,5,6,7,8,9,10,20"
ROWS = 320_000

STEPS = 100
NO_EVAL = 1_000_000
EVAL_EVERY = 50          # V24-realistic cadence, scaled down to fit the probe
EVAL_GAMES = 24          # exactly V24's ladder size
ARMS = [
    (16, NO_EVAL),
    (32, NO_EVAL),
    (16, EVAL_EVERY),
    (32, EVAL_EVERY),
]

OUT = ROOT / "runs/_probe/eval_overhead_real_trainer.json"


def run_arm(batch: int, eval_every: int) -> dict:
    tag = f"_eo{batch}_{eval_every}"
    run_dir = ROOT / "runs/_probe" / tag
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    shutil.copy2(ROOT / CORPUS, run_dir / "corpus.txt")
    shutil.copy2(ROOT / SIDECAR, run_dir / "perply_targets.npz")

    cmd = [
        PY, "-u", "-m", "src.c4.train",
        "--run-dir", str(run_dir),
        "--perply-targets", str(run_dir / "perply_targets.npz"),
        "--load-corpus", "--transcript", "v2", "--move-loss", "7way",
        "--soft-w", "1.0",
        "--steps", str(STEPS), "--batch", str(batch), "--ctx", "192",
        "--lr", "0.0006", "--seed", "32",
        "--eval-every", str(eval_every), "--save-every", str(NO_EVAL),
        "--eval-games", str(EVAL_GAMES),
        "--max-legacy-frac", "0", "--corpus-source", "teacher",
        "--seed-games", "0", "--selfplay-games", "0", "--teacher-games", "0",
        "--selfplay-every", "1000000",
        "--hard-start", "0", "--hard-ramp", "1000", "--hard-w", "0.5",
        "--table-bias", "0.0", "--lags", DENSE, "--col-prior", "0.0",
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, timeout=7200)
    mj = run_dir / "metrics.jsonl"
    rows = []
    if mj.exists():
        for line in mj.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "step" in r and "elapsed_s" in r:
                    rows.append((int(r["step"]), float(r["elapsed_s"])))
    return {"batch": batch, "eval_every": eval_every, "rc": p.returncode,
            "rows": rows, "stderr": p.stderr[-1500:]}


def train_only_seconds(rows) -> float:
    """Total time minus the per-eval cost, derived from the metric rows.

    The trainer writes a metric row at every eval step.  The elapsed time
    between two consecutive eval rows is (gap_steps * s_per_train_step) +
    (eval cost).  With one eval in the window we cannot separate them, so
    instead we use the TOTAL elapsed for the whole run and subtract the
    number of evals times the measured eval cost (obtained from the
    eval/no-eval arm pair).
    """
    return rows[-1][1] if rows else float("nan")


def main() -> int:
    if not (ROOT / CORPUS).exists() or not (ROOT / SIDECAR).exists():
        print("SKIP: corpus/sidecar missing")
        return 0

    results = []
    for b, e in ARMS:
        print(f"=== batch {b}  eval_every {e} ===", flush=True)
        r = run_arm(b, e)
        results.append(r)
        print(f"    rc={r['rc']}  rows={r['rows']}", flush=True)
        if r["rc"] != 0:
            print(r["stderr"][-800:], flush=True)

    by = {(r["batch"], r["eval_every"]): r for r in results}
    out = {"steps": STEPS, "rows": ROWS, "arms": []}

    for b in (16, 32):
        ne = by.get((b, NO_EVAL))
        ee = by.get((b, EVAL_EVERY))
        if not ne or not ne["rows"]:
            continue
        train_total = ne["rows"][-1][1]
        train_sps = STEPS / train_total if train_total > 0 else float("nan")
        train_rows_s = b * train_sps

        eval_cost = None
        if ee and ee["rows"]:
            # evals at step 0, 50, 100 -> 3 eval events, 2 of them inside
            # the timed window; the run's total includes all 3.
            eval_total = ee["rows"][-1][1]
            n_evals = len(ee["rows"]) - 1        # evals after the step-0 one
            extra = eval_total - train_total
            eval_cost = extra / n_evals if n_evals > 0 else None

        # V24-realistic cadence: eval_every = 2000
        amortised = (eval_cost / 2000.0) if eval_cost else 0.0
        v24_sps = train_sps
        if eval_cost:
            v24_sps = 1.0 / (1.0 / train_sps + amortised)
        v24_rows_s = b * v24_sps

        arm = {
            "batch": b,
            "train_s_per_step": 1.0 / train_sps if train_sps else None,
            "train_rows_per_s": train_rows_s,
            "eval_cost_s": eval_cost,
            "v24_amortised_eval_s_per_step": amortised,
            "v24_s_per_step": 1.0 / v24_sps if v24_sps else None,
            "v24_rows_per_s": v24_rows_s,
            "v24_hours_per_epoch": ROWS / v24_rows_s / 3600.0 if v24_rows_s else None,
            "v24_hours_5_epochs": (ROWS * 5) / v24_rows_s / 3600.0 if v24_rows_s else None,
        }
        out["arms"].append(arm)
        print(json.dumps(arm, indent=2))

    if len(out["arms"]) == 2:
        a, c = out["arms"]
        gain = c["v24_rows_per_s"] / a["v24_rows_per_s"]
        out["v24_batch32_gain_vs_batch16"] = gain
        out["claude_claimed_gain"] = 2.0
        print(f"\n=== V24-realistic batch 32 gain: {gain:.3f}x "
              f"(Claude claimed {2.0}x) ===")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
