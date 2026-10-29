"""MEASURE (carefully): rows/second vs batch size in the REAL trainer.

Why this exists
---------------
Claude's review proposes batch 32 with the table:

    batch | s/step | rows/s | VRAM
      16  |  0.69  |  ~23   | 2.7 GiB
      32  |  0.66  |  ~48   | 5.1 GiB
      64  |  1.27  |  ~50   | 9.9 GiB

i.e. "batch 32 gives twice the throughput for almost no extra time per
step", which would shrink a 5-epoch run from ~37 h to ~9 h.

That table came from `tg_scale4096.py` -- a DIFFERENT probe (4,096 rows, the
V22 architecture, its own training loop, no per-ply sidecar, no dense lag
set, ctx 128). The real V24 path is `src.c4.train` with `--perply-targets`,
the 12-lag dense set, ctx 192 and the hard-export ramp.

The question that actually matters for the V24 budget is NOT "s/step" but
**rows/second**, because one epoch is a fixed number of ROWS. If batch 32
costs proportionally more time per step, the epoch time does not halve.

This script runs the real trainer at each batch size for enough steps to
amortise warm-up, then reports the per-step time over a warm window
(steps after the first WARMUP), rows/s, and the projected wall time for
one epoch and for five epochs over the 320,000-row corpus.

Run: ./.venv-c4/Scripts/python.exe tmp/measure_batch_scaling.py
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

STEPS = 40
WARMUP = 8          # steps excluded from the timing window
BATCHES = [16, 32, 64]

OUT = ROOT / "runs/_probe/batch_scaling_real_trainer.json"


def run_arm(batch: int) -> dict:
    run_dir = ROOT / "runs/_probe" / f"_bs{batch}"
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
        "--eval-every", "1000000", "--save-every", "1000000",
        "--eval-games", "0",
        "--max-legacy-frac", "0", "--corpus-source", "teacher",
        "--seed-games", "0", "--selfplay-games", "0", "--teacher-games", "0",
        "--selfplay-every", "1000000",
        "--hard-start", "0", "--hard-ramp", "1000", "--hard-w", "0.5",
        "--table-bias", "0.0", "--lags", DENSE, "--col-prior", "0.0",
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, timeout=5400)
    mj = run_dir / "metrics.jsonl"
    steps = []
    if mj.exists():
        for line in mj.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "step" in r and "elapsed_s" in r:
                    steps.append((int(r["step"]), float(r["elapsed_s"])))
    return {"batch": batch, "rc": p.returncode, "steps": steps,
            "stderr": p.stderr[-800:]}


def main() -> int:
    if not (ROOT / CORPUS).exists() or not (ROOT / SIDECAR).exists():
        print("SKIP: corpus/sidecar missing")
        return 0

    results = []
    for b in BATCHES:
        print(f"=== batch {b} ===", flush=True)
        r = run_arm(b)
        results.append(r)
        print(f"    rc={r['rc']} steps_parsed={len(r['steps'])}", flush=True)
        if r["rc"] != 0:
            print(r["stderr"][-600:])

    OUT.write_text(json.dumps(results, indent=2))

    print()
    print("=== REAL TRAINER: batch scaling (per-ply sidecar, ctx 192, dense lags) ===")
    print(f"{'batch':>6} {'s/step*':>9} {'rows/s*':>9} {'vs b16':>8} "
          f"{'1 epoch':>10} {'5 epochs':>10}")
    base_rps = None
    rows_out = []
    for r in results:
        st = r["steps"]
        if r["rc"] != 0 or len(st) <= WARMUP + 2:
            print(f"{r['batch']:>6}   (failed: rc={r['rc']})")
            continue
        win = st[WARMUP:]
        dt = win[-1][1] - win[0][1]
        dn = win[-1][0] - win[0][0]
        sps = dt / max(1, dn)
        rps = r["batch"] / sps
        if base_rps is None:
            base_rps = rps
        ep_h = ROWS / rps / 3600
        five_h = 5 * ep_h
        rows_out.append((r["batch"], sps, rps, ep_h, five_h))
        print(f"{r['batch']:>6} {sps:>9.3f} {rps:>9.2f} {rps/base_rps:>7.2f}x "
              f"{ep_h:>9.2f}h {five_h:>9.2f}h")
    print(f"* per-step time over steps {WARMUP + 1}+ (warm-up excluded).")
    print()
    if rows_out:
        b16 = next((x for x in rows_out if x[0] == 16), None)
        b32 = next((x for x in rows_out if x[0] == 32), None)
        if b16 and b32:
            print(f"VERDICT on Claude's 'batch 32 = 2x throughput': "
                  f"measured {b32[2]/b16[2]:.2f}x rows/s "
                  f"(epoch {b16[3]:.1f}h -> {b32[3]:.1f}h, "
                  f"5 epochs {b16[4]:.1f}h -> {b32[4]:.1f}h).")
    print()
    print("Claude's table (tg_scale4096: 4096 rows, V22 arch, own loop):")
    print("  16 -> 0.69 s/step ~23 rows/s | 32 -> 0.66 s/step ~48 rows/s "
          "| 64 -> 1.27 s/step ~50 rows/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
