#!/usr/bin/env python
"""PROOF OF FIX: relabel a small corpus with TIE SETS and retrain.

The D14 finding: 99% of the training labels come from a strong negamax
teacher that returns ONE column, and the exact solver proves that 80% of
positions have SEVERAL equally-optimal moves. The model therefore learns
the teacher's centre-biased tie-break instead of optimal play.

This script tests the fix end-to-end on a SMALL corpus so it runs in
minutes, not hours:

  1. take N rows from the training corpus
  2. relabel every ply with the exact solver's TIE SET (uniform over the
     set) wherever the solver can prove it
  3. write a matching per-ply sidecar
  4. train the SAME model on the OLD labels and on the NEW labels
  5. score both on the UNIQUE-optimum subset of the independent holdout

If the tie-set arm wins on the unique subset, the data is the bottleneck
and the fix is worth the full relabel.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board, Outcome  # noqa: E402
from src.c4.exact_solver import ExactSolver  # noqa: E402
from src.c4.fmt import PLY_LEN, MOVE_OFFSET  # noqa: E402
from relabel_v23 import exact_best_set  # noqa: E402
from relabel_v23 import decode_header_to_board  # noqa: E402


def relabel_rows(rows, solver, max_empty, budget, depth, sbudget):
    """Return (targets, ply_of, n_plies, teacher_src) with TIE-SET targets."""
    from src.c4.strong_solver import strong_policy, clear_tt

    n = len(rows)
    P = 19
    target = np.zeros((n, P, 7), dtype=np.float32)
    ply_of = np.full((n, P), -1, dtype=np.int16)
    n_plies = np.zeros(n, dtype=np.int16)
    teacher = np.zeros((n, P), dtype=np.int8)
    stats = collections.Counter()

    for i, seq in enumerate(rows):
        nrec = len(seq) // PLY_LEN
        for s in range(min(P, nrec)):
            p = nrec - 1 - s
            hdr = seq[p * PLY_LEN:p * PLY_LEN + PLY_LEN - 1]
            try:
                b = decode_header_to_board(hdr)
            except Exception:
                continue
            if b.outcome() != Outcome.ONGOING:
                continue
            legal = list(b.valid_actions())
            if not legal:
                continue
            empty = 42 - int(sum(b.heights))
            best = None
            src = 0
            if empty <= max_empty:
                try:
                    best, _v = exact_best_set(b, solver)
                except TimeoutError:
                    best = None
                if best:
                    src = 1
            if not best:
                clear_tt()
                c = strong_policy(b, [], depth=depth, budget=sbudget)
                if c is None or c not in legal:
                    c = legal[0]
                best = [c]
                src = 2
            for c in best:
                if 0 <= c <= 6:
                    target[i, s, c] = 1.0 / len(best)
            ply_of[i, s] = p
            teacher[i, s] = src
            stats[f"src{src}"] += 1
            stats[f"tie{len(best)}"] += 1
        lab = ply_of[i][ply_of[i] >= 0]
        if len(lab):
            n_plies[i] = int(lab.max()) + 1
        if (i + 1) % 200 == 0:
            print(f"  relabel {i+1}/{n}", flush=True)
    return target, ply_of, n_plies, teacher, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--outdir", default="runs/_probe/tieset_ab")
    ap.add_argument("--max-empty", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--depth", type=int, default=13)
    ap.add_argument("--sbudget", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--skip-relabel", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    src_corpus = ROOT / "runs/_probe/perply_v23_full/corpus_320k_fixed.txt"
    src_side = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"

    rows = []
    with open(src_corpus, "rb") as fh:
        for i, line in enumerate(fh):
            if i >= args.rows:
                break
            rows.append(line.rstrip(b"\r\n"))
    print(f"rows: {len(rows)}")

    # --- OLD (one-hot) sidecar, sliced to the same rows ---
    z = np.load(src_side)
    old_target = z["target"][:args.rows].copy()
    old_ply = z["ply_of"][:args.rows].copy()
    old_nply = z["n_plies"][:args.rows].copy()
    old_teacher = z["teacher"][:args.rows].copy()

    # --- NEW (tie-set) sidecar ---
    new_path = out / "tieset_targets.npz"
    if args.skip_relabel and new_path.exists():
        zz = np.load(new_path)
        new_target, new_ply = zz["target"], zz["ply_of"]
        new_nply, new_teacher = zz["n_plies"], zz["teacher"]
        print("reusing existing tie-set sidecar")
    else:
        solver = ExactSolver(max_empty=args.max_empty, budget=args.budget)
        t0 = time.time()
        (new_target, new_ply, new_nply, new_teacher,
         stats) = relabel_rows(rows, solver, args.max_empty, args.budget,
                               args.depth, args.sbudget)
        print(f"relabel done in {time.time()-t0:.0f}s: {dict(stats)}")
        np.savez_compressed(new_path, target=new_target, ply_of=new_ply,
                            n_plies=new_nply, teacher=new_teacher)

    # --- write the two training corpora ---
    # The trainer expects corpus.txt INSIDE --run-dir when --load-corpus is
    # used, so each arm gets its own copy.
    old_path = out / "old_targets.npz"
    np.savez_compressed(old_path, target=old_target, ply_of=old_ply,
                        n_plies=old_nply, teacher=old_teacher)

    arms = [("old_onehot", old_path), ("new_tieset", new_path)]
    for name, side in arms:
        rundir = out / f"arm_{name}"
        rundir.mkdir(parents=True, exist_ok=True)
        with open(rundir / "corpus.txt", "wb") as fh:
            for r in rows:
                fh.write(r + b"\n")
        # the sidecar must line up with the corpus rows
        z2 = np.load(side)
        assert z2["target"].shape[0] == len(rows), (
            f"sidecar {side} has {z2['target'].shape[0]} rows, "
            f"corpus has {len(rows)}")

    # --- tie statistics comparison ---
    def tie_stats(tag, tgt, ply):
        tot = 0
        tied = 0
        for i in range(0, tgt.shape[0], 3):
            for s in range(tgt.shape[1]):
                if ply[i, s] < 0:
                    continue
                v = tgt[i, s]
                mx = v.max()
                tot += 1
                if (v >= mx - 1e-6).sum() > 1:
                    tied += 1
        print(f"  {tag}: {tied}/{tot} = {100*tied/max(1,tot):.1f}% multi-column targets")
        return 100 * tied / max(1, tot)

    print()
    print("=== TARGET TIE STRUCTURE ===")
    tie_stats("OLD one-hot ", old_target, old_ply)
    tie_stats("NEW tie-set ", new_target, new_ply)

    print()
    print("=== TRAIN TWO ARMS (same data, same steps, same seed) ===")
    for name, side in arms:
        rundir = out / f"arm_{name}"
        cmd = [
            "./.venv-c4/Scripts/python.exe", "-u", "-m", "src.c4.train",
            "--load-corpus",
            "--perply-targets", str(side),
            "--run-dir", str(rundir),
            "--steps", str(args.steps),
            "--seed", "32",
            "--ctx", "192",
            "--batch", "16",
            "--table-bias", "0.0",
            "--hard-start", "0",
            # The corpus is FIXED for this A/B (we load 3000 rows and
            # supervise them with a per-ply sidecar).  The trainer
            # refuses to grow the corpus when a sidecar is present,
            # because growth would desynchronise sidecar row ids from
            # corpus rows -- so BOTH growth knobs must be 0.  Without
            # these the trainer exits 1 with:
            #   "--load-corpus with a per-ply sidecar requires
            #    --selfplay-games 0 --teacher-games 0"
            "--selfplay-games", "0",
            "--teacher-games", "0",
        ]
        print(f"\n--- arm {name} ---")
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=7200)
        log = rundir / "train.log"
        log.write_text(r.stdout + "\n=== STDERR ===\n" + r.stderr,
                       encoding="utf-8", errors="replace")
        print(f"  exit={r.returncode}")
        for line in (r.stdout or "").splitlines()[-8:]:
            print("   ", line)
        if r.returncode != 0:
            print("   STDERR tail:")
            for line in (r.stderr or "").splitlines()[-12:]:
                print("     ", line)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
