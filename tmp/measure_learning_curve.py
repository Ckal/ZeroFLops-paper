#!/usr/bin/env python
"""LEARNING CURVE: does the model actually improve with training?

The user asked: "does it learn, can it learn, is it getting better?"

The answer must be a curve, not a single checkpoint. This script scores
every saved checkpoint of an arm on the SAME solver-proven positions and
prints the trajectory against the baselines.

Baselines are included at every step, because a model that ends at 75%
when always-c4 scores 75.7% has learned nothing usable even if its loss
went down.

Measurements (all on identical position sets):
  1. solver-optimal %    -- ground truth: exact solver's optimal set
  2. teacher agreement % -- what the trainer optimises
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.agent import prompt_tokens, trainer_window  # noqa: E402
from src.c4.board import Outcome  # noqa: E402
from src.c4.exact_solver import ExactSolver  # noqa: E402
from src.c4.fmt import PLY_LEN, MOVE_OFFSET, COL_BYTE_OFFSET  # noqa: E402
from tg_solver_truth import optimal_moves  # noqa: E402

HOLD = "runs/_probe/disjoint_true"


def parse_steps(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--steps", default="0,500,1000,1500,2000,2500,3000,"
                                       "3500,4000,4500,5000,5500,6000")
    ap.add_argument("--max-empty", type=int, default=24)
    ap.add_argument("--max-positions", type=int, default=1500)
    ap.add_argument("--budget", type=int, default=120_000_000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from relabel_v23 import decode_header_to_board

    lines = [l.rstrip(b"\r\n")
             for l in open(f"{HOLD}.corpus.txt", "rb") if l.strip()][:args.max_positions]
    z = np.load(f"{HOLD}.npz")

    jobs = []
    for i, ln in enumerate(lines):
        nply = len(ln) // PLY_LEN
        if nply == 0:
            continue
        slots = sorted((int(p), int(s))
                       for s, p in enumerate(z["ply_of"][i]) if p >= 0)
        mv_all = [ln[k * PLY_LEN + MOVE_OFFSET] - 0x41 for k in range(nply)]
        if len(mv_all) != nply:
            continue
        for ply, slot in slots:
            mv = mv_all[:ply]
            off = ply * PLY_LEN
            try:
                b = decode_header_to_board(ln[off:off + PLY_LEN - 1])
            except Exception:
                continue
            if b.outcome() != Outcome.ONGOING:
                continue
            legal = list(b.valid_actions())
            if len(legal) < 2:
                continue
            t = int(np.argmax(z["target"][i][slot]))
            if t not in legal:
                continue
            jobs.append((b, t, legal, ply, tuple(mv)))

    print(f"candidate positions: {len(jobs)}", flush=True)

    solver = ExactSolver(max_empty=args.max_empty, budget=args.budget)
    truth = []
    unproved = 0
    for b, t, legal, ply, mv in jobs:
        side = 1 + (ply % 2)
        try:
            st, opt = optimal_moves(solver, b, side)
        except TimeoutError:
            unproved += 1
            continue
        if st is None or not opt:
            unproved += 1
            continue
        truth.append((b, t, legal, opt, side, ply, mv))

    n = len(truth)
    print(f"PROVEN positions: {n} (unproved {unproved})", flush=True)
    if n < 20:
        print("too few proven positions")
        return 1

    def const_pick(col):
        return [col if col in legal else legal[0]
                for _b, _t, legal, _o, _s, _p, _m in truth]

    def score(picks):
        return sum(1 for i in range(n) if picks[i] in truth[i][3])

    bases = {}
    for col in range(7):
        bases[f"always_c{col}"] = round(100 * score(const_pick(col)) / n, 1)
    bases["chance"] = round(
        100 * score([legal[0] for _b, _t, legal, _o, _s, _p, _m in truth]) / n, 1)
    best_base = max(bases, key=lambda k: bases[k])

    print()
    print("=== BASELINES (solver-optimal %, same positions) ===")
    for k in sorted(bases):
        print(f"  {k:<12} {bases[k]:>6}%")
    print(f"  BEST CONSTANT: {best_base} at {bases[best_base]}%")
    print()

    from src.c4.ui_server import _load_agent

    rows = []
    for step in parse_steps(args.steps):
        ck = pathlib.Path(args.arm_dir) / "checkpoints" / f"step{step}_hard.pt"
        if not ck.exists():
            print(f"  step {step}: MISSING {ck.name}")
            continue
        agent = _load_agent(str(ck), device="cpu")
        picks = []
        for i in range(n):
            b, t, legal, opt, side, ply, mv = truth[i]
            stream = prompt_tokens(list(mv), "v2").tolist()
            x, ridx = trainer_window(stream, agent.ctx)
            with torch.no_grad():
                lg = agent.model.hard_forward(
                    torch.tensor([x], dtype=torch.long), agent.hard)
            v = lg[0, ridx, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
            mask = torch.tensor([c in legal for c in range(7)])
            v = torch.where(mask, v, torch.tensor(-1e9, dtype=v.dtype))
            picks.append(int(torch.argmax(v)))

        opt_hits = sum(1 for i in range(n) if picks[i] in truth[i][3])
        teach_hits = sum(1 for i in range(n) if picks[i] == truth[i][1])
        r = {
            "step": step,
            "solver_optimal_pct": round(100 * opt_hits / n, 1),
            "teacher_agreement_pct": round(100 * teach_hits / n, 1),
            "vs_best_constant_pp": round(100 * opt_hits / n - bases[best_base], 1),
            "vs_always_c3_pp": round(100 * opt_hits / n - bases["always_c3"], 1),
            "col_dist": {str(k): v for k, v in
                         sorted(collections.Counter(picks).items())},
        }
        rows.append(r)
        print(f"  step {step:>5}: solver-opt {r['solver_optimal_pct']:>6}%  "
              f"teacher {r['teacher_agreement_pct']:>6}%  "
              f"(vs best {r['vs_best_constant_pp']:+.1f}pp, "
              f"vs c3 {r['vs_always_c3_pp']:+.1f}pp)")

    if args.out:
        out = {"arm_dir": args.arm_dir, "n_proven": n, "unproved": unproved,
               "max_empty": args.max_empty, "budget": args.budget,
               "baselines": bases, "best_constant": best_base,
               "curve": rows}
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
