#!/usr/bin/env python
"""DECISIVE test: does the model learn, measured on positions with a
UNIQUE optimal move?

Why this subset matters
-----------------------
The authoritative solver proves optimal moves for 560 positions, but the
optimal-SET SIZE distribution is:

    1 optimal move : 112 positions (20.0%)
    2 optimal moves:  55
    3 optimal moves:  41
    4 optimal moves:  31
    5 optimal moves:  68
    6 optimal moves: 101
    7 optimal moves: 152   <-- EVERY column is optimal

On 80% of positions the solver cannot distinguish a good policy from a
bad one, because several moves are equally good. A constant policy can
score ~75% "solver-optimal" there without knowing anything.

So the ONLY positions where the metric has teeth are the 112 with a
UNIQUE optimal move. There a random policy scores ~1/7 of the time and
the model must actually READ THE BOARD to get it right.

Solver truth is expensive, so it is computed ONCE and cached to
truth_cache.json, keyed by (max_empty, budget, n_positions, corpus hash).
Every arm reuses that cache, so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
CACHE = "runs/_probe/unique_optimum_truth_cache.json"


def build_truth(args):
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
    out = []
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
        out.append({
            "legal": list(legal), "teacher": int(t),
            "opt": sorted(int(o) for o in opt),
            "moves": list(mv), "ply": int(ply),
            "hdr": bytes(ln[off:off + PLY_LEN - 1]).hex(),
        })
    return out, unproved, len(jobs)


def load_or_build_truth(args):
    cp = ROOT / CACHE
    key = {
        "max_empty": args.max_empty, "budget": args.budget,
        "max_positions": args.max_positions,
        "corpus_sha": hashlib.sha256(
            (ROOT / f"{HOLD}.corpus.txt").read_bytes()[:4_000_000]).hexdigest()[:16],
    }
    if cp.exists():
        d = json.loads(cp.read_text())
        if d.get("key") == key:
            print(f"reusing cached solver truth: {len(d['truth'])} positions",
                  flush=True)
            return d["truth"], d["unproved"], d["candidates"]
    truth, unproved, cands = build_truth(args)
    cp.write_text(json.dumps({"key": key, "truth": truth,
                               "unproved": unproved, "candidates": cands}))
    print(f"cached solver truth -> {CACHE}", flush=True)
    return truth, unproved, cands


def r1(x):
    return None if x is None else round(x, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--steps", default="0,1000,2000,3000,4000,5000,6000")
    ap.add_argument("--max-empty", type=int, default=24)
    ap.add_argument("--max-positions", type=int, default=1500)
    ap.add_argument("--budget", type=int, default=120_000_000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    truth, unproved, cands = load_or_build_truth(args)
    n = len(truth)
    opts = [set(t["opt"]) for t in truth]
    unique_idx = [i for i in range(n) if len(opts[i]) == 1]
    tied_idx = [i for i in range(n) if len(opts[i]) > 1]
    print(f"PROVEN: {n} (unproved {unproved})   "
          f"UNIQUE optimum: {len(unique_idx)}   TIED: {len(tied_idx)}",
          flush=True)
    if n < 20:
        print("too few proven positions")
        return 1

    def const_pick(col):
        return [col if col in t["legal"] else t["legal"][0] for t in truth]

    def sub_score(picks, idx):
        if not idx:
            return None
        return 100 * sum(1 for i in idx if picks[i] in opts[i]) / len(idx)

    bases = {}
    for col in range(7):
        pc = const_pick(col)
        bases[f"always_c{col}"] = {
            "all": r1(sub_score(pc, range(n))),
            "unique": r1(sub_score(pc, unique_idx)),
            "tied": r1(sub_score(pc, tied_idx))}
    chance = [t["legal"][0] for t in truth]
    bases["chance"] = {
        "all": r1(sub_score(chance, range(n))),
        "unique": r1(sub_score(chance, unique_idx)),
        "tied": r1(sub_score(chance, tied_idx))}
    best_u = max((k for k in bases if k.startswith("always_")),
                 key=lambda k: bases[k]["unique"])

    print()
    print("=== BASELINES by subset (solver-optimal %) ===")
    print(f"{'policy':<12} {'ALL':>8} {'UNIQUE':>8} {'TIED':>8}")
    for k in sorted(bases):
        v = bases[k]
        print(f"{k:<12} {v['all']:>7}% {v['unique']:>7}% {v['tied']:>7}%")
    print(f"  best on UNIQUE: {best_u} at {bases[best_u]['unique']}%")
    print()

    from src.c4.ui_server import _load_agent

    rows = []
    for step in [int(s) for s in args.steps.split(",") if s.strip()]:
        ck = pathlib.Path(args.arm_dir) / "checkpoints" / f"step{step}_hard.pt"
        if not ck.exists():
            print(f"  step {step}: MISSING")
            continue
        agent = _load_agent(str(ck), device="cpu")
        picks = []
        for t in truth:
            stream = prompt_tokens(list(t["moves"]), "v2").tolist()
            x, ridx = trainer_window(stream, agent.ctx)
            with torch.no_grad():
                lg = agent.model.hard_forward(
                    torch.tensor([x], dtype=torch.long), agent.hard)
            v = lg[0, ridx, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
            mask = torch.tensor([c in t["legal"] for c in range(7)])
            v = torch.where(mask, v, torch.tensor(-1e9, dtype=v.dtype))
            picks.append(int(torch.argmax(v)))

        u = sub_score(picks, unique_idx)
        r = {
            "step": step,
            "all_pct": r1(sub_score(picks, range(n))),
            "unique_pct": r1(u),
            "tied_pct": r1(sub_score(picks, tied_idx)),
            "teacher_all_pct": r1(
                100 * sum(1 for i in range(n)
                          if picks[i] == truth[i]["teacher"]) / n),
            "teacher_unique_pct": r1(
                100 * sum(1 for i in unique_idx
                          if picks[i] == truth[i]["teacher"]) / len(unique_idx)),
            "vs_best_unique_pp": r1(u - bases[best_u]["unique"]),
            "col_dist": {str(k): v for k, v in
                         sorted(collections.Counter(picks).items())},
        }
        rows.append(r)
        print(f"  step {step:>5}: ALL {r['all_pct']:>6}%  "
              f"UNIQUE {r['unique_pct']:>6}%  TIED {r['tied_pct']:>6}%  "
              f"teacher(all) {r['teacher_all_pct']:>6}%  "
              f"teacher(uniq) {r['teacher_unique_pct']:>6}%  "
              f"(vs best-unique {r['vs_best_unique_pp']:+.1f}pp)")

    if args.out:
        out = {"arm_dir": args.arm_dir, "n_proven": n, "n_unique": len(unique_idx),
               "n_tied": len(tied_idx), "unproved": unproved, "candidates": cands,
               "max_empty": args.max_empty, "budget": args.budget,
               "baselines": bases, "best_on_unique": best_u, "curve": rows}
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
