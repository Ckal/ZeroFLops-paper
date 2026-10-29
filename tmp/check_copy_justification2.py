"""Is the GROUND TRUTH symmetric, and is the model's copy justified?

Three questions, one probe. Two methodological requirements, both of
which were violated by the first version of this script:

1. The solver's answer must be the FULL SET of optimal moves. A single
   returned move breaks ties toward the lowest column index, which does
   not commute with the reflection c -> 6-c, so a single-move comparison
   manufactures asymmetries that are properties of the tie-break, not of
   the position. Use `tg_solver_truth.optimal_moves`.

2. The MODEL must be measured on the path that actually plays: the hard
   integer export, `use_hard=True`. The float path gives different
   answers, and mixing the two produces conclusions that contradict
   themselves.

Also use the CANONICAL `src.c4.mirror.mirror_board` rather than a
hand-rolled reflection -- the hand-rolled one forgot to reflect `mask`
and `last_move`, which silently produced an illegal board.

Q1. For a position where the opponent has stacked one column, is playing
    that same column (the "copy") actually optimal?
Q2. Is the GROUND TRUTH symmetric? solve(b) vs solve(mirror(b)).
Q3. Is the MODEL mirror-consistent on the same pair of positions?

An opponent always stacks column `sc`; the model plays P2.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board                     # noqa: E402
from src.c4.exact_solver import ExactSolver         # noqa: E402
from src.c4.strong_solver import strong_policy      # noqa: E402
from src.c4.mirror import mirror_board              # noqa: E402
from src.c4.ui_server import _load_agent            # noqa: E402
from tg_solver_truth import optimal_moves           # noqa: E402


def build(sc: int, height: int):
    """Position: opponent has `height` pieces stacked in column sc.

    The opponent is P1 (it moves first), so the model is P2.
    """
    b = Board()
    moves = []
    for _ in range(height):
        if sc not in b.valid_actions():
            return None, None
        b.apply_move(sc)
        moves.append(sc)
        if b.is_win() != 0:
            return None, None
    return b, moves


def solve_set(solver, b: Board, side: int):
    """(set_of_optimal_cols, source) using the FULL optimal set."""
    try:
        status, cols = optimal_moves(solver, b, side)
    except Exception:
        return set(), "unproved"
    if status is None or not cols:
        return set(), "unproved"
    return set(int(c) for c in cols), "exact"


def strong_set(b: Board):
    """The strong solver's single move (a heuristic, recorded as such)."""
    try:
        mv = strong_policy(b, None)
    except Exception:
        return set(), "unproved"
    if mv is None:
        return set(), "unproved"
    return {int(mv)}, "strong"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt")
    ap.add_argument("--out", default="runs/_probe/copy_justification.json")
    args = ap.parse_args()

    agent = _load_agent(str(ROOT / args.model), device="cpu")
    exact = ExactSolver(max_empty=12, budget=2_000_000)
    print(f"model: {args.model}")
    print("model path: HARD export (use_hard=True)\n")

    def model_pick(b, moves):
        """The model's move on the hard path, with the legal mask applied."""
        import torch
        lg = agent.hard_column_logits(b, list(moves))
        mask = agent.legal_mask(b).to(lg.device)
        NEG = torch.tensor(-1e9, dtype=lg.dtype, device=lg.device)
        p = torch.softmax(torch.where(mask, lg, NEG), dim=-1).cpu().numpy()
        return int(p.argmax()), float(p.max())

    rows = []
    print("=== Q1: is the copy optimal? (per stack column, per height) ===")
    print(f"{'col':>4} {'h':>2} | {'model':>5} | {'optimal set':>22} | "
          f"{'src':>8} | {'copy optimal?':>13} | {'model optimal?':>14}")
    print("-" * 84)
    for sc in (0, 1, 3, 5, 6):
        for height in (1, 2, 3, 4, 5):
            b, moves = build(sc, height)
            if b is None:
                continue
            legal = b.valid_actions()
            if not legal:
                continue
            choice, pmax = model_pick(b, moves)
            opt, src = solve_set(exact, b, b.turn)
            if not opt:
                opt, src = strong_set(b)
            in_set = sc in opt
            rows.append({
                "stack_col": sc, "height": height,
                "model_choice": choice, "p_max": pmax,
                "optimal": sorted(opt), "src": src,
                "copy_legal": sc in legal, "copy_in_optimal": in_set,
                "model_optimal": (choice in opt) if opt else None,
            })
            print(f"{sc:>4} {height:>2} | {choice:>5} | {str(sorted(opt)):>22} | "
                  f"{src:>8} | {str(in_set):>13} | "
                  f"{str(choice in opt) if opt else '?':>14}")

    # ---- Q2: is the GROUND TRUTH symmetric? ----
    print("\n=== Q2: ground-truth symmetry (solve b vs solve mirror(b)) ===")
    sym_rows = []
    for sc in (0, 1, 3, 5, 6):
        for height in (1, 2, 3, 4):
            b, moves = build(sc, height)
            if b is None:
                continue
            mb = mirror_board(b)
            opt, src = solve_set(exact, b, b.turn)
            if not opt:
                opt, src = strong_set(b)
            mopt, msrc = solve_set(exact, mb, mb.turn)
            if not mopt:
                mopt, msrc = strong_set(mb)
            if not opt or not mopt:
                continue
            reflected = {6 - c for c in opt}
            agree = (mopt == reflected)
            sym_rows.append({
                "stack_col": sc, "height": height, "src": src,
                "opt": sorted(opt), "mirror_opt": sorted(mopt),
                "reflected_opt": sorted(reflected),
                "ground_truth_symmetric": agree,
            })
            flag = "OK  " if agree else "ASYM"
            print(f"  {flag} stack c{sc} h{height}: b->{sorted(opt)}  "
                  f"mirror->{sorted(mopt)}  reflected->{sorted(reflected)}"
                  f"  [{src}/{msrc}]")
    n_asym = sum(1 for r in sym_rows if not r["ground_truth_symmetric"])
    print(f"\n  ground truth symmetric : {len(sym_rows)-n_asym}/{len(sym_rows)}")
    print(f"  ground truth ASYMMETRIC: {n_asym}")

    # ---- Q3: mirror-paired MODEL behaviour ----
    print("\n=== Q3: is the MODEL mirror-consistent on the same pairs? ===")
    model_rows = []
    for sc in (0, 1, 3, 5, 6):
        for height in (1, 2, 3, 4, 5):
            b, moves = build(sc, height)
            if b is None:
                continue
            mb = mirror_board(b)
            mmoves = [6 - c for c in moves]
            c1, _p1 = model_pick(b, moves)
            c2, _p2 = model_pick(mb, mmoves)
            consistent = (c2 == 6 - c1)
            model_rows.append({
                "stack_col": sc, "height": height,
                "choice_on_b": c1, "choice_on_mirror": c2,
                "mirror_consistent": consistent,
            })
            flag = "OK  " if consistent else "INCONSISTENT"
            print(f"  {flag} stack c{sc} h{height}: b->c{c1}  mirror->c{c2}"
                  f"  (expected c{6-c1})")
    n_inc = sum(1 for r in model_rows if not r["mirror_consistent"])
    print(f"\n  model mirror-consistent : {len(model_rows)-n_inc}/{len(model_rows)}")
    print(f"  model INCONSISTENT      : {n_inc}")

    out = {
        "per_stack": rows,
        "ground_truth_symmetry": sym_rows,
        "model_mirror_behaviour": model_rows,
        "ground_truth_asymmetric": n_asym,
        "model_mirror_inconsistent": n_inc,
    }
    (ROOT / args.out).write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
