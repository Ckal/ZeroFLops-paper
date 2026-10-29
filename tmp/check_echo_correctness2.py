"""Is the model's "echo" CORRECT PLAY, or a defect?

The echo diagnostic says the model answers column X with column X about
half the time against a one-column opponent, while the training labels
copy only 17% of the time. This script asks the question that actually
matters:

    in the positions where the model copies, is the copy CORRECT?

If the opponent has stacked 3+ pieces in column X, then X is a vertical
threat and blocking there is often right. If the opponent has stacked
1-2 pieces, copying is usually a wasted move.

Ground truth comes from the canonical exact solver (conclusive positions
only) and the canonical strong solver otherwise -- NEVER from the labels
and NEVER from the model. The source of each verdict is recorded, and a
position the solvers cannot prove is reported as unproved rather than
guessed.

The solver is asked for the SET of optimal moves, so a position with
several equally good moves is judged correctly.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board, P1, P2                                # noqa: E402
from src.c4.ui_server import _load_agent                            # noqa: E402
from src.c4.exact_solver import ExactSolver                         # noqa: E402
from src.c4 import strong_solver as STRONG                          # noqa: E402
from tg_solver_truth import optimal_moves                           # noqa: E402

STACK_COL = 6          # internal column 6 == user-visible column 7


def ground_truth(b, solver):
    """(set_of_optimal_columns | None, source). None = unproved."""
    legal = b.valid_actions()
    if not legal:
        return None, "terminal"
    side = b.turn
    status, cols = optimal_moves(solver, b, side)
    if status is not None and cols:
        return set(cols), "exact"
    # strong solver: one move, so the "optimal set" is just that move --
    # record the source so it is never presented as exact.
    STRONG.clear_tt()
    c = STRONG.strong_policy(b, [], depth=6, budget=0.05)
    if c is None or c not in legal:
        return None, "unproved"
    return {c}, "strong"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-col", type=int, default=STACK_COL)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stack_col = args.stack_col

    ckpt = ROOT / "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"
    agent = _load_agent(str(ckpt), device="cpu")
    solver = ExactSolver(max_empty=12, budget=2_000_000)
    print(f"model: {ckpt}")
    print(f"opponent stacks internal column {stack_col} "
          f"(user-visible {stack_col + 1})\n")

    rows = []
    for game in range(args.games):
        b = Board()
        moves = []
        while True:
            legal = b.valid_actions()
            if not legal:
                break
            # the opponent always stacks the chosen column
            opp = stack_col if stack_col in legal else int(
                np.random.default_rng(game).choice(legal))
            b.apply_move(opp)
            moves.append(opp)
            if b.is_win() != 0:
                break
            legal = b.valid_actions()
            if not legal:
                break
            # --- the model's decision ---
            height_before = b.heights[stack_col]
            lg = agent.hard_column_logits(b, list(moves))
            mask = agent.legal_mask(b).to(lg.device)
            NEG = torch.tensor(-1e9, dtype=lg.dtype, device=lg.device)
            p = torch.softmax(torch.where(mask, lg, NEG), dim=-1).cpu().numpy()
            choice = int(np.argmax(p))
            best, src = ground_truth(b, solver)
            rows.append({
                "game": game,
                "n_moves": len(moves),
                "stack_height_before": height_before,
                "model_choice": choice,
                "echoed": choice == stack_col,
                "optimal": sorted(best) if best else None,
                "src": src,
                "correct": (choice in best) if best else None,
                "p_choice": float(p[choice]),
                "p_stack": float(p[STACK_COL]),
            })
            b.apply_move(choice)
            moves.append(choice)
            if b.is_win() != 0:
                break

    n = len(rows)
    print("=== IS THE COPY CORRECT? (opponent stacks column 7) ===")
    print(f"  model decisions analysed : {n}")
    proved = [r for r in rows if r["src"] == "exact"]
    strong = [r for r in rows if r["src"] == "strong"]
    print(f"  proven exactly           : {len(proved)}")
    print(f"  strong-solver only       : {len(strong)}")
    print(f"  unproved                 : {n - len(proved) - len(strong)}")

    echo = [r for r in rows if r["echoed"]]
    print(f"\n  the model copied column 7: {len(echo)} "
          f"({100.0*len(echo)/max(1,n):.1f}%)")
    for label, grp in (("EXACT-PROVEN", proved), ("ALL PROVEN+STRONG", None)):
        if label == "EXACT-PROVEN":
            g = [r for r in echo if r["src"] == "exact"]
        else:
            g = [r for r in echo if r["src"] in ("exact", "strong")]
        if not g:
            continue
        c = sum(1 for r in g if r["correct"])
        print(f"  of those copies, CORRECT ({label}): {c}/{len(g)} "
              f"({100.0*c/len(g):.1f}%)")

    print("\n  copies by the opponent's stack height "
          "(height: copies, of which correct):")
    by_h = collections.Counter(r["stack_height_before"] for r in echo)
    by_h_c = collections.Counter(r["stack_height_before"] for r in echo
                                 if r["correct"])
    for h in sorted(by_h):
        n_h = by_h[h]
        c_h = by_h_c.get(h, 0)
        print(f"      height {h}: {n_h:>3} copies, {c_h:>3} correct "
              f"({100.0*c_h/max(1,n_h):.0f}%)")

    print("\n  split by threat level:")
    for name, pred in (("THREAT (stack>=3)", lambda r: r["stack_height_before"] >= 3),
                       ("NO THREAT (stack<3)", lambda r: r["stack_height_before"] < 3)):
        g = [r for r in rows if pred(r) and r["src"] in ("exact", "strong")]
        if not g:
            continue
        ec = sum(1 for r in g if r["echoed"])
        cr = sum(1 for r in g if r["correct"])
        ec_c = sum(1 for r in g if r["echoed"] and r["correct"])
        print(f"    {name}: {len(g)} decisions | echoed {ec} "
              f"({100.0*ec/len(g):.0f}%) | correct {cr} "
              f"({100.0*cr/len(g):.0f}%) | copies correct {ec_c}/{ec}")

    allc = sum(1 for r in rows if r["correct"] is True)
    judged = sum(1 for r in rows if r["correct"] is not None)
    print(f"\n  overall correctness (where judged): {allc}/{judged} "
          f"({100.0*allc/max(1,judged):.1f}%)")

    out_path = args.out or (ROOT / "runs/_probe/echo_correctness.json")
    (ROOT / out_path).write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
