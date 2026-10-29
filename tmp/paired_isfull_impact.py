"""PAIRED impact measurement of the is_full() bug on the strength matrix.

The problem: the pre-fix strength logs and the post-fix logs used
different seeds (999 vs 2468) and different harnesses, so their numbers
are NOT directly comparable. Any delta I report would be confounded by
seed and by sample.

This script removes the confound entirely. It runs the CANONICAL duel
harness TWICE over the SAME games with the SAME seed:

    A: terminal test = the HISTORICAL buggy one (bit 41 only)
    B: terminal test = the CANONICAL one (no legal action remains)

Same checkpoints, same opponents, same eps, same seat alternation, same
RNG stream, same move sequences up to the point where the buggy run
diverts into a phantom draw.

The ONLY difference is when the game is declared over. So the delta
between the two columns is attributable to the bug and nothing else.

To do this without editing the shared harness, the buggy run uses a
local copy of `play_one` that reproduces the historical semantics:
    while not b.is_full():   # the ONE-BIT test
Both variants share the same `duel()` driver, the same policy objects
and the same RNG, so the game sequences are identical until the
divergence point.

Usage:
    python -m tmp.paired_isfull_impact --games 300 --seed 2468
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import torch

from src.c4 import P1, P2
from src.c4.board import Board, COLS, is_terminal
from src.c4.duel import (_explore, _load_hard_agent, _model_policy,
                         duel, play_one)
from src.c4.opponents import get as get_opponent


def _buggy_is_full(mask: int) -> bool:
    """The historical one-bit fullness test (bit 41)."""
    return bool((mask >> 41) & 1)


def play_one_buggy(model_pol, opp_pol, model_is_p1: bool):
    """play_one with the HISTORICAL terminal test.

    Identical to src.c4.duel.play_one except for the loop condition.
    Returns (1/0/None, move_sequence).
    """
    b = Board()
    moves: List[int] = []
    while not _buggy_is_full(b.mask):
        model_side = P1 if model_is_p1 else P2
        pol = model_pol if b.turn == model_side else opp_pol
        col = pol(b, moves)
        if not (0 <= col < COLS) or b.heights[col] >= 6:
            raise ValueError(
                f"illegal move col={col} heights={b.heights} moves={moves}")
        b.apply_move(col)
        moves.append(col)
        winner = b.is_win()
        if winner:
            model_side = P1 if model_is_p1 else P2
            return (1 if winner == model_side else 0), tuple(moves)
    return None, tuple(moves)


def _duel_with_play(a_pol, b_pol, games: int, seed: int, play_fn):
    """Mirror of src.c4.duel.duel with a pluggable play function.

    The seat alternation, the RNG usage and the accounting are copied
    verbatim from the canonical implementation so the two runs differ
    ONLY in the terminal test.
    """
    res: Dict[str, Any] = {
        "games": games, "A_wins": 0, "B_wins": 0, "draws": 0,
        "A_as_P1": 0, "A_as_P2": 0, "B_as_P1": 0, "B_as_P2": 0,
    }
    seqs = set()
    plies = 0
    for i in range(games):
        model_is_p1 = (i % 2 == 0)
        r, seq = play_fn(a_pol, b_pol, model_is_p1)
        seqs.add(seq)
        plies += len(seq)
        if r == 1:
            res["A_wins"] += 1
            res["A_as_P1" if model_is_p1 else "A_as_P2"] += 1
        elif r == 0:
            res["B_wins"] += 1
            res["B_as_P1" if model_is_p1 else "B_as_P2"] += 1
        else:
            res["draws"] += 1
    res["unique_games"] = len(seqs)
    res["A_score_pct"] = 100.0 * (res["A_wins"] + 0.5 * res["draws"]) / max(1, games)
    res["mean_plies"] = plies / max(1, games)
    return res


def build_policies(ck_a: str, ck_b: Optional[str], opponent: str,
                   eps: float, seed: int):
    """Build the policy objects ONCE so both runs share the same wrappers.

    NOTE: the policies are stateful in the sense that the epsilon
    wrapper draws from an RNG. To keep the two runs comparable we build
    FRESH policy objects per run, seeded identically. The underlying
    policies (agent, opponent) are deterministic, so the same seed
    reproduces the same exploration sequence.
    """
    agent_a = _load_hard_agent(ck_a)
    agent_a.greedy = True
    agent_b = None
    if ck_b:
        agent_b = _load_hard_agent(ck_b)
        agent_b.greedy = True

    def make():
        rng_a = random.Random(seed)
        if ck_b:
            rng_b = random.Random(seed + 1)
            a_pol = _model_policy(agent_a, hard=True, eps=0.0, rng=rng_a)
            b_pol = _model_policy(agent_b, hard=True, eps=0.0, rng=rng_b)
        else:
            a_pol = _model_policy(agent_a, hard=True, eps=0.0, rng=rng_a)
            opp = get_opponent(opponent)
            b_pol = _explore(opp, eps, random.Random(seed + 1))
        return a_pol, b_pol
    return make


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2468)
    ap.add_argument("--out", default="runs/_probe/paired_isfull_impact.json")
    args = ap.parse_args()

    CK_FC = "runs/_probe/fc_a_s6000/checkpoints/step6000_hard.pt"
    CK_MIR = "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"
    CK_TAU = "runs/_probe/fc_a_tau0/checkpoints/step3000_hard.pt"

    matchups = []
    for arm in ("fc_a_s6000", "mir_a_s6000"):
        ck = CK_FC if arm == "fc_a_s6000" else CK_MIR
        for opp in ("centre", "heuristic", "lookahead"):
            matchups.append((f"{arm} vs {opp}", ck, None, opp))
    matchups.append(("mir_a_s6000 vs fc_a_s6000", CK_MIR, CK_FC, None))
    matchups.append(("mir_a_s6000 vs fc_a_tau0", CK_MIR, CK_TAU, None))
    matchups.append(("mir_a_s6000 vs mirrored copy", CK_MIR, CK_MIR, None))

    out = {"protocol": {
        "games": args.games, "eps": args.eps, "seed": args.seed,
        "note": "PAIRED: same seed, same policies, same seat alternation; "
                "ONLY the terminal test differs between the two columns.",
        "historical_buggy_test": "bool((mask >> 41) & 1)",
        "canonical_test": "is_win() != 0 or not valid_actions()",
    }, "matchups": {}}

    print(f"{'matchup':36s} {'BUGGY score':>12s} {'CANON score':>12s} "
          f"{'delta':>7s} | {'BUG uniq':>8s} {'CAN uniq':>8s} | "
          f"{'BUG drw':>7s} {'CAN drw':>7s}")
    print("-" * 128)

    for name, ck_a, ck_b, opp in matchups:
        make = build_policies(ck_a, ck_b, opp, args.eps, args.seed)
        a1, b1 = make()
        bug = _duel_with_play(a1, b1, args.games, args.seed, play_one_buggy)
        a2, b2 = make()
        can = _duel_with_play(a2, b2, args.games, args.seed, play_one)

        d = can["A_score_pct"] - bug["A_score_pct"]
        print(f"{name:36s} {bug['A_score_pct']:>11.1f}% {can['A_score_pct']:>11.1f}% "
              f"{d:>+7.1f} | {bug['unique_games']:>8d} {can['unique_games']:>8d} | "
              f"{bug['draws']:>7d} {can['draws']:>7d}")
        out["matchups"][name] = {
            "buggy": {k: bug[k] for k in
                      ("games", "A_wins", "B_wins", "draws", "A_as_P1",
                       "A_as_P2", "unique_games", "A_score_pct", "mean_plies")},
            "canonical": {k: can[k] for k in
                          ("games", "A_wins", "B_wins", "draws", "A_as_P1",
                           "A_as_P2", "unique_games", "A_score_pct", "mean_plies")},
            "delta_score_pct": d,
            "delta_unique_games": can["unique_games"] - bug["unique_games"],
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    print("PAIRED_ISFULL_IMPACT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
