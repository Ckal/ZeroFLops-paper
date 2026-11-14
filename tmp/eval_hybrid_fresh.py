"""Paired, deterministic hybrid-vs-rule games from fresh, corpus-disjoint boards."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

import torch

from test_gf_generalization import position_key, training_overlap
from tools.test_hybrid_override import hybrid_policy
from src.c4.board import Board
from src.c4.duel import _load_hard_agent
from src.c4.opponents import heuristic_policy
from src.c4.strong_solver import _imminent_win_columns


def new_openings(seed, count, plies=12):
    rng = random.Random(seed)
    result = {}
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("Could not generate enough distinct quiet openings")
        board = Board()
        moves = []
        for _ in range(plies):
            board.apply_move(rng.choice(board.valid_actions()))
            moves.append(board.last_move)
            if board.is_win():
                break
        if len(moves) != plies or board.is_win() or not board.valid_actions():
            continue
        if any(_imminent_win_columns(board, side) for side in (1, 2)):
            continue
        key, _ = position_key(moves)
        result.setdefault(key, moves)
    return result


def play(opening, hybrid, hybrid_side):
    board = Board()
    moves = list(opening)
    for move in opening:
        board.apply_move(move)
        if board.is_win():
            raise ValueError("Opening contains a terminal position")
    while board.valid_actions():
        policy = hybrid if board.turn == hybrid_side else heuristic_policy
        col = policy(board, moves)
        if col not in board.valid_actions():
            raise ValueError(f"Illegal move {col}")
        board.apply_move(col)
        moves.append(col)
        winner = board.is_win()
        if winner:
            return (1 if winner == hybrid_side else -1), moves
    return 0, moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=int, default=200)
    parser.add_argument("--seed", type=int, default=872361)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/_probe/v24/checkpoints/best_hard.pt")
    args = parser.parse_args()
    if args.openings <= 0:
        parser.error("--openings must be positive")
    torch.set_num_threads(4)
    candidates = new_openings(args.seed, args.openings * 3)
    print(f"Generated {len(candidates)} distinct quiet opening boards", flush=True)
    known = torch.load(ROOT / "runs/_probe/v24_surgery/positions.pt",
                       map_location="cpu", weights_only=False)
    prior = {position_key(moves)[0] for split in ("tr", "ho")
             for moves, _ in known[split]}
    print(f"Prior surgery dataset board overlap: {len(set(candidates) & prior)}", flush=True)
    overlap = set(prior)
    for name, count in (("v24", 320000), ("v25b", 640000)):
        found = training_overlap(ROOT / "runs/_probe" / name / "corpus.txt",
                                 set(candidates), count)
        print(f"{name} training-board overlaps: {len(found)}", flush=True)
        overlap.update(found)
    selected = [moves for key, moves in candidates.items() if key not in overlap][:args.openings]
    if len(selected) != args.openings:
        raise RuntimeError(f"Only {len(selected)} fresh openings, need {args.openings}")
    agent = _load_hard_agent(str(args.checkpoint))
    hybrid = hybrid_policy(agent)
    games = []
    for index, opening in enumerate(selected):
        for side in (1, 2):
            outcome, moves = play(opening, hybrid, side)
            games.append({"opening": opening, "hybrid_side": side,
                          "outcome": outcome, "moves": moves})
        if (index + 1) % 25 == 0:
            print(f"played {index + 1}/{len(selected)} paired openings", flush=True)
    wins = sum(g["outcome"] == 1 for g in games)
    losses = sum(g["outcome"] == -1 for g in games)
    draws = len(games) - wins - losses
    pair_scores = [(games[2*i]["outcome"] + games[2*i+1]["outcome"] + 2) / 4
                   for i in range(len(selected))]
    bootstrap = random.Random(args.seed + 1)
    estimates = sorted(sum(bootstrap.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
                       for _ in range(10000))
    summary = {
        "seed": args.seed, "opening_plies": 12, "opening_pairs": len(selected),
        "games": len(games), "candidate_openings": len(candidates),
        "excluded_prior_or_training": len(candidates) - sum(k not in overlap for k in candidates),
        "model": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "win": wins, "loss": losses, "draw": draws,
        "score_percent": 100 * (wins + draws / 2) / len(games),
        "paired_bootstrap_95_pct": [100 * estimates[250], 100 * estimates[9750]],
        "as_p1": {str(r): sum(g["outcome"] == r and g["hybrid_side"] == 1 for g in games)
                  for r in (-1, 0, 1)},
        "as_p2": {str(r): sum(g["outcome"] == r and g["hybrid_side"] == 2 for g in games)
                  for r in (-1, 0, 1)},
        "distinct_continuations": len({tuple(g["moves"]) for g in games}),
        "notes": "Fresh random quiet openings; each board and mirror absent from both "
                 "training corpora and prior surgery dataset. Subsequent boards may "
                 "overlap. Both policies deterministic; no exploration after opening.",
    }
    out = ROOT / "runs/_probe/hybrid_fresh_games.json"
    out.write_text(json.dumps({"summary": summary, "games": games}, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved: {out}", flush=True)


def test_fresh_openings():
    first = new_openings(5, 10)
    assert first == new_openings(5, 10)
    assert len(first) == 10
    for moves in first.values():
        _, board = position_key(moves)
        assert len(moves) == 12 and not board.is_win()
        assert not any(_imminent_win_columns(board, side) for side in (1, 2))


def test_hybrid_tactics_and_quiet():
    class Agent:
        def choose(self, board, moves, use_hard):
            assert use_hard
            return 6

    hybrid = hybrid_policy(Agent())
    for moves in ([0, 1, 0, 1, 0, 2], [1, 0, 1, 0, 2, 0]):
        _, board = position_key(moves)
        assert hybrid(board, moves) == heuristic_policy(board, moves) == 0
    board = Board()
    assert hybrid(board, []) == 6
    assert heuristic_policy(board, []) == 3


def test_same_policy_pairs():
    opening = next(iter(new_openings(18, 1).values()))
    first, first_moves = play(opening, heuristic_policy, 1)
    second, second_moves = play(opening, heuristic_policy, 2)
    assert first_moves == second_moves
    assert first == -second


if __name__ == "__main__":
    main()
