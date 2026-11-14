"""Compare hard Connect Four models on training-board-disjoint tactical positions."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from eval_tactics import hard_cols
from readout_surgery import legal_mask
from src.c4.board import Board
from src.c4.duel import _load_hard_agent
from src.c4.strong_solver import _imminent_win_columns
from src.c4.transcript_v2 import encode_state


def key(columns: bytes, turn: int) -> bytes:
    return min(columns, columns[::-1]) + bytes([turn])


def position_key(moves):
    board = Board()
    for move in moves:
        board.apply_move(move)
    state = encode_state(board, None, board.turn)
    return key(state[:7], state[8]), board


def checked_wins(board, side):
    reported = set(_imminent_win_columns(board, side))
    actual = set()
    for column in board.valid_actions():
        after = deepcopy(board)
        after.turn = side
        after.apply_move(column)
        if after.is_win() == side:
            actual.add(column)
    if reported != actual:
        raise ValueError(f"Tactical detector disagrees with board rules: {reported} != {actual}")
    return sorted(actual)


def training_overlap(path: Path, candidates: set[bytes], expected_rows: int):
    found = set()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            rows += 1
            line = line.rstrip(b"\n")
            if len(line) % 10:
                raise ValueError(f"{path}: malformed row {rows}, length {len(line)}")
            for offset in range(0, len(line), 10):
                header = line[offset:offset + 9]
                if len(header) != 9 or any(not (b & 0x80) for b in header[:7]):
                    raise ValueError(f"{path}: invalid v2 board at row {rows}")
                candidate = key(header[:7], header[8])
                if candidate in candidates:
                    found.add(candidate)
    if rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {rows}")
    return found


def score(picks, examples):
    result = {}
    for group in ("win", "block", "quiet"):
        subset = [(pick, ex) for pick, ex in zip(picks, examples) if ex["group"] == group]
        if not subset:
            result[group] = {"n": 0}
            continue
        if group == "quiet":
            hits = sum(pick in ex["teacher"] for pick, ex in subset)
            measure = "teacher_agreement"
        else:
            hits = sum(pick in ex["correct"] for pick, ex in subset)
            measure = "exact_tactical_success"
        result[group] = {"n": len(subset), "hits": hits,
                         "percent": round(100 * hits / len(subset), 2),
                         "measure": measure}
    return result


def main():
    torch.set_num_threads(4)
    data = torch.load(ROOT / "runs/_probe/v24_surgery/positions.pt",
                      map_location="cpu", weights_only=False)
    unique = {}
    for moves, target in data["ho"]:
        board_key, board = position_key(moves)
        if board_key not in unique:
            unique[board_key] = (moves, target, board)
    print(f"holdout rows={len(data['ho'])}, unique mirror-canonical boards={len(unique)}",
          flush=True)
    candidates = set(unique)
    overlaps = {}
    for name, count in (("v24", 320000), ("v25b", 640000)):
        overlaps[name] = training_overlap(ROOT / "runs/_probe" / name / "corpus.txt",
                                          candidates, count)
        print(f"{name}: scanned {count} rows; overlapping boards={len(overlaps[name])}",
              flush=True)
    unseen = candidates - overlaps["v24"] - overlaps["v25b"]
    positions = []
    examples = []
    for board_key, (moves, target, board) in unique.items():
        if board_key not in unseen:
            continue
        win = checked_wins(board, board.turn)
        opponent = 2 if board.turn == 1 else 1
        threats = checked_wins(board, opponent)
        if win:
            group, correct = "win", win
        elif len(threats) == 1:
            group, correct = "block", threats
        elif threats:
            continue
        else:
            group, correct = "quiet", []
        positions.append((moves, target))
        examples.append({"group": group, "correct": correct,
                         "teacher": target, "legal": board.valid_actions()})
    if not positions:
        raise ValueError("No unseen positions remain after filtering")
    legal, _, _ = legal_mask(positions)
    results = {
        "holdout_rows": len(data["ho"]),
        "unique_mirror_canonical_boards": len(unique),
        "v24_training_overlaps": len(overlaps["v24"]),
        "v25b_training_overlaps": len(overlaps["v25b"]),
        "common_unseen_before_excluding_multiple_threats": len(unseen),
        "evaluated_positions": len(positions),
        "notes": "Existing surgery holdout, board- and mirror-disjoint from both "
                 "training corpora; not a fresh untouched test set. Quiet uses "
                 "time-limited teacher labels, not exact solver proof.",
        "scores": {},
    }
    centre_order = (3, 2, 4, 1, 5, 0, 6)
    centre = [next(c for c in centre_order if c in ex["legal"]) for ex in examples]
    results["scores"]["centre"] = score(centre, examples)
    for name in ("v24", "v25b"):
        path = ROOT / "runs/_probe" / name / "checkpoints/best_hard.pt"
        agent = _load_hard_agent(str(path))
        logits = hard_cols(agent, positions).masked_fill(~legal, -1e9)
        results["scores"][name] = score(logits.argmax(1).tolist(), examples)
        print(f"{name}: {results['scores'][name]}", flush=True)
    for group in ("win", "block", "quiet"):
        subset = [ex for ex in examples if ex["group"] == group]
        if subset:
            successes = [len(set(ex["correct"] if group != "quiet" else ex["teacher"])
                             & set(ex["legal"])) / len(ex["legal"]) for ex in subset]
            results["scores"].setdefault("uniform_random", {})[group] = {
                "n": len(subset), "expected_percent": round(100 * sum(successes) / len(subset), 2)}
    out = ROOT / "runs/_probe/gf_generalization_tactics.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved: {out}", flush=True)


def test_mirrored_board_key():
    left, _ = position_key([0, 1, 0, 2, 4])
    right, _ = position_key([6, 5, 6, 4, 2])
    assert left == right


def test_training_overlap(tmp_path):
    columns = bytes([129] * 7)
    turn = ord("X")
    row = columns + bytes([0, turn, ord("A")]) + b"\n"
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(row)
    wanted = key(columns, turn)
    assert training_overlap(corpus, {wanted}, 1) == {wanted}


def test_tactical_wins():
    _, board = position_key([0, 1, 0, 1, 0, 2])
    assert checked_wins(board, board.turn) == [0]
    assert not checked_wins(board, 2)


if __name__ == "__main__":
    main()
