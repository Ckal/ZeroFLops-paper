"""AUTHORITATIVE: false-last-move causality + mirror consistency on the
real 1500-position holdout.

The synthetic single-stack probe (n=25) was too narrow: the holdout
numbers say mirror training HELPED (16.4% -> 36.7% mirror consistency),
while the synthetic probe said it did not. The holdout is the population
that matters, so run the same two measurements there.

MEASUREMENT 1 -- FALSE LAST-MOVE BYTE
    Build the canonical token stream for a real holdout position.
    Build a second stream where ONLY the final header's "last" byte names
    a different legal column.  Assert the two streams differ in exactly
    one byte and that byte is the last-move field.
    If the model's choice changes to the FAKE column, the model reads the
    last-move byte.  If it does not change, the board dominates.

MEASUREMENT 2 -- MIRROR CONSISTENCY
    For each holdout position, evaluate the mirrored position and check
    that the model's choice is the reflection of the original choice.
    This is the population-level mirror number the release gate uses.

Both run on the HARD export (the path that actually plays).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.board import Board                          # noqa: E402
from src.c4.fmt import PLY_LEN, MOVE_OFFSET             # noqa: E402
from src.c4.mirror import mirror_board                   # noqa: E402
from src.c4.ui_server import _load_agent                 # noqa: E402
from src.c4 import fmt                                   # noqa: E402
from src.c4.agent import trainer_window                  # noqa: E402


def decode_moves(line: str):
    """The transcript line -> the move list (columns, 0-6)."""
    from src.c4.transcript_v2 import decode_game
    return decode_game(line)


def stream(moves, fake_last=None, version="v2"):
    """Canonical prompt stream; optionally the final header's last byte
    is replaced by `fake_last`."""
    mod = fmt.get(version)
    raw = mod.encode_game(moves).replace(b"\n", b"")
    b = Board()
    for c in moves:
        b.apply_move(c)
    last = moves[-1] if fake_last is None else fake_last
    hdr = list(mod.encode_state(b, last=last, side_to_move=b.turn))
    pre = bytearray(raw + bytes(hdr))
    assert len(pre) == len(moves) * PLY_LEN + (PLY_LEN - 1)
    return bytes(pre), b


def pick(agent, tokens, board):
    x, read_idx = trainer_window(list(tokens), agent.ctx)
    with torch.no_grad():
        lg = agent.model.hard_forward(
            torch.tensor([x], dtype=torch.long), agent.hard)[0, read_idx]
    sub = lg[0x41:0x41 + 7].float()
    mask = agent.legal_mask(board).to(sub.device)
    NEG = torch.tensor(-1e9, dtype=sub.dtype, device=sub.device)
    p = torch.softmax(torch.where(mask, sub, NEG), dim=-1).cpu().numpy()
    return int(p.argmax()), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="runs/_probe/ab1500/corpus.txt")
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agent = _load_agent(str(ROOT / args.model), device="cpu")
    print(f"model : {args.model}")
    print(f"corpus: {args.corpus}")
    print("path  : HARD export\n")

    raw = (ROOT / args.corpus).read_bytes()
    lines = [ln for ln in raw.split(b"\n") if ln]
    lines = lines[: args.max_rows]

    # ---- MEASUREMENT 1: false last-move byte ----
    print("=== MEASUREMENT 1: false last-move byte ===")
    print(f"{'row':>5} {'ply':>3} | {'real':>4} | {'fake':>4} | "
          f"{'pick(real)':>10} | {'pick(fake)':>10} | {'followed':>8}")
    print("-" * 66)
    n_t = n_chg = n_follow = 0
    m1 = []
    for r, line in enumerate(lines):
        try:
            moves = decode_moves(line)
        except Exception:
            continue
        if not moves:
            continue
        try:
            canon, b = stream(moves, None)
        except Exception:
            continue
        legal = b.valid_actions()
        real_last = moves[-1]
        fake = next((c for c in legal if c != real_last), None)
        if fake is None:
            continue
        try:
            tampered, _ = stream(moves, fake)
        except Exception:
            continue
        diff = [i for i in range(len(canon)) if canon[i] != tampered[i]]
        if len(diff) != 1 or diff[0] != len(canon) - 2:
            continue
        p_real, _ = pick(agent, canon, b)
        p_fake, _ = pick(agent, tampered, b)
        n_t += 1
        chg = p_real != p_fake
        fol = (p_fake == fake)
        n_chg += chg
        n_follow += fol
        m1.append({"row": r, "ply": len(moves), "real_last": real_last,
                   "fake_last": fake, "pick_real": p_real, "pick_fake": p_fake,
                   "choice_changed": bool(chg), "followed_fake": bool(fol)})
        if n_t <= 25:
            print(f"{r:>5} {len(moves):>3} | c{real_last:<3} | c{fake:<3} | "
                  f"c{p_real:<8} | c{p_fake:<8} | {str(fol):>8}")

    print(f"\n  positions tested                     : {n_t}")
    print(f"  choice CHANGED when the last byte lied: {n_chg} "
          f"({100.0*n_chg/max(1,n_t):.2f}%)")
    print(f"  model FOLLOWED the false byte         : {n_follow} "
          f"({100.0*n_follow/max(1,n_t):.2f}%)")

    # ---- MEASUREMENT 2: mirror consistency ----
    print("\n=== MEASUREMENT 2: mirror consistency (hard path) ===")
    m2 = []
    n_inc = 0
    for r, line in enumerate(lines):
        try:
            moves = decode_moves(line)
        except Exception:
            continue
        if not moves:
            continue
        try:
            tok_a, b = stream(moves, None)
            mb = mirror_board(b)
            tok_b, _ = stream([6 - c for c in moves], None)
        except Exception:
            continue
        pa, _ = pick(agent, tok_a, b)
        pb, _ = pick(agent, tok_b, mb)
        inc = (pb != 6 - pa)
        n_inc += inc
        m2.append({"row": r, "ply": len(moves), "pick_orig": pa,
                   "pick_mirror": pb, "mirror_consistent": not inc})
        if n_t and len(m2) <= 25:
            print(f"  row {r:>5} ply {len(moves):>3}: orig->c{pa}  "
                  f"mirror->c{pb}  (expected c{6-pa})  "
                  f"{'INCONSISTENT' if inc else 'ok'}")

    print(f"\n  positions tested             : {len(m2)}")
    print(f"  mirror-consistent            : {len(m2)-n_inc} "
          f"({100.0*(len(m2)-n_inc)/max(1,len(m2)):.2f}%)")
    print(f"  mirror-INCONSISTENT          : {n_inc} "
          f"({100.0*n_inc/max(1,len(m2)):.2f}%)")

    out = {"model": args.model, "corpus": args.corpus,
           "false_last_move": m1,
           "false_last_move_positions": n_t,
           "false_last_move_choice_changed": n_chg,
           "false_last_move_followed": n_follow,
           "mirror_pairs": m2,
           "mirror_inconsistent": n_inc}
    (ROOT / args.out).write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
