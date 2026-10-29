"""DECISIVE TEST: does the model copy a FALSE last-move byte?

Everything measured so far is consistent with two competing mechanisms:

  (M1) The model reads the BOARD and correctly decides the opponent's
       stack must be blocked or built on.  The "copy" is then a real
       tactical judgement, and the observed asymmetry (copying c6 but
       not c0) is explained by the position, not by the last-move byte.

  (M2) The model reads the LAST-MOVE byte and echoes it.  The observed
       asymmetry is then a property of the last-move encoding, not of
       the position.

These make OPPOSITE predictions on a position whose last-move byte lies.
If the model plays the column named by a byte that is factually wrong
about where the opponent just played, M2 holds and M1 fails.

METHOD
------
Build a real position.  Build the canonical token stream for it.  Then
build a SECOND stream in which only the "last" byte of the final header
is changed to a different column -- everything else identical, so the
BOARD the model sees is byte-identical.

    canonical : ... [c0..c6, last=REAL, turn]
    tampered  : ... [c0..c6, last=FAKE, turn]

If the model's choice changes to the FAKE column, it is reading the
last-move byte.  If it does not change, the board is what it reads.

The token stream is built the same way `prompt_tokens` builds it, so the
tampered stream differs from the canonical one in exactly one byte.  The
tampered stream is asserted to differ from the canonical in exactly one
position, and to be identical elsewhere.

Also run the mirror pair for the same positions, on the hard path.
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

from src.c4.board import Board                        # noqa: E402
from src.c4.fmt import PLY_LEN, MOVE_OFFSET           # noqa: E402
from src.c4.mirror import mirror_board                 # noqa: E402
from src.c4.ui_server import _load_agent               # noqa: E402
from src.c4 import fmt                                 # noqa: E402
from src.c4.agent import trainer_window                # noqa: E402


def build_stack(sc: int, height: int):
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


def stream_with_last(moves, fake_last: int | None, version="v2"):
    """The canonical prompt stream, optionally with the final header's
    'last' byte replaced by `fake_last`."""
    mod = fmt.get(version)
    raw = mod.encode_game(moves).replace(b"\n", b"")
    k = len(moves)
    b = Board()
    for c in moves:
        b.apply_move(c)
    last = moves[-1] if fake_last is None else fake_last
    hdr = list(mod.encode_state(b, last=last, side_to_move=b.turn))
    pre = bytearray(raw + bytes(hdr))
    assert len(pre) == k * PLY_LEN + (PLY_LEN - 1)
    return bytes(pre), b


def pick(agent, tokens, board, ctx):
    x, read_idx = trainer_window(list(tokens), ctx)
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
    ap.add_argument("--model",
                    default="runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt")
    ap.add_argument("--out", default="runs/_probe/false_last_move.json")
    args = ap.parse_args()

    agent = _load_agent(str(ROOT / args.model), device="cpu")
    print(f"model: {args.model}")
    print("path : HARD export\n")

    rows = []
    print("=== Does the model follow a FALSE last-move byte? ===")
    print(f"{'stack':>6} {'h':>2} | {'real last':>9} | {'fake':>4} | "
          f"{'pick(real)':>11} | {'pick(fake)':>11} | {'follows fake?':>13}")
    print("-" * 74)
    n_follow = 0
    n_changed = 0
    n_tot = 0
    for sc in (0, 1, 3, 5, 6):
        for height in (1, 2, 3, 4, 5):
            b, moves = build_stack(sc, height)
            if b is None:
                continue
            legal = b.valid_actions()
            real_last = moves[-1]
            # a fake last-move byte naming a DIFFERENT legal column
            fake = next((c for c in legal if c != real_last), None)
            if fake is None:
                continue
            canon, _ = stream_with_last(moves, None)
            tampered, _ = stream_with_last(moves, fake)
            diff = [i for i in range(len(canon)) if canon[i] != tampered[i]]
            assert len(diff) == 1, (
                f"tampered stream differs in {len(diff)} bytes, expected 1: "
                f"{diff}")
            assert diff[0] == len(canon) - 2, (
                f"the changed byte is at {diff[0]}, expected the LAST byte "
                f"of the final header ({len(canon) - 2})")
            p_real, _ = pick(agent, canon, b, agent.ctx)
            p_fake, _ = pick(agent, tampered, b, agent.ctx)
            n_tot += 1
            follows = (p_fake == fake)
            changed = (p_real != p_fake)
            n_follow += follows
            n_changed += changed
            rows.append({
                "stack_col": sc, "height": height, "real_last": real_last,
                "fake_last": fake, "pick_real": p_real, "pick_fake": p_fake,
                "follows_fake": bool(follows), "choice_changed": bool(changed),
            })
            print(f"c{sc:<5} {height:>2} | c{real_last:<8} | c{fake:<3} | "
                  f"c{p_real:<10} | c{p_fake:<10} | {str(follows):>13}")
            if sc == 6 and height == 1:
                pass

    print(f"\n  positions tested          : {n_tot}")
    print(f"  choice CHANGED with the fake byte : {n_changed} "
          f"({100.0*n_changed/max(1,n_tot):.1f}%)")
    print(f"  model followed the FAKE byte      : {n_follow} "
          f"({100.0*n_follow/max(1,n_tot):.1f}%)")

    # ---- mirror pair on the hard path, canonical reflection ----
    print("\n=== Mirror pair (hard path, canonical mirror_board) ===")
    mrows = []
    for sc in (0, 1, 3, 5, 6):
        for height in (1, 2, 3, 4, 5):
            b, moves = build_stack(sc, height)
            if b is None:
                continue
            mb = mirror_board(b)
            mmoves = [6 - c for c in moves]
            tok_a, _ = stream_with_last(moves, None)
            tok_b, _ = stream_with_last(mmoves, None)
            pa, _ = pick(agent, tok_a, b, agent.ctx)
            pb, _ = pick(agent, tok_b, mb, agent.ctx)
            mrows.append({
                "stack_col": sc, "height": height,
                "pick_orig": pa, "pick_mirror": pb,
                "mirror_consistent": (pb == 6 - pa),
            })
            print(f"  stack c{sc} h{height}: orig->c{pa}  mirror->c{pb}  "
                  f"(expected c{6-pa})")
    n_inc = sum(1 for r in mrows if not r["mirror_consistent"])
    print(f"\n  model mirror-consistent : {len(mrows)-n_inc}/{len(mrows)}")
    print(f"  model INCONSISTENT      : {n_inc}")

    out = {
        "false_last_move": rows,
        "mirror_pairs": mrows,
        "positions_tested": n_tot,
        "choice_changed": n_changed,
        "followed_fake": n_follow,
        "model_mirror_inconsistent": n_inc,
    }
    (ROOT / args.out).write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
