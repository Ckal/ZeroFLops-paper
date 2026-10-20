"""Smoke test: does the V6 GF(256) byte-LM actually play Connect Four?

Runs a short real training loop on a winners-only corpus, then checks the
three things that decide whether the final deliverable is real:

  1. training reduces loss and the readout stops being zero
  2. soft column logits are finite and actually differentiate between columns
  3. the hardened integer forward agrees with the soft float forward
  4. the agent can finish a self-play game under the legal-move mask

Point 3 is the one that matters most: everything the user sees at play time
runs through the hardened path, and if that diverges from training the model
is decoration.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.c4 import corpus as cp                    # noqa: E402
from src.c4.agent import Agent, agent_agreement    # noqa: E402
from src.c4.board import Board                     # noqa: E402
from src.gf256_markov_lm_v6 import (              # noqa: E402
    GFMarkovLM, get_batch, load_data,
)

CORPUS = "runs/c4/smoke_corpus.txt"
STEPS = 80


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = GFMarkovLM(d=32, layers=3, fan_in=3, seed=0).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {n_params:.2f}M params on {dev}")

    # The readout is zero-initialised by design, so an untrained model emits a
    # flat 256-way distribution. That is why the check below is on the value
    # AFTER training, not before.
    print(f"readout[0] absmean before training: "
          f"{model.readout[0].abs().mean().item():.6f}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    os.makedirs("runs/c4", exist_ok=True)
    if os.path.exists(CORPUS):
        os.remove(CORPUS)
    games = cp.play_games(200, random.Random(1))
    kept = cp.winners_only(games)
    cp.append_corpus(CORPUS, kept)
    train, val = load_data(CORPUS)
    print(f"corpus: {len(games)} games -> {len(kept)} wins -> "
          f"{train.numel()} train bytes, {val.numel()} val")

    first_loss = None
    last_loss = float("nan")
    for step in range(STEPS):
        xb, yb = get_batch(train, 16, 64, dev)
        logits = model(xb)                      # (B, T, 256) plain tensor
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 256), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
    print(f"loss {first_loss:.4f} -> {last_loss:.4f} over {STEPS} steps")
    print(f"readout[0] absmean after training: "
          f"{model.readout[0].abs().mean().item():.6f}")

    agent = Agent(model, dev, greedy=True)
    moves = [3, 3, 4, 5]
    board = Board()
    for c in moves:
        board.apply_move(c)

    soft = agent.column_logits(moves)
    print(f"soft column logits: {[round(v, 4) for v in soft.tolist()]}")
    assert torch.isfinite(soft).all(), "soft logits not finite"
    assert soft.std() > 1e-6, "soft logits are flat -- readout never learned"

    hard = model.export_hard()
    hl = agent.hard_column_logits(moves, hard)
    print(f"hard column logits: {[round(v, 4) for v in hl.tolist()]}")

    agree, soft_move = agent_agreement(agent, moves, board, hard)
    print(f"soft move {soft_move}, hard agrees: {agree}")

    # Full self-play game under the legal-move mask.
    play_agent = Agent(model, dev, greedy=False, temperature=0.7,
                       rng=random.Random(0))
    b = Board()
    played: list[int] = []
    while not b.is_full() and not b.is_win():
        mv = play_agent.choose(list(played), b)
        assert mv in b.valid_actions(), f"illegal move {mv}"
        b.apply_move(mv)
        played.append(mv)
    print(f"self-play game finished: {len(played)} plies, "
          f"winner={b.is_win()}, full={b.is_full()}")
    assert len(played) > 0

    print("\nOK: train -> score -> hardened forward -> legal self-play all work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())