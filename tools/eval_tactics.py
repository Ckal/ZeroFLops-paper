"""Score hard checkpoints on cached solver-labelled positions, by tactic class.

    .venv-c4\\Scripts\\python.exe tools\\eval_tactics.py <positions.pt> <ckpt> [<ckpt> ...]
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from readout_surgery import by_class, fmt_cls, legal_mask  # noqa: E402
from src.c4 import COLS  # noqa: E402
from src.c4.agent import COL_BYTE_OFFSET, prompt_tokens, trainer_window  # noqa: E402
from src.c4.duel import _load_hard_agent  # noqa: E402


@torch.no_grad()
def hard_cols(agent, positions, chunk=64):
    out = []
    for i in range(0, len(positions), chunk):
        xs, ri = [], []
        for moves, _ in positions[i:i + chunk]:
            x, r = trainer_window(prompt_tokens(moves, agent.version).tolist(),
                                  agent.ctx)
            xs.append(x)
            ri.append(r)
        lg = agent.model.hard_forward(torch.tensor(xs), agent.hard)
        out.append(lg[torch.arange(len(xs)), torch.tensor(ri),
                      COL_BYTE_OFFSET:COL_BYTE_OFFSET + COLS])
    return torch.cat(out)


def main():
    blob = torch.load(sys.argv[1], weights_only=False)
    pos = blob["ho"]
    legal, target, cls = legal_mask(pos)
    print(f"n={len(pos)}  class counts win1/block1/strict/tie: "
          f"{[int((cls == k).sum()) for k in range(4)]}")
    for ck in sys.argv[2:]:
        agent = _load_hard_agent(ck)
        lg = hard_cols(agent, pos).masked_fill(~legal, -1e9)
        pick = lg.argmax(1)
        hit = target[torch.arange(len(pick)), pick].float()
        print(f"{ck}\n   {fmt_cls(hit, cls)}", flush=True)


if __name__ == "__main__":
    main()
