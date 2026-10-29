#!/usr/bin/env python
"""Score the tie-set A/B arms on the UNIQUE-optimum subset.

Reuses the cached solver truth from the D14 curve run, so the comparison
is on the SAME 560 proven positions (112 unique-optimum) that the shipped
arm was measured on.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.agent import prompt_tokens, trainer_window  # noqa: E402
from src.c4.fmt import COL_BYTE_OFFSET  # noqa: E402

CACHE = ROOT / "runs/_probe/unique_optimum_truth_cache.json"


def load_truth():
    d = json.loads(CACHE.read_text())
    return d["truth"]


def score_arm(ckpt, truth):
    from src.c4.ui_server import _load_agent

    agent = _load_agent(str(ckpt), device="cpu")
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
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma-separated name=checkpoint-path")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    truth = load_truth()
    n = len(truth)
    opts = [set(t["opt"]) for t in truth]
    uniq = [i for i in range(n) if len(opts[i]) == 1]
    tied = [i for i in range(n) if len(opts[i]) > 1]
    print(f"truth: {n} proven, {len(uniq)} unique-optimum, {len(tied)} tied")

    def sub(picks, idx):
        return round(100 * sum(1 for i in idx if picks[i] in opts[i]) / len(idx), 1)

    def sub_t(picks, idx):
        return round(100 * sum(1 for i in idx
                               if picks[i] == truth[i]["teacher"]) / len(idx), 1)

    def const(col):
        return [col if col in t["legal"] else t["legal"][0] for t in truth]

    base_all = {f"always_c{c}": sub(const(c), range(n)) for c in range(7)}
    base_uni = {f"always_c{c}": sub(const(c), uniq) for c in range(7)}
    best_u = max(base_uni, key=base_uni.get)
    print(f"best constant on UNIQUE: {best_u} at {base_uni[best_u]}%")
    print()

    results = {}
    for spec in args.arms.split(","):
        name, path = spec.split("=", 1)
        ck = pathlib.Path(path)
        if not ck.exists():
            print(f"  {name}: MISSING {ck}")
            continue
        picks = score_arm(ck, truth)
        r = {
            "checkpoint": str(ck),
            "all_pct": sub(picks, range(n)),
            "unique_pct": sub(picks, uniq),
            "tied_pct": sub(picks, tied),
            "teacher_all_pct": sub_t(picks, range(n)),
            "teacher_unique_pct": sub_t(picks, uniq),
            "vs_best_unique_pp": round(sub(picks, uniq) - base_uni[best_u], 1),
            "col_dist": {str(k): v for k, v in
                         sorted(collections.Counter(picks).items())},
        }
        results[name] = r
        print(f"  {name:<18} ALL {r['all_pct']:>6}%  UNIQUE {r['unique_pct']:>6}%  "
              f"TIED {r['tied_pct']:>6}%  teacher(uniq) {r['teacher_unique_pct']:>6}%  "
              f"(vs best-unique {r['vs_best_unique_pp']:+.1f}pp)")

    out = {"n_proven": n, "n_unique": len(uniq), "n_tied": len(tied),
           "baselines_all": base_all, "baselines_unique": base_uni,
           "best_constant_unique": best_u, "arms": results}
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
