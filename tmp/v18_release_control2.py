"""Apples-to-apples control for the RELEASED step18000 public artifact.

Does the released model beat the best legal constant-column policy on a
real holdout? Uses the EXACT scoring protocol of tg_v18_holdout_control.py
(same holdout corpus + sidecar, same rng(12345) row sampling, same CTX=191
window, same legality mask via transcript_v2.decode_state, same tie rule:
argmax-equality, equally optimal move counts as correct).

The released artifact is a FLAT hard export (emb/coef/sbox/wiring/readout/
bias/lags), so it is re-imported into a soft V7 model via the project's own
GFMarkovLM.import_hard() and scored on the soft path.

torch.load(weights_only=False) is required: the payload holds plain Python
values (lists/ints) alongside tensors, and torch's safe unpickler cannot
rebuild the meta dict. This mirrors src/c4/soft_loader.py and train.py.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.c4.fmt import COL_BYTE_OFFSET, MOVE_OFFSET, PLY_LEN

CTX = 191


def legal_from_header(full_bytes):
    from src.c4.transcript_v2 import decode_state
    cols, _last, _side = decode_state(bytes(full_bytes))
    out = np.zeros(7, dtype=bool)
    for c, (h, _owners) in enumerate(cols):
        if h < 6:
            out[c] = True
    return out


def build_model(ckpt):
    from src.gf256_markov_lm import GFMarkovLM
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = raw.get("meta", {}) or {}
    model = GFMarkovLM(d=int(meta.get("d", 64)),
                       layers=int(meta.get("layers", 3)),
                       fan_in=int(meta.get("fan_in", 3)),
                       lags=tuple(int(x) for x in raw["lags"]),
                       seed=int(meta.get("seed", meta.get("model_seed", 0))),
                       n_coef=int(meta.get("n_coef", 16)))
    model.import_hard(raw)
    model.eval()
    return model, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release/artifacts/connect_four_gf256_best_hard_step18000.pt")
    ap.add_argument("--split", default="runs/c4/posdata_score_split")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--out", default="runs/_probe/release_v24_const_control.json")
    args = ap.parse_args()

    model, raw = build_model(args.ckpt)
    print("lags:", raw["lags"])
    print("meta:", {k: v for k, v in (raw.get("meta") or {}).items()
                     if isinstance(v, (int, float, str))})

    sp = Path(args.split)
    seqs = [l for l in (sp / "holdout_corpus.txt").read_bytes().split(b"\n") if l]
    T = np.load(sp / "holdout_targets.npz")["target"]
    assert len(seqs) == len(T), (len(seqs), len(T))
    best = T.max(axis=1, keepdims=True)
    optimal = np.isclose(T, best, rtol=0.0, atol=1e-6)

    n_total = len(seqs)
    idxs = list(range(n_total))
    if n_total > args.max_rows:
        rs = np.random.default_rng(12345)
        idxs = rs.choice(n_total, size=args.max_rows, replace=False).tolist()

    model_hit = tot = 0
    const_hit = [0] * 7
    const_legal = [0] * 7
    centre_hit = 0
    model_pick = [0] * 7

    with torch.no_grad():
        for i in idxs:
            seq = seqs[i]
            nply = len(seq) // PLY_LEN
            if nply < 1:
                continue
            ctx = min(CTX, len(seq) - 1)
            start = max(0, len(seq) - (ctx + 1))
            window = seq[start:start + ctx + 1]
            window = window + b"\n" * (ctx + 1 - len(window))
            arr = np.frombuffer(window, dtype=np.uint8)
            x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
            logits = model(x)[0]
            pos = len(seq) - 1 - start - 1
            if not (0 <= pos < ctx):
                continue
            lg = logits[pos][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].clone()
            hdr_at = pos + 1 - MOVE_OFFSET
            legal = (legal_from_header(x[0, hdr_at:hdr_at + 9])
                     if 0 <= hdr_at < ctx else np.ones(7, dtype=bool))
            lg_masked = lg.masked_fill(~torch.tensor(legal), float("-inf"))
            pick = int(lg_masked.argmax())
            opt = optimal[i]

            tot += 1
            model_hit += int(opt[pick])
            model_pick[pick] += 1
            for c in range(7):
                if legal[c]:
                    const_legal[c] += 1
                    if opt[c]:
                        const_hit[c] += 1
            legal_idx = [c for c in range(7) if legal[c]]
            if legal_idx:
                cpick = 3 if legal[3] else min(legal_idx, key=lambda c: (abs(c - 3), c))
                centre_hit += int(opt[cpick])

    def pct(a, b):
        return 100.0 * a / max(b, 1)

    best_c = max(range(7), key=lambda c: const_hit[c])
    res = {
        "ckpt": args.ckpt, "split": args.split, "lags": list(raw["lags"]),
        "rows_scored": tot,
        "model_optimal_pct": pct(model_hit, tot),
        "model_pick_dist": model_pick,
        "model_max_pick_share": pct(max(model_pick), tot),
        "constant_pct": {str(c): pct(const_hit[c], tot) for c in range(7)},
        "centre_else_nearest_pct": pct(centre_hit, tot),
        "best_constant_col": best_c,
        "best_constant_pct": pct(const_hit[best_c], tot),
        "model_minus_best_constant_pts": pct(model_hit, tot) - pct(const_hit[best_c], tot),
    }
    print(json.dumps(res, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
