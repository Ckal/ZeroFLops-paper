"""Apples-to-apples: does the RELEASED step18000 model beat the constant-column baseline?

Uses the same scoring protocol as tg_v18_holdout_control.py / the trainer:
same holdout corpus + sidecar targets, same row sampling, same legality mask,
same tie rule. The released artifact has a FLAT key layout (bias/coef/emb/
readout/sbox/wiring, no 'state'/'meta' sub-dict), so it is rebuilt via the
project's own soft loader rather than load_state_dict.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release/artifacts/connect_four_gf256_best_hard_step18000.pt")
    ap.add_argument("--split", default="runs/c4/posdata_score_split")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--out", default="runs/_probe/release_v24_const_control.json")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print("artifact keys:", sorted(raw.keys()))
    print("lags:", raw.get("lags"), "step:", raw.get("step"),
          "transcript:", raw.get("transcript"))

    from src.c4.soft_loader import load_soft_model
    model = load_soft_model(args.ckpt)
    model.eval()

    sp = Path(args.split)
    corpus = (sp / "holdout_corpus.txt").read_bytes().split(b"\n")
    corpus = [c for c in corpus if c]
    z = np.load(sp / "holdout_targets.npz")
    print("split files:", sorted(p.name for p in sp.iterdir()))
    print("npz keys:", list(z.keys()))
    opt = z["target"]                      # (n, 19, 7) optimality scores
    print("opt shape:", opt.shape)

    rng = np.random.default_rng(12345)
    n = min(args.max_rows, len(corpus))
    idx = rng.choice(len(corpus), size=n, replace=False)

    picks = np.zeros(7, dtype=np.int64)
    model_hits = 0
    const_hits = np.zeros(7, dtype=np.int64)
    const_center = 0
    legal_any = 0
    rows_used = 0

    for i in idx:
        raw_row = corpus[i]
        ply = raw_row[0] if isinstance(raw_row[0], int) else raw_row[0]
        try:
            head = int(raw_row[0])
        except Exception:
            head = None
        # The corpus line format: the trainer slices the last ply's header.
        # Reconstruct exactly as the trainer does.
        try:
            nply = int(raw_row[0])
        except Exception:
            nply = None
        if nply is None or nply <= 0:
            continue
        # header = 9 bytes before the move byte of the final ply
        start = (nply - 1) * PLY_LEN
        hdr = raw_row[start + 0: start + 9]
        if len(hdr) < 9:
            continue
        legal = legal_from_header(hdr)
        if not legal.any():
            continue
        legal_any += 1
        rows_used += 1

        # optimal set for the scored ply
        t = opt[i] if opt.ndim == 3 else opt
        # target rows are per-ply; use the final ply
        prow = t[min(nply - 1, t.shape[1] - 1)]
        best = prow.max()
        oset = (prow >= best - 1e-6)

        # model pick
        seq = raw_row[-(CTX + 1):]
        if len(seq) < CTX + 1:
            seq = b"\n" * (CTX + 1 - len(seq)) + seq
        x = torch.from_numpy(np.frombuffer(seq, dtype=np.uint8).astype(np.int64))
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0] if callable(model) else None
        if logits is None:
            raise SystemExit("model forward did not return logits")
        lg = logits.float().numpy()
        lg = np.where(legal, lg, -1e9)
        pick = int(np.argmax(lg))
        picks[pick] += 1
        if oset[pick]:
            model_hits += 1

        for c in range(7):
            if legal[c] and oset[c]:
                const_hits[c] += 1
        for c in (2, 3, 4):
            if legal[c] and oset[c]:
                const_center += 1
                break

    print("rows used:", rows_used, "legal_any:", legal_any)
    print("model pick distribution:", picks.tolist())
    print("model pick max share: %.4f" % (picks.max() / max(1, rows_used)))
    print("model holdout optimal pct: %.4f" % (model_hits / max(1, rows_used)))
    for c in range(7):
        print("  always-col%d: %.4f" % (c, const_hits[c] / max(1, rows_used)))
    print("best constant: %.4f" % (const_hits.max() / max(1, rows_used)))
    print("center-3 (c2/3/4, lowest wins): %.4f" % (const_center / max(1, rows_used)))

    out = {
        "ckpt": args.ckpt, "split": args.split, "rows_used": rows_used,
        "model_pick_dist": picks.tolist(),
        "model_max_pick_share": float(picks.max() / max(1, rows_used)),
        "model_holdout_optimal_pct": float(model_hits / max(1, rows_used)),
        "const_col_holdout_optimal_pct": [float(v / max(1, rows_used)) for v in const_hits],
        "best_const_pct": float(const_hits.max() / max(1, rows_used)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[written]", args.out)


if __name__ == "__main__":
    main()
