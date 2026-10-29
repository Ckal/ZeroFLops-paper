"""Compare the promoted `best_hard.pt` with the step snapshot it came from.

Purpose: verify whether the promoted artifact is byte-equal to its source
step snapshot, so a damaged `best_hard.pt` can be restored exactly rather
than approximately.

Also used to verify the fx_a restore after an accidental overwrite.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def payload_of(p):
    r = torch.load(p, map_location="cpu", weights_only=False)
    return r["model"] if isinstance(r, dict) and "model" in r else r


def weight_signature(payload):
    """A comparable signature of every tensor in the payload."""
    sig = {}
    for k, v in payload.items():
        if torch.is_tensor(v):
            sig[k] = v
        elif isinstance(v, list):
            flat = []
            for x in v:
                if torch.is_tensor(x):
                    flat.append(x)
                elif isinstance(x, tuple):
                    flat.append(tuple(y for y in x if torch.is_tensor(y)))
                else:
                    flat.append(x)
            sig[k] = tuple(flat)
    return sig


def compare(a_path, b_path):
    pa, pb = payload_of(a_path), payload_of(b_path)
    sa, sb = weight_signature(pa), weight_signature(pb)
    diffs = []
    if set(sa) != set(sb):
        diffs.append(f"key sets differ: only_a={sorted(set(sa)-set(sb))} "
                     f"only_b={sorted(set(sb)-set(sa))}")
    for k in sorted(set(sa) & set(sb)):
        va, vb = sa[k], sb[k]
        if torch.is_tensor(va):
            if not torch.equal(va, vb):
                diffs.append(f"{k}: tensor differs")
        elif isinstance(va, tuple) and len(va) == len(vb):
            for i, (x, y) in enumerate(zip(va, vb)):
                if torch.is_tensor(x) and torch.is_tensor(y):
                    if not torch.equal(x, y):
                        diffs.append(f"{k}[{i}]: tensor differs")
                elif isinstance(x, tuple) and isinstance(y, tuple):
                    if len(x) != len(y):
                        diffs.append(f"{k}[{i}]: tuple len differs")
                    else:
                        for j, (xx, yy) in enumerate(zip(x, y)):
                            if torch.is_tensor(xx) and torch.is_tensor(yy):
                                if not torch.equal(xx, yy):
                                    diffs.append(f"{k}[{i}][{j}]: tensor differs")
                            elif xx != yy:
                                diffs.append(f"{k}[{i}][{j}]: {xx!r} != {yy!r}")
                elif x != y:
                    diffs.append(f"{k}[{i}]: {x!r} != {y!r}")
        elif va != vb:
            diffs.append(f"{k}: {va!r} != {vb!r}")
    meta_keys = [k for k in pa if k not in ("coef", "sbox", "wiring", "readout",
                                           "emb", "bias", "lags", "d", "layers",
                                           "fan_in", "coef_set")]
    return diffs, {k: pa.get(k) for k in meta_keys}


if __name__ == "__main__":
    for arm in ("fx_b_s6000", "fx_c_s6000"):
        best = ROOT / f"runs/_probe/{arm}/checkpoints/best_hard.pt"
        step = ROOT / f"runs/_probe/{arm}/checkpoints/step6000_hard.pt"
        diffs, meta = compare(best, step)
        print(f"{arm}: best_hard vs step6000")
        print(f"   weights: {'IDENTICAL' if not diffs else diffs[:4]}")
        print(f"   best_hard meta: {meta}")
        print(f"   step6000  meta: {compare(step, best)[1]}")
        print()
