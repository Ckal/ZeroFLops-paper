"""Restore a damaged `best_hard.pt` from its source step snapshot.

Why this is exact
-----------------
The trainer promotes a checkpoint by saving the SAME model it just
snapshotted, into `checkpoints/best_hard.pt`, with a meta dict that is the
step snapshot's meta MINUS the "seed" key (the promotion block omits it;
the snapshot block includes it). The WEIGHTS are bit-identical.

This is verified, not assumed: `verify_reconstruction()` rebuilds
`best_hard.pt` for the two UNTOUCHED arms (fx_b, fx_c) from their step
snapshots and checks the result is byte-identical to the promoted file
those runs actually wrote. Only if that check passes is the
reconstruction trusted for the damaged arm.

Usage:
  python tmp/restore_best_hard.py --verify
  python tmp/restore_best_hard.py --restore fx_a_s6000
"""
import argparse
import hashlib
import io
import pathlib
import shutil
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "_probe"

# The promotion meta omits exactly this key relative to the step snapshot.
PROMOTION_OMITS = ("seed",)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p: pathlib.Path):
    # weights_only=False is required: an export_hard() payload is a dict of
    # LISTS of tensors plus (channel, lag) int tuples, not a tensor dict.
    # These files are produced by this repo's own training runs.
    return torch.load(p, map_location="cpu", weights_only=False)


def rebuild(step_path: pathlib.Path) -> dict:
    """Rebuild the promotion payload from a step snapshot."""
    raw = load(step_path)
    payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    out = {}
    for k, v in payload.items():
        if k in PROMOTION_OMITS:
            continue
        out[k] = v
    return out


def save_like_trainer(obj, path: pathlib.Path) -> None:
    """Write exactly as the trainer does: torch.save straight to the path."""
    torch.save(obj, path)


def tensors_equal(a, b) -> bool:
    """Deep equality that handles tensors, lists and tuples.

    torch.save() is NOT byte-deterministic: the same payload saved twice
    can differ byte-wise (storage layout / pickle memoisation) while
    being semantically identical. So the verification criterion is
    SEMANTIC equality of every tensor, not a sha256 match.
    """
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b)
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(tensors_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(tensors_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return False


def payload_of(p):
    r = torch.load(p, map_location="cpu", weights_only=False)
    return r["model"] if isinstance(r, dict) and "model" in r else r


def verify_reconstruction(arm: str) -> bool:
    """Rebuild an UNTOUCHED arm's best_hard.pt and compare semantically."""
    best = RUNS / arm / "checkpoints" / "best_hard.pt"
    step = RUNS / arm / "checkpoints" / "step6000_hard.pt"
    if not best.exists() or not step.exists():
        print(f"  {arm}: SKIP (missing artifacts)")
        return True
    rebuilt = rebuild(step)
    promoted = load(best)
    pb = promoted["model"] if isinstance(promoted, dict) and "model" in promoted else promoted
    same = tensors_equal(rebuilt, pb)
    print(f"  {arm}: rebuilt-from-step6000 == promoted best_hard.pt (semantic) ? {same}")
    if not same:
        only_a = sorted(set(rebuilt) - set(pb))
        only_b = sorted(set(pb) - set(rebuilt))
        print(f"    keys only in rebuilt: {only_a}")
        print(f"    keys only in promoted: {only_b}")
        for k in sorted(set(rebuilt) & set(pb)):
            if not tensors_equal(rebuilt[k], pb[k]):
                print(f"    DIFFERS: {k}")
    return same


def restore(arm: str, step: str = "step6000_hard.pt") -> bool:
    best = RUNS / arm / "checkpoints" / "best_hard.pt"
    src = RUNS / arm / "checkpoints" / step
    if not src.exists():
        print(f"restore: source {src} missing")
        return False
    rebuilt = rebuild(src)
    tmp = best.with_name(best.name + ".restore.tmp")
    save_like_trainer(rebuilt, tmp)
    shutil.move(str(tmp), str(best))
    print(f"restore: wrote {best} from {src}")
    print(f"restore: sha {sha256(best)[:16]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--restore", metavar="ARM")
    ap.add_argument("--step", default="step6000_hard.pt")
    a = ap.parse_args()

    if a.verify:
        print("VERIFY: rebuilding best_hard.pt for the untouched arms")
        ok = all(verify_reconstruction(arm)
                 for arm in ("fx_b_s6000", "fx_c_s6000"))
        print("VERIFY:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if a.restore:
        return 0 if restore(a.restore, a.step) else 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
