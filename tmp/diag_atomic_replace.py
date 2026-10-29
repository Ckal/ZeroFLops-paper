"""Diagnose the Windows atomic-rename failure modes.

Two distinct failures were observed:
  1. PermissionError (WinError 5)  -- destination open by a reader.
  2. FileNotFoundError (WinError 2) -- the SOURCE temp file vanished.

This script reproduces both and reports the exact sequence, so the fix is
based on the observed cause rather than on a guess.
"""
import os
import pathlib
import threading
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMPDIR = ROOT / "runs" / "_probe" / "_atomic_diag"
TMPDIR.mkdir(parents=True, exist_ok=True)
target = TMPDIR / "ckpt.pt"

PAYLOAD = {"state": {"w": torch.randn(4096, 4096)}, "meta": {"step": 2}}


def atomic_v1(obj, path):
    """The OLD implementation: fixed 100 x 0.02 s retry."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)
    return path


def atomic_v2(obj, path, budget=30.0):
    """The NEW implementation: deadline + jitter, dedicated RNG."""
    import random
    rng = random.Random(0xA70)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    deadline = time.monotonic() + budget
    while True:
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.005 + rng.random() * 0.04)


def run(atomic, label):
    for f in TMPDIR.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    atomic(PAYLOAD, target)

    stop = threading.Event()
    errors = []
    reads = [0]
    src_missing = [0]

    def reader():
        while not stop.is_set():
            try:
                d = torch.load(target, map_location="cpu", weights_only=False)
                if "state" not in d:
                    errors.append("missing state")
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
            reads[0] += 1

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.05)

    outcome = "ok"
    detail = ""
    t0 = time.time()
    try:
        for i in range(40):
            atomic(PAYLOAD, target)
    except Exception as e:
        outcome = type(e).__name__
        detail = str(e)[:300]
    elapsed = time.time() - t0
    time.sleep(0.05)
    stop.set()
    t.join(timeout=10)

    print(f"--- {label} ---")
    print(f"  outcome      : {outcome} {detail}")
    print(f"  elapsed      : {elapsed:.1f} s for 40 writes")
    print(f"  reader reads : {reads[0]}")
    print(f"  reader errors: {len(errors)}  first: {errors[:1]}")
    leftover = sorted(p.name for p in TMPDIR.glob('*'))
    print(f"  leftover     : {leftover}")
    print()


run(atomic_v1, "OLD: fixed 100 x 0.02 s retry")
run(atomic_v2, "NEW: 30 s deadline + jitter")

# Now: does the tmp file actually disappear?  Watch it directly.
print("=== direct observation: is the tmp file present at replace time? ===")
for f in TMPDIR.glob("*"):
    try:
        f.unlink()
    except OSError:
        pass
tmp = TMPDIR / "watch.tmp"
torch.save(PAYLOAD, tmp)
print(f"  after torch.save: tmp exists = {tmp.exists()}  size = {tmp.stat().st_size}")

stop = threading.Event()
seen = []


def reader2():
    while not stop.is_set():
        try:
            torch.load(target, map_location="cpu", weights_only=False)
        except Exception:
            pass


t = threading.Thread(target=reader2, daemon=True)
t.start()
time.sleep(0.05)
for i in range(5):
    torch.save(PAYLOAD, tmp)
    present = tmp.exists()
    try:
        os.replace(tmp, target)
        r = "replace ok"
    except OSError as e:
        r = f"{type(e).__name__}"
    seen.append((present, r))
    if r != "replace ok":
        # after a failure, is the source still there?
        seen.append(("src_exists_after_fail", tmp.exists()))
        break
stop.set()
t.join(timeout=5)
print(f"  observed: {seen}")
for f in TMPDIR.glob("*"):
    try:
        f.unlink()
    except OSError:
        pass
TMPDIR.rmdir()
