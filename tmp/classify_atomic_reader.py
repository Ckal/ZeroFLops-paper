"""Characterise exactly what a concurrent reader observes on Windows.

Question: during an atomic temp+rename publication, can a reader ever
observe a CORRUPT (partially written) checkpoint, or only a TRANSIENT
error?

This distinction decides what the production UI must handle:
  - corrupt payload  -> the atomicity guarantee has FAILED (real bug)
  - transient error  -> the reader must retry (Windows share-mode
                        semantics, not a data problem)

The reader here distinguishes the two by opening the file first and only
then unpickling, and by classifying the exception.
"""
import os
import pathlib
import sys
import threading
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.c4.train import atomic_torch_save  # noqa: E402

TMPDIR = pathlib.Path(ROOT / "runs" / "_probe" /
                      f"_atomic_classify_{os.getpid()}")
TMPDIR.mkdir(parents=True, exist_ok=True)
target = TMPDIR / "ckpt.pt"

PAYLOAD = {"state": {"w": torch.zeros(4096, 4096)}, "meta": {"step": 2}}
atomic_torch_save(PAYLOAD, target)

stop = threading.Event()
stats = {"ok": 0, "permission": 0, "notfound": 0, "corrupt": 0, "other": 0}
corrupt_examples = []
first_errors = []


def classify_exc(e):
    """Separate 'could not open' from 'opened but bad content'."""
    name = type(e).__name__
    if name == "PermissionError":
        return "permission"
    if name == "FileNotFoundError":
        return "notfound"
    # An unpickling error means the file WAS opened but its content is
    # broken -> that is a genuine atomicity failure.
    return "corrupt"


def reader():
    while not stop.is_set():
        try:
            # Separate the OPEN from the UNPICKLE so we can tell a
            # transient lock apart from a corrupt payload.
            with open(target, "rb") as fh:
                data = fh.read()
            d = torch.load(io_bytes(data), map_location="cpu", weights_only=False)
            if "state" not in d:
                stats["corrupt"] += 1
                corrupt_examples.append("loaded but missing 'state'")
            else:
                stats["ok"] += 1
        except Exception as e:
            kind = classify_exc(e)
            stats[kind] += 1
            if len(first_errors) < 5:
                first_errors.append(f"{kind}: {type(e).__name__}: {str(e)[:140]}")


def io_bytes(data):
    import io
    return io.BytesIO(data)


t = threading.Thread(target=reader, daemon=True)
t.start()
time.sleep(0.1)

t0 = time.time()
for i in range(40):
    atomic_torch_save(PAYLOAD, target)
elapsed = time.time() - t0
time.sleep(0.1)
stop.set()
t.join(timeout=15)

print(f"writes            : 40 in {elapsed:.1f}s")
print(f"reader OK         : {stats['ok']}")
print(f"reader Permission : {stats['permission']}   (transient Windows lock)")
print(f"reader FileNotFnd : {stats['notfound']}   (transient swap window)")
print(f"reader CORRUPT    : {stats['corrupt']}   <-- the atomicity property")
if corrupt_examples:
    print(f"  examples: {corrupt_examples[:3]}")
if first_errors:
    print("first errors:")
    for e in first_errors:
        print(f"  {e}")

verdict = "ATOMICITY HOLDS" if stats["corrupt"] == 0 else "ATOMICITY VIOLATED"
print(f"\nVERDICT: {verdict}")

for f in TMPDIR.glob("*"):
    try:
        f.unlink()
    except OSError:
        pass
TMPDIR.rmdir()
