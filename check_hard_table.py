"""Verify that runs/p_e8_tab_hard.pt is the table the published checkpoints need.

The ground truth for "does this table match this checkpoint" is the shape of
the checkpoint's memory-input weights. Each checkpoint was trained with a fixed
set of experts and a fixed number of evidence statistics, and those consume
exactly

    n_extra = 256 * len(experts) + n_stats

extra input dimensions next to the token embedding. So the check is: load the
table the way the training code does, and see whether the resulting expert
count and statistics count reproduce the checkpoint's `gpt.extra.weight`
shape.

IMPORTANT (learned 2026-11-17): the table file itself stores fewer experts
than the trained model uses. The extra ones are supplied at load time by
`HardModel.set_runtime(args)` -- the cache expert and its per-order
statistics come from the checkpoint's own args, not from the table. Comparing
the raw table's `names`/`n_stats` against the runtime model's therefore
reports a false mismatch. Always compare after set_runtime.

torch.load(weights_only=False) is intentional here: the checkpoints are local
files produced by this repo's own training script and they store an
argparse.Namespace of run arguments, which weights_only=True cannot unpickle.
"""
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_residual_lm import HardModel  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
HARD = os.path.join(REPO, "runs", "p_e8_tab_hard.pt")
CKPTS = ["runs/p_e8b_cache_best.pt", "runs/p_e8l_cache_best.pt"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== hard table file ===")
    print("  path : %s" % HARD)
    print("  size : %.1f MB" % (os.path.getsize(HARD) / 1048576))
    print("  sha256: %s" % sha256(HARD))

    print("\n=== load it (as stored) ===")
    hm = HardModel.load(HARD, "cpu")
    print("  experts : %s" % (getattr(hm, "names", None),))
    print("  main    : %s" % (getattr(hm, "main", None),))
    print("  n_stats : %s" % (getattr(hm, "n_stats", None),))
    table_hash = sha256(HARD)

    ok = True
    for rel in CKPTS:
        ck_path = os.path.join(REPO, rel)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        a = ck["args"]
        hp_name = ck.get("hard_path")
        table_path = os.path.join(os.path.dirname(os.path.abspath(ck_path)), hp_name)

        # The runtime model: the table plus what the checkpoint's args add.
        loaded = HardModel.load(table_path, "cpu").set_runtime(a)

        # The ground truth: how many memory-input dimensions the checkpoint
        # was actually trained with.
        extra = ck["model"]["gpt.extra.weight"]
        n_extra = extra.shape[1]
        expected = 256 * len(loaded.names) + loaded.n_stats

        print("\n=== %s ===" % os.path.basename(ck_path))
        print("  hard_path        : %s" % hp_name)
        print("  ckpt args        : cache=%s cache_orders=%s order=%s words=%s"
              % (a.get("cache"), a.get("cache_orders"), a.get("order"), a.get("words")))
        print("  runtime experts  : %s (main=%s)" % (loaded.names, loaded.main))
        print("  runtime n_stats  : %s" % loaded.n_stats)
        print("  table hash match : %s" % (sha256(table_path) == table_hash))
        print("  trained extra dim: %d" % n_extra)
        print("  expected extra dim: 256 * %d experts + %d stats = %d"
              % (len(loaded.names), loaded.n_stats, expected))

        match = (n_extra == expected)
        print("  -> table matches this checkpoint: %s" % match)
        ok = ok and match

    print("\n=== RESULT ===")
    print("  all published checkpoints accept the published table: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
