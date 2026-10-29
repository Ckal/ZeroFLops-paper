"""Merge sharded sidecar repairs back into one file.

Each shard writes a FULL-SIZE copy of the arrays with only its own row
range modified. Merging therefore means: for each shard, take its rows for
its own range verbatim (they are a superset of the original for that range),
and keep the original everywhere else.

Verification is done here, not just in the shards:
  - every pre-existing label must be byte-identical to the original;
  - the merged file must contain strictly more labels;
  - no slot may be labelled at a position that the original left as a hole
    AND that is still a hole in the merge (i.e. nothing lost).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    tmp = pathlib.Path(args.shards_dir)
    orig = np.load(args.orig)
    target = orig["target"].copy()
    ply_of = orig["ply_of"].copy()
    teacher = orig["teacher"].copy()
    n_plies = orig["n_plies"].copy()

    pre = ply_of >= 0
    orig_labels = int(pre.sum())

    reports = []
    for rp in sorted(tmp.glob("report_*.json")):
        reports.append(json.loads(rp.read_text()))
    reports.sort(key=lambda r: r["row_offset"])

    added = 0
    for r in reports:
        off = r["row_offset"]
        n = r["rows_processed"]
        sp = np.load(tmp / f"shard_{r['row_offset'] // max(1, r['rows_processed'])}.npz") \
            if False else None
        # shard file name is shard_<s>.npz where s is the shard index; find it
        # by matching the report's row range among the files present.
        cand = sorted(tmp.glob("shard_*.npz"))
        shard = None
        for c in cand:
            d = np.load(c)
            if d["ply_of"][off:off + n].shape[0] == n:
                # accept the shard whose filled count in this range matches
                new_in_range = int(((d["ply_of"][off:off + n] >= 0)
                                    & (~pre[off:off + n])).sum())
                if new_in_range == r["slots_filled"]:
                    shard = d
                    break
        if shard is None:
            print(f"FATAL: no shard file matches report offset {off}")
            return 1
        sl = slice(off, off + n)
        added += int(((shard["ply_of"][sl] >= 0) & (~pre[sl])).sum())
        # take the shard's rows for its own range
        target[sl] = shard["target"][sl]
        ply_of[sl] = shard["ply_of"][sl]
        teacher[sl] = shard["teacher"][sl]
        n_plies[sl] = shard["n_plies"][sl]

    # --- verification ----------------------------------------------------
    changed = int((np.abs(target[pre] - orig["target"][pre]) > 1e-6).any(axis=-1).sum())
    ply_changed = int((ply_of[pre] != orig["ply_of"][pre]).sum())
    after = int((ply_of >= 0).sum())
    lost = int((pre & (ply_of < 0)).sum())

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, target=target, ply_of=ply_of, teacher=teacher,
                        n_plies=n_plies)

    rep = {
        "orig": args.orig, "out": str(out),
        "shards": len(reports),
        "labels_before": orig_labels,
        "labels_after": after,
        "labels_added": after - orig_labels,
        "shard_reported_added": sum(r["slots_filled"] for r in reports),
        "pre_existing_labels_changed": changed,
        "pre_existing_ply_of_changed": ply_changed,
        "labels_lost": lost,
    }
    rep["VERDICT"] = ("MERGE_SOUND" if (changed == 0 and ply_changed == 0
                                        and lost == 0
                                        and after > orig_labels)
                      else "MERGE_PROBLEM")
    print(json.dumps(rep, indent=2))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(rep, indent=2),
                                             encoding="utf-8")
    print("MERGE_SIDECAR_DONE")
    return 0 if rep["VERDICT"] == "MERGE_SOUND" else 1


if __name__ == "__main__":
    sys.exit(main())
