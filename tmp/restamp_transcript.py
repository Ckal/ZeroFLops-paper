"""Re-stamp existing checkpoints with the transcript version they were
trained on.

`src/c4/train.py`'s final-save path omitted `"transcript"` from the meta it
stamped onto `hard_model_final.pt` and `final_float.pt`.  `_load_agent`
treats a checkpoint with no transcript stamp as **v1 BY DEFINITION**, so a
v2 model served from that artifact reads a different byte alphabet: no
crash, just a model that plays badly and looks untrained.

This script repairs already-written artifacts in place (it does not retrain
anything).  It takes the version from the run's own `best_hard.pt` /
`best_soft.pt` sibling, which DOES carry the stamp, and copies it onto the
final exports.  If no sibling carries it, the script refuses -- guessing
the alphabet is exactly the failure it is repairing.

Usage:
  ./.venv-c4/Scripts/python.exe tmp/restamp_transcript.py <run_dir> [...]
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    # weights_only=False: the file is produced by this repo's own training
    # run, so it is trusted input.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def find_version(run_dir: pathlib.Path):
    """The version the run was trained on, from a sibling that records it."""
    for name in ("checkpoints/best_hard.pt", "checkpoints/best_soft.pt",
                 "checkpoints/step0_hard.pt", "checkpoints/step0_soft.pt"):
        p = run_dir / name
        if not p.exists():
            continue
        raw = load(p)
        meta = raw.get("meta") if isinstance(raw, dict) else None
        if not isinstance(meta, dict):
            meta = raw if isinstance(raw, dict) else {}
        ver = meta.get("transcript")
        if ver:
            return ver, name
    return None, None


def restamp(run_dir: pathlib.Path) -> dict:
    version, src = find_version(run_dir)
    out = {"run_dir": str(run_dir), "version": version, "version_source": src,
           "files": []}
    if not version:
        out["error"] = ("no sibling checkpoint records a transcript version; "
                        "refusing to guess the byte alphabet")
        return out

    for name in ("hard_model_final.pt", "final_float.pt"):
        p = run_dir / name
        if not p.exists():
            out["files"].append({"file": name, "status": "absent"})
            continue
        raw = load(p)
        before = None
        if isinstance(raw, dict):
            meta = raw.get("meta")
            if not isinstance(meta, dict):
                meta = raw
            before = meta.get("transcript")
            if before == version:
                out["files"].append({"file": name, "status": "already-correct",
                                     "transcript": before})
                continue
            meta["transcript"] = version
            # The float export also needs the architecture hyper-parameters,
            # otherwise the loader falls back to d32/L3/fan_in3 defaults and
            # silently mis-builds the model.
            for key, default in (("d", 32), ("layers", 3), ("fan_in", 3)):
                meta.setdefault(key, default)
        torch.save(raw, p)
        out["files"].append({"file": name, "status": "restamped",
                             "was": before, "now": version})
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = []
    for a in sys.argv[1:]:
        d = pathlib.Path(a)
        if not d.is_absolute():
            d = ROOT / a
        results.append(restamp(d))
        print(json.dumps(results[-1], indent=2))

    op = ROOT / "runs/_probe/restamp_transcript.json"
    op.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {op}")
    bad = [r for r in results if r.get("error")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
