"""Measure the SHIPPED per-ply label-agreement scorer on real checkpoints.

This is the independent cross-check of the promotion gate: the scorer lives
in `src/c4/train.py` and is used by the trainer's promotion key.  This script
loads checkpoints through the canonical loader and reports the scorer's
number, so the promotion gate's metric can be verified against the
independent `tmp/measure_label_agreement.py` measurement (91.29 % on the
corrected final checkpoint).

Usage:
    ./.venv-c4/Scripts/python.exe -u tmp/measure_promotion_scorer.py <run_dir> [...]
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.c4.train import PerPlyAgreement, _align_perply_targets  # noqa: E402
from src.c4.agent import hard_to_cpu  # noqa: E402


def load_corpus(run: pathlib.Path):
    raw = []
    with open(run / "corpus.txt", "rb") as fh:
        for ln in fh.read().splitlines():
            if ln:
                raw.append(ln)
    return raw


def main() -> int:
    runs = [pathlib.Path(a) for a in sys.argv[1:]]
    if not runs:
        print("usage: measure_promotion_scorer.py <run_dir> [...]")
        return 2

    failed = False
    for run in runs:
        corpus = run / "corpus.txt"
        sidecar = run / "perply_targets.npz"
        print(f"\n=== {run} ===")
        if not run.is_dir():
            # A checkpoint file was passed instead of a run directory. Say so
            # loudly and FAIL: a silent "SKIP" here looks like a passing
            # measurement of zero checkpoints, which is exactly the kind of
            # vacuous result this project forbids.
            print(f"  ERROR: {run} is not a directory.")
            print("         usage: measure_promotion_scorer.py <run_dir> [...]")
            print("         (a run dir must contain corpus.txt + perply_targets.npz)")
            failed = True
            continue
        if not corpus.exists() or not sidecar.exists():
            missing = [p.name for p in (corpus, sidecar) if not p.exists()]
            print(f"  ERROR: missing {missing} in {run}")
            failed = True
            continue
        raw_lines = load_corpus(run)

        class A:
            pass

        a = A()
        a.perply_targets = str(sidecar)
        a.transcript = "v2"
        a.max_legacy_frac = 0.0
        a.corpus_source = "teacher"
        pt, po, nply = _align_perply_targets(a, raw_lines, "v2")
        if pt is None:
            print("  SKIP: sidecar not aligned")
            continue
        print(f"  corpus rows: {len(raw_lines)}  sidecar: {tuple(pt.shape)}")

        ckpts = []
        # The PROMOTED artifact is `checkpoints/best_hard.pt`. It must be
        # measured explicitly: the `step*_hard.pt` glob does NOT match it
        # (fnmatch("best_hard.pt","step*_hard.pt") is False), so a
        # run-root-only lookup silently never measured the promoted model.
        for p in (run / "hard_model_final.pt",
                  run / "checkpoints" / "best_hard.pt",
                  run / "best_hard.pt"):
            if p.exists() and p not in ckpts:
                ckpts.append(p)
        if (run / "checkpoints").exists():
            for p in sorted((run / "checkpoints").glob("step*_hard.pt")):
                if p not in ckpts:
                    ckpts.append(p)

        for ck in ckpts:
            try:
                # Load the HARD payload straight from the file.  Do NOT use
                # `_load_agent` + `export_hard()`: GFMarkovLM_V6 has no
                # `import_hard`, so a V6 checkpoint re-loaded that way keeps
                # ZERO float parameters, and re-exporting from them yields an
                # all-zero model (constant column 0).  The saved payload IS
                # the shipped artefact, so read it directly.
                raw = torch.load(str(ck), map_location="cpu", weights_only=False)
                payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
                hard = hard_to_cpu(payload)
                # hard_forward is a method on GFMarkovLM; build one with the
                # payload's own architecture.  It reads ONLY the integer
                # tables in `hard`, so the float params are irrelevant.
                from src.gf256_markov_lm_v6 import GFMarkovLM as V6
                from src.gf256_markov_lm import GFMarkovLM as V7
                d = int(payload["emb"].shape[1])
                layers = len(payload["coef"])
                fan_in = int(payload["coef"][0].shape[1])
                lags = tuple(int(x) for x in payload["lags"])
                cls = V7 if d >= 64 else V6
                mdl = cls(d=d, layers=layers, fan_in=fan_in, lags=lags)
                mdl.eval()
            except Exception as exc:
                print(f"  {ck.name}: LOAD FAILED {exc!r}")
                continue
            # ctx must match the trainer's --ctx (192 for the acceptance
            # run): trainer_window reads at index ctx-1, so a different ctx
            # shifts the read position.
            scorer = PerPlyAgreement(raw_lines, pt, po, nply,
                                     torch.device("cpu"), n_rows=512,
                                     hard=hard, model=mdl, ctx=192)
            res = scorer.score(mdl, max_positions=4000)
            print(f"  {ck.name:28s} LABEL {res['perply_agreement_pct']:6.2f}%"
                  f"  n={res['perply_n']}  mass={res['perply_mass']:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
