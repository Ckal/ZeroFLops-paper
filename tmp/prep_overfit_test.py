"""DECISIVE EXPERIMENT: can this architecture fit ANYTHING?

The measured state of affairs
-----------------------------
* Model-vs-label agreement on its OWN TRAINING rows: 20.2 %
  (GOLD-verified: encode_game_dense(moves) == raw corpus bytes, 1500/1500)
* legal_acc on the training corpus: 0.133 (step 0) -> 0.277 (step 6000),
  then PLATEAU (0.265 at 5000, 0.286 at 5500, 0.277 at 6000).
* 6000 steps x batch 16 = 96,000 samples drawn from 320,000 rows /
  ~4.46 M per-ply labels.  That is ~2 % of ONE epoch.

So there are two competing explanations, and they lead to OPPOSITE actions:

  (A) "Under-trained."  The model has seen 2 % of one epoch.  Fix: train
      much longer / on less data per step.
  (B) "Cannot fit."  The architecture or the optimisation cannot represent
      this mapping at all.  Fix: change the architecture.

THIS TEST SEPARATES THEM
------------------------
Memorisation test: take a TINY corpus (NROWS rows), so the same rows are
seen over and over (many epochs), and train with the identical settings
fd_a used.  Then measure model-vs-label agreement on those same rows.

  * If agreement climbs towards 100 %  -> the architecture CAN fit.
    Explanation (A).  The fix is scale of training, not architecture.
  * If agreement stays near 20 %        -> the architecture CANNOT fit.
    Explanation (B).  More data, more steps, better labels are all
    pointless until the architecture changes.

Neither the "worse labels" nor the "tie sets" hypothesis can explain a
failure to MEMORISE 500 rows.  That is why this is the right probe.

Artifact: runs/_probe/overfit_test/
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_CORPUS = ROOT / "runs/_probe/perply_v23_full/corpus_320k_fixed.txt"
SRC_SIDE = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"
OUTDIR = ROOT / "runs/_probe/overfit_test"
NROWS = 500


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- carve a small corpus + matching sidecar slice ----------------
    rows = []
    with open(SRC_CORPUS, "rb") as fh:
        for i, line in enumerate(fh):
            if i >= NROWS:
                break
            rows.append(line.rstrip(b"\r\n"))
    (OUTDIR / "corpus.txt").write_bytes(b"\n".join(rows) + b"\n")

    z = np.load(SRC_SIDE, allow_pickle=True)  # repo's own pipeline output
    small = {k: z[k][:NROWS] for k in z.files}
    np.savez(OUTDIR / "perply_targets.npz", **small)
    print(f"corpus rows : {len(rows)}")
    print(f"sidecar keys: {sorted(small.keys())}")
    for k, v in small.items():
        print(f"  {k}: {getattr(v, 'shape', None)}")
    tot = int((small["ply_of"] >= 0).sum())
    print(f"labelled per-ply slots: {tot}")

    # ---- how many steps for many epochs? ------------------------------
    n_labels = tot
    per_step = 16
    steps_for_10_epochs = int(10 * n_labels / per_step)
    print(f"\nsteps for ~10 epochs over {n_labels} labels: "
          f"{steps_for_10_epochs}")

    cfg = {
        "nrows": NROWS,
        "labelled_slots": tot,
        "steps": steps_for_10_epochs,
        "batch": per_step,
        "epochs": round(steps_for_10_epochs * per_step / max(1, n_labels), 1),
        "note": ("identical trainer flags to fd_a except --steps; "
                 "same corpus bytes, same sidecar, same seed 32"),
    }
    (OUTDIR / "config.json").write_text(json.dumps(cfg, indent=2),
                                        encoding="utf-8")
    print(f"\nwrote {OUTDIR/'config.json'}")
    print(f"RUN: python -m src.c4.train --run-dir runs/_probe/overfit_test "
          f"--perply-targets runs/_probe/overfit_test/perply_targets.npz "
          f"--load-corpus --transcript v2 --move-loss 7way --soft-w 1.0 "
          f"--steps {steps_for_10_epochs} --batch 16 --ctx 192 --lr 0.0006 "
          f"--seed 32 --eval-every 100 --save-every 1000000 "
          f"--eval-games 0 --max-legacy-frac 0 --corpus-source teacher "
          f"--seed-games 0 --selfplay-games 0 --teacher-games 0 "
          f"--selfplay-every 1000000 --hard-start 0 --hard-ramp 1000 "
          f"--hard-w 0.5 --table-bias 0.0 --lags 0,1,2,3,4,5,6,7,8,9,10,20 "
          f"--col-prior 0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
