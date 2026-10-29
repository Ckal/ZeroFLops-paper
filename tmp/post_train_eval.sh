#!/usr/bin/env bash
# Post-training evaluation battery for the CORRECTED arms.
#
# Runs only after the trainers have finished, and only if they finished
# correctly.  Every metric is computed with the SAME loader, transcript
# version, seat order and trajectory accounting.
set -u
cd "$(dirname "$0")/.."
PY=./.venv-c4/Scripts/python.exe
OUT=runs/_probe
mkdir -p "$OUT"

# ---- one-writer guard ----
# Two concurrent battery chains used to write the SAME log and JSON paths.
# The result was NUL-padded logs and a silently overwritten JSON: last
# writer won, the other result vanished with no error. Refuse to run
# twice at once instead.
LOCK="$OUT/.post_train_eval.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: another post_train_eval chain holds $LOCK" >&2
  echo "       (if that chain is dead, remove the directory and retry)" >&2
  exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "lock acquired: $LOCK"

# ---- the trainers must have finished VERIFIABLY, not merely exited ----
RC=0
for tag in fx_a_s6000 fx_b_s6000 fx_c_s6000; do
  LOG="$OUT/$tag.log"
  fail=""
  [ -f "$LOG" ] || fail="no log"
  if [ -z "$fail" ]; then
    grep -qE "^\s*step\s+6000" "$LOG" || fail="never reached step 6000"
  fi
  if [ -z "$fail" ]; then grep -q "elapsed:" "$LOG" || fail="no completion marker"; fi
  if [ -z "$fail" ]; then grep -q "Traceback" "$LOG" && fail="traceback in log"; fi
  if [ -z "$fail" ]; then
    for f in "$OUT/$tag/final_float.pt" "$OUT/$tag/hard_model_final.pt" \
             "$OUT/$tag/checkpoints/best_hard.pt"; do
      [ -s "$f" ] || fail="missing/empty $f"
    done
  fi
  if [ -n "$fail" ]; then
    echo "ARM NOT USABLE  $tag : $fail"; RC=1
  else
    echo "ARM USABLE      $tag"
  fi
done
[ $RC -eq 0 ] || { echo "ABORT: a corrected arm did not complete"; exit 1; }

# ---- 1. the shipped promotion scorer, per arm ----
# This is the AUTHORITATIVE promotion metric. It measures the promoted
# artifact (checkpoints/best_hard.pt) as well as the final export.
# It runs FIRST because it is cheap and bounded (n_rows=512, max_positions
# 4000) -- unlike the generic scanner below, which loads the whole corpus.
for tag in fx_a_s6000 fx_b_s6000 fx_c_s6000; do
  $PY -u tmp/measure_promotion_scorer.py "$OUT/$tag" \
    > "$OUT/${tag}_promotion_scorer.log" 2>&1
  rc=$?
  echo "--- $tag / shipped scorer (exit=$rc) ---"
  tail -8 "$OUT/${tag}_promotion_scorer.log"
  [ $rc -eq 0 ] || RC=1
done

# ---- 2. independent cross-check with the generic scanner ----
# measure_label_agreement.py takes: <corpus> <sidecar> <checkpoint> [...]
# The corrected arms each carry their own corpus.txt and perply_targets.npz
# (copied from the shared source), so the measurement is run against the
# arm's OWN copy -- not a different corpus.
#
# NOTE: the generic scanner loads the WHOLE corpus into memory and has been
# observed to die under memory pressure. It runs LAST among the measurement
# steps, and its failure is reported without aborting the battery -- the
# shipped scorer above is the authoritative promotion metric.
for tag in fx_a_s6000 fx_b_s6000 fx_c_s6000; do
  for ck in checkpoints/best_hard hard_model_final; do
    f="$OUT/$tag/$ck.pt"
    [ -s "$f" ] || continue
    name=$(echo "$ck" | tr '/' '_')
    $PY -u tmp/measure_label_agreement.py \
      "$OUT/$tag/corpus.txt" "$OUT/$tag/perply_targets.npz" "$f" \
      > "$OUT/${tag}_${name}_label_agreement.log" 2>&1
    rc=$?
    echo "--- $tag / $ck (generic scanner, exit=$rc) ---"
    grep -iE "agreement|%|error" "$OUT/${tag}_${name}_label_agreement.log" | tail -4
  done
done

# ---- 3. in-game strength with TRUE constant opponents ----
for tag in fx_a_s6000 fx_b_s6000 fx_c_s6000; do
  ck="$OUT/$tag/hard_model_final.pt"
  [ -s "$ck" ] || continue
  $PY -u tmp/audit_ingame_strength_fd_a.py --ckpt "$ck" --games 200 \
    --out "$OUT/ingame_strength_${tag}.json" \
    > "$OUT/ingame_strength_${tag}.log" 2>&1
  echo "--- $tag / in-game strength ---"
  grep -E "MODEL|best constant|VERDICT" "$OUT/ingame_strength_${tag}.log" | head -20
done

# ---- 4. the full regression sweep, one more time, after everything ----
bash tmp/run_regression_sweep.sh > "$OUT/regression_sweep_final.log" 2>&1
tail -3 "$OUT/regression_sweep_final.log"

echo "EVALUATION BATTERY DONE"
