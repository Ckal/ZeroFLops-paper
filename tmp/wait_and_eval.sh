#!/usr/bin/env bash
# Wait for the corrected trainers to finish, then run the evaluation battery.
#
# "The process exited" is NOT "the arm finished": the battery's own
# pre-flight re-checks the log for the final step, the completion marker,
# the absence of a traceback, and the presence of the final artifacts
# before it measures anything.  If an arm failed, the battery aborts
# rather than reporting a number from a half-trained model.
set -u
cd "$(dirname "$0")/.."
PY=./.venv-c4/Scripts/python.exe
LOG=runs/_probe/post_train_wait.log
mkdir -p runs/_probe

# One-writer guard: two waiter chains used to truncate the SAME
# post_train_wait.log and both drive the same post_train_eval.log. The
# post_train_eval battery now has its own lock, but the waiter log itself
# was still clobbered. Guard it here too.
LOCK=runs/_probe/.wait_and_eval.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: another wait_and_eval chain holds $LOCK" >&2
  exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

: > "$LOG"

echo "waiting for the corrected trainers..." >> "$LOG"
while true; do
  n=$($PY -c "
import psutil
n=0
for p in psutil.process_iter(['pid','cmdline']):
    try:
        cl=p.info['cmdline'] or []
        if 'src.c4.train' in cl and any(t in c for c in cl for t in ('fx_a_s6000','fx_b_s6000','fx_c_s6000')):
            n+=1
    except Exception: pass
print(n)
")
  echo "  trainers alive: $n   $(date +%H:%M:%S)" >> "$LOG"
  [ "$n" -eq 0 ] && break
  sleep 60
done

echo "all trainers exited; running the evaluation battery" >> "$LOG"
bash tmp/post_train_eval.sh > runs/_probe/post_train_eval.log 2>&1
echo "BATTERY_EXIT=$?" >> "$LOG"
tail -40 runs/_probe/post_train_eval.log >> "$LOG"
echo "DONE" >> "$LOG"
