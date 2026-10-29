#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_v24_chain.sh -- start the V24 convergence run AND keep the UI
#                       serving the model it is training.
#
# Why this exists
# ---------------
# `run_v24.sh` trains.  `play_v24.sh` serves the newest promoted checkpoint
# and hot-reloads it when the trainer promotes a better one.  Neither knows
# about the other: the watcher needs a checkpoint to exist before it can
# start, and the trainer needs ~18 hours to produce one.  This chain wires
# the two together and is the thing a scheduled task actually invokes.
#
# It is deliberately resilient:
#
#   * a disk-headroom guard refuses to start with too little free space
#     (an 18 h run that dies at hour 17 because the volume filled is worse
#     than not starting);
#   * the training runs as a child, and the chain waits for the FIRST
#     promoted checkpoint before starting the watcher -- the watcher exits
#     with code 2 if the checkpoint is missing, so it must not be started
#     too early;
#   * if the trainer dies before producing a checkpoint, the chain stops
#     waiting instead of sleeping for hours against a dead process;
#   * when training finishes the chain does NOT exit.  It stays alive
#     holding the UI up, so the final model remains playable.  The
#     scheduled task therefore runs for the whole life of the run plus
#     however long you want to play with the result.
#
# Usage
# -----
#   bash tmp/launch_v24_chain.sh
#   V24_TAG=my_run PLAY_PORT=7860 bash tmp/launch_v24_chain.sh
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.."

TAG="${V24_TAG:-v24}"
PORT="${PLAY_PORT:-7860}"
TRAIN_LOG="runs/_probe/${TAG}_chain.log"
PLAY_LOG="runs/_probe/${TAG}_play.log"
DISK_MIN_MB="${V24_DISK_MIN_MB:-1500}"

mkdir -p runs/_probe

# ---- disk headroom guard -------------------------------------------------
free_mb="$(df -Pm . 2>/dev/null | awk 'NR==2{print $4}')"
if [ -z "${free_mb:-}" ] || [ "${free_mb:-0}" -lt "$DISK_MIN_MB" ]; then
  echo "ABORT: only ${free_mb:-unknown} MB free on the volume, need >= ${DISK_MIN_MB} MB" >&2
  echo "       The default --keep-last 3 retention needs ~0.9 GB for checkpoints" >&2
  exit 1
fi
echo "disk headroom: ${free_mb} MB free (need >= ${DISK_MIN_MB} MB)"

# ---- start the training --------------------------------------------------
echo "=== launch_v24_chain: starting V24 training (tag=$TAG) ===" >> "$TRAIN_LOG"
echo "=== at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$TRAIN_LOG"

bash run_v24.sh >> "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "training pid=$TRAIN_PID  log=$TRAIN_LOG"

# ---- wait for the first promoted checkpoint, then start the watcher ------
PLAY_PID=""
for _ in $(seq 1 240); do
  if [ -f "runs/_probe/$TAG/checkpoints/best_hard.pt" ]; then
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "training exited before producing a promoted checkpoint" >&2
    wait "$TRAIN_PID"
    exit $?
  fi
  sleep 30
done

if [ -f "runs/_probe/$TAG/checkpoints/best_hard.pt" ]; then
  PLAY_PORT="$PORT" V24_TAG="$TAG" bash play_v24.sh > "$PLAY_LOG" 2>&1 &
  PLAY_PID=$!
  echo "watcher pid=$PLAY_PID  port=$PORT  log=$PLAY_LOG"
else
  echo "no promoted checkpoint after 2 h -- watcher not started" >&2
fi

# ---- training finishes; keep the UI serving the final model -------------
wait "$TRAIN_PID"
RC=$?
echo "TRAINER_CHAIN_EXIT=$RC"
if [ -n "$PLAY_PID" ] && kill -0 "$PLAY_PID" 2>/dev/null; then
  echo "training done -- the UI keeps serving the final model on port $PORT"
  echo "(Ctrl-C here to stop the watcher and release the port)"
  wait "$PLAY_PID"
else
  echo "training done -- no watcher running"
fi
