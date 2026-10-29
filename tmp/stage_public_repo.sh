#!/usr/bin/env bash
# Stage a clean public repository for the Connect Four release.
#
# WHY A FRESH REPO AND NOT A PUSH
# -------------------------------
# The current history contains a 118,405,919-byte blob
# (artifacts/gf256_hard_model.pt, commit 186d176). GitHub rejects any
# blob larger than 100 MB and the limit is NOT raised by default, so a
# plain `git push` to GitHub WILL fail. This script therefore builds a
# NEW repository with a clean history containing only the Connect Four
# release, and leaves the text-generation paper and its 118 MB artifact in
# the private working repository.
#
# It also refuses to stage anything that would leak:
#   - the private training corpus / sidecar
#   - historical runs, logs, probe scratch
#   - credentials
#
# Usage:
#   bash tmp/stage_public_repo.sh <target-dir>
#
# Then:
#   cd <target-dir>
#   git init -b main
#   git add -A && git commit -m "feat: GF(256) Connect Four release"
#   gh auth switch --user Ckal
#   gh repo create Ckal/<name> --public --source=. --remote=origin --push
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
  echo "usage: bash tmp/stage_public_repo.sh <target-dir>" >&2
  exit 2
fi

case "$TARGET" in
  /*|?:*) : ;;                       # absolute path (native or msys)
  *) TARGET="$ROOT/$TARGET" ;;
esac

echo "=== staging public repo ==="
echo "  source : $ROOT"
echo "  target : $TARGET"

if [ -e "$TARGET" ]; then
  echo "ABORT: target already exists: $TARGET" >&2
  exit 1
fi

mkdir -p "$TARGET"
cd "$TARGET"

# ---------------------------------------------------------------- files
copy() {  # copy <src-relative-to-root> <dst-relative-to-target>
  mkdir -p "$(dirname "$2")"
  cp "$ROOT/$1" "$2"
}

echo "-- release documents"
for f in README.md RELEASE_RECORD.md CONNECT_FOUR_PAPER.md FOLLOWUP_PLAN.md MODEL_CARD.md LICENSE; do
  copy "release/$f" "$f"
done

echo "-- source"
# The whole src/c4 package is copied, not a hand-picked subset: the
# modules import each other (agent -> patterns, train -> corpus,
# corpus_aug, corpus_behavior, corpus_quality, movemask, symmetry,
# teacher_corpus, transcript, ...), and a missing sibling breaks the
# package at import time.  src/c4 is 1.3 MB and contains no data.
mkdir -p src/c4 src
cp -r "$ROOT/src/c4/." "$TARGET/src/c4/"
find "$TARGET/src/c4" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
for f in \
  src/gf256_markov_lm_v6.py \
  src/gf256_markov_lm.py \
  src/generate.py ; do
  [ -f "$ROOT/$f" ] && copy "$f" "$f" || echo "   (skip missing $f)"
done
# NOTE: src/gf256_markov_lm_v6_1.py is intentionally NOT copied.  It is a
# dead branch -- the shipped architecture and every loader/trainer path
# use gf256_markov_lm_v6.py.  Shipping both invites the reader to patch
# the wrong file.

echo "-- curated tests and authoritative tools"
mkdir -p tmp
for f in \
  tmp/measure_promotion_scorer.py \
  tmp/measure_label_agreement.py \
  tmp/audit_ingame_strength_fd_a.py \
  tmp/test_resume_state.py \
  tmp/test_resume_equivalence.py \
  tmp/test_promotion_gate.py \
  tmp/test_atomic_checkpoint_save.py \
  tmp/test_ui_hot_reload.py \
  tmp/test_ui_single_server.py \
  tmp/test_transcript_stamp.py \
  tmp/test_play_v24_watcher.py \
  tmp/test_live_parity_v3.py \
  tmp/test_live_parity_v4.py \
  tmp/test_live_parity_v5.py \
  tmp/test_live_ui_fd_a.py \
  tmp/run_regression_sweep.sh ; do
  [ -f "$ROOT/$f" ] && copy "$f" "$f" || echo "   (skip missing $f)"
done

echo "-- launchers"
for f in run_v24.sh play_v24.sh ; do
  [ -f "$ROOT/$f" ] && copy "$f" "$f" || echo "   (skip missing $f)"
done

echo "-- docs"
mkdir -p docs
for f in docs/CURRENT_TRUTH.md docs/CONNECT_FOUR_HANDOVER.md docs/CONNECT_FOUR_PLAIN_ENGLISH.md docs/CONNECT_FOUR_OVERVIEW.md ; do
  [ -f "$ROOT/$f" ] && copy "$f" "$f" || echo "   (skip missing $f)"
done

# ---------------------------------------------------------------- hygiene
cat > .gitignore <<'GITIGNORE'
# probe / scratch — never published
runs/
*.npz
*.log
!runs/.gitkeep

# local env
.venv*/
__pycache__/
*.pyc

# secrets
.env
*.key
*.pem
GITIGNORE

# ---------------------------------------------------------------- checks
echo "=== leak check: no corpus / sidecar / large artifacts ==="
BAD=0
for pat in 'corpus.txt' '*.corpus.txt' '*.npz' '*.pt' '*.bin'; do
  found="$(find . -type f -name "$pat" 2>/dev/null || true)"
  if [ -n "$found" ]; then
    echo "  NOTE: present -> $found"
  fi
done

BIG="$(find . -type f -size +40M 2>/dev/null || true)"
if [ -n "$BIG" ]; then
  echo "ABORT: files > 40 MB staged:" >&2
  echo "$BIG" >&2
  BAD=1
fi

echo "=== credential pattern check ==="
CREDS="$(grep -rIlE "(api[_-]?key|secret|password|passwd|token|bearer|private[_-]?key)[[:space:]]*[:=][[:space:]]*['\"][A-Za-z0-9_/+=-]{12,}" . 2>/dev/null || true)"
if [ -n "$CREDS" ]; then
  echo "ABORT: possible credentials in:" >&2
  echo "$CREDS" >&2
  BAD=1
else
  echo "  clean"
fi

echo "=== private-corpus reference check ==="
for probe in "corpus.txt" "disjoint_holdout" "perply-targets"; do
  if grep -rIl -- "$probe" . >/dev/null 2>&1; then
    echo "  NOTE: references '$probe' (expected in launcher scripts; corpus itself is absent)"
  fi
done

if [ "$BAD" -ne 0 ]; then
  echo "STAGING ABORTED — fix the above before publishing." >&2
  exit 1
fi

echo "=== import integrity check (staged tree must import standalone) ==="
PY="$(command -v python3 || command -v python || true)"
if [ -n "$PY" ]; then
  if (cd "$TARGET" && "$PY" -c "
import sys; sys.path.insert(0,'.')
import src.c4.board, src.c4.agent, src.c4.opponents, src.c4.transcript_v2
import src.c4.movemask, src.c4.symmetry, src.c4.patterns, src.c4.exact_solver
import src.c4.mirror, src.c4.corpus, src.c4.corpus_integrity
print('  staged package imports OK')
" 2>&1); then
    :
  else
    echo "ABORT: the staged package does not import standalone." >&2
    exit 1
  fi
else
  echo "  (skipped: no python on PATH)"
fi

echo "=== staged tree ==="
find . -type f -not -path './.git/*' | sort | sed 's|^\./||'
echo
echo "total files: $(find . -type f -not -path './.git/*' | wc -l)"
echo "total size : $(du -sh . | cut -f1)"
echo
echo "STAGING OK -> $TARGET"
echo "Next:"
echo "  cd \"$TARGET\""
echo "  git init -b main && git add -A && git commit -m 'feat: GF(256) Connect Four release'"
echo "  gh auth switch --user Ckal"
echo "  gh repo create Ckal/<name> --public --source=. --remote=origin --push"
