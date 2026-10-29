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
for f in README.md RELEASE_RECORD.md CONNECT_FOUR_PAPER.md GF256_LANGUAGE_MODEL_PAPER.md FOLLOWUP_PLAN.md MODEL_CARD.md SOCIAL_COPY.md LICENSE; do
  copy "release/$f" "$f"
done
# The release documents also live under release/ in the source tree, so a
# reader who finds the repo can see they came from a release package.
mkdir -p "$TARGET/release"
cp -R "$ROOT/release/." "$TARGET/release/" 2>/dev/null || true
find "$TARGET/release" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
# The mirror must actually contain the documents -- a silent cp failure
# would ship an empty release/ dir.
for f in README.md RELEASE_RECORD.md CONNECT_FOUR_PAPER.md GF256_LANGUAGE_MODEL_PAPER.md MODEL_CARD.md SOCIAL_COPY.md; do
  [ -f "$TARGET/release/$f" ] || echo "   (WARNING: release/$f missing from the mirror)"
done

echo "-- verification scripts"
# The README documents the isolation gate as a published artifact, so it
# must ship with the repo -- otherwise the reader cannot re-run the check.
copy "scripts/verify_public_isolation.py" "scripts/verify_public_isolation.py"

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
  tmp/test_ui_multiuser.py \
  tmp/test_sessions_unit.py \
  tmp/test_transcript_stamp.py \
  tmp/test_play_v24_watcher.py \
  tmp/test_live_parity_v3.py \
  tmp/test_live_parity_v4.py \
  tmp/test_live_parity_v5.py \
  tmp/test_live_ui_fd_a.py \
  tmp/test_ui_api.py \
  tmp/test_ui_terminal.py \
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

echo "=== attribution check: must be Christof Ckal, never Alex ==="
# The release is authored by Christof Ckal. A leftover "Alex" attribution
# in a public document is a real defect: it misattributes the work.
ALEX="$(grep -rInE "\bAlex\b|alexchen|AlexChen" . --include='*.md' --include='*.py' --include='*.sh' --include='*.cff' 2>/dev/null || true)"
if [ -n "$ALEX" ]; then
  echo "NOTE: possible non-Christof attribution:" >&2
  echo "$ALEX" >&2
else
  echo "  clean -- no 'Alex' attribution in the staged tree"
fi
# The author of record must actually be present.
if grep -rIq "Christof Ckal" . --include='*.md' 2>/dev/null; then
  echo "  author of record present: Christof Ckal"
else
  echo "ABORT: 'Christof Ckal' does not appear in the staged documents." >&2
  BAD=1
fi

echo "=== no file over GitHub's 100 MB blob limit ==="
LIMIT="$(find . -type f -size +100M 2>/dev/null || true)"
if [ -n "$LIMIT" ]; then
  echo "ABORT: files over the GitHub 100 MB limit:" >&2
  echo "$LIMIT" >&2
  BAD=1
else
  echo "  clean -- largest staged file is $(find . -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
fi

echo "=== sanitize local paths out of the public tree ==="
# The working tree is on a personal Windows account. Absolute paths like
# C:\Users\<name>\... leak the local username into a PUBLIC repository.
# Replace the user segment with a generic <user>; the commands stay
# readable and copy-pasteable for anyone else.
# NOTE: the pattern is deliberately loose on the name so it also catches
# paths with a different separator style. Verification below is what
# actually gates the publish -- the detection pattern here must never be
# trusted on its own.
san_one() {
  python - "$1" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    s = p.read_text(encoding="utf-8")
except (UnicodeDecodeError, OSError):
    sys.exit(0)
orig = s
s = re.sub(r'[Cc]:[\\/]{1,2}Users[\\/]{1,2}[A-Za-z0-9_.\-]+', r'C:\\Users\\<user>', s)
s = re.sub(r'/[cC]/Users/[A-Za-z0-9_.\-]+', '/c/Users/<user>', s)
if s != orig:
    p.write_text(s, encoding="utf-8", newline='')
    print("S", end="")
else:
    print(".", end="")
PY
}
N_SAN=0
find . -type f \( -name '*.md' -o -name '*.py' -o -name '*.sh' \) -not -path './.git/*' -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do san_one "$f"; done
echo
echo "  sanitization pass complete"

# The gate: does the local username still appear anywhere?
LEFT="$(grep -rInE '[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+|/[cC]/Users/[A-Za-z0-9_.-]+' . 2>/dev/null | grep -v '<user>' || true)"
if [ -n "$LEFT" ]; then
  echo "ABORT: local paths remain in the staged tree:" >&2
  echo "$LEFT" | head -10 >&2
  BAD=1
else
  echo "  clean -- no local user paths remain"
fi

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

# The import-integrity check above compiles the staged package, which
# writes __pycache__ INTO the staged tree. Remove it AFTER the check, not
# only before it, or the published repo ships .pyc files.
find "$TARGET" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name '*.pyc' -type f -delete 2>/dev/null || true

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
