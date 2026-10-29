# ZeroFLops — Public Release Plan (Paper, Code, Model, Game, Marketing)

> **For Hermes:** execute task-by-task with `subagent-driven-development`; two-stage
> review per task. Do not publish anything before its gate says `GO`.

**Goal:** take the Connect Four GF(256) work from "measured on one laptop" to a
public, auditable release across five surfaces — paper, GitHub, Hugging Face, a
public playable game, and a marketing wave — with every published number traced
to a measured artifact.

**Architecture:** One frozen release artifact (`best_hard.pt`, step 18000) is the
single source of truth. A clean-history staging script produces the public repo
from a verified source tree (the existing history cannot be pushed: it contains a
118 MB blob over GitHub's 100 MB limit). `release/RELEASE_RECORD.md` is the
ledger every document cites. Nothing is published until its gate passes.

**Tech Stack:** Python 3.11 + PyTorch (CPU inference), Git, GitHub CLI, Hugging
Face Hub CLI, Nginx + Docker on Netcup (159.195.150.113), Blogger API, X API.

---

## 0. Current state (measured 2026-10-29 08:34)

### The training run — alive, and PLATEAUED

| | |
|---|---|
| Live step | **41,040 / 100,000** (py-spy, PID 42856) |
| Last evaluation | step 40,000 at t=21,175 s |
| Rate | ~0.42 s/step effective |
| Remaining | ~59,000 steps ≈ **6.9 h** → ETA ≈ 15:30 |
| Promoted artifact | `best_hard.pt` = **step 18000**, unchanged since 04:41 |
| LABEL curve | 59.55 (18k) → 59.0 → 58.5 → 58.6 → 58.6 → 59.3 → 59.0 — **flat for 10 evals** |
| H-LABEL | 41.0 (18k) → 40.8 (40k), peak 42.2 at 32k — **flat** |
| vs random | 41-7 at 18k, 35-13 at 40k (48-game noise) |
| vs 1-ply heuristic | 16-32 at 18k, 17-31 at 40k — **net negative throughout** |
| Disk | **6.1 GB free, C: at 100 %** ← real constraint |

**Decision D1 — do not block the release on 100k steps.** Label agreement has
been flat since step 18,000 across 10 consecutive evaluations. The remaining
59,000 steps will almost certainly not promote a new artifact. Keep the run
alive as a *separate* "does more training help?" experiment; release step 18,000
now.

### The playable versions (answering "new version to play against?")

| Port | Checkpoint | LABEL | Verdict |
|---|---|---|---|
| 7860 | `best_hard.pt` = step **18000** | **59.55 %** | **best authoritative — keep as the playable** |
| 7861 | `step20000_hard.pt` = step 20000 | 59.00 % | kept only for the A/B head-to-head; **worse** |
| — | step 36000 / 38000 / 40000 exist | 58.58 / 59.33 / 59.00 | **none beat 18000** |

**There is no strictly better new version to play.** The honest answer to the
user is: *the promoted model at :7860 is already the best one; the run is
improving the score on paper, not on the board.*

**Caveat C1 — both running servers predate the session-isolation refactor.**
They still run the old single-global-game code. They are fine for private
play but are **not** evidence that isolation works, and they must not be
deployed publicly.

**Caveat C2 — the plateau makes the "newest = best" claim false.** Task 1.3
below sets up the experiment that tests it directly.

### Publishing identities (verified)

| Surface | Account | State |
|---|---|---|
| GitHub | `Ckal` | ✅ exists, authenticated but **inactive** (active = `alexchen1403-prog`) |
| Hugging Face | `ckal` | ❌ **Bob Kaleugher — a different person. Do not use.** |
| Hugging Face | `AlexChen1403` | ⚠️ authenticated, unsuitable for public attribution |
| X | `zX14_7` | ✅ Christof Kaller — **no API session configured** |
| Blogger | `ckaller.blogspot.de` | ✅ token available (`[REDACTED]`) |
| Netcup | 159.195.150.113 | ✅ SSH works with the deploy key |

### Blockers

- **B1** GitHub blob limit: `artifacts/gf256_hard_model.pt` is 118,405,919 B at
  commit `186d176`; limit is 104,857,600 B → existing history is unpushable.
  Workaround exists: `tmp/stage_public_repo.sh` (clean history).
- **B2** `tmp/test_sessions_unit.py` was written but **never executed**.
- **B3** Reload/new-session edge: `main()`'s `_new_game` closure captures the
  *initial* agent, so a session created after a hot reload gets the old model.
- **B4** Hugging Face namespace: no Christof-owned namespace exists or is
  authenticated.
- **B5** X API: no credential/session; `xurl` skill unsupported on this platform.
- **B6** Disk: 6.1 GB free. Staging + HF upload + deploy + video all need room.
- **B7** `release/*.md` cite step-18000 numbers — consistent today, but every
  doc must cite from the ledger so a future promotion does not desync them.

---

## PHASE 1 — Freeze the evidence (blocking, ~45 min)

Nothing publishes until this phase is done. Its job is to make every later
number checkable.

### Task 1.1 — Run the never-executed session unit test — ✅ DONE 08:37

**Objective:** close blocker B2 before anything ships on the new server code.

**RESULT:** the test **failed 3/46 on first run** and exposed two real product
bugs plus one test bug. All fixed; now **47/47 PASS**.

| Failure | Kind | Fix |
|---|---|---|
| "session is gone after the TTL" | **real bug** | `get()` never checked the TTL — expiry was only enforced by the background sweeper, so a session could be revived between expiry and the next sweep. `get()` now treats an idle session as gone and drops it. |
| "parser reads the session" | **real bug** | a repeated cookie name resolved to the **last** value, letting a broader-scoped cookie override the session cookie. RFC 6265 sends the most specific path first → the **first** value now wins. |
| "sweep reports the expired session" | test bug | `get()` had already evicted the session, so the sweeper correctly found nothing. The test now exercises the sweeper on its own session. |

Committed in `b286aee`.

### Task 1.2 — Fix the reload/new-session agent closure (B3) — ✅ DONE 08:37

**Objective:** a session created *after* a hot reload must use the new model.

**Fix applied:** a module global `_CURRENT_AGENT` replaces the closure over the
startup agent.

- `main()` sets `_CURRENT_AGENT = agent` at startup.
- `_reload_agent` sets `_CURRENT_AGENT = agent` **before** swapping sessions, so
  a session created concurrently picks up the new model.
- `_new_game()` reads `_CURRENT_AGENT`.

**Still to verify:** extend `tmp/test_ui_hot_reload.py` with the case "reload to
checkpoint B, then a NEW visitor's `/probs` must match B". Not yet written.

### Task 1.3 — Test whether "newer" plays better than "better-scoring"

**Objective:** settle C2 with a measurement instead of an assumption.

**Files:** `tmp/audit_ingame_strength_fd_a.py` (reuse, do not fork).

```bash
# 200 games per arm, canonical harness, identical loader/seat/trajectory rules
./.venv-c4/Scripts/python.exe -u tmp/audit_ingame_strength_fd_a.py \
  --ckpt runs/_probe/compare/step20000_hard.pt \
  --out runs/_probe/ingame_strength_step20000.json --games 200
./.venv-c4/Scripts/python.exe -u tmp/audit_ingame_strength_fd_a.py \
  --ckpt runs/_probe/v24/checkpoints/step40000_hard.pt \
  --out runs/_probe/ingame_strength_step40000.json --games 200
```

**Note:** the trainer's own ladder uses 48 games per arm (24 vs_ref + 48 vs
each opponent) — too small to separate 85 % from 91 %. The 200-game matrix is
the number that goes in the paper.

**Record** in `release/RELEASE_RECORD.md`: model-vs-random %, model-vs-centre %,
model-vs-heuristic %, with n and the exact command.

**Commit:** `git add runs/_probe/ingame_strength_*.json release/RELEASE_RECORD.md && git commit -m "measure(c4): 200-game strength matrix for steps 20000 and 40000"`

### Task 1.4 — Reconcile every published number against the ledger

**Objective:** one source of truth; no doc carries a hand-typed number.

**File:** `release/RELEASE_RECORD.md` — extend with a machine-readable block.

Add at the top:

```markdown
## Authoritative numbers (release gate: these and only these may be published)

| Key | Value | Source |
|---|---|---|
| `artifact_step` | 18000 | `best_hard.pt` top-level `step` |
| `artifact_bytes` | 33,830,949 | `stat` on the released file |
| `parameters` | 8,422,496 | `sum(p.numel() for p in model.parameters())` |
| `label_agreement_train` | 59.55 % | `tmp/measure_promotion_scorer.py --corpus-run runs/_probe/v24` |
| `label_agreement_holdout` | 41.02 % | step-18000 evaluation |
| `corpus_games` | 320,000 | preflight row count |
| `corpus_labels` | 4,462,571 | per-ply label count |
| `boards_scanned` | 5,100,175 | corpus integrity scan |
| `strength_vs_random` | <from 1.3> | 200-game matrix |
| `strength_vs_centre` | <from 1.3> | 200-game matrix |
| `strength_vs_heuristic` | <from 1.3> | 200-game matrix |
| `transcript_version` | v2 | checkpoint meta |
| `context_width` | 192 | trainer args |
| `batch_size` | 16 | trainer args |
| `seed` | 32 | trainer args |
| `lags` | 0..10, 20 | checkpoint meta |
| `table_bias` | 0.0 | trainer args |
| `inference` | CPU-only | `exp_t`/`log_t` are CPU tensors |
```

**Then** rewrite every number in `release/CONNECT_FOUR_PAPER.md`,
`release/README.md`, `release/MODEL_CARD.md`, `release/SOCIAL_COPY.md` to match
this table. Where a doc currently says "beats random ~85–90 %", replace it
with the measured value from Task 1.3.

**Verify:** grep every numeric claim and check each appears in the ledger.
```bash
grep -nE "[0-9]+\.[0-9]+ ?%|step [0-9]+|[0-9]{3},[0-9]{3}" release/*.md
```

**Commit:** `git commit -m "docs(c4): single authoritative number ledger for the release"`

### Task 1.5 — Freeze the release artifact

**Objective:** make the published bytes immutable and checksummed.

```bash
mkdir -p release/artifacts
cp runs/_probe/v24/checkpoints/best_hard.pt release/artifacts/connect_four_gf256_best_hard_step18000.pt
sha256sum release/artifacts/connect_four_gf256_best_hard_step18000.pt
./.venv-c4/Scripts/python.exe -c "
import torch
c=torch.load('release/artifacts/connect_four_gf256_best_hard_step18000.pt',weights_only=False)
print('step',c.get('step'));print('meta keys',sorted((c.get('meta') or {}).keys()))"
```

Write the SHA-256 into `release/RELEASE_RECORD.md`. The model card, the paper,
and the HF upload all cite this hash.

**Do NOT** commit the 33 MB file to this repo (it is a release *artifact*, not
source). It goes to Hugging Face; the repo carries the hash and the loader.

**Commit:** `git commit -m "chore(c4): pin the release artifact hash"`

---

## PHASE 2 — Finish the public-safe server (blocking, ~90 min)

### Task 2.1 — Re-run the full regression suite against the new code

**Objective:** the session refactor touched every handler; prove nothing else broke.

```bash
bash tmp/run_regression_sweep.sh
```
Expected: `24 PASS, 0 FAIL`. This sweep owns its own UI server and uses the
PID-aware lock — **do not `rm -rf` the lock** (that was the cause of the
rotating-failure episode).

If it fails, run each test individually against a single sweep-owned server.

### Task 2.2 — Re-run the multi-user isolation test

```bash
./.venv-c4/Scripts/python.exe -u tmp/test_ui_multiuser.py --port 7871
```
Expected: `ALL PASS`, `26/26 checks passed`. **Already verified once** — re-run
after Task 1.2's fix to confirm the reload change did not regress isolation.

### Task 2.3 — Restart the local play servers on the NEW code

**Objective:** kill caveat C1 — the running servers use the old global-game code.

```bash
# find and kill the ui_server processes on 7860/7861 (psutil tree kill,
# children first) -- do NOT touch the trainer PID 42856
./.venv-c4/Scripts/python.exe tmp/cleanup_leaked_watchers.py --dry-run   # inspect first
```

Then relaunch:
```bash
# promoted model, with admin routes (local use only)
PYTHONPATH=$PWD ./.venv-c4/Scripts/python.exe -u -m src.c4.ui_server \
  --model release/artifacts/connect_four_gf256_best_hard_step18000.pt --port 7860
```

Verify: `GET /modelinfo` → `step: 18000`; two browsers → two independent boards.

### Task 2.4 — Commit the session work

```bash
git add src/c4/sessions.py src/c4/ui_server.py tmp/test_ui_multiuser.py tmp/test_sessions_unit.py
git commit -m "feat(c4): per-visitor session isolation for the public game"
```

---

## PHASE 3 — GitHub go-live (gate: Phase 1 + 2 green)

### Task 3.1 — Switch the CLI to the Christof account

```bash
gh auth switch --user Ckal
gh auth status        # MUST show "Active account: Ckal"
```
Verify before any push. If `Ckal` cannot create a repo, stop and report — do
not fall back to `alexchen1403-prog` for a Christof-attributed release.

### Task 3.2 — Produce and verify the clean-history release tree

```bash
rm -rf "$TMPDIR/tg_release_stage"
bash tmp/stage_public_repo.sh "$TMPDIR/tg_release_stage"
```

**Gate checks on the staged tree:**
- [ ] `src/c4/__init__.py` present; the whole `src/c4/` package copied
- [ ] `python -c "import sys;sys.path.insert(0,'<stage>');from src.c4 import agent, board, sessions, ui_server"` succeeds **standalone**
- [ ] no `corpus*`, `*.npz`, `*.log`, `runs/`, `checkpoints/`
- [ ] no file > 100 MB (`find <stage> -size +100M`)
- [ ] no secrets: grep for `password|token|api[_-]?key|secret|BEGIN .*PRIVATE KEY` → only `[REDACTED]`
- [ ] no attribution to Alex: `grep -ril "alex" <stage>` → empty
- [ ] `release/` docs present and reconciled
- [ ] `LICENSE` present
- [ ] the release artifact is NOT in the tree (hash only)

### Task 3.3 — Create and push the public repository

```bash
cd "$TMPDIR/tg_release_stage"
git init -b main
git remote add origin git@github.com:Ckal/<repo>.git
git add -A && git commit -m "feat: GF(256) Connect Four -- exact-solver supervision, per-ply labels, playable UI"
git push -u origin main
```

**Repo name decision:** the existing root repo is the *text-generation* paper
(`paper.md`, `paper/`). The Connect Four work is a distinct project with its
own paper, model card, and UI. **Recommendation: two repositories.**

| Repo | Contents |
|---|---|
| `Ckal/zeroflops` | the existing text-generation paper, unchanged |
| `Ckal/connect-four-gf256` | this release (new, clean history) |

Rationale: the two papers have different audiences, different model cards, and
different release gates. Merging them into one README makes the GitHub landing
page ambiguous and makes the "which paper is this?" question unanswerable.

**Alternate (if the user prefers one repo):** publish a single repo with
`papers/text-generation/` and `papers/connect-four/` subtrees and a landing
README that links both — but this requires editing the *existing* README, which
the text-generation paper's provenance depends on.

### Task 3.4 — Repository polish

- [ ] `README.md` = `release/README.md` (the good one), first viewport = result table + "play it" link
- [ ] Topics: `connect-four`, `gf256`, `integer-arithmetic`, `game-ai`, `pytorch`, `zero-flops`
- [ ] Description: one line, Christof-attributed
- [ ] Screenshots/GIF of the game (from Task 6.4)
- [ ] `CITATION.cff` with author `Ckal, Christof`
- [ ] Issue templates, `SECURITY.md` (report privately, no public CVE issue for a game server)
- [ ] CI: run the shipped tests (`tmp/test_sessions_unit.py`, `tmp/test_ui_multiuser.py`, `tmp/test_promotion_gate.py`) on push — these are CPU-only and fast
- [ ] `SECURITY.md` must state: the public game runs the model on CPU, holds no user data beyond an ephemeral session cookie, and has no accounts

---

## PHASE 4 — Hugging Face go-live (gate: Phase 1 green)

### Task 4.1 — Establish a Christof-owned HF namespace

**Objective:** find or create an HF account actually belonging to Christof.

```bash
curl -s https://huggingface.co/api/models?author=ckal | head -c 400   # confirm ckal is NOT ours
hf auth whoami
```

`ckal` is **Bob Kaleugher** — a different person. `AlexChen1403` is authenticated
locally but is the wrong public identity for a Christof-attributed release.

**Options, in order of preference:**
1. **An existing Christof-owned HF namespace** — ask the user for the handle, or
   check `huggingface.co/<handle>` for each plausible one.
2. **Register a new namespace** — `huggingface.co/join` requires an email
   confirmation. The user must complete this in a browser.
3. **Deploy a gated/private repo under the authenticated account** — only as an
   explicitly labelled fallback, with the attribution mismatch noted in the
   model card. **Not preferred.**

**Gate:** the namespace resolves to Christof, verified by fetching
`https://huggingface.co/<ns>` and reading the display name. If it cannot be
verified, **stop** — do not upload under an unverified namespace.

### Task 4.2 — Upload the model

```bash
hf repo create <ns>/connect-four-gf256 --repo-type model --private   # public only after the gate
hf upload <ns>/connect-four-gf256 \
  release/artifacts/connect_four_gf256_best_hard_step18000.pt \
  --path-in-repo model/connect_four_gf256_step18000.pt
hf upload <ns>/connect-four-gf256 release/MODEL_CARD.md --path-in-repo README.md
```

Also upload (small, useful):
- `release/LICENSE`
- the loader snippet from `src/c4/agent.py` (a standalone `loader.py`)
- `runs/_probe/v24/metrics.jsonl` — the full step-by-step curve (small, high value)

### Task 4.3 — Model-card reconciliation

The model card must state, without hedging:
- the artifact SHA-256 from Task 1.5
- step 18000, transcript v2, ctx 192, lags, seed 32, table_bias 0.0
- train LABEL 59.55 %, holdout LABEL 41.02 %
- **game strength from Task 1.3** — and the honest line: *beats random; does not
  beat a simple centre-column policy; loses to a 1-ply heuristic*
- CPU-only inference, ~100 ms/move measured on this host
- licence: MIT for code/weights, paper text under separate terms
- known limitations: exact-solver holdout coverage is only 560/1,500 (37.3 %)
  positions; the model is **not** a strong player
- a "how to reproduce" block with the exact commands

---

## PHASE 5 — Public game go-live (gate: Phases 1–4 green + B1 resolved)

### Task 5.1 — Decide the public URL and DNS

**Recommendation:** `play.tensorgarten.de` → 159.195.150.113, Nginx TLS,
proxying to a container on `127.0.0.1:7860` **inside the Netcup host**.

Note from memory: `*.tensorgarten.de` already resolves to 159.195.150.113, and
the host runs Docker + Nginx. Verify the exact vhost layout before adding
anything — do not assume a directory.

**Gate:** the user confirms the hostname before DNS is touched.

### Task 5.2 — Deploy privately and smoke-test

**Objective:** deploy to the host, but **not** publicly, until the smoke tests pass.

- Build a small image: `python:3.11-slim` + `torch (CPU)` + the `src/c4`
  package + the released checkpoint (32 MB, not the 118 MB historical one).
- Expose the container on `127.0.0.1:7860` **on the host**, not `0.0.0.0`.
- Run it with `--public` so `/modelinfo` and `/reload` are refused and the
  cookie is `Secure`.
- Nginx: TLS (certbot), `proxy_pass`, `proxy_set_header X-Forwarded-For`,
  request body cap, and the security headers (the app sets them, but the
  reverse proxy should too).

**Smoke tests against the private URL:**
- [ ] `/` → 200, HTML contains the game
- [ ] `/state` → 200 with an empty board
- [ ] `/modelinfo` → **403**
- [ ] `/reload` → **403**
- [ ] two independent clients → two independent boards
- [ ] `Set-Cookie` has `HttpOnly`, `SameSite=Lax`, **`Secure`**
- [ ] `Strict-Transport-Security` present, HTTP→HTTPS redirect works
- [ ] `X-Frame-Options`, `CSP`, `X-Content-Type-Options`, `Referrer-Policy` present
- [ ] a malformed `tg_sid` → fresh session, not a shared board
- [ ] 30 sequential moves → no leak, latency stable
- [ ] server memory stable over 100 requests

### Task 5.3 — Flip DNS and re-run the public smoke

Only after Task 5.2 is fully green, and only after the user says go.

### Task 5.4 — Add a "play the model" link + a reset-to-model button

Small but high-impact: the public page should say which checkpoint is being
served and let the visitor restart against a fresh model.

---

## PHASE 6 — Paper go-live (gate: Phases 1–4 green)

### Task 6.1 — Decide the paper's shape

**Decision D2:** `paper.md` (text generation, existing) and
`release/CONNECT_FOUR_PAPER.md` (Connect Four, new) stay **separate papers**.
The Connect Four paper is standalone and cites the earlier work as prior art.
It is NOT a chapter appended to `paper.md` — the two have different
architectures, different evaluations, and different venues.

### Task 6.2 — The Connect Four paper must be self-contained

Currently 365 lines. Before submission it needs:
- [ ] an **abstract** stating both measurements (agreement and strength) and
      the gap between them — the paper's actual contribution
- [ ] a **related-work** section (integer neural nets, GF(256) arithmetic,
      exact-solver distillation for game AI)
- [ ] the **architecture section** with exact tensor shapes and the GF(2⁸)
      operation inventory, plus the honest note that "zero FLOPs" applies to the
      *trunk*, not the whole model
- [ ] the **experiment table**: every logged evaluation (all 21 rows of the
      LABEL curve) + the 200-game strength matrix from Task 1.3
- [ ] a **negative-results section**: the six defects found and fixed
      (corrupt transcript field 126,537/320,000 rows; sidecar row-index bug
      that zeroed 41.27 % of targets; non-atomic checkpoint writes; promotion
      metric that degenerated to byte-bpb; board-blind initialisation;
      dense-lag resume `IndexError`) — this is the most interesting part
- [ ] **limitations**: holdout coverage 37.3 %, net-negative vs. the heuristic,
      single corpus, no cross-validation of the training run itself
- [ ] a **reproducibility statement** with exact commands and the artefact hash
- [ ] **figures**: (a) the LABEL/H-LABEL curve, (b) the strength ladder,
      (c) the per-column recall heatmap (c1 recall is the known weak spot)

### Task 6.3 — Venue choice

| Venue | Fits? |
|---|---|
| **arXiv** (cs.LG / cs.AI) | ✅ fastest, no peer review, no code/data requirement — **recommended first step** |
| **OpenReview** (ICLR/NeurIPS workshop) | ✅ the "teacher agreement ≠ game strength" framing is a good workshop paper |
| **Journal** (JAIR / AI Magazine) | ✅ if the follow-up research lands first |

**Recommendation:** post to arXiv **after** the GitHub repo and HF model are
public, so the paper's links resolve. Then aim at a workshop with the extended
negative-results + follow-up work.

### Task 6.4 — Generate the figures

```bash
# (a) learning curve from metrics.jsonl
# (b) strength ladder bar chart
# (c) per-column recall heatmap from perply_col_recall
```
Save to `release/figures/`, reference from the paper and the GitHub README.

---

## PHASE 7 — Marketing wave (gate: Phases 3–6 published — URLs must exist)

**Order matters.** Marketing that links to a 404 is worse than no marketing.
Each item below is blocked on a specific URL existing.

| # | Channel | Account | Blocked on | Content |
|---|---|---|---|---|
| 7.1 | **Blog post** | `ckaller.blogspot.de` | 3.3 (repo URL) | The long version: the gap between agreement and strength, the six bugs, the negative result. This is the *reference* post — everything else links to it. |
| 7.2 | **X** | `zX14_7` | 3.3 + 5.3 + 6.3 | Thread: the result, the honest caveat, the repo, the model, the playable game, the paper. |
| 7.3 | **LinkedIn** | `de.linkedin.com/in/christof-kaller-6b043733` | 7.1 | Professional framing: "I built a model that agrees with an exact solver 59 % of the time and still loses to a one-ply heuristic. Here's what that taught me about measuring AI." |
| 7.4 | **Product Hunt** | new account or existing | 3.3 + 5.3 | **"Play the model"** as the product. Tagline: *"A GF(256) integer model that learned to read a Connect Four board. It beats random. It does not beat the centre column. That's the interesting part."* Needs a 60 s GIF. |
| 7.5 | **Video** | YouTube / X / PH | 5.3 | 60–90 s scripted demo — see below. |
| 7.6 | **Reddit** | r/MachineLearning, r/Connect4 | 6.3 | A genuine "here's my model and it isn't great, here's the measurement" post. Highest technical engagement per unit effort. |
| 7.7 | **Hacker News** | — | 6.3 | Submit the paper. Expect criticism about the strength result; the negative-result framing is the defense. |
| 7.8 | **GitHub Sponsors / README CTA** | — | 3.3 | Optional. |

### Task 7.5 detail — the video

**Script (90 s):**
1. (0–10 s) Terminal: the model file, the SHA-256, the parameter count.
2. (10–25 s) The game UI. The model plays. Caption: *"It agrees with the exact
   solver on 59 % of training positions."*
3. (25–45 s) The **counter-example**: the same model vs. the centre-column
   policy, side by side, with the score. Caption: *"…and loses to a policy that
   just plays the middle column."*
4. (45–65 s) The learning curve on screen: 16 % → 59 %. Caption: *"Teacher
   agreement is not game strength."*
5. (65–80 s) The repo + model + paper links, and the honest one-liner.
6. (80–90 s) "Follow-up plan is public. If you can close the gap, I'd like to
   hear how."

**Production:** deterministic scripted match (no live randomness) via
`src/c4/duel.py`; record with a deterministic seed; use the real UI for the
gameplay shots so the video matches what a visitor sees.

### Task 7.6–7.8 — a rule for the copy

Every claim in every channel must trace to `release/RELEASE_RECORD.md`. The
social copy in `release/SOCIAL_COPY.md` currently contains the provisional
phrase *"beats random ~87 %"* — replace it with the Task 1.3 value. **No
channel publishes until the copy is reconciled.**

---

## PHASE 8 — "What else?"

### 8.1 — Hugging Face Space (the demo that runs itself)

Docker Space, CPU, `src/c4/ui_server.py` behind Gradio or the raw server. A
visitor can play **without waiting for the Netcup deployment**, and it is a
second independent proof that the artifact loads and plays on a clean machine.
Keep `--public` on so the operational routes are refused. Serve the release
artifact from the repo.

**Gate:** the Space must pass the same multi-user isolation test (Task 2.2)
against its public URL.

### 8.2 — A public evaluation endpoint

`GET /eval?position=<base64 board>` → the model's next-move distribution on a
given position, read-only, no session needed. It makes the model inspectable
and is exactly what a technical audience wants to poke at. Rate-limit it hard.

### 8.3 — Reproducibility as a first-class artifact

A `Makefile` (or `justfile`) in the public repo:

```make
install:  pip install -e .[dev]
test:     pytest tests/ -q
corpus:  python tools/build_corpus.py --out data/corpus.txt
train:   python -m src.c4.train --run-dir runs/demo --corpus data/corpus.txt --steps 2000
score:   python tmp/measure_promotion_scorer.py --corpus-run runs/demo runs/demo/checkpoints/best_hard.pt
play:    python -m src.c4.ui_server --model runs/demo/checkpoints/best_hard.pt
```

A stranger must be able to clone, install, train briefly, and play — without
asking a question. **This is the single highest-value addition to the repo
after the model card.**

### 8.4 — A published dataset (carefully)

The 320,000-game corpus is a derived artifact of the exact solver. It is large
and its ownership is the user's call. **Recommendation:** publish only a small
**held-out evaluation set** with per-ply labels (e.g. 4,000 positions, the
disjoint holdout) plus a script to regenerate the full corpus. Do not publish
the full corpus without an explicit decision — it is training data, not a
result, and it is large.

### 8.5 — Version the release

Tag the GitHub repo `v1.0.0-connect-four`. Tag the HF repo the same. The paper
cites both tags, not `main`, so the artifact is reproducible forever.

### 8.6 — A second-paper plan (the follow-up)

`release/FOLLOWUP_PLAN.md` already lays out Phases 1–3 of the follow-up
research. The natural second paper is: **why does teacher agreement not convert
into game strength?** — with decision regret under real opponent responses,
a different supervision target (outcome / policy distillation), and a
Pons-graded-target arm that is not yet trained. Publish the *plan* now as a
roadmap; the paper follows the data.

### 8.7 — A community channel

A GitHub Discussions tab with pinned seeds: "post your strongest position
against this model" — it produces exactly the kind of adversarial data the
follow-up needs.

---

## Gates summary

| Gate | Blocks | Verified by |
|---|---|---|
| **G1 Evidence** | everything | ledger reconciled; artifact hash pinned; 200-game matrix measured |
| **G2 Server** | 3, 4, 5 | sweep 24/24; isolation 26/26; sessions unit N/N; new code serving on 7860 |
| **G3 GitHub** | 6, 7 | `Ckal` active; staged tree passes all checks; pushed |
| **G4 HF** | 7 | namespace verified as Christof; model card reconciled; upload done |
| **G5 Public game** | 7 | private smoke 12/12 green; DNS confirmed by the user; flipped |
| **G6 Paper** | 7 | figures generated; abstract + negative results + limitations; arXiv posted |
| **G7 Marketing** | — | every URL resolves; every number traced to the ledger |

**G1 and G2 are the only gates I can close autonomously.** G3 needs the HF
namespace decision, G5 needs the hostname decision, and G7 needs the real URLs.

---

## Immediate next actions (in order)

| # | Action | Status |
|---|---|---|
| 1 | `tmp/test_sessions_unit.py` — run it | ✅ **47/47** — found 2 real bugs, both fixed |
| 2 | Fix the reload/new-session closure (B3) | ✅ code fixed; test case still to write |
| 3 | 200-game strength matrix for step 20000 | 🔄 running (`proc_117ae5eece0a`) |
| 4 | 200-game strength matrix for step 40000 | ⏳ after #3 |
| 5 | Reconcile every number into `release/RELEASE_RECORD.md` | ⏳ needs #3/#4 |
| 6 | Freeze the artifact + SHA-256 | ⏳ needs #5 |
| 7 | Full regression sweep against the new code | ⏳ needs #2's test case |
| 8 | Restart 7860/7861 on the new code | ⏳ needs #7 |
| 9 | Commit the session work | ✅ `b286aee` |
| 10 | Switch `gh` to `Ckal`; stage + verify the release tree | ⏳ after #7 |
| 11 | **Ask the user:** HF namespace handle + public hostname | 🔴 blocking G3/G4/G5 |

### Two questions only the user can answer

**Q1 — Hugging Face namespace.** The `ckal` HF account is **Bob Kaleugher**, a
different person. The locally authenticated account is `AlexChen1403`, which is
the wrong identity for a Christof-attributed release. Options:
(a) an existing Christof-owned HF handle, (b) register a new one (needs an email
confirmation in a browser), (c) publish privately under the authenticated
account as a labelled fallback.

**Q2 — public hostname.** Proposed `play.tensorgarten.de` → 159.195.150.113.
`*.tensorgarten.de` already resolves there and the host runs Docker + Nginx,
but the exact vhost layout must be inspected before anything is added.

---

## Answers to the questions asked

**"Is it still all running and improving?"**

*Still running:* yes — the trainer is alive at **step 41,620 / 100,000**, ~0.42
s/step, ETA ≈ 6.9 h.

*Still improving:* **no, not in a way that changes the release.** The label
agreement is flat:

```
step 18000  LABEL 59.55   H-LABEL 41.02
step 20000  LABEL 59.00   H-LABEL 38.87
step 22000  LABEL 58.52   H-LABEL 40.82
step 24000  LABEL 58.55   H-LABEL 40.43
step 26000  LABEL 58.70   H-LABEL 39.06
step 28000  LABEL 58.80   H-LABEL 40.82
step 30000  LABEL 59.40   H-LABEL 41.99
step 32000  LABEL 58.85   H-LABEL 42.19
step 34000  LABEL 58.55   H-LABEL 41.80
step 36000  LABEL 58.58   H-LABEL 37.50
step 38000  LABEL 59.33   H-LABEL 39.45
step 40000  LABEL 59.00   H-LABEL 40.82
```

Ten consecutive evaluations, no improvement. The holdout metric peaked at
**42.19 %** at step 32,000 — also flat. Against the 1-ply heuristic the model
is **net negative at every single evaluation** (best case 23-25 at step 12,000,
now 17-31).

*This is a real research finding, not a failure to report as such:* teacher
agreement saturates near 60 % on this corpus and stops converting into game
strength. That is precisely the thesis `release/FOLLOWUP_PLAN.md` was written
to investigate. The paper's contribution is the **measurement of the gap**,
not the score.

**"New version to play against?"**

| Port | Checkpoint | LABEL | Use it? |
|---|---|---|---|
| **7860** | step **18000** (promoted) | **59.55 %** | ✅ **yes — this is the best one** |
| 7861 | step 20000 | 59.00 % | ❌ worse; kept only for the documented A/B |

**There is no better new version to play.** Steps 36k/38k/40k all score *below*
step 18,000. The honest answer is: the model at `:7860` is already the one to
play, and more training is not producing a stronger opponent.

**Caveat:** both running servers predate the session-isolation refactor (they
still run the old shared-global code). They are fine for private play but must
be restarted on the new code before anything public — Task 2.3.
