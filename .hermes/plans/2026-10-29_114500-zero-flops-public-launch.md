# ZeroFLops Public Launch — Execution Plan

**Goal:** Take the GF(256) Connect Four release from "code and weights are
published" to "a stranger can read it, play it, and see the honest numbers —
with every claim traceable."

**Status at plan time (2026-10-29 ~11:40 CEST):**

| Item | State |
|---|---|
| GitHub `Ckal/connect-four-gf256` | **LIVE**, public, verified via API |
| HuggingFace `Chris4K/connect-four-gf256` | **LIVE**, public, SHA-256 re-verified after download |
| Model checkpoint | Step 18,000 hard export, `134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2` |
| Local playable server | **RUNNING** on `127.0.0.1:7860`, session isolation verified with 2 real HTTP clients |
| Public deployment `c4.tensorgarten.de` | Compose service wired on the Netcup host, **image not yet built** |
| Regression sweep | Re-running after the reboot — 10 PASS / 0 FAIL at time of writing |
| V24 trainer | Killed by the reboot at step 58,000. **Nothing lost** — see §1.1 |
| Second paper (`GF256_LANGUAGE_MODEL_PAPER.md`) | Written, needs review + publication |

---

## 0. Constraints (non-negotiable)

1. **Never claim an unfinished action happened.** No "posted to X" until X
   actually accepted the post. No "deployed" until the public URL returns
   200 and the session-isolation smoke test passes against the *deployed*
   process.
2. **Public attribution is `Christof Ckal`.** Never Alex. The release
   staging script enforces this (`tmp/stage_public_repo.sh`).
3. **No credentials, tokens, keys or connection strings** in any committed
   or published file. Replace with `[REDACTED]`.
4. **Do not delete anything.** Archived probe scripts stay on disk in
   `%LOCALAPPDATA%\Temp\zeroflops-probe-archive-20261029\` (71 files, 702 KB)
   and the private corpus stays at `data/corpus_c4_bytes.txt`. "Clean" means
   *gitignore + move*, never `rm`.
5. **Every published number** must exist in `release/RELEASE_RECORD.md` with
   its command and artifact. If a number is not in the ledger, it is not
   publishable.
6. **The private corpus is never published.** Only the checkpoint ships.

---

## 1. What the reboot changed (read first)

### 1.1 The V24 training run was NOT lost

The trainer reached step 58,000 before the reboot. The label-agreement
curve had already **plateaued and turned down**:

```
step 14000  57.55
step 18000  59.55   <- peak, this is the release artifact
step 30000  59.40
step 44000  57.85
step 58000  57.63   <- killed by reboot
```

The strength matrix independently confirms step 18,000 is the best
checkpoint: step 20,000 scored worse on **every** opponent arm
(84.67 / 86.67 / 30.17 / 24.00 / 41.33 % vs 85.17 / 90.50 / 37.17 / 24.33 /
41.83 %). Continuing to step 58,000 could not have improved the release.

**Decision:** do not resume training. Ship step 18,000. The run's purpose
was to answer "does more training help?" and the answer is recorded: **no**.

### 1.2 What did NOT survive the reboot

- The V24 trainer process (intentionally not resumed).
- The earlier regression sweep (re-running now).
- The two pre-refactor UI servers on ports 7860/7861 — **this was a real
  hazard**: they were running the *old* shared-board code. The current 7860
  server is a fresh process from the current tree, so its behaviour is
  evidence about the new code.

### 1.3 The failure that was found and fixed

`tmp/test_ui_api.py` hard-coded `--port 7860` and ignored the `UI_PORT`
environment variable that `tmp/run_regression_sweep.sh` exports. The sweep
was therefore testing **the wrong server** — a stale pre-refactor process —
and the failures looked like product bugs.

**Fix:** `tmp/test_ui_api.py` now defaults to `int(os.environ.get("UI_PORT",
"7860"))`. Verified: the test now reports **ALL PASS** against a fresh
session-aware server on port 7903.

---

## 2. Task list

### Task 1: Commit the current work (no deletion)

**Objective:** get the cleanup, the second paper, the README fixes and the
`test_ui_api.py` port fix into git history.

**Files:**
- Modify: `.gitignore` — private corpora + loose probe scripts
- Modify: `tmp/test_ui_api.py` — read `UI_PORT` from the environment
- Modify: `release/README.md` — real HF URL, `--model` not `--checkpoint`,
  dead `Ckal/zeroflops` link → `GF256_LANGUAGE_MODEL_PAPER.md`
- Modify: `release/SOCIAL_COPY.md` — corrected strength matrix, real URLs,
  the "100k steps" claim replaced with the honest 18,000 plateau statement
- Create: `release/GF256_LANGUAGE_MODEL_PAPER.md` — the second paper
- Create: `.hermes/plans/2026-10-29_114500-zero-flops-public-launch.md`
  (this file)

**Step 1: verify nothing large or private is staged**

```bash
cd C:/Users/Chris4K/projects/ZeroFLops-paper
git status --porcelain
git check-ignore -v data/corpus_c4_bytes.txt
```

Expected: `data/corpus_c4_bytes.txt` matched by a `.gitignore` rule.

**Step 2: stage and inspect**

```bash
git add .gitignore tmp/test_ui_api.py release/ .hermes/plans/2026-10-29_114500-zero-flops-public-launch.md
git diff --cached --stat
```

Expected: no `.pt` file, no `data/` entry, no file over ~40 KB except the
paper.

**Step 3: commit**

```bash
git commit -m "docs(c4): second paper, corrected README links, gitignore the private corpus

- Add release/GF256_LANGUAGE_MODEL_PAPER.md: the language-model paper as a
  standalone, publication-ready manuscript. The build narrative is removed;
  the technical content is unchanged.
- README: real HuggingFace URL, --model (not --checkpoint) for ui_server,
  and the dead Ckal/zeroflops link replaced by the second paper.
- SOCIAL_COPY: corrected strength matrix and the 100k-step claim.
- .gitignore: the private 27 MB corpus and 292 loose probe scripts.
- tmp/test_ui_api.py: read UI_PORT so the sweep tests the server it starts."
```

**Step 4: verify**

```bash
git log --oneline -3
git status --short          # expect: clean except untracked plan
```

---

### Task 2: Finish the public deployment on `c4.tensorgarten.de`

**Objective:** a public, session-safe, playable Connect Four at a stable
URL under the user's own domain.

**Current state:** the `c4-game` service is appended to
`/opt/netcup-infra/docker-compose.yml` and `docker compose config`
validates. The image is **not built** and the service is **not running**.
`DOMAIN_C4=c4.tensorgarten.de` is in `/opt/netcup-infra/.env`.

**Why `c4.tensorgarten.de`:** the wildcard DNS `*.tensorgarten.de →
159.195.150.113` is already live (verified via the Google DNS-over-HTTPS
resolver). `https://c4.tensorgarten.de/` currently returns **503** —
nginx-proxy is up with no backend. No DNS decision is needed.

**Step 1: build the image**

```bash
ssh -i ~/.ssh/netcup_deploy_ed25519 root@159.195.150.113 \
  "cd /opt/netcup-infra && docker compose build c4-game"
```

The Dockerfile is at `/opt/netcup-infra/apps/c4-game/app/Dockerfile`:
`python:3.12-slim` + `torch==2.5.1+cpu` + `numpy==1.26.4`. CPU-only is
sufficient — the GF(256) trunk is integer arithmetic and the float32
read-out needs no GPU.

**Step 2: start it**

```bash
ssh -i ~/.ssh/netcup_deploy_ed25519 root@159.195.150.113 \
  "cd /opt/netcup-infra && docker compose up -d c4-game && docker compose logs -f c4-game"
```

Expected: the container is `healthy` (the healthcheck hits `/state`).

**Step 3: verify the checkpoint on the server is the published one**

```bash
ssh -i ~/.ssh/netcup_deploy_ed25519 root@159.195.150.113 \
  "sha256sum /opt/netcup-infra/apps/c4-game/app/model/best_hard.pt"
```

Expected: `134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2`.
Already verified once — re-check after the rebuild.

**Step 4: verify the public URL**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://c4.tensorgarten.de/
```

Expected: **200** (not 503). If 503, nginx-proxy has not picked up the
backend: check `docker logs netcup-infra-nginx-proxy-1` for the upstream
resolution.

**Step 5: session-isolation smoke test against the DEPLOYED process**

This is the gate. A local 200 is not evidence of a safe public deploy.
Run the two-client isolation check against the public URL:

```bash
./.venv-c4/Scripts/python.exe tmp/verify_public_isolation.py https://c4.tensorgarten.de
```

Assertions (from the `public-interactive-demo-server` skill):
- Client A and client B get **different** session cookies.
- A move in A is **invisible** in B.
- The cookie carries `HttpOnly` and `Secure`.
- `/modelinfo` and `/reload` return **403** (public mode disables them).
- Security headers present: CSP, `X-Frame-Options`, `X-Content-Type-Options`,
  `Referrer-Policy`.

**Step 6: check the cert**

```bash
curl -sI https://c4.tensorgarten.de/ | grep -i "strict-transport\|HTTP/"
```

Expected: HTTP/2 and `strict-transport-security`. acme-companion issues the
cert via the `LETSENCRYPT_HOST` variable already set on the service.

**Step 7: update the ledger**

Record in `release/RELEASE_RECORD.md`: the public URL, the date, the
isolation-test result, and the checkpoint SHA-256 served. Only then is the
deployment claim publishable.

---

### Task 3: Publish the second paper to GitHub and HuggingFace

**Objective:** the language-model paper becomes a first-class public
artifact instead of an orphan file in the working tree.

**Step 1: read it before publishing**

```bash
cd C:/Users/Chris4K/projects/ZeroFLops-paper
wc -l release/GF256_LANGUAGE_MODEL_PAPER.md
grep -nE "TODO|TBD|coming soon|XXX|placeholder" release/GF256_LANGUAGE_MODEL_PAPER.md
```

Expect: no matches for the open markers. Code fences must be even in
number (currently 66).

**Step 2: add it to the public repo**

The public repo is a *clean-history* repo staged by
`tmp/stage_public_repo.sh`. The second paper must be added to the staging
script's file list:

```
release/README.md RELEASE_RECORD.md CONNECT_FOUR_PAPER.md
GF256_LANGUAGE_MODEL_PAPER.md FOLLOWUP_PLAN.md MODEL_CARD.md
SOCIAL_COPY.md LICENSE
```

**Step 3: decide the destination**

Two options, both valid:
- **(a) Same repo** — `Ckal/connect-four-gf256` gains the second paper.
  Simplest; the repo is already about the GF(256) programme.
- **(b) Separate repo** — `Ckal/gf256-lagrange-field-transformer` for the
  language-model paper alone. Cleaner separation; the two papers share an
  architecture but not their findings.

**Recommendation: (b).** The two works are explicitly separate in every
public document ("shares the architecture, not the findings"). A reader
arriving at a Connect Four repo should not have to wonder which paper they
are reading. The staging script already supports it — point it at a
different target dir and a different file list.

**Step 4: publish**

```bash
bash tmp/stage_public_repo.sh /tmp/gf256-lm-repo
cd /tmp/gf256-lm-repo
git init -b main && git add -A && git commit -m "feat: GF(256) Lagrange Field Transformer paper"
gh auth switch --user Ckal
gh repo create Ckal/gf256-lagrange-field-transformer --public --source=. --remote=origin --push
```

The script aborts on: files over 40 MB, credential patterns, remaining local
user paths, and a missing "Christof Ckal" attribution. All four gates must
pass.

**Step 5: mirror the model artifact to HF**

The 118 MB `artifacts/gf256_hard_model.pt` exceeds nothing on HF (which
allows far larger), but it was **deliberately not uploaded** — it is
currently only in the private working repo. Decide explicitly:

- Upload it to a `Chris4K/gf256-lagrange-field-transformer` model repo, or
- Ship code + paper only and let readers retrain.

**Recommendation: upload it.** The artifact is the result; a paper whose
weights are unavailable is weaker than one whose weights are a 118 MB
download away. Check the disk headroom on the HF side first.

---

### Task 4: Deploy the paper as the project README

**Objective:** `Ckal/connect-four-gf256`'s README should read like a
publication, not like a work log.

Current state: the README is already strong — honest TL;DR table, the
convergence curve, "What the model did not learn", the bug table, the
reproducibility table, limitations, and a BibTeX citation. It needs:

- A **"Play it"** link at the top once `c4.tensorgarten.de` is live.
- The `BEST_HARD` SHA-256 next to the download command (it is already
  there).
- The release date and the artifact version.

---

### Task 5: Marketing — the actual public launch

**Objective:** put the work in front of people, in the right order, with
honest claims.

**Ordering matters.** Publish the playable URL *before* announcing it, and
announce the *finding*, not the model.

**The hook is the negative result.** "A model that agrees with an exact
solver 59.6 % of the time still loses to a three-line heuristic" is a
better story than "I trained a Connect Four model". Lead with the gap.

**5.1 X** — `https://x.com/zX14_7`

Prerequisite: **no stored X session exists** (`xurl` is not installed, no
config). The user will need to authenticate interactively. Status: blocked
on the user's action.

Copy is ready in `release/SOCIAL_COPY.md` — already corrected for the
strength matrix and the step-18,000 plateau.

**5.2 Blog** — `ckaller.blogspot.de`

Prerequisite: a Blogger token exists at
`~/AppData/Local/hermes/google_token.json`. The `blogger` skill covers the
API. Write the post as a technical write-up of the central finding, then
link the repo, the model and the playable URL.

**5.3 LinkedIn** — `de.linkedin.com/in/christof-kaller-6b043733`

Profile found. Posting needs a session. Status: blocked on the user's
action.

**5.4 Product Hunt**

Needs a dedicated landing page and an account decision. **Recommendation:
defer.** A Product Hunt launch of a research artifact whose central claim is
a negative result is a poor fit for the platform's format, and it needs
real-time community engagement to succeed. Do it only if the user wants it
specifically.

**5.5 Demo video**

**Recommendation: defer until the public URL is live.** A short screen
recording of a real game against the deployed model is worth more than a
synthetic animation, and it needs the public URL anyway.

---

### Task 6: Close the loop on the trainer

**Objective:** decide what to do with a killed run, and record it.

- **Do not resume.** The curve says more training does not help.
- Record in `release/RELEASE_RECORD.md` that the run was terminated by a
  host reboot at step 58,000, that the peak was step 18,000, and that the
  termination is *not* a confound for any published number — because every
  published number comes from the step-18,000 checkpoint, which was written
  long before the reboot.
- If a future run is wanted, it must start from scratch with a **different
  hypothesis** (e.g. value-targeting instead of teacher-agreement), not with
  more steps of the same recipe.

---

## 3. Files likely to change

| File | Change |
|---|---|
| `.gitignore` | private corpora + loose probe scripts |
| `tmp/test_ui_api.py` | read `UI_PORT` from env |
| `release/README.md` | real URLs, `--model`, live link |
| `release/SOCIAL_COPY.md` | corrected matrix, real URLs |
| `release/GF256_LANGUAGE_MODEL_PAPER.md` | **new** — second paper |
| `release/RELEASE_RECORD.md` | deployment status, run termination |
| `release/CONNECT_FOUR_PAPER.md` | live URL reference |
| `tmp/stage_public_repo.sh` | add the second paper to the file list |
| `tmp/verify_public_isolation.py` | **new** — public-deploy gate |
| `/opt/netcup-infra/docker-compose.yml` | *(host)* `c4-game` service |
| `/opt/netcup-infra/.env` | *(host)* `DOMAIN_C4` |

## 4. Validation

| Check | Command | Expected |
|---|---|---|
| Private corpus not tracked | `git ls-files data/` | empty |
| Nothing large staged | `git diff --cached --stat` | no `.pt`, no >40 KB |
| Sweep green | `bash tmp/run_regression_sweep.sh` | 24 PASS, 0 FAIL |
| Local session isolation | 2 clients, separate cookie jars | boards independent |
| Public game up | `curl -o /dev/null -w '%{http_code}' https://c4.tensorgarten.de/` | 200 |
| Public isolation | `tmp/verify_public_isolation.py https://c4.tensorgarten.de` | all assertions pass |
| Admin routes closed | `curl -o /dev/null -w '%{http_code}' https://c4.tensorgarten.de/modelinfo` | 403 |
| Checkpoint identity | `sha256sum` on the server | `13403563…72e2` |
| GH repo public | `gh api repos/Ckal/connect-four-gf256` | `private: false` |
| HF model public | `GET huggingface.co/api/models/Chris4K/connect-four-gf256` | 200, anon |

## 5. Risks and tradeoffs

| Risk | Mitigation |
|---|---|
| The public deploy leaks boards between visitors | The isolation gate (Task 2, step 5) is **blocking**. No announcement until it passes. |
| A stale process on 7860/7861 gets mistaken for the new code | Cite only processes started from the current tree. The old ones were killed by the reboot. |
| The `c4-game` compose service breaks the host stack | A backup of the compose file exists (`docker-compose.yml.bak.c4-<ts>`). `docker compose config` validates before `up`. |
| The private corpus leaks into a public commit | `.gitignore` + the staging script's own checks + a manual `git ls-files data/` before pushing. |
| The second paper contradicts the Connect Four paper | They are separate works sharing an architecture. Both documents already say so explicitly. |
| Marketing posts claim the model is strong | `release/SOCIAL_COPY.md` forbids it and states the negative result. The central claim is the *gap*. |

## 6. Open questions for the user

1. **Second paper: same repo or separate repo?** Recommendation: separate
   (`Ckal/gf256-lagrange-field-transformer`).
2. **Upload the 118 MB language-model artifact to HuggingFace?** Recommendation:
   yes.
3. **Product Hunt — actually wanted?** Recommendation: defer.
4. **Demo video — after the public URL is live?** Recommendation: yes, defer.
5. **X and LinkedIn need your interactive login.** No stored session exists.
   I can drive the browser once you are logged in.
