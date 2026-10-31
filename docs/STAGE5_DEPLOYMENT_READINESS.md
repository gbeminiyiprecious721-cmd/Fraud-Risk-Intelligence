# Stage 5 — Final Production-Demo Readiness and Deployment Preparation

**Branch reviewed:** `render-dashboard-integration`, now at commit `<see
git log>` (Stage 5 commit, created from `8831440`).
**Currently deployed / known-working branch:** `render-demo-preparation`
at commit `700152e` (unchanged — verified below).
**Status: verification complete, deployment NOT authorized or triggered.**
Nothing in this stage was pushed, merged into `main`, merged into
`render-demo-preparation`, or deployed. This document is the gate; a
human decision is still required before anything goes live.

---

## 1. Gate-by-gate results

| # | Gate | Result | Evidence |
|---|---|---|---|
| 1 | Integration branch preserves the live demo, model artifact, artifact-delivery mechanism, Python 3.12.15 compat, and `/v1/predict` | **PASS** | `git diff render-demo-preparation HEAD` touches only `docs/DASHBOARD.md`, `src/api.py`, `src/render_demo.py`, `tests/test_dashboard_endpoint.py`, `tests/test_render_dashboard_integration.py`, `ui/dashboard.{css,html,js}`, `ui/index.html` (one line). `deploy/render_fetch_artifact.sh` and `requirements-render.txt` are **byte-identical** to the deployed branch. Line-by-line diff of `src/api.py` and `src/render_demo.py` against the deployed branch shows **zero removed lines** in `src/api.py` and only import/docstring/path-refactor removals in `src/render_demo.py` — no `/v1/predict` logic, middleware, or rate-limiting logic was altered, only added to. Local venv Python is exactly `3.12.15`, matching the documented target; no syntax requiring a newer interpreter was introduced. (The actual Render build environment itself remains unverified — this was already a disclosed, pending item in `docs/RENDER_DEPLOYMENT_PREPARATION.md` before this stage, and still is.) |
| 2 | Homepage presents the dashboard; single-transaction interface remains accessible as a separate route; navigation, static assets, API paths, same-origin requests, deep links all work | **PASS** (one deliberate, confirmed routing change) | Per explicit confirmation, `/` and `/dashboard.html` now both serve the dashboard (`src/render_demo.py::dashboard_page`); the single-transaction demo moved to `/index.html` only (`home()`). Verified live: `curl /` and `curl /dashboard.html` return byte-identical HTML with the dashboard's `<title>`; `curl /index.html` returns the single-transaction page's `<title>`. A **direct deep link** to `/index.html` (no prior visit to `/`) was tested in a real browser and the "Score this transaction" button was really clicked, returning a real score. All static assets (`dashboard.css`, `dashboard.js`, Chart.js from cdnjs) and API calls (`/health`, `/v1/dashboard/queue`, `/v1/predict`) returned 200 from the **same origin** — confirmed via `Network.responseReceived` events during a live browser session, not assumed. 37/37 tests in `tests/test_render_demo.py` + `tests/test_render_dashboard_integration.py` pass with the new routing (3 pre-existing tests from the original Render-prep stage were updated to target `/index.html`, since they tested exactly the content that moved there — not deleted, redirected to the still-correct behavior). |
| 3 | Real browser tests at desktop/tablet/mobile; all 5 sequences, every control, chart selection, sorting, KPIs, details, Kanban, reset | **PASS** | Chrome DevTools Protocol sessions against `src.render_demo:app` at 1440px (desktop), 768px (tablet), and 390px (mobile). All 5 fixed sequences exercised through the **real dropdown control** (not just `curl`), each confirming the correct seed displayed and a bounded 24-row batch. Filter dropdowns, search box, both chart click-to-filter interactions (arc geometry computed from Chart.js's own data, not guessed — see Stage 4 notes below), column-header sorting with `aria-sort`, KPI scope-reactivity, transaction detail drawer (table AND Kanban entry points), Kanban selection consistency, and the Reset-filters button (confirmed it returns the exact pre-filter row count and hides the filter-count badge) were all exercised with real DOM events. Zero console errors or exceptions in any pass. |
| 4 | KPI/chart/queue counts internally consistent with the scored batch; Stage 10 metrics clearly historical; no fabricated real-world claims | **PASS** | KPI values were cross-checked against the actual filtered/unfiltered transaction arrays in every test run (e.g. filtering to ProductCD "C" correctly dropped High-risk-transactions from 8/48 to 1/9, with the sub-label switching from "of full batch" to "of 9 filtered"). The historical panel is visually separated (dashed border, its own heading, "distinct from, and not comparable to, the live scores below" copy) and sources only already-published `README.md` figures — `docs/_stage10_final_test_evaluation_raw.json` (the actual `final_test`-named artifact) was never opened in this project. No copy anywhere claims a live fraud detection, a verified loss, a demographic fairness result, or a real investigator outcome; the Responsible-AI panel explicitly disclaims all of these. |
| 5 | API protections: synthetic-only restriction, bounded batch/seeds, rate limits, 16KB cap, information exposure, proxy attribution, artifact integrity, failure states; safe bounded local load test | **PASS** | See §2 (Security review) and §3 (Resource measurements) below for full detail and raw numbers. |
| 6 | Correct model version/checksum; scoring behavior unchanged; no retraining; `final_test` untouched | **PASS** | `/version`, `/v1/dashboard/queue`'s `model` field, and the committed manifest (`artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json`) all report the identical sha256 `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04` and `artifact_version: 1.0.0`. No training, tuning, or `final_test`-reading code exists anywhere in this stage's diff (confirmed by the same line-by-line diff review as gate 1 — `predict_fraud_probability` itself was never touched). |
| 7 | Full test suite; console; network; mobile overflow; keyboard a11y; frontend library security | **PASS** | Full fast suite: **387 passed, 0 failed** (see exact command/output in §4). Console: zero errors/exceptions across every real-browser pass this stage (desktop, tablet, mobile, both routing changes). Network: every request observed returned 2xx from the same origin. Mobile/tablet: `document.body.scrollWidth` exactly equals `document.documentElement.clientWidth` at both 390px and 768px — no horizontal overflow. Keyboard: a real `Tab`-reachable, `Enter`-activatable sort header was exercised, confirming `aria-sort` updates correctly; skip-link, `aria-expanded`/`aria-controls` on the nav toggle, and labeled form controls were re-confirmed. Frontend library: Chart.js is pinned at `4.4.1` with a cdnjs-API-sourced SRI hash; checked the GitHub Advisory Database directly (`api.github.com/advisories?ecosystem=npm&affects=chart.js`) — the only published advisory (`CVE-2020-7746`, prototype pollution) was fixed in `2.9.4`, two major versions before the pinned `4.4.1`. No advisory affects the pinned version. |
| 8 | Genuine screenshots at desktop/mobile; compared to the approved target; missing/simplified features listed honestly | **PASS** | See §5 (Visual comparison to target) below. |
| 9 | Rollback plan to known-working commit `700152e`, verified restorable | **PASS (plan verified, not executed)** | See §6 (Rollback plan) below. |

**No gate failed.** Two things are explicitly **not yet verified** and are called out as residual risk in §7, not papered over: the actual Render build environment (dependency resolution, Python interpreter match) has never been exercised against a real Render build in this project, and the private-artifact-repo fetch (`deploy/render_fetch_artifact.sh`) was last diagnosed, not re-run against a live Render deploy, in the prior Render-prep stage. Neither was touched or re-tested in this stage because doing so would require an actual Render deployment, which this stage is explicitly not authorized to trigger.

---

## 2. Security review (gate 5 detail)

- **Synthetic-only input restriction on `/v1/predict`**: unchanged, re-verified — an arbitrary payload still returns `403`, only the fixed fixture scores.
- **Bounded batch size / seed allowlist on the public dashboard endpoint**: `PUBLIC_DASHBOARD_BATCH_SIZE = 24` (client-supplied `n` is ignored outright); `seed` only ever selects into `PUBLIC_DASHBOARD_SEED_ALLOWLIST = (101, 202, 303, 404, 505)` — re-confirmed this round via `curl` with out-of-range and negative seeds, always resolving to one of the 5 allowlisted values.
- **Rate limits**: re-confirmed *actively engaging*, not just present in code — see §3's load test, where combined test traffic within the same ~60s window correctly triggered `429`s on both `/v1/predict` (20/min per client, 120/min global) and `/v1/dashboard/queue` (30/min per client, 180/min global), with `/health` and `/version` confirmed never rate-limited.
- **16KB request-body cap on `/v1/predict`**: unchanged, test-covered (`test_oversized_request_body_is_rejected_with_413` in the local suite; the public-demo body-size branch of `synthetic_only_public_demo` untouched this stage).
- **Information exposure**: error responses were spot-checked this stage — `/v1/dashboard/queue` and `/v1/predict` failure paths return only a generic `detail` string and an `error_category` enum value (e.g. `ArtifactUnavailable`), never a stack trace, file path, or internal exception message. No new log statement added this stage prints a request body, header, or credential.
- **Proxy attribution (`X-Forwarded-For`)**: unchanged from the prior stage's disclosed limitation — Render's exact header semantics behind its proxy remain unverified against a live deployment; the global rate-limit backstop exists specifically to remain protective even if per-client attribution is wrong.
- **Artifact integrity**: unchanged — `load_artifact` still refuses to serve on a checksum mismatch, verified by the existing `test_queue_returns_503_when_artifact_unavailable`-style tests plus a fresh manual check this stage (§1, gate 6).
- **Failure states**: `/v1/dashboard/queue` on both the local and public app correctly returns `503` with `error_category: "ArtifactUnavailable"` when the artifact isn't loaded (re-verified by temporarily nulling `app.state.artifact` in a test, matching the existing pattern already used for `/v1/predict`).

---

## 3. Resource measurements (local, bounded, safe — never against a real deployed URL)

No Render URL exists for this project to test against, and the task
explicitly prohibits stress-testing a public endpoint irresponsibly.
Every measurement below is against **this machine's own local
`src.render_demo:app` process**, run with the exact same code that would
run on Render.

**Process memory**, freshly started, before vs. after 25 real (HTTP 200)
`/v1/dashboard/queue` requests (each one a genuine RandomForest inference
over a 24-row batch), via `ps -o rss=`:

| | RSS |
|---|---|
| Baseline (process just started) | 205.3 MB |
| After 25 sequential real requests | 209.2 MB |
| **Growth** | **+3.9 MB over 25 requests (~1,200 scored rows)** |

Render Free's limit is 512 MB; this leaves roughly **300 MB of headroom**
even before any growth, and the growth pattern (a few MB, not a
per-request multiple) is consistent with normal, bounded Python/glibc
heap behavior, not a leak. This was not run long enough to rule out a
*very* slow leak over hours of uptime — stated as a residual risk, not
claimed as proven.

**Concurrent load** (separate run, 10 concurrent workers, 60 requests per
endpoint — this is what triggered the rate-limit confirmation above,
since it ran in the same ~60s window as other same-stage testing against
the same long-running process):

| Endpoint | 200s | 429s | Mean latency (200s) | Max latency |
|---|---|---|---|---|
| `/v1/dashboard/queue` | 22 | 38 | 168 ms | 538 ms |
| `/v1/predict` | 19 | 41 | 144 ms | 459 ms |

No request errored (5xx) or timed out under this load. Sub-second
latency even at peak concurrency, on ordinary laptop hardware, suggests
no obvious performance cliff at a volume far beyond what a portfolio demo
would realistically see.

---

## 4. Full test suite

```
.venv/bin/python -m pytest -q -m "not slow"
...
387 passed, 5 warnings in 141.58s
```

387 = 386 (end of Stage 4) + 1 net new test from the homepage-routing
split (one old alias test removed, two new routing tests added, three
pre-existing single-transaction-page tests renamed to target their new
correct path — not deleted). Zero failures, zero skips beyond the
pre-existing `-m "not slow"` deselection.

---

## 5. Visual comparison to the approved target

Screenshots this round: full desktop (1440px) and mobile (390px) passes
of the dashboard now served at `/`, plus a dedicated tablet (768px)
capture, all against the live `src.render_demo:app` process.

**Present and matching the target's major visible sections:** dark
premium theme; colorful, icon-coded KPI cards; coordinated filter
controls with a reset; an investigation table; linked/clickable charts;
a transaction detail drawer; a four-column Kanban workflow board;
left navigation.

**Intentionally simplified or different from the original reference
mockup, listed plainly (not hidden):**

- **No real date-range picker.** The reference mockup showed a calendar
  date range ("Jan 1 – Jan 31, 2024"). This batch has no real elapsed-
  time dimension (only synthetic queue order), so a working calendar
  control was deliberately not built — it would have been an interactive
  element with no real data behind it. The "Demo batch" sequence
  selector replaces it as the header's temporal-feeling control, and it
  is real and functional.
- **No "Review Capacity" / reviewer-workload KPI.** The mockup's
  dashboard showed a reviewer-capacity metric ("85/100 utilized"). This
  project has no real reviewer-capacity data, so no such KPI was added —
  inventing one would be exactly the kind of fabricated metric this
  stage's instructions forbid.
- **No period-over-period trend arrows** ("↑ 12.4% vs previous period")
  on the KPI cards. There is no real previous period to compare against.
- **Kanban board is read-only**, not drag-and-drop — there is no backend
  write path for transaction status, and a fake local-only reorder would
  misrepresent a real workflow action. Confirmed again this stage: no
  `draggable`/`drag*` code exists anywhere in `ui/dashboard.html` or
  `ui/dashboard.js`.
- **No live/per-transaction SHAP explainability.** The detail drawer's
  "feature highlights" are explicitly labeled as the model's global
  (Stage 11) importance ranking, not a per-prediction explanation, since
  no live explainability was ever implemented for serving.
- **No genuine historical trend chart** (e.g., a real "fraud over time"
  line spanning weeks/months) — "Risk score by queue order" shows
  sequence within the current batch only, labeled as such.

No backend capability was invented to make the page look more complete
than it actually is; every one of the above is a disclosed, deliberate
scope boundary, not an oversight.

---

## 6. Rollback plan to the known-working commit (`700152e`)

**Nothing below has been executed.** This is the verified, written
procedure for the human operator to use if a deployment from
`render-dashboard-integration` ever needs to be reverted.

**Precondition check (already true today, re-verified this stage):**
```
git diff render-demo-preparation origin/render-demo-preparation --stat   # must be empty
```
Confirmed empty — the branch actually live on Render (if any deploy from
it has happened) and its local copy are identical, and neither has been
touched by any of this dashboard work.

**If Render is configured to auto-deploy from a branch, and that branch
is ever pointed at `render-dashboard-integration` or a descendant of it,
and the deployment needs to be rolled back:**

1. **Fastest path — Render's own rollback, if available on the plan in
   use:** in the Render dashboard, open the service's Deploys tab and
   redeploy the last known-good deploy (the one built from `700152e`).
   This is the lowest-risk option because it doesn't touch git state at
   all — confirm this is available before using the git-level steps
   below.
2. **Git-level rollback (if the service deploys from a branch ref and
   that ref needs to move back):**
   ```
   git fetch origin
   git checkout render-demo-preparation
   git reset --hard 700152e   # only if render-demo-preparation itself was advanced past 700152e -- it has NOT been, as verified above
   ```
   Given the precondition check above already confirms
   `render-demo-preparation` is still exactly at `700152e`, this step is
   a no-op today — it's documented for the future case where that branch
   *has* moved and needs to be put back.
3. **If Render is instead pointed directly at a commit SHA or at
   `render-dashboard-integration`**, the rollback is simply repointing
   Render's deploy source back to `render-demo-preparation` /
   `700152e` in the Render dashboard's service settings, then triggering
   a manual deploy from that ref. No force-push, no history rewrite, and
   no secret rotation is required for this rollback, since `700152e`
   never contained the dashboard work and nothing about its artifact-
   delivery secrets changes between the two.
4. **Verify the rollback**: `curl https://<service>.onrender.com/health`
   should report `artifact_loaded: true`, and `curl
   https://<service>.onrender.com/version` should report the same
   `artifact_sha256` as today (`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`)
   — the model artifact and its delivery mechanism are identical on both
   branches, so this should never change across a rollback.

No destructive git operation (`--force` push, history rewrite) is part
of this plan under any branch state this project is actually in today.

---

## 7. Remaining risks (not gate failures, disclosed for the deployment decision)

1. **The actual Render build has never been exercised in this project.**
   Dependency resolution and Python-version match on Render's real build
   image remain unverified — this was already flagged as "pending" in
   `docs/RENDER_DEPLOYMENT_PREPARATION.md` before this stage and still
   is; nothing in this stage could close that gap without actually
   deploying, which is outside this stage's authorization.
2. **`deploy/render_fetch_artifact.sh`'s 403-handling fix (prior round)
   was never re-confirmed against a live GitHub rate-limit event or a
   live Render build** — only against local mocks and a real (non-rate-
   limited) failure case, as disclosed at the time.
3. **Memory growth was only measured over 25 requests in a few seconds**,
   not sustained hours-long uptime; a slow leak over a much longer window
   can't be ruled out by this test, though nothing in the request path
   (stateless synthetic-row generation + scikit-learn inference) suggests
   one.
4. **`X-Forwarded-For` attribution behind Render's actual proxy remains
   unverified.** The global rate-limit backstop is specifically designed
   to stay protective even if this is wrong, but the *per-client*
   fairness of rate limiting on Render specifically is unconfirmed.
5. **The doughnut chart's thinnest slice remains a harder click target**
   than the others, mitigated (not eliminated) by the Stage 4 legend-click
   fix.

None of these are new to this stage, and none were found to have gotten
worse — they're restated here, explicitly, so the deployment decision is
made with the real, current picture rather than an optimistic one.

---

## 8. Exact safe deployment steps (for if/when authorized)

Not executed. Provided so the next authorized step is unambiguous:

1. Confirm the Stage 5 commit hash on `render-dashboard-integration`
   (`git log --oneline -1`).
2. Push the branch: `git push origin render-dashboard-integration`
   (creates/updates the remote branch; does **not** affect `main` or
   `render-demo-preparation`).
3. In the Render dashboard, either point the existing service at
   `render-dashboard-integration` for a trial deploy, or (lower-risk)
   create a **second, separate** Render service from this branch so the
   currently-live service on `render-demo-preparation`/`700152e` is
   never touched during evaluation.
4. After a successful build, verify `/health`, `/version`, `/`,
   `/index.html`, `/dashboard.html`, and `/v1/dashboard/queue` on the
   real `*.onrender.com` URL before considering it ready.
5. Only after that verification would merging
   `render-dashboard-integration` into `render-demo-preparation` (or
   repointing the live service) be the next decision — and that merge
   itself is a separate, explicit authorization this stage does not
   grant.
