# Investigation Dashboard — what it is and isn't

`ui/dashboard.html` is an additive presentation-layer upgrade on top of the
existing Stage 14 API (`src/api.py`) and Stage 15 single-transaction demo
(`ui/index.html`). It is **not** part of the project's approved Stage 0–19
ML workflow (see `docs/ML_WORKFLOW_STATE.md`, closed/approved) — no model
logic, evaluation, or responsible-AI finding was touched, re-run, or
re-validated to build it. It was built the same way the Render deployment
prep work was: as a separate, additive task, out of the formal roadmap.

## What's real

- **The model is real.** Every risk score shown is produced by calling the
  same, unmodified `predict_fraud_probability` the single-transaction demo
  and the production API use, against the same frozen, checksummed
  artifact (`artifacts/frozen_rf_fraud_risk_v1.joblib`, sha256
  `2fb0cbc4457e...`). Nothing about the model, its preprocessing, or its
  training was changed or re-run.
- **The global feature-importance ranking is real.** The "feature
  highlights" shown in the transaction detail drawer are the top-15
  entries of Stage 11's own `global_ranking_by_original_column_top15`
  (`docs/_stage11_explainability_raw.json`), copied verbatim into
  `src/api.py` as `STAGE11_GLOBAL_FEATURE_IMPORTANCE`. It is never
  recomputed by the dashboard.
- **The risk bands are honest about what they are.** Low/Medium/High/
  Critical are computed as quantiles (60th/85th/97th percentile) of each
  batch's own score distribution, returned to the client as
  `risk_band_quantiles` — never a fixed, absolute threshold. This matches
  the project's standing position that the score is a *relative ranking
  signal*, not a calibrated probability.
- **The single-transaction flow is unchanged.** `/v1/predict` and
  `ui/index.html` were not modified in their behavior (one footer link was
  added to `index.html`; see `tests/test_dashboard_endpoint.py::
  test_single_transaction_predict_endpoint_is_unaffected` and the full
  existing `tests/test_api.py` suite, both passing unchanged).

## What's illustrative / synthetic

- **Every transaction row is fabricated.** `GET /v1/dashboard/queue`
  generates `n` rows from the same neutral baseline as the existing
  `SYNTHETIC_TRANSACTION` fixture (392 raw columns, mostly pinned at a
  placeholder value), then varies a deliberately small set of fields —
  `TransactionAmt`, `ProductCD`, `card4`/`card6`, `P_emaildomain`, and the
  ~14 numeric columns from Stage 11's own top-15 importance ranking — using
  a seeded `random.Random`. None of these values are drawn from, or
  modeled on, any real or training dataset. The response is marked
  `"synthetic": true` and every transaction ID is prefixed `TXN-DEMO-`.
- **The investigation `status` field (New/In Review/Escalated/Resolved) is
  a display-only construct.** It carries no scientific content, is never
  derived from the risk score, and exists purely so the queue reads like a
  live workflow.
- **The "feature highlights" in the detail drawer are global, not
  per-transaction.** No SHAP or other live, per-prediction explainability
  was implemented for serving in this project, so none is faked here. The
  drawer is explicit in its own copy that this is a model-level ranking,
  shown for context, not an explanation of that specific score.
- **The risk-score distribution chart auto-ranges to the batch's observed
  min/max**, rather than a fixed 0–1 axis, so its shape stays legible even
  when a batch's scores cluster narrowly (as they typically do here, since
  most of the 392 raw columns are held at a fixed neutral baseline and only
  a handful of fields vary per synthetic row). The chart's caption states
  the real observed range for whichever batch is currently displayed.

## Architecture notes

- The new endpoint (`GET /v1/dashboard/queue`) is attached to the
  module-level `app` object in `src/api.py` **after** `create_app()`
  returns — not inside the factory function. `src/render_demo.py` (on the
  separate `render-demo-preparation` branch, not yet merged to `main`)
  builds its own, separately rate-limited and size-limited public app by
  calling `create_app()` itself; a route added inside the factory would
  have silently also appeared, unrestricted, on that public service. This
  keeps the dashboard endpoint local-only by construction. It has not been
  verified by running both apps side-by-side, since `render_demo.py` does
  not exist on this branch (built from `main`).
- CORS is unchanged — `ALLOWED_UI_ORIGINS` already covered
  `http://127.0.0.1:8080` / `http://localhost:8080`, which is where
  `ui/dashboard.html` is served from in the documented local setup.
- Chart.js 4.4.1 is loaded from cdnjs with a Subresource Integrity hash
  fetched from cdnjs's own API at implementation time (not guessed).

## Running it

```
.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000
python3 -m http.server 8080 --directory ui
```

Then open `http://127.0.0.1:8080/dashboard.html`.

## Verified this round

- `tests/test_dashboard_endpoint.py` (10 new tests): response shape, score
  bounds, determinism by seed, non-determinism across seeds, `n` clamping,
  quantile-band correctness (highest-scoring transaction always gets the
  worst available band), the global-importance payload is stable across
  seeds, real artifact version/checksum reporting, a 503 when the artifact
  is unavailable, and a regression check that `/health` still works.
- Full fast suite including the new file: 350 passed, 0 failed (10 of
  which are the new dashboard-endpoint tests).
- Real end-to-end verification against a locally running server
  (`uvicorn` + `python3 -m http.server`), not just unit tests: `curl`
  against `/v1/dashboard/queue` and `/v1/predict`; a CORS check from the
  documented origin; a headless-Chrome screenshot of the rendered
  dashboard; and a real DOM click (via the Chrome DevTools Protocol, not a
  simulated function call) that opened the detail drawer, confirmed its
  contents matched the clicked row, exercised the risk-band filter and
  transaction-ID search, and confirmed the Refresh button fetches a new,
  differently-seeded batch — with zero console errors or exceptions
  throughout.
- Not independently re-verified: Subresource Integrity protects against a
  compromised cdnjs response, but the hash was fetched from cdnjs's own
  API rather than verified against an independent source.

## Render integration (public-facing variant)

`src/render_demo.py` now also serves the dashboard, at the same hosted
HTTPS origin as the single-transaction demo. It is **not** the same
endpoint behavior as the local-only version above — the public version is
deliberately more restricted, mirroring how `/v1/predict` is already
restricted to its one fixed fixture on that same service.

### What's different on the public deployment

- **`GET /v1/dashboard/queue` is bounded and deterministic.** It accepts
  the same `n`/`seed` query parameters the local version does (so
  `ui/dashboard.js` needs no awareness of which backend it's talking to),
  but neither is used the way the local route uses them:
  - `n` is **ignored outright** — the public batch is always exactly
    `PUBLIC_DASHBOARD_BATCH_SIZE` (24) transactions, never caller-chosen,
    to bound the inference cost of every public request.
  - `seed` only ever **selects an index** (via modulo) into the small,
    fixed `PUBLIC_DASHBOARD_SEED_ALLOWLIST` (5 values) — it is never
    passed through as the actual RNG seed. The full set of possible
    public responses is therefore finite, fixed, and reviewable ahead of
    time, regardless of what a caller sends. The Refresh button still
    works (cycles through those 5 fixed batches) but will not produce
    arbitrary-looking output the way the local dashboard's arbitrary-seed
    version can.
- **It has its own rate limiting**, separate from `/v1/predict`'s
  (`_dashboard_rate_limit_by_client`/`_dashboard_rate_limit_global`, 30
  per client / 180 global per 60s window) — generous enough for one page
  load plus a few Refresh clicks, but independent so a burst of dashboard
  traffic can never starve the single-transaction demo's budget, or vice
  versa.
- **Static assets are served from the same origin.** `GET /dashboard.html`,
  `/dashboard.css`, and `/dashboard.js` are new routes on the Render app.
  `/index.html` is now also a route (alias for `/`), so the two pages'
  existing relative links to each other (`dashboard.html` ↔ `index.html`)
  resolve correctly once both are hosted. `ui/dashboard.js` itself needed
  one change to work unmodified in both contexts: it now resolves its API
  base from `window.location.origin` whenever it is *not* running on the
  documented local dev split-origin setup (`127.0.0.1:8080`), rather than
  hardcoding the local API port — the same problem `render_demo.py`
  already solved for `index.html`'s inlined `API_BASE`, solved here
  without server-side text injection since `dashboard.js` is a plain
  static file.

### What's unchanged

- `/v1/predict`'s fixed-fixture restriction, rate limiting, and
  request-body size cap (`src/render_demo.py`'s existing
  `synthetic_only_public_demo` middleware) — untouched, and re-verified by
  the full existing `tests/test_render_demo.py` suite (19/19 passing,
  unmodified).
- `deploy/render_fetch_artifact.sh`'s verified artifact-fetch/checksum
  process — not touched by this integration at all.
- The model, the frozen artifact, and the Stage 11 global importances
  used by the dashboard — same ones described above; the public batch is
  real inference against the real artifact, just bounded in cardinality.

### Verified this round

- Both branches (`investigation-dashboard`, `render-demo-preparation`)
  were confirmed to touch disjoint files before merging; the merge into
  a new `render-dashboard-integration` branch (based on
  `origin/render-demo-preparation`, not `main`) was clean, with no
  conflicts.
- Full fast suite on the integration branch: **385 passed**, 0 failed
  (340 original + 10 local dashboard-endpoint tests + 19 pre-existing
  `render_demo` tests, unmodified + 16 new integration tests covering the
  bounded/deterministic public endpoint, its separate rate limiting, and
  static asset serving).
- Real end-to-end verification against `src.render_demo:app` run as a
  single server on one port (mirroring how Render hosts it) — not just
  `src.api:app` locally: `curl` confirmed `n` is ignored and `seed` maps
  into the fixed allowlist; a real headless-Chrome session (via the
  Chrome DevTools Protocol) loaded `/dashboard.html` from that single
  origin, confirmed Chart.js loaded, performed a real DOM click that
  opened the detail drawer, exercised the risk-band filter, clicked
  Refresh and confirmed the seed cycled within the allowlist, captured a
  390px-wide mobile screenshot to confirm the responsive layout holds,
  and — separately — performed a real click on the *original*
  single-transaction demo's "Score this transaction" button at that same
  origin and confirmed it still returns a real score, with zero console
  errors or exceptions across all of it.
- `main` and `render-demo-preparation` (both local and `origin/`) were
  not touched by any of this — confirmed via `git status`/`git diff`
  against both before committing.

### Deployment blockers before this reaches the live service

1. **This branch has not been pushed or merged**, and no Render config
   was changed — per instructions, nothing here goes live until you
   decide to merge `render-dashboard-integration` and Render redeploys
   from it.
2. **The bounded/deterministic design choices above (24 rows, 5 fixed
   seeds, the specific rate-limit numbers) are reasoned defaults, not
   load-tested against Render's actual free-tier CPU budget.** Worth a
   first real deployment check before calling this final.
3. **`X-Forwarded-For` attribution behind Render's proxy remains
   unverified against a live deployment** (a pre-existing, disclosed
   limitation of `/v1/predict`'s rate limiting, inherited as-is by the
   dashboard's new rate limiter via the same `_client_key` helper).

## Stage 3: visual refinement

A visual pass toward a "premium dark fraud investigation command center"
feel, against a reference mockup used as design inspiration only (not a
source of factual metrics — its kanban "Review Capacity" KPI and
period-over-period trend arrows were deliberately NOT copied, since this
project has no real reviewer-capacity data and no real previous period to
compare against).

### What changed

- **Responsive left navigation** (`ui/dashboard.html`/`.css`/`.js`):
  anchor links to this single page's real sections only (Overview,
  Investigation queue, Analytics, Workflow board, Responsible use, plus a
  link to the single-transaction demo) — no fake multi-page items. Fixed
  sidebar on desktop; an off-canvas, toggleable panel on narrow viewports.
- **Header redesign**: a real, functional search box (promoted from the
  filters panel) plus a "Demo batch" selector — 5 fixed, labeled
  sequences replacing the old arbitrary-random "Refresh" button. These 5
  seeds (101/202/303/404/505) are the exact same values
  `src/render_demo.py`'s public endpoint already enforces as its bounded
  allowlist, so the control behaves identically on both the local app and
  the public Render deployment rather than implying more variety locally
  than Render can actually provide. A deliberate, disclosed deviation
  from the literal "date" control: a fake calendar date-range picker was
  **not** added, since this batch has no real elapsed-time dimension
  (only synthetic queue order) — adding one would have been an
  interactive control with no real backing data.
- **Historical model performance panel**: the real, already-approved
  Stage 10 `final_test` metrics (AP 0.4278, ROC-AUC 0.8664, 95% CI,
  Precision/Recall@1%, no-skill baseline) quoted verbatim from the
  project's own published `README.md` — visually and structurally
  separated (dashed border, its own heading, explicit "distinct from...
  the live scores below" copy) from the live per-batch KPI grid, so the
  two can never be read as the same kind of number. The raw
  `docs/_stage10_final_test_evaluation_raw.json` artifact itself was
  **not** opened — only the figures already public in `README.md`.
- **7 KPI cards**, each with a small inline SVG icon and a distinct
  accent-colored left border, for visual differentiation without
  fabricating trend deltas (no previous period exists to compare against,
  so none is shown).
- **Click-to-filter charts**: clicking a slice of the risk-band doughnut
  or a bar in the by-ProductCD chart sets the matching filter — real
  wiring via Chart.js's own `onClick`, not decorative.
- **Click-sortable table headers** (Transaction ID / Amount / Risk score)
  with `aria-sort` and a caret indicator, replacing the old separate
  "Sort by" dropdown.
- **Read-only Kanban workflow board** (New / In Review / Escalated /
  Resolved), grouping the SAME filtered batch by its existing synthetic
  `status` field. Not drag-and-drop: there is no backend write path for
  status changes, and a fake local-only reorder would misrepresent this
  as a working update — the panel's own copy says so. Clicking a card
  opens the same detail drawer as the table.
- **Accessibility basics**: a skip-to-content link, `aria-sort` on
  sortable headers, `aria-expanded`/`aria-controls` on the nav toggle,
  a labeled search input, `aria-label` on every chart canvas, and
  `:focus-visible` outlines already present project-wide.

### Two real bugs found and fixed by actually testing in a browser

1. **Chart.js re-entrancy exception.** Destroying and recreating a chart
   synchronously from inside that same chart's own `onClick` handler (the
   natural way to implement "click a slice to filter") threw a real,
   reproducible `TypeError` from inside `chart.umd.min.js` — not a test
   artifact, a bug any real user clicking a chart would have hit. Fixed by
   deferring the re-render one tick (`setTimeout(fn, 0)`) so Chart.js
   finishes its own event dispatch first.
2. **Mobile horizontal overflow.** A real 390px-wide headless-Chrome
   render showed `document.body.scrollWidth` at 742px against a 390px
   viewport. Root-caused (not just patched) by walking the ancestor
   chain: CSS Grid/flex items default to `min-width: auto`, so a
   descendant's intrinsic content (`table.queue`'s `min-width: 620px`)
   forced `.layout-grid`'s grid items wider than their track even though
   that table scrolls internally — fixed with the standard `min-width: 0`
   override on the grid items. A second contributor (the Kanban board's
   mobile layout using `grid-auto-flow: column` with explicit item
   widths) didn't reliably clip at its own container boundary; switched
   to the standard `display: flex; overflow-x: auto` pattern for
   horizontally-scrollable card rows. Verified fixed by re-measuring:
   `bodyScrollWidth` now equals `windowInnerWidth` exactly (390 = 390)
   with the mobile nav open.
3. **(Backend) seed-dropdown misalignment.** The public endpoint's
   `seed % 5` fallback mapped every one of the 5 dropdown values to a
   *different* allowlist entry (e.g. sending `303` returned batch `404`)
   — the exact-match behavior documented above was not actually true
   until a real browser test against the live server caught the mismatch
   (`health-row-seed` showing `404` after selecting "Sequence 3").
   Fixed in `src/render_demo.py`: an allowlisted seed now passes through
   unchanged; only a non-allowlisted value falls back to the modulo
   mapping. Covered by a new regression test
   (`test_public_queue_allowlisted_seeds_pass_through_unchanged`).

### Verified this round

- Full fast suite: **386 passed**, 0 failed (385 prior + 1 new seed-
  alignment regression test). No backend/model files were touched except
  the one-line `render_demo.py` seed-mapping fix above.
- Real browser verification (Chrome DevTools Protocol, not just unit
  tests) against **both** `src.api:app` (local split-origin) and
  `src.render_demo:app` (single origin, mirroring Render): page load
  sanity (7 KPI cards, Chart.js loaded, historical panel populated, 4
  Kanban columns), a real click-sort on the Amount column header with
  `aria-sort` verified, a real canvas click on the risk-band doughnut
  that set the band filter, Kanban card total matching the filtered row
  count, a real Kanban-card click opening the correct transaction in the
  drawer, the header search box filtering to an exact match, the Demo
  batch dropdown changing the displayed seed correctly (and, after the
  fix above, to the *correct* seed), accessibility attribute checks, a
  real 390px mobile render with the nav toggled open, and zero console
  errors or exceptions throughout every pass on both servers. The
  original single-transaction flow was re-verified with a real button
  click on `src.render_demo:app`'s `/` page, returning a real score.
- Not independently re-verified: the exact doughnut-chart click
  coordinate used in automated testing lands reliably on the ring in
  this environment, but click-to-filter on a doughnut is inherently
  geometry-sensitive (a click near the hollow center won't hit any
  slice) — real users have the same constraint Chart.js itself imposes,
  this isn't a defect introduced here, just not independently
  re-confirmed across every possible click position.

### Remaining limitations (unchanged from the prior round)

- Not pushed, merged, or deployed. `main` and the original
  `render-demo-preparation` branch remain untouched.
- The bounded-batch size and rate-limit thresholds are still reasoned
  defaults, not load-tested against Render's actual free-tier CPU
  budget.
- `X-Forwarded-For` behavior behind Render's proxy remains unverified
  against a live deployment.
