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

## Stage 4: Kanban and dashboard interaction quality review

A review pass against items 1-8 below (Kanban/drawer correctness,
selection consistency, filter/chart/sort/search coherence, accessibility,
visual hierarchy, the 5 fixed sequences on both backends, and the
no-fake-interactivity constraint). No backend scoring logic, the frozen
model, synthetic-only restrictions, or approved Stages 0-19 were touched.

### Gap analysis against the target (checklist order)

| # | Target | Finding | Resolution |
|---|---|---|---|
| 1 | Kanban columns New/In Review/Escalated/Resolved with accurate counts from the current batch | Already correct — columns sum to the batch size (verified 19+11+3+15=48 and 7+8+2+7=24 on the bounded public batch) | No change needed |
| 2 | Kanban card click opens the matching drawer with correct ID/score/metadata | Already correct, re-verified with exact field-by-field comparison (card's own displayed amount/band vs. the drawer's) | No change needed |
| 3 | Table and Kanban selections stay consistent | **Gap found**: selection highlighting only existed on table rows, not Kanban cards, and did not survive a filter-triggered re-render of either view | Added `state.selectedId` plus a shared `applySelectionVisuals()` applied after every table/Kanban render and from open/close-drawer, so both views highlight the same transaction and the highlight survives filtering |
| 4 | Filters/charts/sort/search coherent; KPI and queue counts accurately reflect their stated scope | **Gap found**: the top KPI row always described the full batch even when the table/charts/Kanban were filtered to a subset — `renderKpis` received the filtered rows as a parameter but never used it | 4 of 7 KPIs (the ones that are naturally "what am I looking at" numbers) now compute from the active filter scope, with an explicit "of N filtered" / "of full batch" sub-label; "Transactions scored" deliberately stays batch-wide since it states a true, filter-independent fact (how many real inferences happened), labeled as such |
| 4 (cont.) | — | **Gap found**: clicking the risk-band doughnut's legend swatch triggered Chart.js's *default* behavior (hiding that slice) instead of the chart's own click-to-filter — inconsistent with clicking the arc itself, and would have let a slice's data silently disappear from view, contradicting "accurately reflect scope" | Legend clicks now call the identical filter toggle the arc uses; the default hide-on-click behavior is overridden so no slice can be hidden |
| 5 | Keyboard navigation, semantic labels, visible focus, contrast, mobile layouts | Re-verified: skip link, `aria-expanded`/`aria-controls` on the nav toggle, `aria-sort` + real `<button>` sort headers reachable and activatable by keyboard, labeled search input, `aria-label` on every chart canvas, Kanban cards are real `<button>` elements. Computed WCAG contrast ratios for every text/background pair in the palette: worst case 5.18:1 (Critical risk pill), best 15.42:1 — all pass AA (4.5:1) with margin. | No change needed; documented instead of just asserted |
| 5 (cont.) | — | **Real bug found**: a genuine desktop-width horizontal overflow (`document.body.scrollWidth` 1476px against a 1425px viewport) — missed entirely in the prior round, which only checked mobile. Root cause: the `#chart-by-product` canvas was stuck at Chart.js's hardcoded 300px default width instead of sizing to its grid column, visibly clipping the "High-risk by ProductCD" title and overflowing the page. | Wrapped every chart canvas in a dedicated, explicitly-sized `.chart-canvas-wrap` (Chart.js's own documented pattern for `responsive: true` to shrink, not just grow). Re-measured: `bodyScrollWidth` now exactly equals viewport width on both desktop and mobile. |
| 6 | Polished visual hierarchy, consistent risk colors, professional spacing | Re-reviewed via full desktop and mobile screenshots; no remaining clipping, consistent risk-pill coloring across table/Kanban/drawer/charts, consistent card/panel spacing | No further change |
| 7 | All 5 fixed sequences work on both local and Render-style apps, no false persistence/monitoring/outcome claims | Re-verified via direct API calls and full UI interaction on both `src.api:app` (48-row batches, exact seed) and `src.render_demo:app` (24-row bounded batches, same 5 seeds pass through unchanged — the fix from the prior round holds) | No change needed this round |
| 8 | No fake drag-and-drop, no invented backend, no unverified explanations/metrics | Confirmed by grep: no `draggable`/`drag*` code anywhere in `ui/dashboard.html`/`.js`; the historical Stage 10 panel quotes only already-published `README.md` figures; the drawer's "feature highlights" are still explicitly labeled global, not per-transaction | No change needed |

### A real finding about the doughnut-chart click specifically

Testing this properly (per the explicit instruction) meant computing the
exact click coordinate from Chart.js's own arc geometry
(`chart.getDatasetMeta(0).data[i]`'s `startAngle`/`endAngle`/radii),
rather than guessing a screen position as the prior round did. Result,
tested in isolation across all 4 bands on a fresh page load:

- Low, Medium, and High: the geometrically-exact click reliably selects
  the correct band every time.
- Critical (2 of 48 transactions in this batch, ≈15° of arc): an exact
  midpoint click sometimes lands on the adjacent slice. This is standard
  doughnut-chart behavior for a very thin wedge, not a bug in this page's
  click handling — the same geometry problem exists on any doughnut
  chart, anywhere. The legend-click fix above gives a reliable,
  large-target alternative for exactly this case.

### Also found and fixed: a screenshot-tooling artifact (not a dashboard bug)

Capturing a true full-page screenshot by resizing the CDP viewport to the
page's full content height, after the charts had already rendered at a
smaller height, left every chart visually blank in the screenshot —
despite `Chart.getChart(id)` showing correct, non-empty data and zero
console errors at that exact moment. Confirmed this was specific to the
screenshot method (not a real defect) by: rendering at a single, stable
viewport size matching how the page is actually used (no mid-session
resize) and getting correct charts every time; and by directly
introspecting the live `Chart` instances, which always had correct data.
Documented here rather than silently discarded, since it shaped how the
evidence in this review was captured.

### Verified this round

- Full fast suite: **386 passed**, 0 failed — unchanged from the prior
  round, since this review's fixes were frontend-only (`ui/dashboard.html`
  /`.css`/`.js`); no backend file was modified.
- Real browser verification (Chrome DevTools Protocol) against **both**
  `src.api:app` and `src.render_demo:app`, run as one consolidated pass
  covering checklist items 1-6 and 8 plus the dedicated doughnut-geometry
  test for item 8: Kanban counts summing to the batch size on both
  servers; a real Kanban-card click opening the drawer with exact
  field-for-field matches; selection consistency across table and Kanban
  surviving a filter change; KPI values changing correctly (and the
  "Transactions scored" KPI correctly *not* changing) when a filter is
  applied; search and sort producing matching, consistent row counts
  across the table and Kanban board; accessibility attribute checks; a
  real Tab-and-Enter keyboard activation of a sortable column header; no
  horizontal overflow on a real 1440px-wide or 390px-wide render; and the
  doughnut click correctly toggling the filter on and off. Zero console
  errors or exceptions in any pass.

### Remaining limitations

- Everything listed at the end of the prior "Render integration" section
  still applies unchanged (not pushed/merged/deployed; rate-limit/batch-
  size thresholds untested under real load; `X-Forwarded-For` behavior
  unverified live).
- The doughnut chart's thinnest possible slice remains a harder click
  target than the others by the nature of the chart type; mitigated, not
  eliminated, by the legend-click fix.

## Stage 8: multi-page redesign (branch: dashboard-multiview-redesign)

Built on a new local branch off `d413486` (the then-currently-deployed
integration commit), **not pushed, merged, or deployed** — this stage
delivers and documents the redesign; it does not ship it. The long
scrolling single page with anchor links was replaced with a real
multi-view application: a hash-based client-side router
(`#/overview`, `#/queue`, `#/analytics`, `#/workflow`, `#/insights`,
`#/responsible-ai`) across six views, sharing one JS `state` object as
the single source of truth for the current demo batch, the selected
transaction, and all filters.

### Why hash routing, not path-based routing (confirmed decision)

Path-based routes (`/queue`, not `/#/queue`) look cleaner but require a
new FastAPI catch-all on Render so a hard refresh/direct link resolves
server-side — something local dev's plain `python3 -m http.server`
cannot replicate, which would make direct-link behavior differ between
local dev and Render. Hash routing needs zero new server routes and
behaves identically in both environments; confirmed with the project
owner before implementing.

### Why filters are global, not per-view (confirmed decision)

The detailed filter panel lives on the Investigation Queue view, but its
*state* affects every view: Overview's preview, Analytics' charts, and
the Workflow board all read the same `filteredSorted()` result the Queue
table does. Confirmed with the project owner specifically to satisfy
"avoid inconsistent KPI, chart and queue counts" literally — every view
is always looking at the same scope, never a silently different one.

### View-by-view content redistribution

- **Overview** — compact 7-card KPI grid, two small non-interactive
  preview charts (risk band breakdown, risk score distribution), and an
  8-row investigation preview with a link to the full queue. The large
  historical Stage 10 panel was removed from here per the explicit
  instruction and relocated to Model Insights.
- **Investigation Queue** — the full filter panel + the complete,
  sortable, searchable table (unchanged logic from Stage 4).
- **Analytics** — all 5 full-size charts in a dedicated 2-column grid
  with real room to breathe, instead of being squeezed into a narrow
  column beside the table.
- **Workflow Board** — the 4-column Kanban board, now filling most of
  the viewport instead of a fixed 520px box.
- **Model Insights** (new) — the relocated Stage 10 historical panel,
  plus a new "Top model-level risk drivers" list built from the same
  `global_feature_importance` data the detail drawer already used
  (genuinely derived from the batch response, not invented), plus the
  relocated Model & service health panel.
- **Responsible Use & Limitations** — unchanged content, now its own
  standalone view instead of the page's last scrollable section.
- **`/index.html`** (single-transaction demo) — untouched, still a
  separate page outside the SPA, linked from the sidenav and footer
  exactly as before.

### Desktop viewport-fit shell, with a mobile fallback

At `min-width: 901px`, the app shell is fixed to `100vh`: the header
stays compact and visible, and only `.view-container` scrolls
internally (confirmed via `document.body.scrollWidth`/computed overflow
checks — `overflow: hidden` on `body`, `overflow-y: auto` on
`.view-container`). Below that width, the base (mobile-first) rules
apply instead: the whole document scrolls normally, matching the
already-proven mobile pattern from Stages 3-4, satisfying the explicit
"don't clip essential content on narrow screens" requirement.

### Real bugs found and fixed while building this

1. **A visible focus-outline box appeared on the view heading after every
   navigation.** Each route change moves keyboard focus to the new
   view's `<h2>` (for screen-reader announcement), and the browser's
   default `:focus-visible` outline rendered a stark white box around
   it — correct for an interactive control, wrong for a heading a mouse
   user just watched get swapped in. Fixed by removing the outline
   specifically on `.view-heading`, while leaving it in place on every
   genuinely interactive control.
2. **My own test harness's port choice broke every data-dependent
   browser test.** `ui/dashboard.js`'s local-dev origin detection
   (`IS_LOCAL_DEV_SPLIT_ORIGIN`) hardcodes "static server on 8080, API
   on 8000" — the project's own documented convention. The first version
   of the new committed browser-test fixture used dynamically-assigned
   random ports instead, so the browser's own fetch silently went to the
   wrong origin and the dashboard never loaded any data — not a
   dashboard bug, a test-harness bug, root-caused by checking exactly
   what the browser actually fetched rather than assuming the test
   infrastructure was correct. Fixed by using the project's real
   documented ports, with a clear skip if they're already occupied.
3. **A test helper's URL fragment was silently dropped.** `requests.put(f"{base}/json/new?{url}")` strips everything after `#` before
   the HTTP request is even sent (a URL fragment is a client-side-only
   concept) — so a direct-hash-navigation test was never actually
   testing a hash URL. Fixed by opening `about:blank` first and using
   `Page.navigate` with the full URL as a JSON body value instead, which
   has no such stripping problem.

None of these were dashboard defects that would affect a real user —
items 2 and 3 were bugs in the new test code written to verify the
dashboard, caught by checking actual behavior rather than trusting the
test's own first result.

### Automated tests: routing and shared state (new)

`tests/test_dashboard_routing_browser.py` — a real, committed,
pytest-discoverable browser-integration suite (12 tests), using only
already-pinned project dependencies (`requests`, `websocket-client`;
see `requirements.txt`) to drive a real headless Chrome instance via the
Chrome DevTools Protocol. No new test framework (Selenium/Playwright/
Jest) was introduced. Marked `@pytest.mark.browser` (registered in
`pytest.ini`, alongside the project's existing `heavy`/
`data_contract_full` markers) and skips cleanly via `pytest.skip` if
Chrome isn't found or its required ports are occupied, rather than
failing a run on a machine without Chrome.

Covers: default route with no hash; each of the 6 nav links updating the
hash, active view, and `aria-current`; a direct hash navigation on a
*fresh* page load resolving the correct view (the literal "direct link"
requirement); real `history.back()`/`forward()` restoring the correct
prior views; the core shared-state contract — selecting a transaction,
then navigating to a different view, and confirming the drawer stays
open on the *same* transaction; global filter state propagating from the
Queue view to the Overview preview and the Workflow board's card counts;
and a full navigation sweep asserting zero console exceptions.

### Verified this round

- Full suite: **399 passed**, 0 failed (387 prior + 12 new browser
  tests). No backend, model, or `src/` Python file was touched this
  stage — confirmed via `git diff --stat` — only `ui/dashboard.{html,css,js}`,
  `pytest.ini` (new marker), and the new test file.
- Real browser verification beyond the committed suite: desktop
  screenshots at **1440px and 1280px**, tablet at **768px**, and mobile
  at **390px**, for every one of the 6 views — zero horizontal overflow
  at any width (`document.body.scrollWidth` exactly equals
  `document.documentElement.clientWidth` at all four), confirmed by
  direct measurement, not visual inspection alone.
- The original single-transaction demo (`/index.html`) re-verified with
  a real button click at the end of this stage, returning a real score —
  confirmed unaffected by the SPA restructuring.

### Visual fidelity vs. the approved target mockup

Present and matching: dark navy/charcoal theme with cyan/purple/amber/
teal/red accents; colorful icon-coded KPI cards; a left sidebar nav
(now routing to real views, not scrolling to anchors); coordinated,
globally-consistent filters; linked/clickable charts; a transaction
detail drawer; a 4-column Kanban board; a compact, professional header.

Intentionally different, listed plainly (carried over from Stage 5's
same honest accounting, now re-confirmed against the restructured page):
no fabricated date-range picker (no real elapsed-time dimension to
support one — the Demo batch selector is the real, functional
header control instead); no invented reviewer-capacity KPI; no
period-over-period trend arrows (no real previous period exists); the
Kanban board is explicitly read-only, not drag-and-drop; the detail
drawer's "feature highlights" are the model's global importance ranking,
not fabricated live SHAP.

### Deployment blockers

- This branch is local-only (`dashboard-multiview-redesign`), not
  pushed, not merged into `render-dashboard-integration` or PR #2, and
  not deployed. The live dashboard (deployed separately, if/when PR #2
  is merged) is unaffected by any of this work.
- Everything listed in the Stage 5 readiness report's "remaining risks"
  section still applies unchanged (the real Render build environment has
  never been exercised; `X-Forwarded-For` behind Render's proxy remains
  unverified live).
- The new browser-integration test suite has only been run on this
  machine (macOS, a specific Chrome version); it has not been exercised
  in a CI environment, and its `CHROME_PATH` override / common-path
  detection has not been tested on Linux (the likely CI/Render build
  platform) — it is designed to skip cleanly there, not fail, but that
  skip path itself has not been independently verified on Linux.

## Stage 9: professional audit and refinement (same branch)

An audit pass on top of Stage 8's architecture, continuing directly on
`dashboard-multiview-redesign` (not a new branch — this is refinement of
the same work, per the task's own stated option). Confirmed by real
browser testing first that the Stage 8 architecture was already
functionally sound (routing, cross-view selection, global filters, all
working correctly against `src.render_demo:app`); this stage's actual
work was a visual/UX audit against screenshots at all 4 required widths,
closing the real gaps found, not rebuilding anything that already
worked.

### What the audit found and fixed

1. **The Status column (Queue table, Overview preview) had no color
   coding** — plain gray text, while Risk band already used color pills.
   Fixed by styling the existing `status` field with the exact same
   color language the Kanban column headers already use (New=accent,
   In Review=accent-2, Escalated=warn, Resolved=ok) — no new data, just
   making an already-real field scannable. Computed WCAG contrast for
   all 4: 4.66–5.89:1, all pass AA.
2. **The "Amount vs. risk score" scatter and "Risk score by queue order"
   line charts used a fixed 0–1 y-axis**, which — since scores in this
   project's synthetic batches cluster narrowly, as already documented —
   visually wasted most of the chart's height. Fixed by applying the
   exact same honest auto-ranging the risk-score distribution histogram
   already used (zoom to the batch's own observed range, captioned with
   the real numbers), not a new technique.
3. **The Responsible AI view was mostly empty space** at normal desktop
   height — real, correct content, just not enough of it to fill a
   dedicated view on its own, confirmed by direct screenshot review, not
   assumed. Fixed by adding a companion "At a glance" panel that reuses
   only already-real elements (the same badges shown in the header, real
   functional links to Model Insights and the single-transaction demo) —
   explicitly not filled with invented content.
4. **The Model Insights "Model & service health" panel** (6 short facts
   in a plain list) had the same empty-space problem. Restyled as a
   stat-tile grid reusing the historical panel's own `.k`/`.v` visual
   language, so 6 real facts read as a deliberately designed panel.
5. **"Top model-level risk drivers" was a plain numbered text list.**
   Added a small proportional horizontal bar per entry, scaled against
   the top-ranked value — the exact same numbers already shown as text,
   visualized as an actual mini feature-importance chart rather than
   invented or re-derived.

### Confirmed NOT broken: the architecture itself

Re-verified before touching anything: all 6 views genuinely distinct
(not scroll anchors); routing, direct hash navigation, back/forward, the
5 fixed sequences (exact-match seed pass-through re-confirmed against
`src.render_demo:app`), cross-view selected-transaction persistence, and
global filter propagation all still correct. Chart click-to-filter was
re-tested with precise Chart.js arc/bar geometry; one apparent bar-chart
click "miss" during this testing was root-caused to the test script
clicking exactly on a bar's 1px top boundary pixel (a test-precision
artifact — real users click within the bar body, confirmed to work
reliably on retest), not a dashboard defect, and is disclosed as such
rather than silently dropped.

### Verified this round

- Full suite: **399 passed**, 0 failed (387 Python + 12 browser-
  integration, both re-run cleanly after freeing the ports the browser
  fixture needs). No backend, model, or `src/` Python file touched —
  confirmed via `git diff --stat`; only `ui/dashboard.{html,css,js}`
  changed.
- Real browser screenshots re-captured at **1440px, 1280px, 768px, and
  390px** for every view, before and after the fixes, with zero
  horizontal overflow at any width on either pass (`document.body.
  scrollWidth` exactly equal to `document.documentElement.clientWidth`).
- The new "At a glance" panel's live link to Model Insights was clicked
  for real and confirmed to navigate correctly, not just visually
  present.

### Remaining limitations

Unchanged from Stage 8: this work is local-only on
`dashboard-multiview-redesign`, not pushed again this round, not merged,
Render untouched. Everything in the Stage 5 readiness report's
"remaining risks" section still applies.
