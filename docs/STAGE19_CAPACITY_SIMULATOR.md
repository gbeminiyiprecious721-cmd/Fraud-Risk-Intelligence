# Portfolio enhancement Stage 19 — Investigation Capacity Simulator

This is an extension of the existing application, separate from the original
closed ML Stages 0–19. The owner authorized this stage on 2026-10-30.
Base: `3569473e649ed27ce21771d4b2d2c8cb41fa6547` on the verified Render
integration branch. Development branch: `stage19-investigation-capacity`.
No push, merge or deployment is authorized by this stage.

## Business problem and intended user

Fraud operations analysts and risk managers have finite review capacity.
The simulator shows how a fixed historical ranking concentrates verified
fraud labels as workload increases. It supports human investigation
prioritization. Capacity is a fraction/count of this historical evaluation
window, never an estimate of daily staff capacity, prevention, savings,
automated declines or guaranteed future performance.

## Evidence inventory and authorization boundary

| Evidence | Permitted use |
|---|---|
| Published Stage 10 final-test summaries in README/protocol: AP 0.42784; Precision@1% 0.8583, Recall@1% 0.2461; Precision@5% 0.3037, Recall@5% 0.4351 | Historical context only; unchanged. Never used to generate a new curve. |
| Stage 7 validation AP 0.41101 and Stage 9 chronological-CV AP 0.51236 | Different models/populations/configurations; not interchangeable with this simulator. No persisted row-level score source from those stages was adopted. |
| Stage 10 readiness rehearsal: validation AP approximately 0.447 | Published development evidence establishing the authorized population and train-only scoring procedure; not final-test evidence. |
| Trusted Stage 13 artifact plus original split-assignment metadata | Artifact fitted on **train only**, 438125 rows, verified train-ID fingerprint. Enables inference on disjoint validation without retraining. |
| Original validation rows/labels | 75560 rows, 2617 positives, prevalence 0.03463472737. Used only for this historical aggregate. |
| Existing synthetic dashboard batches (actually 48 rows in the current UI, rather than the brief's older 24) | No verified outcomes; never used for observed fraud capture, recall or confusion counts. |

`docs/PRODUCTION_PIPELINE_REPORT.md`, the trusted artifact manifest and
`scripts/build_stage13_artifact.py` establish train-only provenance. The
artifact is a reconstruction of the approved frozen configuration, not
the original Stage 10 Python object. Validation was previously consulted
in model development, including Stage 7 selection decisions: this is
**development backtesting with possible selection optimism**, not a new
independent held-out estimate. No fresh tuning or threshold optimization
was performed.

No closed final-test row-level predictions or derived final-evaluation
artifacts were opened for this stage. The builder reads split metadata
with **TransactionID and partition only**, never its label column. Its
initial scan of `train_transaction.csv` parses only IDs to locate physical
rows. A second CSV read uses `skiprows` **before feature/label parsing** to
load validation rows only. It does not use the legacy read-all-labels-
then-filter helper. ID alignment, unique IDs and training fingerprint are
checked before inference; feature-schema validation is delegated to the
unchanged production pipeline. No row-level predictions/labels are saved.

## Definitions and independent validation

For a fixed validation population of N rows, P positives and chosen K:

- Rank descending by the frozen artifact's score, canonicalized to 12
  decimal places to remove the project's documented ~1e-15 parallel
  summation noise. Resolve ties by ascending TransactionID. This is a
  reproducibility convention, not calibration or a model change.
- TP is the count of positive labels among the first K rows; FP = K − TP;
  FN = P − TP; TN = N − P − FP. Reviewed workload is K and K/N;
  remaining unreviewed is N − K. Precision = TP/K; recall/fraud capture = TP/P.
- K=0 has undefined precision (`null` / “Not defined”), with zero reviews.
  A zero-positive population has undefined recall; the pure metric code
  and unit tests handle this without division by zero.
- Uniform random sampling without replacement has expected TP = KP/N,
  expected precision P/N for K>0, and expected recall K/N if P>0. These
  are **expectations, not actual draws or guaranteed counts**. No confidence
  interval or independence assumption for model performance is claimed.
- Publish K = ceil(N × i/200), i=0…200: 201 exact, precomputed checkpoints
  across the full 0–100% range. Numeric input snaps to the nearest supported
  K, clamped to the endpoints; exact equidistance selects the lower K.
  All displayed counts/metrics come from the selected checkpoint, never
  interpolated from a line. This ceiling rule is explicit and does not
  rewrite the historical Stage 7/10 round-based K convention.
- Marginal benefit is the additional TP in the block between the preceding
  checkpoint and selected K, divided by additional reviews for block
  precision. It is not a claim about the next individual transaction.
- Each of the **201 checkpoints** is independently recomputed using a
  separate sorted index, binary selection mask, and sklearn confusion
  matrix/precision/recall. Unit tests also check a hand-sized independent
  example, monotonic cumulative capture, invalid inputs, ties, boundaries
  and denominator invariants.

Current canonical validation AP is **0.4470019466**, distinct from every
published historical evaluation. Tiny differences from unrounded-score AP
(about 0.4470030919 in the initial read-only inference pass) arise from
explicit score-tie canonicalization. No earlier number is overwritten.
At 1% (K=756), TP=618, FP=138, FN=1999, TN=72805; precision=81.7460%,
recall=23.6148%; expected random TP=26.1839.
At 5% (K=3778), TP=1316, precision=34.8332%, recall=50.2866%.
These are historical validation estimates, not the Stage 10 final-test results.

## Architecture and API protections

- `src/capacity_simulator.py`: pure ranking/count/metric calculations.
- `scripts/build_capacity_aggregate.py`: trusted-artifact inference in
  batches of 2048, source isolation, provenance checks, independent
  metric checks and aggregate generation. **No fit calls.**
- `ui/capacity-v1.json`: compact versioned, nonsensitive aggregate; contains
  population totals, provenance hashes and checkpoint counts/ratios.
  It contains no transaction IDs, per-row labels, per-row scores or raw
  feature values. Runtime does not need raw data or additional inference.
- `src/capacity_routes.py`: fixed GET `/v1/dashboard/capacity`, reused in
  local and public apps via `create_app`. No user-supplied file paths or
  model selection. A missing aggregate returns explicit 503; POST is 405.
- `ui/capacity.js`: lazy aggregate loading, synchronized capacity controls,
  five KPI cards, recall and precision curves with random expectations,
  selected-point markers, click-to-select, exact confusion counts,
  marginal block yield, expandable confusion-count/assumption detail, Reset,
  explicit unavailable/retry state and a
  Chart.js-unavailable fallback retaining numeric controls/results.
- Existing hash router adds `#/capacity`; both APIs share the exact same
  static aggregate. The Render adapter adds only a fixed `capacity.js`
  asset route. Existing prediction and security handlers, CORS, fixture-only
  public scoring, request-size caps, seed allowlist and rate limiting
  remain unchanged and covered by regression tests.

Only one eligible evaluation population is offered: no misleading selector.
The shared population badge changes to “Historical validation / portfolio”
in this view and returns to its synthetic label in the other views. The
footer distinguishes synthetic queue records from historical aggregates;
Responsible Use explicitly scopes the synthetic-record disclosure to the queue.
Shared synthetic search/batch controls are hidden in this view and replaced
by an explicit historical-population message. Existing global synthetic
state is preserved when navigating back. Simulator selection is independent
of that state. Server-provided strings use textContent, not HTML injection.

## Reproduction and browser evidence

```sh
.venv/bin/python scripts/build_capacity_aggregate.py
.venv/bin/python -m pytest tests/test_capacity_simulator.py tests/test_dashboard_routing_browser.py -q
.venv/bin/python scripts/verify_capacity_browser.py
.venv/bin/python -m pytest tests/ -q
```

Generation requires the existing gitignored trusted artifact, original raw
source and split assignment; it uses existing pinned dependencies only.
The runtime consumes the committed aggregate and never requires labelled
row-level data. Existing heavy regression tests fit temporary in-memory
train-only reference models for equivalence; they do not replace or retrain
the production artifact or access closed final-test outcomes for evaluation.

Focused verification: **44 passed** (18 metric/provenance/API tests and
26 real-Chrome cases, including all 21 prior browser regressions).
Five new browser cases cover four viewport sizes and deliberate evidence
failure/retry with native ArrowRight slider activation. Controls cover zero
capacity, full capacity, over-limit count clamping, Reset and route return;
desktop cases also use actual pointer input on chart geometry and require
readable plot areas of at least 70px. The decision-detail disclosure opens
and closes natively without losing the selected capacity. Existing
six-view navigation, selection, filters, charts/Kanban and real
single-transaction scoring/Back to Dashboard tests remain present.

| Viewport | Simulator evidence |
|---|---|
| 1440×900 | [Screenshot](stage19_capacity_evidence/capacity-1440x900.png) |
| 1280×800 | [Screenshot](stage19_capacity_evidence/capacity-1280x800.png) |
| 768×1024 | [Screenshot](stage19_capacity_evidence/capacity-768x1024.png) |
| 390×844 | [Screenshot](stage19_capacity_evidence/capacity-390x844.png) |

[Measurements](stage19_capacity_evidence/viewport-metrics.json) record actual
scroll dimensions and browser exception output. Mobile/tablet screenshots
show the first viewport; remaining panels are reachable with natural document
scrolling. Desktop keeps all primary controls, KPIs and readable charts visible without
outer scrolling; the concise decision summary expands counts/assumptions
on demand, allowing intentional scrolling when that detail is opened.
Desktop retains the premium dark shared navigation, panels,
accent colors and Chart.js typography.

## Remaining scientific and technical limitations

- Validation reuse/selection optimism; no fresh independent accuracy claim,
  future capture guarantee, calibration study or statistical uncertainty band.
- Anonymous feature availability, historical temporal shift and possible
  transaction dependence remain unresolved original project limitations.
- No economically optimal K is claimed without verified review costs and
  business constraints; the controls expose the workload/capture trade-off.
- Fixed-batch top-K ranking assumes the full evaluation window is available;
  it is not an online arrival process, staffing schedule or automated decision.
- Checkpoint discretization and deterministic tie convention affect exact K
  and tie ordering; predictions themselves are never published.
- Public aggregate counts reveal only coarse ranked blocks, without record
  identity or features; they are not transaction-level API output.
- Chart.js still uses the existing integrity-pinned CDN dependency. Numeric
  metrics remain available without it. Actual screen-reader behavior,
  Safari/Firefox, arbitrary zoom levels and physical devices are not certified.
- No live Render verification or deployment is part of this stage. The
  confirmed live commit stays `3569473`; all work remains local.

Owner approval remains pending after implementation verification. No Stage
20–25 work is authorized or begun by this stage.

## Reproducibility and preservation evidence

Rebuilding from the unchanged artifact and validation source produced a
byte-identical aggregate: SHA-256
`c2a0ebd0bca815a278c6c88ca9105bcd02ccc3e491e722177bb1c39b9ee6ca7d`
(56709 bytes). See [provenance checks](stage19_capacity_evidence/provenance-checks.json)
and [aggregate-build summary](stage19_capacity_evidence/aggregate-build.json).
All 201 independent metric checks pass on both builds. There is no random
resampling, refitting or invented performance interpolation in generation.

The original workflow record from `## Project Identity` onward remains
byte-for-byte identical to the base commit. The separate Stage 19 state
note above it records only this portfolio enhancement. Frozen artifact
SHA-256 remains
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
Original preprocessing, production pipeline, leakage controls, model
configuration, artifact manifest, scientific reports and deployment
configuration are unchanged. API factory/Render changes are additive
route registration only; existing security middleware is unchanged.

## Final verification and local delivery

The final `.venv/bin/python -m pytest tests/ -q` run completed with
**431 passed, 5 warnings in 187.35s (3:07)**, exit code 0; no exclusions,
skips or failures. This includes all six heavy/data-contract tests and
all **26 real-Chrome cases**. Five existing warnings are the Starlette/
httpx deprecation and the original small-SAGA-test sklearn parameter/
convergence warnings. [Final regression log](stage19_capacity_evidence/full-regression-tests.txt).
An earlier full run also passed all 431 tests before the final shared
population-badge clarification; the final run covers that correction and
public JavaScript delivery, final disclosure copy and the numerical
API assertion correction documented below too.

All four screenshots were inspected. Both desktop sizes have exactly
zero outer scroll overflow and retain readable chart plot areas (about
197px at 1440×900 and 94px at 1280×800). Tablet/mobile have natural document
scrolling and no horizontal overflow. No JavaScript console errors or
exceptions were observed in the recorded normal-path Chrome sessions.
[Image checksums](stage19_capacity_evidence/screenshot-checksums.json).

The source/metric tests passed red-to-green: the new metric module was
absent initially and the new endpoint returned 404 before implementation.
The independent calculations and source isolation/negative tests now pass.
Two desktop layout failures were reproduced and corrected; screenshot
review additionally caught plotting areas that were too shallow despite
passing width checks. The regression now enforces a readable plot-height
floor and the disclosure keeps detailed counts reachable by keyboard.

The work is ready for owner review, with the limitations above preserved.
No scientific or technical implementation blocker remains in the verified
scope. It is committed locally only; no push, merge or deployment is part
of this delivery. No additional stage is begun.

### Exact changed-file inventory relative to the integration checkpoint

- `docs/ML_WORKFLOW_STATE.md`
- `docs/STAGE19_CAPACITY_SIMULATOR.md`
- `docs/stage19_capacity_evidence/aggregate-build.json`
- `docs/stage19_capacity_evidence/api-tests.txt`
- `docs/stage19_capacity_evidence/base-api-float-reproduction.json`
- `docs/stage19_capacity_evidence/capacity-1280x800.png`
- `docs/stage19_capacity_evidence/capacity-1440x900.png`
- `docs/stage19_capacity_evidence/capacity-390x844.png`
- `docs/stage19_capacity_evidence/capacity-768x1024.png`
- `docs/stage19_capacity_evidence/float-assertion-failure.txt`
- `docs/stage19_capacity_evidence/focused-tests.txt`
- `docs/stage19_capacity_evidence/full-regression-tests.txt`
- `docs/stage19_capacity_evidence/full-tests-initial.txt`
- `docs/stage19_capacity_evidence/provenance-checks.json`
- `docs/stage19_capacity_evidence/screenshot-checksums.json`
- `docs/stage19_capacity_evidence/viewport-metrics.json`
- `docs/superpowers/plans/2026-10-30-investigation-capacity.md`
- `scripts/build_capacity_aggregate.py`
- `scripts/verify_capacity_browser.py`
- `src/api.py`
- `src/capacity_routes.py`
- `src/capacity_simulator.py`
- `src/render_demo.py`
- `tests/test_api.py`
- `tests/test_capacity_simulator.py`
- `tests/test_dashboard_routing_browser.py`
- `ui/capacity-v1.json`
- `ui/capacity.js`
- `ui/dashboard.css`
- `ui/dashboard.html`
- `ui/dashboard.js`

## Existing API float-equality regression found during final verification

A later full run exposed an existing flaky exact-equality assertion:
`test_repeated_identical_requests_return_identical_probabilities` compared
0.5201984944879682 with 0.5201984944879683 (difference 1.11e-16), although
both requests succeeded with the approved artifact identity. That run had
430 passes and one failure; [failure evidence](stage19_capacity_evidence/float-assertion-failure.txt)
is preserved rather than hidden behind a retry.

The production inference function and its model bytes are unchanged.
The existing `test_production_pipeline.py` repeated-inference and full-
training tests explicitly document macOS parallel-RF summation variation
at about 1e-15 and already use allclose (rtol=1e-9, atol=1e-12). The API test
was inconsistent with that established numerical contract. Its assertion
now requires both requests to return HTTP 200 and equality within **absolute
1e-12, relative 0**, a tighter bound than that existing pipeline convention.
No prediction, probability formatting, security restriction, model parameter
or scientific result was modified to make the test pass.

A read-only base-commit API replay (100 requests, 4 concurrent workers) did
not reproduce the tiny variation in that attempt; [its result](stage19_capacity_evidence/base-api-float-reproduction.json)
is recorded accurately, not claimed as a reproduced base failure. The
unchanged inference path, actual 1e-16 failing responses and existing
pipeline numerical-consistency tests support the test-only correction.
The complete API test module then passed: **30 passed, 1 existing warning**.
[API verification](stage19_capacity_evidence/api-tests.txt).


After that test-only correction the complete suite was rerun: **431 passed,
5 existing warnings in 187.35s**, with no skips or exclusions. The final
regression log above records this successful run on the delivered tree.
The extra API test-file change is retained because it corrects a demonstrated
flaky regression assertion using the project's established numerical
contract; no unrelated product change is included.
