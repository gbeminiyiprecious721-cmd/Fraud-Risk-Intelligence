# Stage 22 — Model Monitoring and Drift Analysis

Status: implementation verified locally; awaiting owner review. No push, merge or deployment.

## Inspection checkpoint — 2026-10-31

Owner request explicitly authorises Stage22. Branch `stage22-model-monitoring`
was created from verified `origin/render-dashboard-integration` at
`cad3c70ead38ec50623f0743f40d998ef1dd6326`, with a clean starting tree.
No implementation, monitoring calculation or model inference has started.
The supplied brief is truncated at Phase F item4: “Validate aggregate schemas
and”. Remaining engineering, testing and delivery requirements were requested.

## Verified development provenance

Inspection used existing reports, artifact manifest and split metadata columns
TransactionID, TransactionDT and partition only. No outcome columns or closed
final-test records/predictions were opened. No published final-test metrics
will be used to derive monitoring outputs.

| Authorised population | Rows | Relative TransactionDT range | ID SHA256 |
|---|---:|---|---|
| train |438125|86400–11093269|a39d2eeedd2b1c6f2f26bb26670e6a4b53973480806294524887ff56dbdc4470|
| validation |75560|11093842–13452406|14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc|

Development IDs are unique. Populations are strictly chronologically separated.
TransactionDT is relative time, not an established calendar timestamp. Verified
model inputs include TransactionAmt, ProductCD, card4, C1, dist1 and TransactionDT.
Card network semantics must retain the dataset's documented encoding.

Frozen artifact SHA256 remains
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
The artifact was trained on the full train partition. A training score reference
is in-sample; validation scores are out-of-sample for that model. This difference
must be disclosed when comparing score distributions. Validation was repeatedly
consulted during prior development; it cannot become a pristine holdout.

## Existing components and proposed direction

Stage18 already implements pure PSI/reference-distribution functions in
`src/monitoring.py`, train-based aggregate references and local demonstration
scripts. Preserve those results and inspect numerical contracts before reuse.
Stage19/20 establish fixed read-only aggregate endpoints and historical views.
Reuse that architecture and the existing chart library; do not introduce
runtime historical inference, backend case writes or production-score changes.

Proposed direction, pending complete brief and design review: fixed historical
training reference, tie-safe successive validation windows, reference-only
numeric bins and categorical vocabularies, explicit missing/unknown categories,
score quantiles and a fixed reference-score tail criterion. Report drift
magnitude and quality changes separately from any inferential significance.
Never infer actual model deterioration or new fraud-capture performance without
verified new labels. No dependency additions are currently proposed.

Before any calculation, fix windows, sample policy, monitored features,
binning/smoothing, adequacy rules, interpretation thresholds and reproducibility
checks in a concrete protocol. Synthetic scenarios are optional and must remain
separate from historical observations. No results are claimed at this checkpoint.

## Completed brief and protocol proposal

The owner supplied the complete Phases F–H and delivery requirements. Work is
local-only on this feature branch: no push, merge, deployment, integration/main
change, new paid infrastructure or access to closed final-test material.
All six heavy/data-contract tests, prior Stage19–21 regressions and real-Chrome
checks at1440×900,1280×800,768×1024,390×844 are required before completion.
The existing scientific record must remain verbatim below the new stage note.

### Reference alternatives

The original training-reference proposal is feasible for feature drift but
confounds score comparisons with in-sample versus out-of-sample model scoring.
A second, recommended design uses the earliest validation third as reference
and the two subsequent thirds as comparisons. All scores then have the same
out-of-sample relationship to the frozen model. It is a shorter historical
baseline and still reused development data, not prospective production data.
The owner has been asked which reference to use; the following concrete
protocol describes the recommended validation-only design for review.

### Populations fixed before monitoring calculations

Sort authorised validation metadata by TransactionDT then TransactionID.
Take floor(n/3) and floor(2n/3), advancing any boundary past its timestamp tie.
Use the full eligible windows, not label-stratified or score-selected samples.
No labels are needed or requested. Verified metadata boundaries:

| Window | Rows | Relative seconds inclusive | ID SHA256 |
|---|---:|---|---|
| Reference V1 |25186|11093842–11915077|8b9e394395fe3759b08927ac56a95235058590f0f60938862f885db0bdb04470|
| Comparison V2 |25187|11915125–12739542|270ecc5d1fa436a5db474473da41a52474582e55da52ac337b5b63541299c0ea|
| Comparison V3 |25187|12739588–13452406|841b7d86a9aad08ebb83a64545e5e2bf11b93e04f56a0ddba624208b7af85ec4|

Load source IDs only to locate authorised rows, then skip all other CSV rows
before parsing selected feature values. Reuse the established physical-row
selection approach, without requesting isFraud. Verify exact IDs, timestamps,
unique identifiers, population fingerprints and strict chronological separation.
Inference loads the checksum-verified frozen artifact, in bounded batches, with
no fit/refit and no artifact writes. Check checksum before and after.

### Features and quality

Monitor TransactionAmt (amount), ProductCD (coded product), card4 (documented
card-network category), C1 (anonymised count input; no invented business meaning),
and dist1 (dataset distance field; units/interpretation are not established).
A derived TransactionDT modulo86400 distribution is explicitly relative daily
phase, not verified local clock-hour or customer behaviour. Absolute elapsed
time is shown as window context, not flagged as drift merely for increasing.
These are a selected diagnostic subset, not a complete explanation of813
transformed model inputs. Rankings show distribution change, not causality or
model feature importance.

Track missing and non-finite numeric counts separately; missingness uses all
rows as denominator. Numeric drift uses finite observations only. Categorical
histograms include separate missing and reference-unseen buckets, with
reference vocabulary fixed before examining comparison categories. A category
unseen in the reference is not automatically an invalid schema value.
Report source schema/required-column checks, duplicate-ID counts, numeric
parse failures/infinities and negative TransactionAmt as quality checks.
Do not assert undocumented ranges for anonymised C1 or dist1. All rows must
remain accounted for; malformed schema aborts aggregate generation rather than
publishing partially misaligned scores.

### Fixed numerical methods and interpretation

For numeric features and scores, use NumPy linear-interpolated quantiles and
derive up to10 quantile bins from finite
reference values only, deduplicating edges and using open-ended extreme bins.
For a constant reference, use an explicit reference-value versus other-value
comparison. Persist finite interior bin edges; never serialize infinity/NaN.
Separate missing/nonfinite rates from finite-value PSI. Use additive proportion
smoothing epsilon=0.0001, renormalise each vector, and compute
PSI=sum((q-p)*ln(q/p)). Zero-frequency bins remain finite.

Numeric comparisons also report the two-sample empirical KS distance on finite
values. No IID p-value or significance claim: temporally dependent transactions,
multiple selected features and reused validation invalidate a naive significance
interpretation. Categorical comparisons report Jensen–Shannon divergence in
base2, bounded0–1, on the common missing/known/unseen vocabulary with zero terms
handled by their limit. Reference/comparison proportions and counts are shown.

Require at least200 finite observations in both windows for numeric drift;
otherwise publish null metrics and explicit insufficient-evidence status.
For categorical comparisons require at least200 total rows in each window and
show sparse categories without relying on chi-square approximations. Empty
windows and unsupported features must produce controlled errors/empty states.

Investigation signals are demonstration heuristics, not universal laws:
PSI>=0.20 or KS>=0.10 for numeric/score changes; JS>=0.05 for categorical change;
missingness increase>=5 percentage points or any invalid numeric/duplicate-ID
issue warrants quality investigation. Below those criteria say “No material
shift under selected rules”, never “no drift” or “model healthy”. Insufficient
evidence takes precedence. Display magnitude and the chosen rule together.
These thresholds are fixed before outputs and are not tuned to force alerts.

Score summaries include finite n, mean, q01/q10/q50/q90/q95/q99, common-reference
histogram and proportion strictly above the reference q95 threshold. Ties mean
the reference fraction need not equal exactly5%. No calibrated probabilities,
synthetic batch-relative Low/High labels, fraud outcomes or performance metrics.
Trend charts compare V1,V2,V3 descriptive points; no forecast or robust seasonal
trend claim from three windows.

### Dashboard and safety design

Reuse hash navigation, Chart.js, existing dark panels and historical-workspace
filter isolation. Add Model Monitoring with window, feature and monitoring-tab
selectors and a reset action affecting only monitoring controls. Show overview,
ranked drift, score histograms/quantiles, quality counts, feature details,
window provenance and compact methodology/limitations. Provide readable values
alongside chart canvases and native keyboard-operable controls. Historical
signals are visibly labelled analytical demonstrations, not operational alerts.
No synthetic scenarios are necessary for the historical deliverable; controlled
synthetic unit fixtures exercise shift and quality edge cases separately.

A versioned small JSON aggregate and fixed GET endpoint carry only approved
counts, distributions, summaries, boundaries and provenance hashes. Validate
schemas, finite bounds, count sums, windows and permitted fields before serving;
reject corrupt evidence gracefully. No arbitrary paths, record retrieval,
training imports or heavy request-time inference. Add only the required static
asset/read-only route integration; preserve prediction/security middleware,
rate limits, CORS and local case-store schema/keys. No new dependencies planned.

### Verification and acceptance

Independent hand-computable fixtures and separate NumPy/SciPy calculations
check PSI, KS, JS, missingness and histogram counts. Verify reproducibility by
rerunning aggregate construction and comparing canonical outputs, with model
checksum and population identities fixed. Test malformed/absent aggregates,
unsupported selectors, insufficient samples and aggregate-only responses.
Browser checks exercise selectors/reset, charts, errors, keyboard, viewports,
no desktop outer overflow, navigation to capacity/benchmark/cases and no case
storage changes. Run the complete suite including all six heavy/data-contract
checks, preserve existing scientific text and inventories, capture screenshots,
then commit verified work locally only. Completion remains pending evidence.

## Implemented protocol and historical findings

The owner selected **earliest validation third** on2026-10-31. This supersedes
the training-reference suggestion in the inspection checkpoint. V1,V2,V3 and
all methods/thresholds above were fixed before monitoring calculation. The
completed continuation is the binding scope; the earlier truncated-brief note
is preserved as inspection history, not an outstanding blocker.

Implementation adds `src/drift_intelligence.py` instead of changing the approved
Stage18 monitor/references. This isolates the new protocol's categorical/quality
contracts and validation-only provenance from earlier scientific evidence.
`build_monitoring_aggregate.py` reads ID metadata, parses only authorised
validation features, uses the unchanged artifact for bounded inference and
saves only a14KiB aggregate. Scores are rounded to12 decimal places solely to
canonicalise established parallel summation noise (same convention as Stage19).
No outcome columns are requested; labels are neither needed nor measured.

Two independent full builds produced byte-identical canonical JSON:
`715f88fdd67c6065d7bbe9702241c28041aab67ada72096283561d58372f3f37`.
Each build independently checked14 feature/score distribution comparisons,
using NumPy histograms/formulas and SciPy KS/JS rather than reusing the serving
calculation functions. See `stage22_monitoring_evidence/aggregate-build-first.json`
and `aggregate-build-repeat.json`. Source feature rows/scores exist in memory
only; no new row-level data/prediction artifact is persisted.

### Distribution observations

| Indicator | V2 vs V1 | V3 vs V1 |
|---|---:|---:|
| Amount PSI / KS |0.001968 /0.012776|0.007247 /0.033138|
| C1 PSI / KS |0.000561 /0.007863|0.001431 /0.005481|
| dist1 finite-value PSI / KS |0.003802 /0.015413|0.003400 /0.013901|
| ProductCD JS divergence |0.000609|0.000932|
| card4 JS divergence |0.000121|0.000155|
| Relative daily phase JS divergence |0.002323|0.002817|
| Frozen score PSI / KS |0.016737 /0.055496|0.028846 /0.072251|

All are below the predeclared demonstration investigation rules. This means
**no material shift under these selected rules**, not a conclusion of model
health, distribution equivalence or no deterioration. Ranking uses the largest
indicator divided by its corresponding heuristic: a display aid, not a common
statistical effect scale, significance level or model-importance attribution.

| Score summary | V1 reference | V2 | V3 |
|---|---:|---:|---:|
| Mean uncalibrated score |0.033384|0.031151|0.033824|
| Median |0.013787|0.013470|0.013454|
| q95 |0.095643|0.090463|0.099449|
| Fraction strictly above fixed V1 q95 |5.0028%|4.6333%|5.2607%|

Fixed threshold is0.0956431245855. No score is interpreted as a calibrated fraud
probability; no new AP, precision, recall, fraud-capture or performance claim.
Summary displays disclose rounding; aggregate counts and unrounded numerical
results remain available. Ties and interpolated reference quantiles explain
why the strict-tail reference share need not equal exactly5%.

### Observed quality

`dist1` missingness is51.2705% (12,913/25,186) in V1,54.2740%
(13,670/25,187) in V2 and54.6830% (13,773/25,187) in V3: increases of
3.0035 and3.4124 percentage points. This is a measured completeness change,
although below the fixed5pp quality-investigation heuristic. It remains visible
separately from finite-value drift. Other selected inputs have no missingness
except card4's two missing reference values (0.00794%); later windows have none.
No reference-unseen categorical values, parse failures, infinities, duplicate
validation identifiers or negative amounts were observed. Required source
columns and timestamp/ID alignment checks passed. These checks concern the
selected monitoring inputs and source alignment, not universal proof of data
quality or legitimate transaction behaviour.

### UI, API and security

New hash view `#/monitoring` reuses the existing navigation, theme and Chart.js.
Native window/feature/dimension selectors drive feature investigation ranking,
common-bin distributions, exact bucket counts, score summaries/fixed-tail trend,
missingness/quality counts and three-window trends. Reset changes only these
three monitoring controls; it never clears synthetic filters or browser-local
case data. Methodology is available in a contained expandable panel. Missing,
corrupt or unavailable evidence produces an explicit empty state and Retry.
Unsupported selections explain recovery; insufficient finite samples have null
metrics/chart shares, not zero drift, NaN or invented scores.

`GET /v1/dashboard/monitoring` serves only the fixed schema-validated aggregate;
unsupported write methods return405. It enforces a256KiB file bound, fixed
allowlists, count sums, chronology, finite numerical ranges, score[0,1] bounds,
score-summary consistency, tail consistency and histogram shape. Corrupt or
absent evidence fails closed with503 and a generic message. No dynamic file
path, record lookup, request-time inference or backend case-write endpoint.
The API modifications are router registration and static asset delivery only;
existing prediction implementation, middleware, rate limits and public demo
restrictions are preserved. Existing Render requirements already contain every
library used. No dependency change, telemetry, scheduler, paid service or
Render configuration change; runtime serves a small fixed aggregate.

### Scientific references and interpretation limits

The independent check squares SciPy's Jensen–Shannon **distance** to obtain
base2 divergence; see [SciPy JS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).
KS is used only as the descriptive empirical CDF distance; the independence and
continuous-distribution assumptions for inferential testing are not claimed
for temporally dependent transaction data with ties; see
[SciPy KS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
No p-values, multiple-testing significance claims or automatic retraining.
No synthetic drift scenarios are shipped; deliberately synthetic test fixtures
exercise known shifts/quality failures and never enter historical summaries.

For genuine production monitoring, this demonstration would require authorised
live input/score capture with privacy/retention controls, reliable timestamps,
validated delayed outcomes, representative reference selection, dependency-aware
uncertainty and alert calibration, seasonal baselines, accountable investigation
and escalation owners, service/error telemetry and controlled model governance.
None of those capabilities or live performance is claimed here.

### Final verification evidence

Initial complete regression passed525 tests, zero failures/skips, five inherited
warnings,301.88 seconds. It is retained in
[pre-final viewport run](stage22_monitoring_evidence/pre-final-viewport-regression.log).
Final runtime refinements then improved desktop plot readability and contained
methodology expansion; six focused real-Chrome tests passed in17.37 seconds.
Final complete fresh run against the settled changes: **525 passed, 0 failed,
0 skipped, five existing warnings,292.42 seconds**. See
[full regression](stage22_monitoring_evidence/full-regression.log). All **77 real-Chrome
cases** and all **six heavy/data-contract tests** ran. The warnings are inherited
Starlette deprecation and sklearn logistic penalty/convergence warnings; they do
not concern the deployed frozen Random Forest. The heavy reference regression
uses a disposable independently fitted train-only model and never refits or
overwrites the frozen artifact.

The numerical/API suite passes25 tests. New coverage totals31 tests relative
to the approved494 baseline:20 numerical/provenance/edge/schema tests, five
read-only API tests and six real-Chrome tests. Browser checks cover four required
sizes, native selection/Enter reset, all dimensions and charts, contained desktop
plot areas/methodology expansion, retry/unsupported selections, null shares for
all-missing comparisons and navigation/storage preservation across Stages19–21.
Existing complete-suite tests retain original scoring and public security checks.

An independent source/scientific/security reviewer found and rechecked fixes for
score-bound/cross-field validation and undefined chart shares. Final re-review
found no blockers. See
[independent review](stage22_monitoring_evidence/independent-review.json).

The local public Render-entrypoint check passes all three dimensions at all four
viewports, preserves case storage exactly, and retains the24-record synthetic
queue. Eighteen screenshots include12 exact-viewport captures and six additional
full-page mobile/tablet images, with hashes recorded. Desktop content overflow
is zero. Selected evidence:

- [Desktop feature drift1280×800](stage22_monitoring_evidence/monitoring-features-1280x800.png)
- [Desktop scores1440×900](stage22_monitoring_evidence/monitoring-scores-1440x900.png)
- [Mobile score view, full page](stage22_monitoring_evidence/monitoring-scores-390x844-full.png)
- [Browser metrics](stage22_monitoring_evidence/public-browser-verification.json)
- [Preservation](stage22_monitoring_evidence/preservation.json)
- [Exact changed-file inventory](stage22_monitoring_evidence/changed-files.txt)

A third aggregate build independently reconstructs empirical CDFs with NumPy
searchsorted to verify KS distances as well as checking SciPy; all14 comparisons
pass and the aggregate remains byte-identical. See
[final aggregate verification](stage22_monitoring_evidence/aggregate-build-verification.json).

No synthetic historical scenarios are shipped. Remaining limitations are reused
validation, three short unequal-duration historical windows, selected diagnostic
inputs, heuristic thresholds, missing verified new outcomes and no live capture,
seasonal controls, operational alerting or physical-device/assistive-technology
certification. Browser-local cases retain their previous unauthenticated,
user-editable storage limitations. No claim of production monitoring readiness.


## Local delivery and completion gate

All Stage22 acceptance gates pass for this historical portfolio demonstration:
verified validation aggregates, independent drift/quality checks, functioning
read-only dashboard/API, preserved Stage19–21 functionality and production
security, and a complete passing regression. No technical blocking defect is
known. The original closed scientific workflow body remains verbatim; the new
Stage22 note does not revise prior results or approvals.

Branch: `stage22-model-monitoring`. Exact inventory: **48 files**, listed in
[changed-files.txt](stage22_monitoring_evidence/changed-files.txt). This comprises
implementation, tests, a small aggregate, reproducible build/capture scripts,
protocol/plan/workflow documentation, logs, review/preservation evidence and
18 screenshots. No unrelated styling or dependency change. The local commit SHA
is reported with delivery; no push, merge, deployment or Render change. Main and
integration references are unchanged. Stage23 requires separate owner approval.


## Final review and GitHub PR preparation (2026-10-31)

The owner now explicitly authorises pushing `stage22-model-monitoring` and
opening a PR against `render-dashboard-integration`. This supersedes the prior
local-only delivery restriction; merge, deployment, Render changes and Stage23
remain prohibited. Original implementation head:
`d4995276fd4dec6e93ab33e5d18ba62bb24e568e`.

The complete48-file inventory and committed diff were reviewed. Independent
source/scientific/security review found no release blockers. No product code
correction was required. Release adds three verification files: the reproducible
`scripts/verify_stage22_release.py`, fresh `release-regression.log` and
`release-review.json`. Report/workflow/inventory are updated in place; the
current release inventory is **51 files**. The earlier48 count refers to the
original implementation snapshot.

The independent audit passes **395 scalar/vector assertions**, checking exact
V1/V2/V3 populations, disjointness/full validation coverage, timestamps and
fingerprints, reference-only bins/counts, PSI, direct empirical-CDF KS,
Jensen–Shannon divergence, missingness/invalid-value counts, score means and
quantiles, and fixed-reference-tail fractions. It uses no outcome columns,
metric builders or model fitting. Reproduce with
`.venv/bin/python scripts/verify_stage22_release.py`.

Fresh regeneration is byte-identical to the versioned aggregate at SHA256
`715f88fdd67c6065d7bbe9702241c28041aab67ada72096283561d58372f3f37`.
A fresh clean-profile public-entrypoint Chrome run passes all three dimensions,
charts, reset, storage preservation and the24-record synthetic queue at all four
sizes. Existing18 screenshot checksums remain verified; additional fresh captures
were made in disposable temporary storage to avoid unrelated screenshot churn.
See [release review](stage22_monitoring_evidence/release-review.json).

Render Free compatibility requires no new packages, infrastructure or configuration:
only a fixed14KiB aggregate and existing pinned NumPy/pandas/SciPy stack. No
request-time model inference is introduced. The frozen checksum and previous
science, historical aggregates, case schema/storage contracts, scoring and API
safeguards are unchanged. No live performance claims are made; investigation
thresholds remain expressly heuristic. Scientific limitations above still apply.


Fresh release regression: **525 passed, 0 failed, 0 skipped, five existing
warnings,317.00 seconds**. Includes all77 real-Chrome cases and all six heavy/data-
contract tests. See [release regression](stage22_monitoring_evidence/release-regression.log).
No test was excluded. Existing warnings remain the Starlette deprecation and
sklearn logistic penalty/convergence warnings. No blocking defect remains;
controlled merge still requires separate owner approval.
