# Portfolio enhancement Stage 20 — Challenger Model Benchmarking

**Status: Final release review verified. Owner authorised branch push and a GitHub PR targeting `render-dashboard-integration`. Merge, deployment and model promotion remain unauthorised.**

This is a portfolio enhancement, separate from the original closed ML
Stages 0–19. The owner requested Stage 20 on 2026-10-30. Work is confined
to `stage20-challenger-benchmark`, created from the verified Stage 19 merge
`5b6e90d8bf5665d4a316c063e0a6a1b0737feab7` on
`render-dashboard-integration`. The owner identifies that merge as the last
confirmed live deployment. This phase does not query or modify Render.

## 1. Inspection and permitted evidence

The repository was clean when inspected. The existing artifact is a trusted,
train-only reconstruction of the frozen RF configuration, version 1.0.0:
SHA-256 `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
It is loaded for inference only; no fit, overwrite, parameter mutation or
production scoring change is authorized.

| Population | Rows | Relative TransactionDT range | Purpose |
|---|---:|---|---|
| Original train | 438125 | 86400–11093269 | Challenger fitting and internal early stopping only |
| Proposed inner fit: earliest 85% of train | 372406 | 86400–9256765 | Fit inner preprocessing and provisional challengers |
| Proposed inner stop: latest 15% of train | 65719 | 9256767–11093269 | Select boosting round count only |
| Original validation | 75560 | 11093842–13452406 | Same final comparison population for all four models |

These counts/time boundaries were computed from split metadata using ONLY
TransactionID, TransactionDT and partition, not closed final-test labels.
The train and validation ID fingerprints match existing artifact/Stage 19
provenance. The inner boundary does not split a timestamp tie. There is a
strict chronological separation between inner fit, inner stop and validation.
No label-maturity embargo is invented: the project's original label-delay
uncertainty remains unresolved and explicitly carried forward.

**An additional pristine common holdout cannot be claimed.** The incumbent
was fitted on the entire train partition. Any subset carved from train is
therefore in-sample for it. The original validation population was consulted
15 times in Stage 7, including six decision-influencing consultations, then
used in subsequent development/preflight/Stage 19 analysis. Splitting it now
does not erase prior use. Inner stop is a tuning partition, not an independent
incumbent-comparison set. The main comparison will therefore transparently
use the established validation population as **development backtesting**.

Stage 9's budgeted 18-of-20 RF candidate search used chronological, tie-safe
training-only CV. Its CV mean AP, Stage 10's published final-test summary
metrics and Stage 19's validation metrics describe different measurement
contexts. No published final-test number will be inserted into the new
comparison table. No closed final-test rows, predictions or derived evaluation
artifacts will be opened for new experiments.

## 2. Proposed controlled comparison and alternatives

**Recommended:** hold raw feature eligibility and final input representation
constant, then compare the three bounded challenger configurations with the
unchanged incumbent on identical validation rows.

The existing 392 eligible raw fields and approved imputation/one-hot/
missingness-indicator construction remain the reference. Experimental
Stage 6 features and identity fields remain excluded.

- Inner selection: fit a separate approved preprocessing pipeline on
  **inner fit only**, including its missingness-indicator decisions and
  categorical vocabulary. Transform inner stop without fitting. Never use
  the full-train-fitted incumbent transformer to claim an inner-stop-safe
  preprocessing fit.
- Final challenger fitting: use the trusted artifact's already train-fitted
  transformer and exact 813-feature ordering. It has not seen validation.
  Transform in bounded batches; store private dense float32 matrices using
  memory maps. Dense representation preserves actual numeric zeros rather
  than silently making XGBoost's omitted sparse entries act as missing values.
- Score the RF on the same final feature ordering, without altering its
  fitted preprocessor/model. Verify matrix-path RF predictions against the
  unchanged raw inference function within the established numerical tolerance.
- Class imbalance: binary log-loss objectives, no resampling, no automatic
  class weights or scale_pos_weight search. The frozen RF also uses no class
  weighting. Primary ranking metrics and fixed review budgets address the
  rare-positive decision. No threshold or calibration claim is introduced.

**Alternative deferred:** per-library native categorical/missing-value
pipelines. They may better exploit CatBoost/LightGBM capabilities, but would
compare complete feature/pipeline systems as well as algorithms and require
additional leakage-safe preprocessing/selection design. The proposed study
will explicitly be a **shared-representation, budget-limited comparison**,
not a test of the maximum attainable performance of each library.

No random split or newly relabelled "independent" holdout is proposed.

## 3. Proposed configurations fixed before training

Hardware observed: arm64, Python 3.12.15, 10 logical CPUs, 64 GiB RAM.
CPU only; one model process at a time; no paid compute or search service.
All challenger seeds are 42. The unchanged RF retains its original
`n_jobs=-1`, using this host's available cores; challengers use 10 threads
on the same host. Actual effective concurrency will be recorded.

Common challenger budget: maximum **400 boosting rounds**, learning rate
**0.05**, depth at most **6**, early-stopping patience **30** on inner-stop
**binary log loss**, with no external validation supplied to early stopping.
Monitoring log loss is deliberately common to all libraries; AP remains the
primary final comparison metric, not a silently substituted native PR-area
metric. If the round ceiling is reached, report budget limitation rather
than claiming convergence.

| Model | Fixed proposed settings |
|---|---|
| Incumbent RF | Existing artifact only: n_estimators=100, max_depth=15, min_samples_leaf=5, max_features=sqrt, random_state=42, class_weight=None, n_jobs=-1; no retraining |
| LightGBM 4.7.0 | LGBMClassifier; objective=binary, metric=binary_logloss, n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1, max_bin=255, random_state=42, n_jobs=10, deterministic=True, force_col_wise=True; patience-30 callback |
| XGBoost 3.4.1 | XGBClassifier; booster=gbtree, objective=binary:logistic, eval_metric=logloss, tree_method=hist, device=cpu, n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, max_bin=256, random_state=42, n_jobs=10, early_stopping_rounds=30 |
| CatBoost 1.2.10 | CatBoostClassifier; loss_function=Logloss, eval_metric=Logloss, iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=3, bootstrap_type=Bernoulli, subsample=0.8, rsm=0.8, boosting_type=Plain, border_count=128, random_seed=42, thread_count=10, task_type=CPU, allow_writing_files=False; patience=30, use_best_model=True |

Training sequence per challenger: one inner-selection fit, convert the
library's best iteration to an explicit tree count (handle 0/1-based API
conventions), then fit a fresh **challenger** on full train for that fixed
count without external early stopping. Maximum six substantive fits total,
no candidate grid/search and no validation-driven retuning. Record all
resolved parameters, library versions and seeds in the experiment evidence.

Resource ceilings proposed: **15 minutes per fit and 24 GiB process RSS**.
If exceeded, record infeasibility and stop/report the blocker; do not silently
shrink the eligible population or change settings after viewing validation
results. Import/ABI smoke checks precede training. Native-runtime incompatibility
or dataset/schema/fingerprint failure also stops the run rather than producing
placeholder metrics.

## 4. Proposed comparison definitions and uncertainty

All four models use identical ordered validation IDs and verified labels.
Canonicalize ranking scores to 12 decimals and break ties by ascending
TransactionID, matching Stage 19. Preserve the same validation-ID fingerprint.

- Primary: sklearn **Average Precision**, named exactly. Trapezoidal PR-AUC,
  if included, is a separately labelled secondary measurement, never an AP alias.
- Secondary: ROC-AUC, precision/recall at top 1% and top 5%, using
  K=ceil(N×fraction), and exact TP/FP/FN/TN at the same 201 workload checkpoints
  as Stage 19. K=0 precision is undefined; no-positive recall is undefined.
- Independently verify counts/metrics using separate selection masks and
  sklearn calculations. Require nondecreasing cumulative TP and recall.
- Proposed uncertainty: **1000 paired chronological-block bootstrap draws**,
  seed 42. Fix 20 approximately equal-row-count contiguous blocks using validation
  TransactionDT/ID ordering only, advancing boundaries past timestamp ties; use identical sampled indices for all
  models in each draw. Calculate descriptive percentile intervals for AP
  and paired AP differences versus RF; log any degenerate-label draws and
  withhold intervals if fewer than 900 of 1000 draws are valid.
- These are exploratory, **conditional fixed-model historical block-resampling
  intervals**, not unbiased future-performance intervals. They assume blocks
  capture relevant dependence adequately; cross-block/entity dependence,
  temporal nonstationarity, training variability and validation-selection bias
  are not removed. The three paired comparisons are not a basis for an
  unadjusted significance or automatic-promotion claim.

Highest observed validation AP identifies the strongest challenger **within
this protocol and budget**; report AP differences, capture/precision trade-offs,
uncertainty, operational cost and limitations. An interval crossing zero must
not become a claim of proven superiority. No production promotion is authorized.

## 5. Efficiency, reproducibility and private artifacts

Measure preprocessing separately from model fitting. Record inner-selection,
final-fit and total challenger training duration. Frozen RF current training
cost is **not measured / not applicable**, never zero; any manifest historical
training time must be clearly labelled historical and excluded from a current
training-speed ranking.

For comparable inference timing, use the same pretransformed validation
matrix and rows, warm up, then repeat five bounded scoring passes. Record
batch size, wall time, rows/second and effective thread settings. Also record
preprocessing time separately. Sample RSS per isolated model process during
training and scoring; distinguish total peak RSS from incremental peak and
include sampling resolution/measurement limits.

Preserve each challenger in a distinct gitignored local artifact directory,
using its native model format. Never save over the production joblib or
manifest. Keep matrices, row-level predictions, labels and model artifacts
private and out of the public API/repository. Verify reload/inference
consistency and deterministic repeat inference; record platform/thread/
library limitations instead of promising cross-platform bitwise identity.
Before and after all experiments, check the frozen artifact checksum and
unchanged production preprocessing/evaluation/security code.

## 6. Approved optional dependencies

Read-only package-index and resolver checks found ARM64-compatible wheels.
The five owner-approved packages are now installed; no existing package was upgraded. OpenMP's dylib is already
present under `/opt/homebrew/opt/libomp/lib/`; all three native imports passed. `pip check` reports no broken requirements.

Proposed exact additions:

- lightgbm==4.7.0
- xgboost==3.4.1
- catboost==1.2.10
- graphviz==0.21 and plotly==7.1.0 (CatBoost's declared Python dependencies;
  not a dashboard redesign and not a system Graphviz installation)

The resolver dry run proposes **no installed-package upgrades**. Existing
numpy 2.5.4, pandas 3.1.0, sklearn 1.9.1, scipy 1.18.1, psutil 7.2.2 and
other dependencies remain fixed. After approval, use a constrained optional
benchmark requirements manifest and verify no existing distribution changes.
Do not add training libraries to production `requirements-render.txt`, or
make their imports necessary for runtime API/dashboard operation.

## 7. Proposed dashboard, validation and delivery

Add an eighth view, Model Benchmarking, using existing navigation, Chart.js
and premium dark components. Serve one versioned aggregate from a fixed,
read-only endpoint. Include deployed-RF/experimental labels, metric comparison,
capacity precision/recall, training/inference/memory measurements, and concise
findings with the population/selection limitations always visible.

Only measured results may populate the view; no invented fallback or missing-
model zero values. No training or artifact-loading public endpoint. Keep all
current scoring/security paths and the Stage 19 simulator unchanged.

Before declaring completion: add independent metric/split/provenance tests,
same-population and inference-reproducibility tests, aggregate API boundary
tests and real-browser interaction/layout tests; run the complete suite
including heavy/data-contract tests. Preserve the closed scientific workflow
verbatim and record this enhancement separately. Commit verified work locally
only, with no push, merge, deployment, main/live-branch change or promotion.

## 8. Decision gate

The owner explicitly approved this experimental protocol and the five exact
dependency additions above on 2026-10-30. Do not request this approval again.
At that checkpoint, the next step was dependency installation/import checks
and the authorised Stage 20 implementation/training. Resumed execution is
recorded below.
Execution evidence and final verification are recorded below as completed; the original approval is unchanged.

## References reviewed during design

- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): CPU/deterministic settings and bounded model configuration.
- [LightGBM Python introduction](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html): matrix interfaces, categorical support and early stopping.
- [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html): explicit eval_set, early stopping and best-iteration handling; current stable docs identify 3.4.2, while the inspected installable package index offers 3.4.1, so imports/API smoke checks remain required against the pinned installed version.
- [CatBoost overfitting detection](https://catboost.ai/docs/en/references/training-parameters/overfitting-detection): patience-based stopping.
- Existing local reports: `docs/STAGE9_TUNING_DESIGN_REPORT.md`, `docs/STAGE10_EVALUATION_PROTOCOL.md` (published protocol/summary only), `docs/PRODUCTION_PIPELINE_REPORT.md`, `docs/STAGE19_CAPACITY_SIMULATOR.md`.


## Owner approval and historical tool-review interruption

Owner response on 2026-10-30: **“Approve protocol and dependencies.”** This
covers the documented protocol and LightGBM 4.7.0, XGBoost 3.4.1, CatBoost
1.2.10, graphviz 0.21 and plotly 7.1.0, without upgrades to existing packages.

The attempt to run `git branch --unset-upstream` with required elevated
permissions was not executed: automatic approval review could not complete
because its service reported a usage limit and a retry time of **5:24 PM**.
The error explicitly states this is a review failure, not a determination
that the action is unsafe, and directs the agent not to bypass the check.
Consequently the subsequent Git staging/commit calls were not executed.
The branch may still inherit `origin/render-dashboard-integration` as its
upstream; no push is authorized or attempted. The dependency dry run was
read-only and installed nothing. No challenger training, production-model
change, push, merge or deployment occurred.

Protocol/dependency approval persists. Once review access is restored,
continue from this checkpoint without repeating the approval question.
This historical interruption preserved approval; the resumed experiment and
verification evidence below supersede its pre-execution status.


## Resumed execution notes

The usage-limit blocker is resolved. Existing branch/upstream configuration
was left untouched; no unnecessary Git configuration changes were made.
Local commit `bae94ca` preserved the approved pre-training protocol and
inspection/installation evidence. `dependency-installation.json` verifies the
five exact additions, native imports, unchanged existing distributions and
frozen checksum. Training dependencies are optional and absent from Render
requirements and runtime import paths.

The first XGBoost inner-selection fit completed but its evidence serialization
failed because the library exposes a `NaN` missing-value sentinel in its default
parameters. The model was not retained by that failed process. The writer now
records nonfinite **configuration sentinels as strings**; metrics remain finite
strict JSON. The identical fit was repeated with no parameter, population or
selection change and before any challenger validation comparison. Failure log
and resource evidence are retained. This is **seven fit executions for six
planned fits**, including one technical retry, exceeding the nominal six-fit
execution count without adding a candidate or conducting a search. This
implementation deviation and its wasted runtime are disclosed rather than
hidden in training-speed comparisons. Main comparison training times use the
successful six fits; the failed process consumed approximately 30 wall seconds.

RSS is sampled every 100 ms in isolated fresh processes. The recorded baseline
is the first startup sample above 50 MiB, so the increment includes subsequent
imports, model loading and mapped input pages; it is **not** a pure model-only
allocation. Peaks may miss short spikes. Scoring uses the same ordered dense
float32 validation matrix in 4096-row batches, one warm-up and five passes.
Throughput is based on median complete-pass time, excluding preprocessing.
Dense numeric zeros are retained as values for all libraries.


## Measured comparable results

All values below use the same 75,560 validation rows with 2,617 positives
(3.46347% prevalence), not closed final-test outcomes or synthetic labels.
Scores are rounded to 12 decimal places; ties use ascending TransactionID.
Top 1% means K=756, top 5% means K=3,778. AP is sklearn Average Precision;
trapezoidal PR-AUC is separately recorded in the aggregate and is not an AP alias.

| Model | AP | ROC-AUC | P@1% | R@1% | P@5% | R@5% |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.447002 | 0.873333 | 0.817460 | 0.236148 | 0.348332 | 0.502866 |
| LightGBM | 0.545903 | 0.909374 | 0.883598 | 0.255254 | 0.411593 | 0.594192 |
| XGBoost | 0.551777 | 0.910438 | 0.887566 | 0.256400 | 0.411064 | 0.593428 |
| CatBoost | 0.490482 | 0.877544 | 0.865079 | 0.249904 | 0.375596 | 0.542224 |

Each model has 201 independently verified capacity checkpoints including
zero and full workload. Independent binary selection masks/sklearn confusion
matrices verify TP/FP/FN/TN and precision/recall; cumulative TP/recall is
monotonic. The RF matrix path matches the original raw inference path within
2.23e-16 absolute probability (tolerance 1e-12), and its capacity counts match
the unchanged Stage 19 aggregate exactly.

| Model | AP descriptive 95% interval | AP difference vs RF, 95% interval |
|---|---|---|
| Random Forest | 0.401867–0.491920 | 0.000000–0.000000 |
| LightGBM | 0.500218–0.590131 | 0.084464–0.116437 |
| XGBoost | 0.505687–0.595575 | 0.088628–0.123317 |
| CatBoost | 0.447164–0.531096 | 0.031101–0.056416 |

All 1,000 paired draws were valid, zero degenerate draws. Twenty contiguous,
tie-safe chronological blocks and seed 42 are fixed across models. Block
resampling is implemented with row multiplicity weights, exactly equivalent
to repeated sampled rows; independent sklearn tests verify equivalence,
including tied predictions. These conditional intervals exclude training and
selection uncertainty and are not adjusted for three comparisons. Positive
difference intervals must not be presented as proof of future superiority.

## Measured resource costs and reproducibility

| Model | Selected rounds | Inner fit (s) | Full-train fit (s) | Total fit (s) | Median validation scoring (s) | Rows/s | Peak fit RSS (GiB) | Peak scoring RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | N/A | N/A | N/A | N/A | 0.500111 | 151,086 | N/A | 0.434 |
| LightGBM | 400 | 14.886 | 16.055 | 30.941 | 0.213651 | 353,662 | 2.914 | 0.394 |
| XGBoost | 399 | 28.273 | 23.938 | 52.211 | 0.079728 | 947,723 | 2.473 | 0.413 |
| CatBoost | 399 | 41.579 | 48.083 | 89.662 | 0.124883 | 605,046 | 3.429 | 0.466 |

All three inner fits reached the 400-round budget before a full patience window
could establish convergence (selected tree counts: LightGBM 400, XGBoost 399,
CatBoost 399). No validation-driven retuning or additional candidate search
was performed. No current RF training duration is measured; N/A is not zero.
Native reload probabilities and five repeated scoring passes match to absolute
1e-12; platform-level bitwise training reproducibility is not claimed.
All fits stayed below the 15-minute / 24-GiB ceilings. Preparation peak was
14.43 GiB. Inner preprocessing fit took 12.218s; full-train and validation
transforms took 7.838s and 1.598s, respectively, measured separately from fits.

Exact configured parameters, selection/final timings, artifact digests, all
five scoring-pass durations, thread settings, process resource measurements,
provenance and bootstrap definitions are in
[the result summary](stage20_challenger_evidence/result-summary.json).
The full aggregate is `ui/benchmark-v1.json`. The five pinned versions are in
`requirements-benchmark.txt`; Render dependencies are unchanged.

Reproduce locally using the same approved data/environment:

```sh
.venv/bin/python scripts/run_stage20_benchmark.py all
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_benchmark_browser.py
```

The runner executes each phase sequentially in fresh processes with resource
monitoring, fails on dependency/fingerprint/schema/integrity mismatch, and
never loads experimental row-level outputs into the public server. It does
not resume stale matrices or silently downsample. It overwrites only private
Stage 20 caches/native challenger files and the Stage 20 aggregate/evidence;
rerunning the full command is a new execution of the same fixed protocol.
Private matrices, labels, identifiers, scores and native models remain under
gitignored `artifacts/stage20/`; they are not committed or exposed. Data are
not bundled, so reproduction requires the authorised local source files.

## Decision framework and recommendation

**XGBoost is the strongest observed challenger within this exact representation
and compute budget**, with AP improvement 0.104776 over the frozen reference.
At 1% review it captured 671 positives versus RF's 618; at 5%, 1,553 versus
1,316. LightGBM captured two more positives than XGBoost at 5% (1,555), used
less successful-fit time, and is a credible operational alternative. The
small XGBoost–LightGBM AP gap (0.005874) is not evidence of established future
superiority. CatBoost improved over RF but had lower AP and higher fit cost
than the other challengers in this shared one-hot representation; this is
not a verdict on native-categorical CatBoost configurations.

Detection, review capacity, latency, memory, dependency footprint and
maintenance must be assessed jointly. XGBoost had the fastest measured
pretransformed scoring on this host; timings are not a production SLA.
The incumbent has an existing frozen serving contract and no new dependency
cost. Each challenger adds a native runtime/versioning burden and would need
separate artifact/schema/security verification, deployment sizing and
monitoring. CatBoost has two additional approved Python dependencies;
Graphviz's Python package does not install a system executable. Production
requirements and inference imports deliberately remain unchanged.

**Keep Random Forest deployed.** Future promotion requires separate owner
approval, a valid uncontaminated evaluation plan, prospective chronological
and label-maturity assessment, dependence/drift analysis and operational
validation. Existing validation reuse cannot be cured by bootstrapping or
carving another subset from previously consulted data. No final-test access,
new calibration/threshold, financial benefit or automated-decline claim is
part of this stage.

## Dashboard and security scope

`#/benchmark` is the eighth view in the existing premium dark dashboard.
It shows four AP cards, an interactive metric chart, review-capacity recall
curves, exact precision/recall and confusion counts, training/scoring costs,
role labels and expandable paired intervals/limitations. Slider, numeric
input, keyboard, reset and chart pointer selection share exact precomputed
checkpoints. Synthetic filters are hidden in this historical workspace.
The RF/experimental roles and reused-validation warning remain visible.

GET `/v1/dashboard/benchmark` serves a single fixed versioned JSON file with
`Cache-Control: no-store`. Missing evidence returns 503 without invented
fallback results; writes are rejected with 405. Only aggregate counts,
metrics, configurations, provenance hashes and measured resource summaries
are exposed. No identifiers, row labels/scores, user-selected file paths,
training or native model-loading endpoint exists. `src/api.py` only includes
the new router; `src/render_demo.py` only serves the new JavaScript asset.
Existing fixture allowlists, production scoring, batch caps, rate limits,
CORS and security handlers are unchanged.


## Evidence inventory and practical limits

- `preparation.json`: authorised row counts/fingerprints, chronological ranges,
  inner-only preprocessing fit, exact final feature ordering and transform costs.
- `environment.json` and `dependency-installation.json`: all 132 original
  distribution versions unchanged; exactly five approved additions, native imports,
  host/Python, RF effective jobs and dependency health.
- `*-resources.json`: isolated phase timings, sampled RSS baseline/peak/increment,
  100-ms sampling, exit status; separate failed-XGBoost-attempt record.
- `result-summary.json`: per-model exact configurations, metrics, uncertainty,
  training/inference evidence and native artifact hashes; no row-level output.
- `aggregate-reproducibility.json`: byte-identical aggregate recomputed from the
  same private scores with fixed paired bootstrap seed, without retraining.
- `preservation.json`: frozen model checksum, unchanged original scientific
  record hash and protected code/dependencies, local branch references.

The pre-existing Chart.js CDN remains a network dependency. An unavailable
chart library leaves exact aggregate values visible; missing aggregate evidence
shows an explicit unavailable state with retry. Mobile tables scroll internally;
normal document scrolling on narrow screens and expanded explanatory details
is intentional. Default desktop results fit available content height; small
screens/zoom retain content access rather than clipping scientific disclosures.
The five existing regression warnings are carried forward; no package upgrade
was made to silence them.

No known scientific blocker prevents **exploratory development benchmarking**.
Validation contamination, budget-limited convergence, label-delay uncertainty,
platform-specific timings and unmeasured training variance limit conclusions.
A future production release/promotion remains a separate approved stage. This
work creates no PR and does not touch Render settings or trigger deployment.


## Recovery checkpoint

At resume, HEAD was `5b6e90d8bf5665d4a316c063e0a6a1b0737feab7` on
`stage20-challenger-benchmark`. Saved work comprised the modified Stage 20
workflow note, the untracked approved protocol report and design-inspection
JSON. No challenger implementation, installed benchmark packages, fitted
challengers or benchmark UI existed. Those documents were preserved and
extended. No reset, discard, upstream change or unrelated reversion occurred.
The inherited integration upstream was deliberately left unchanged because
no push is authorised. Existing production/scientific files were inspected
and preserved; changes in the shared API/server files are limited to the
aggregate router and static JavaScript route shown in the review diff.


## Final regression and browser verification

The full regression run passed **453 tests, 0 failed, 0 skipped**, with five
pre-existing warnings, in 199.38 seconds. This includes all six heavy/data-
contract tests (inventory recorded separately), all 31 real-Chrome cases,
and all existing API/security/scoring, navigation, filters, charts, Kanban,
Back to Dashboard and Stage 19 simulator regressions. The heavy reference
procedure fits disposable temporary reference models; it does not overwrite
the frozen incumbent or open closed final-test data. There were 431 baseline
tests plus 22 Stage 20 tests (12 numerical/split/reader/bootstrap tests, five
aggregate/public API tests and five additional browser cases).

- [Full regression log](stage20_challenger_evidence/full-regression.log)
- [Heavy/data-contract inventory](stage20_challenger_evidence/heavy-test-inventory.log)
- [Final focused checks](stage20_challenger_evidence/final-focused.log)
- [Screenshot capture log](stage20_challenger_evidence/browser-capture.log)
- [Viewport measurements](stage20_challenger_evidence/viewport-metrics.json)
- [1280×800 benchmark](stage20_challenger_evidence/screenshots/benchmark-1280x800.png)
- [1440×900 benchmark](stage20_challenger_evidence/screenshots/benchmark-1440x900.png)
- [Mobile benchmark](stage20_challenger_evidence/screenshots/benchmark-390x844.png)

Eighteen screenshots under `stage20_challenger_evidence/screenshots/` cover
all eight dashboard views at 1440×900 and 1280×800, plus benchmark at 768×1024
and 390×844. Real rendered measurements show no unnecessary outer content
scrolling for default desktop views and no body horizontal overflow at any
size. Charts, dark panels, existing spacing/typography and active navigation
retain the approved design. Mobile uses intentional document/table scrolling.
Expanding scientific details may require content scrolling; disclosures are
not clipped. The AP cards explicitly retain AP units when another comparison
metric is selected. Physical-device assistive-technology review is not claimed.

Initial browser setup was blocked by sandbox loopback permissions; rerunning
with authorised local-server access resolved that environment issue. Two new
desktop layout tests then identified duplicated header/paragraph space and
unbounded child minimums in the new view. Measured compact layout corrections
removed the overflow; all final browser checks pass. Existing components were
preserved. No failed test is excluded from the final full run.

Final preservation checks confirm the exact frozen SHA256, byte-identical
original scientific record section, unchanged production preprocessing/scoring
and Render dependencies, unchanged main/integration refs and ignored private
artifacts. `git diff --check`, Python compilation and `pip check` pass. The
remaining limitations are scientific/operational disclosures above; there is
no known unresolved functional defect or local implementation blocker.

An independent read-only final review of the complete diff found no Critical,
Important or Minor findings. Its boundary, preservation and protocol checks
are recorded in [independent-review.json](stage20_challenger_evidence/independent-review.json).


## Owner-authorised GitHub release review

The owner explicitly authorised final review, a fresh full regression run,
branch push and a PR targeting `render-dashboard-integration`, starting from
implementation commit `cf4fea1141c7f721f59300a8afb17dae48a73627`.
Merge, deployment, Render settings changes and model promotion are not authorised.
The earlier local-only delivery and its preservation evidence remain historical
records; this section records the subsequent, separately authorised PR phase.

The complete committed diff was reviewed. No implementation defect was found.
The only new work is the reusable independent release verifier, fresh review
evidence and this documentation/state update; trained models, experimental
settings, measured results and dashboard implementation are unchanged.

Fresh `.venv/bin/python -m pytest -q`: **453 passed, 0 failures, 0 skips**,
five existing warnings, 197.72s. This includes all six heavy/data-contract
checks and 31 real-Chrome cases. Full output:
[release-review-regression.log](stage20_challenger_evidence/release-review-regression.log).
The browser cases verify benchmark metric/chart/capacity/reset/navigation,
responsive layouts, failure recovery and all existing dashboard/scoring flows.

`.venv/bin/python scripts/verify_stage20_results.py` independently re-read
**only authorised validation labels**, verified exact chronological ordering,
75,560 eligible rows, 2,617 positives, identical validation-ID fingerprint
`14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc`,
and reloaded all four native models for inference on the same cached matrix.
Reload differences were zero for challengers and 2.23e-16 at most for RF.
Sklearn AP, ROC-AUC, separately labelled trapezoidal PR-AUC, top-1%/top-5%
metrics and all **804** capacity confusion/precision/recall checkpoints match.

The independent verifier reconstructs the 20 chronological tie-safe blocks
and uses sklearn AP on **literal resampled row indices** for all 1,000 paired
draws, rather than invoking the benchmark's weighted AP/bootstrap routines.
All 1,000 draws were valid, none degenerate; every AP and paired-difference
interval matches to absolute tolerance 1e-12. Evidence:
[release-independent-verification.json](stage20_challenger_evidence/release-independent-verification.json).
No challenger was retrained for release review.

Preservation was rechecked against Stage 19 base
`5b6e90d8bf5665d4a316c063e0a6a1b0737feab7`: frozen artifact checksum and
manifest, original scientific record section, production preprocessing/scoring,
Stage 19 simulator and production dependencies are unchanged. Shared server
diffs contain only the new aggregate-router and static-asset additions;
security handlers are otherwise byte-identical. No private model, matrix,
row-level labels/scores or data file is tracked. Code inspection and current
verification preserve the closed final-test boundary. `pip check`, Python
compilation and `git diff --check` pass.

Remote preflight found integration still at the approved Stage 19 base and
main still at `52c01c919cb40cd2bedf1a41794a73e38c6cf737`; the Stage 20 remote
branch did not yet exist. Only the Stage 20 branch will be pushed. PR metadata,
exact head and changed-file inventory will be checked after creation. Existing
selection/generalisation limitations and the explicitly disclosed technical
fit retry remain unchanged. No unresolved implementation blocker was found.
