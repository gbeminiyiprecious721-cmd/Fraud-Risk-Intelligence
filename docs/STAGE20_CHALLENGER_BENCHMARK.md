# Portfolio enhancement Stage 20 — Challenger Model Benchmarking

**Status: Phase A protocol and all five pinned dependency additions explicitly APPROVED by the owner on 2026-10-30. Installation/training have not begun because automatic approval review failed with a usage-limit error. No challenger has been trained, and no benchmark result is claimed.**

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

## 6. Dependency approval required before installation

Read-only package-index and resolver checks found ARM64-compatible wheels.
No packages have been installed or upgraded. OpenMP's dylib is already
present under `/opt/homebrew/opt/libomp/lib/`; successful native imports are
still to be checked after installation, not assumed from the filename.

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
The next step is dependency installation/import checks, followed by the
authorized Stage 20 implementation/training once tool review is available.
No comparative results, winning model, training measurements or functional
benchmark dashboard exist yet; Phase A is not Stage 20 completion.

## References reviewed during design

- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): CPU/deterministic settings and bounded model configuration.
- [LightGBM Python introduction](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html): matrix interfaces, categorical support and early stopping.
- [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html): explicit eval_set, early stopping and best-iteration handling; current stable docs identify 3.4.2, while the inspected installable package index offers 3.4.1, so imports/API smoke checks remain required against the pinned installed version.
- [CatBoost overfitting detection](https://catboost.ai/docs/en/references/training-parameters/overfitting-detection): patience-based stopping.
- Existing local reports: `docs/STAGE9_TUNING_DESIGN_REPORT.md`, `docs/STAGE10_EVALUATION_PROTOCOL.md` (published protocol/summary only), `docs/PRODUCTION_PIPELINE_REPORT.md`, `docs/STAGE19_CAPACITY_SIMULATOR.md`.


## Owner approval and tool-review blocker

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
Stage 20 remains incomplete until actual experiments and all required
regressions/browser checks pass; no comparative results or winner exist.
