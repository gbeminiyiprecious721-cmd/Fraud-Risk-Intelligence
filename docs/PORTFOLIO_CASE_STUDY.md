# Fraud Risk Intelligence — from ranking evidence to investigation decisions

## Problem

A fraud model is useful to an investigator only if it helps decide what to review
within limited capacity. This portfolio project frames that decision explicitly:
rank transactions for human review, show the trade-off between workload and
historically labelled fraud capture, and make model limitations visible.
Automatic declines are outside scope. The setting is simulated; there is no
bank client, real analyst workforce or verified financial return.

## Data

The research uses [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data),
with data provided by Vesta. The labelled transaction table has 590,540 records
and 394 columns. A 144,233-record identity table joins by TransactionID, with
partial coverage. Approximately 3.5% of transaction labels indicate fraud.
Many inputs are anonymised and time is relative, limiting business interpretation
and claims about point-in-time availability.

The project checks unique keys, schema differences, joins and missingness before
modelling. Chronological partitions separate 438,125 training records, 75,560
validation records and the 76,855-record closed final evaluation. Preprocessing
fits imputation values and categorical encodings only on training data. Later
research explicitly reuses validation rather than presenting it as a fresh
independent holdout. Restricted raw data is not published in the repository.
Sources: [validation](DATA_VALIDATION_REPORT.md), [EDA](EDA_REPORT.md) and
[leakage boundaries](LEAKAGE_BOUNDARIES_REPORT.md).

## Methods and model selection

Average Precision measures ranking over the precision–recall curve; precision
and recall at a fixed review budget connect that ranking to the operating
question. Plain accuracy would hide the consequences of class imbalance.
Baseline experiments and expanding-window chronological search led to a frozen
100-tree Random Forest with depth 15, minimum leaf size 5 and square-root feature
subsampling. The approved preprocessing uses 392 raw inputs and yields 813
ordered transformed columns, including training-selected missingness indicators
and one-hot categorical columns. Separate engineered-feature prototypes were
not adopted into the frozen model.

The original fitted Stage 10 object was not serialized. The serving artifact is
a later reconstruction of the same frozen configuration and training procedure,
independently checked for schema/prediction equivalence. Historical final-evaluation
results must not be described as a fresh evaluation of that specific joblib.
The artifact checksum and ordered schema are part of the serving contract.
Sources: [tuning](STAGE9_TUNING_DESIGN_REPORT.md),
[evaluation](STAGE10_EVALUATION_PROTOCOL.md),
[artifact verification](PRODUCTION_PIPELINE_REPORT.md).

## Results and business interpretation

The original held-out evaluation reported **AP 0.42784**, **ROC-AUC 0.86637**,
**precision at top 1% 0.8583** and **recall at top 1% 0.2461**. At that budget,
769 reviews contained 660 labelled positives. These published historical
summaries support investigation prioritisation in that population. They do not
measure prevented loss, current fraud capture or real staffing requirements.
The final-evaluation data and row-level predictions remain closed.

The capacity simulator translates the different, reused validation population
into 201 review-budget checkpoints. At approximately 1% capacity, 756 reviews
contain 618 of its 2,617 labelled positives: precision 81.75% and recall 23.61%.
An analyst can examine how greater workload changes capture and precision,
without pretending that the synthetic dashboard's labels are evaluation truth.
Sources: [capacity protocol](STAGE19_CAPACITY_SIMULATOR.md) and
[versioned aggregate](../ui/capacity-v1.json).

Later benchmarking compares RF, LightGBM, XGBoost and CatBoost on those same
75,560 validation records. Their AP values are **0.44700, 0.54590, 0.55178 and
0.49048**, respectively. There are 804 capacity checkpoints and 1,000 paired
bootstrap draws over 20 chronological blocks. XGBoost has the highest observed
AP under the experiment's representation and resource budget, but validation
reuse, limited tuning and uncertainty prevent a claim of proven production
superiority. The frozen Random Forest remains deployed. Neither these values
nor their differences should be compared directly with the separate original
final-evaluation population. [Benchmark methods and limitations](STAGE20_CHALLENGER_BENCHMARK.md).

## Engineering the evidence into an application

The dashboard separates synthetic operational interactions from historical
research workspaces. Overview, Queue and Analytics show deterministic synthetic
transactions scored by the frozen model. The capacity/benchmark views read
approved aggregates. Case Management adds explicit case creation, fictional
assignment, priority, reasoned status transitions, notes and activity history.
Cases persist in a validated browser-local store isolated by demo scope; no
backend case-write endpoint or authenticated case platform is implied. Status,
Analyst and Priority filters operate on opened cases, while unopened seeded
statuses remain display context. Clearing filters preserves cases; demo reset
requires explicit confirmation.

Historical monitoring divides validation into three disjoint chronological
windows of 25,186, 25,187 and 25,187 records. PSI, descriptive KS distance,
Jensen–Shannon divergence and missingness checks expose distribution changes
without claiming production monitoring or measured accuracy decay. Explainability
maps all 813 columns to 392 raw groups, using global MDI and offline TreeSHAP on
512 label-free validation examples. All 240 synthetic local explanations
reconcile to frozen-model scores; public access is limited to 120 approved
identifiers, with no runtime SHAP calculation.

FastAPI/Uvicorn serve the synthetic public adapter and precomputed evidence on
Render Free. Minimal serving requirements exclude challenger/SHAP training
packages. Public scoring remains constrained; notes stay in the browser, and no
historical row-level evidence is exposed through the new aggregate endpoints.
Versioned tests cover numerical contracts, public safeguards, persistence,
responsive layouts and real-Chrome interactions. Fresh verification is recorded
in the [Stage 25 report](STAGE25_FINAL_RELEASE_REPORT.md), rather than assumed
from the application screenshots.

Sources: [case management](STAGE21_CASE_MANAGEMENT.md),
[queue semantics](STAGE21_QUEUE_FILTER_CORRECTIONS.md),
[monitoring](STAGE22_MODEL_MONITORING.md),
[explainability](STAGE23_MODEL_EXPLAINABILITY.md),
[deployment safeguards](RENDER_DEPLOYMENT_PREPARATION.md) and
[architecture](ARCHITECTURE.md).

## Responsible AI and limitations

The model scores are uncalibrated, including the legacy API field named
`fraud_probability`. Feature attribution is not causation or a fraud verdict.
MDI has known impurity/cardinality biases, and path-dependent TreeSHAP does not
establish how correlated real-world features would behave under intervention.
Anonymised data and incomplete demographic evidence prevent fairness
certification. Unknown label-maturity/feature-timing conditions remain relevant
leakage risks. Historical distribution stability cannot establish current
predictive performance without appropriate outcomes.

Browser-local synthetic cases lack authentication, cross-device collaboration
and a tamper-proof audit trail. A Free-hosted demo is not a banking service or an
availability commitment. No live transaction feed, measured savings or proven
fraud reduction exists. A genuine operational pilot would require separately
approved prospective evaluation, legitimate data access, calibration/fairness
work and governed case operations. [Responsible AI assessment](RESPONSIBLE_AI_ASSESSMENT.md).

## Lessons learned

The most consequential lessons were about evidence identity and interpretation.
A feature set with a different column order can produce different Random Forest
trees; schema identity therefore includes order. A better validation result is
not automatically a deployable improvement, especially when the population has
been reused. A drift indicator is an investigation prompt, not a performance
verdict. A seeded transaction status is not an investigator's completed decision.

Connecting those distinctions to the interface makes the portfolio more useful:
a reviewer can trace a metric to its population, see how workload changes the
historical trade-off, inspect a synthetic case and understand exactly where the
demonstration ends. The outcome is an auditable research-to-application example,
not a claim of completed production fraud prevention.
