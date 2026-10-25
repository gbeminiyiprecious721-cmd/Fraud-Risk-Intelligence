# Fraud-Risk-Intelligence

**A fraud-detection decision-support system built on the IEEE-CIS Fraud Detection dataset — with a rigorous, stage-gated ML engineering process as the actual deliverable, not just a model.**

> **Status: Stages 1–6 approved. Stage 7 (Model Selection) was authorised and began; full-feature modelling experiments were interrupted by a diagnosed computational/convergence problem and are under controlled review. No model has been trained to completion on the full feature set, and `final_test` has not been touched.** See [Current Development Progress](#current-development-progress).

---

## Executive Overview

Credit-card fraud is rare (here, 3.5% of transactions) but expensive to miss and expensive to over-flag. This project builds the decision-support layer a fraud-operations team would actually use: a model that **ranks** transactions by fraud likelihood so a fixed pool of human investigators spends their time on the highest-risk cases first — not a black box that auto-declines transactions.

What makes this project worth a second look isn't a leaderboard score. It's the discipline around *how* the model gets built: every stage — from reading the problem statement to splitting the data to engineering features — passes an explicit approval gate, with its own written report, its own test suite, and its own documented leakage check, **before** the next stage starts. When something went wrong (a slow-converging model, a data-loading inefficiency, a mis-measured benchmark), it's investigated, fixed or explicitly deferred, and written up — including the mistakes. That process, and the paper trail it leaves, is the actual portfolio artifact.

## Business Problem and Intended Users

**Users:** a fraud-operations team with finite investigation capacity, deciding which flagged transactions a human reviews first.

**Problem:** out of every ~590K transactions, roughly 1 in 29 is fraudulent, and the cost of a missed fraud (chargeback, loss) and the cost of an unnecessary investigation (analyst time, customer friction) are asymmetric and not directly comparable. A ranking model that concentrates fraud at the top of a reviewed queue is strictly more useful than an un-triaged one, even before any cost figure is pinned down.

**Scope, stated precisely:** this project targets **human-investigation prioritisation**, not automated transaction decline — declines are explicitly out of scope, noted as a possible future extension. Full reasoning, cost framing, and labelled sensitivity scenarios: [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md).

## Fraud Risk Intelligence — Solution Objectives

1. Produce a transaction-level fraud-risk score, evaluated by **Average Precision (AP)** — the right metric for a severely imbalanced ranking problem (see [Model Development and Evaluation Methodology](#model-development-and-evaluation-methodology)).
2. Build every stateful transformation (imputation, encoding, feature derivation) so it is fit on training data only and never touches the held-out evaluation set — enforced in code, not just by convention.
3. Keep every column's prediction-time availability and leakage risk individually traceable, not asserted at the dataset level.
4. Document every material decision — including ones later corrected — so the project is auditable end to end, not just the final answer.

## Dataset Overview and Source Attribution

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle competition dataset, Vesta Corporation / IEEE Computational Intelligence Society). Not redistributed in this repository — see [Installation and Reproducibility](#installation-and-reproducibility).

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Labelled (`isFraud`); `TransactionDT` is a relative time offset, not a calendar timestamp |
| `train_identity.csv` | 144,233 | 41 | Joins to `train_transaction` on `TransactionID`; 24.42% join coverage |
| `test_transaction.csv` / `test_identity.csv` | 506,691 / 141,907 | 393 / 41 | **Unlabelled** — Kaggle's own hold-out, not used by this project's evaluation |

Verified fraud prevalence: **3.499%** (27.6 : 1 imbalance) — [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md). Full discovery/validation evidence: [`docs/DATA_VALIDATION_REPORT.md`](docs/DATA_VALIDATION_REPORT.md).

## Technology Stack

| Layer | Tools |
|---|---|
| Language / environment | Python 3.12.15, project-local `.venv`, pinned in [`requirements.txt`](requirements.txt) |
| Data handling | pandas 3.1.0, numpy 2.5.4 |
| Modelling / preprocessing | scikit-learn 1.9.1 (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`), scipy 1.18.1 (sparse matrices) |
| EDA | matplotlib 3.11.2, Jupyter/`nbformat`/`nbclient` (notebook built and executed programmatically) |
| Testing | pytest 9.1.1 — 107 tests across 6 modules |
| Dataset retrieval | kagglehub |

No deep-learning or gradient-boosting-specific library (LightGBM/XGBoost/CatBoost) is installed — a deliberate, documented boundary; see [Known Limitations](#known-limitations-and-future-development).

## High-Level ML System Architecture

```
data/raw/ (gitignored, read-only)
   -> src/data_validation.py        Stage 2: schema, integrity, join, missingness checks
   -> src/eda_analysis.py           Stage 3: distribution/association/leakage-lead analysis
   -> src/leakage_boundaries.py     Stage 4: feature-availability register + chronological split
                                     + get_partition()/assert_no_final_test_contamination()
                                       (the code-level guardrail every later stage routes through)
   -> src/preprocessing.py          Stage 5: ColumnTransformer fit on `train` only -> sparse matrix
   -> src/feature_engineering.py    Stage 6: additive engineered features, fit on `train` only
   -> src/model_selection.py        Stage 7: baseline/candidate models, metrics, ablations
                                       (in progress, see status banner above)
```

Each arrow is a stage boundary with its own report and its own tests — nothing upstream is "approved" by a downstream stage merely running without error.

## The Complete 20-Stage Machine Learning Workflow

This project follows a generic, reusable stage-gated ML workflow (not specific to this project — see `~/.claude/skills/ml-engineering-workflow` if you're inspecting the process itself). **`docs/ML_WORKFLOW_STATE.md` is the authoritative, live tracker** — stage status, decisions, carried-forward requirements, and approval dates. This table mirrors it at a glance; if the two ever disagree, the state file wins.

| # | Stage | Status | Artifact |
|---|---|---|---|
| 0 | Roadmap Generation | ✅ Done | this README + `docs/ML_WORKFLOW_STATE.md` |
| 1 | Business Problem Definition | ✅ Approved | [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) |
| 2 | Dataset Discovery & Validation | ✅ Approved | [`docs/DATA_VALIDATION_REPORT.md`](docs/DATA_VALIDATION_REPORT.md), [`src/data_validation.py`](src/data_validation.py) |
| 3 | Exploratory Data Analysis | ✅ Approved | [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md), [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) |
| 4 | Leakage Boundaries & Data Splitting | ✅ Approved | [`docs/LEAKAGE_BOUNDARIES_REPORT.md`](docs/LEAKAGE_BOUNDARIES_REPORT.md), [`docs/feature_availability_register.csv`](docs/feature_availability_register.csv) |
| 5 | Preprocessing | ✅ Approved | [`docs/PREPROCESSING_REPORT.md`](docs/PREPROCESSING_REPORT.md), [`src/preprocessing.py`](src/preprocessing.py) |
| 6 | Feature Engineering | ✅ Approved | [`docs/FEATURE_ENGINEERING_REPORT.md`](docs/FEATURE_ENGINEERING_REPORT.md), [`src/feature_engineering.py`](src/feature_engineering.py), [`docs/engineered_features_register.csv`](docs/engineered_features_register.csv) |
| 7 | Model Selection | 🔶 **In progress, interrupted — under controlled review** | [`src/model_selection.py`](src/model_selection.py), [`scripts/run_stage7_model_selection.py`](scripts/run_stage7_model_selection.py) |
| 8 | Experiment Tracking | ⬜ Not started | — |
| 9 | Hyperparameter Tuning | ⬜ Not started | — |
| 10 | Model Evaluation | ⬜ Not started — `final_test` reserved, untouched | — |
| 11 | Explainability | ⬜ Not started | — |
| 12 | Responsible AI Assessment | ⬜ Not started | — |
| 13 | Production Pipeline | ⬜ Not started | — |
| 14 | API Development | ⬜ Not started | — |
| 15 | Application Integration | ⬜ Not started | — |
| 16 | Automated Testing | ⬜ Not started (note: all stages so far already carry their own pytest suites) | — |
| 17 | Deployment | ⬜ Not started | — |
| 18 | Monitoring Design | ⬜ Not started | — |
| 19 | Documentation & Portfolio | ⬜ Not started | — |

## Current Development Progress

**Approved (Stages 1–6):** problem framing, dataset validation, EDA, a chronological 70/15/15 `train` (438,125) / `validation` (75,560) / `final_test` (76,855) split with a code-enforced final-holdout boundary, a 392-column preprocessing pipeline, and 11 experimental engineered features — each stage closed out with a written report, a review round, and a passing test suite.

**Stage 7 — honestly, not a success story yet:** a no-skill baseline and a small 3-feature Logistic Regression baseline completed validly (see table below). The next planned step — Logistic Regression on the full 813-column preprocessed feature set — was **interrupted**: it ran for over 40 minutes without finishing, and a focused diagnostic traced this to severe multicollinearity (245 of 271 missingness-indicator columns are exact duplicates of one another) interacting badly with the `liblinear` solver, not to dataset size. **This is reported as a diagnosed, unresolved problem, not a completed experiment** — no full-feature model result exists yet.

| Experiment | Status | Average Precision | Notes |
|---|---|---|---|
| No-skill (constant score = train prevalence) | ✅ Completed | 0.03463 | Equals validation's actual prevalence, by construction |
| Simple Logistic Regression (`TransactionAmt`, `ProductCD`, `card4` only) | ✅ Completed | 0.09651 | ~2.8× the no-skill floor from 3 raw columns |
| Logistic Regression, full 813-column pipeline | ❌ Interrupted — no valid result | — | See diagnostic above; a corrected re-run requires separate approval |
| Random Forest / Gradient Boosting candidates | ⬜ Not yet run | — | Blocked behind the Logistic Regression fix |

## Data Preprocessing and Feature-Engineering Approach

**Preprocessing (Stage 5):** a `ColumnTransformer` built *programmatically* from the feature-availability register (never a hand-maintained column list) — median imputation for numeric columns, a missingness indicator for the 271 columns with ≥5% train-missingness, constant-fill + one-hot encoding for the 14 categorical columns. Output is a sparse matrix (813 columns, 41.56% density) — sparse storage was chosen over scikit-learn's own density-based default after *measuring* a 38% memory reduction, not assuming one. Full detail, including a dedicated final-verification round: [`docs/PREPROCESSING_REPORT.md`](docs/PREPROCESSING_REPORT.md).

**Feature engineering (Stage 6):** 11 additive, experimental features — relative temporal phase (hour/day-of-week, explicitly *not* claimed as calendar time), causal point-in-time entity-history counts, train-only frequency encoding, and a few interpretable interaction/amount-derived features — each individually traced in [`docs/engineered_features_register.csv`](docs/engineered_features_register.csv) with its own leakage-risk classification. **None are used in any model result yet** — their predictive contribution is explicitly deferred to a leakage-safe Stage 7 ablation, not assumed.

## Model Development and Evaluation Methodology

**Primary metric: Average Precision (AP)**, not accuracy or plain ROC-AUC — appropriate here because the positive class is rare (3.5%) and the real decision is a *ranked* investigation queue, which AP measures directly; ROC-AUC is tracked as a secondary metric since it doesn't capture how well the model orders the top of the queue. AP's own "no-skill" floor is the positive-class prevalence (not 0.5), a detail easy to misread an AP number without.

**Methodology (as implemented so far):** fit on `train` only, score on `validation` only, via a code-level partition guardrail (`get_partition()`) that refuses `final_test` access unless explicitly overridden — an override that is intended to appear exactly once, in Stage 10. No hyperparameter search has occurred (that's Stage 9's job, and it will need a chronological, not random, cross-validation scheme, since this is time-ordered data).

## Verified Experimental Findings

- **No-skill AP (0.03463) empirically equals validation's own prevalence** — a mathematical identity, numerically confirmed rather than assumed.
- **A 3-feature baseline reaches AP 0.09651**, ~2.8× the no-skill floor — real, if modest, signal from `TransactionAmt`/`ProductCD`/`card4` alone.
- **271 missingness-indicator columns collapse to 26 unique patterns** — 245 columns are exact duplicates, independently reproduced across Stages 5, 6, and the Stage 7 diagnostic.
- **The full-feature Logistic Regression slowdown is root-caused, not just observed**: `liblinear` + that multicollinearity + `class_weight="balanced"` measured at 1,692 seconds for 100 non-converging iterations on a 50,000-row subsample; `saga` on the identical data took 27 seconds for the same (still non-converged) 100 iterations — a measured 63× gap.

No claim beyond these is made. In particular: **no full-feature model result, no feature-engineering lift claim, and no model-family recommendation exist yet.**

## Data Leakage Prevention and Scientific Safeguards

- **`get_partition()` / `assert_no_final_test_contamination()`** ([`src/leakage_boundaries.py`](src/leakage_boundaries.py)) — a code-level guardrail, not a convention: requesting `final_test` without an explicit override raises an exception.
- **Every stateful transform is fit on `train` only**, verified by a dedicated leakage-guard test in every stage's suite (e.g. `tests/test_preprocessing.py::test_leakage_guard_train_only_fit_differs_from_train_plus_validation_fit`).
- **A 434-row feature-availability register** ([`docs/feature_availability_register.csv`](docs/feature_availability_register.csv)) individually classifies every source column as `include`, `include_with_caveat`, `exclude_conservative`, or `exclude_structural` — availability is a per-column, evidence-based decision, never a dataset-wide assumption.
- **`include_with_caveat` (369 columns) means "eligible for controlled experiments," never "confirmed safe"** — stated explicitly and repeatedly because it's the easiest thing to forget once a model produces an encouraging number.
- Point-in-time correctness for engineered historical features is proven by tests constructed to *fail* if a later row's information ever leaked into an earlier row's feature — not just tests that happen to pass.

## Repository Structure and Navigation

```
Fraud-Risk-Intelligence/
├── README.md                          you are here
├── CLAUDE.md, AGENTS.md                cross-agent pointers to the live tracker below
├── requirements.txt                    pinned environment manifest
├── data/raw/                           dataset (gitignored — see Installation)
├── docs/
│   ├── ML_WORKFLOW_STATE.md            *** authoritative, live stage tracker ***
│   ├── HANDOFF.md                      narrative handoff notes / gotchas
│   ├── PROBLEM_STATEMENT.md            Stage 1
│   ├── DATA_VALIDATION_REPORT.md       Stage 2
│   ├── EDA_REPORT.md                   Stage 3
│   ├── LEAKAGE_BOUNDARIES_REPORT.md    Stage 4
│   ├── feature_availability_register.csv
│   ├── PREPROCESSING_REPORT.md         Stage 5
│   ├── FEATURE_ENGINEERING_REPORT.md   Stage 6
│   ├── engineered_features_register.csv
│   └── _stage*_raw.json                generated evidence (gitignored)
├── src/                                 one reusable module per stage (see architecture diagram)
├── scripts/                             one real-data driver per stage (`run_stageN_*.py`)
├── tests/                               one pytest file per src module, 107 tests total
└── notebooks/                           Stage 3 EDA notebook (executed, not hand-run)
```

## Installation and Reproducibility

```bash
# 1. Create and activate a virtual environment (Python 3.12)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Place the dataset (not included in this repository)
#    Download from https://www.kaggle.com/competitions/ieee-fraud-detection
#    and extract into: data/raw/ieee-fraud-detection/

# 4. Run the test suite
pytest tests/ -v

# 5. Run any stage's real-data driver (read-only against data/raw/)
.venv/bin/python3 scripts/run_stage5_preprocessing.py
```

**Gitignored, not published:** `data/raw/` (the dataset itself), `.venv`/`.venv-1`, `__pycache__`/`.pytest_cache`, and every stage's generated `_stageN_*_raw.json` evidence file — these are regenerated by the driver scripts, not hand-edited, so they're excluded the same way compiled artifacts would be. Every *report* (`docs/*_REPORT.md`) and both feature registers are real, tracked deliverables, not generated noise.

**Tracking experiment results:** each stage's driver script writes a structured JSON evidence file plus a human-readable report; `docs/ML_WORKFLOW_STATE.md`'s Decisions log records every material choice with its justification, in chronological order, including corrections. There is no separate experiment-tracking tool yet — that's Stage 8.

## Testing and Quality Assurance

```
pytest tests/ -v
107 passed, 0 failed, 0 skipped
```

| Test file | Tests | Covers |
|---|---|---|
| `test_data_validation.py` | 9 | Schema, target presence, ID uniqueness |
| `test_eda_analysis.py` | 7 | Effect-size and association statistics |
| `test_leakage_boundaries.py` | 32 | Split determinism, register completeness, final-test guardrail |
| `test_preprocessing.py` | 24 | Fit/transform discipline, column accounting, sparse-input compatibility |
| `test_feature_engineering.py` | 20 | Point-in-time correctness, mutation-invariance, no-target-leakage |
| `test_model_selection.py` | 15 | Metric correctness, partition isolation, pipeline structure |

Every stage's non-trivial safeguard (leakage guards, point-in-time correctness, partition isolation) has a test specifically constructed to **fail** if that safeguard were removed — not a test that would pass regardless.

## Known Limitations and Future Development

- **Stage 7 is incomplete.** The full-feature Logistic Regression experiment is diagnosed but not yet re-run correctly; Random Forest and Gradient Boosting candidates haven't started.
- **No gradient-boosting-specific library is installed.** LightGBM/XGBoost/CatBoost were deliberately not added without separate approval — scikit-learn's own sparse-compatible `GradientBoostingClassifier` is the current plan; `HistGradientBoostingClassifier` is incompatible with this pipeline's sparse output without densifying (~2.85GB), a cost assessed but not yet paid.
- **No true customer/card identity exists in this anonymized dataset.** `card1` alone is a weak proxy (one value covers ~2.65% of all transactions); a `card1+card2+addr1` composite is structurally more defensible but still not 1:1 — stated as an open question, not resolved.
- **Peak preprocessing memory (~16GB) is well above this project's 64GB development machine's comfort margin** and would not fit a smaller one — localized to scikit-learn's own internal pipeline construction, a real constraint for future productionization.
- **`TransactionDT`'s real-world meaning (relative offset vs. an actual timestamp) remains unconfirmed** — all temporal features are stated as relative, never calendar-based.
- This system is **not deployed, not production-ready, and not fully validated** — nothing past Stage 7 exists, and Stage 7 itself is paused pending a corrected re-run.

## Links to Research Reports and Documentation

- [`docs/ML_WORKFLOW_STATE.md`](docs/ML_WORKFLOW_STATE.md) — the authoritative, live stage tracker (read this for current status, not this README)
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — narrative handoff notes and known gotchas
- [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) · [`docs/DATA_VALIDATION_REPORT.md`](docs/DATA_VALIDATION_REPORT.md) · [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md) · [`docs/LEAKAGE_BOUNDARIES_REPORT.md`](docs/LEAKAGE_BOUNDARIES_REPORT.md) · [`docs/PREPROCESSING_REPORT.md`](docs/PREPROCESSING_REPORT.md) · [`docs/FEATURE_ENGINEERING_REPORT.md`](docs/FEATURE_ENGINEERING_REPORT.md) — Stage 1–6 reports
- [`docs/feature_availability_register.csv`](docs/feature_availability_register.csv) · [`docs/engineered_features_register.csv`](docs/engineered_features_register.csv) — per-column leakage/availability traceability
