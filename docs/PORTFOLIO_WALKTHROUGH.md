# Fraud Risk Intelligence — a 5–7 minute portfolio walkthrough

This is a presentation guide for a portfolio project, not a claim of banking
employment or a real fraud-operations deployment. Open the
[dashboard](https://fraud-risk-intelligence.onrender.com) before presenting and
allow for a Free-instance cold start. Use the versioned screenshots in the
[README](../README.md#selected-visual-evidence) if the service is unavailable.

## 0:00–0:40 — Start with the decision, on Overview

“I built this project around a constrained investigation queue. If a team can
review only a small percentage of transactions, a useful model should put more
historically labelled fraud near the top. The goal is human investigation
prioritisation, not automatic payment rejection.”

Show the synthetic/portfolio and uncalibrated-score labels. Explain that Overview
is a demonstration batch, not a stream of customer payments or current bank KPIs.
The intended users are a simulated fraud analyst and risk manager.

## 0:40–1:20 — Explain the data and modelling decisions

“The research uses the IEEE-CIS Kaggle dataset. Its labelled transaction table
has 590,540 records, with a smaller identity table joined by TransactionID.
Fraud labels are roughly 3.5%, many fields are anonymised, and missingness is
structural in parts of the data. I checked keys, join coverage and schemas before
modelling.”

“I used chronological partitions, fitted imputation and encoding only on
training data, and recorded feature-availability caveats. The frozen pipeline
uses 392 raw inputs transformed into 813 ordered columns. Missingness indicators
are part of the pipeline; the separate engineered-feature prototypes were not
silently adopted.”

If asked why accuracy is absent: predicting the majority class would look good
while failing the investigation task. AP and precision/recall at a review budget
better describe ranking utility. Do not imply that chronological separation
solves unknown label delays or proves every feature was available at checkout.

## 1:20–2:10 — Show Capacity Simulator, then Model Benchmarking

“The original Stage 10 evaluation reported AP 0.42784 and ROC-AUC 0.86637. At a
1% review budget, precision was 85.83% and recall 24.61%. Those are the historical
held-out summary results; that final-evaluation boundary stays closed.”

Move to **Capacity simulator**. Start at 1%: on the different, reused validation
population, 756 reviews contain 618 labelled positives—81.75% precision and
23.61% recall. Move the slider to show the workload trade-off, then reset to 1%.
Synthetic batch filters do not change these historical aggregates. Do not compare
the two populations as if they were repeated evaluations on identical records.

Move to **Model benchmarking**. Show RF AP 0.44700, LightGBM 0.54590, XGBoost
0.55178 and CatBoost 0.49048 on the same 75,560 validation records. Open the
uncertainty/limitations details rather than calling the highest bar a winner.

“XGBoost was strongest by observed AP within this bounded experiment. Validation
was reused, tuning was limited, and the small lead over LightGBM does not prove
future superiority. The paired block bootstrap describes uncertainty for these
fixed models and historical records. I retained the frozen Random Forest.”

## 2:10–3:15 — Demonstrate Queue and Case Management

Go to **Investigation queue**. Clear filters, select a demo sequence and inspect
a synthetic record. Show the distinction between an unopened transaction's
seeded label and an actual opened case. Open a case, choose a fictional analyst,
set a priority and save a short clearly fictional note.

Apply a Case Analyst or Case Priority filter, then visit **Workflow board** to
show the same local case state. A resolution/reopening reason is required for
those transitions. Clear case filters and explain that this keeps notes/history.
Show the Reset Demo Cases confirmation; cancel it during the presentation unless
you deliberately intend to discard that demo scope's local changes.

“These are synthetic browser-local cases. They survive refresh when storage is
available, but are not authenticated identities, shared case records or a secure
audit trail. Notes do not go to the public scoring API.”

## 3:15–4:05 — Show Model Monitoring

Select **Model monitoring**, then compare V2 and V3 with reference V1. Change
between feature drift, score drift and data quality, and reset the controls.

“This is historical monitoring over three disjoint validation windows: 25,186
records in the reference and 25,187 in each comparison window. PSI, descriptive
KS distance, categorical Jensen–Shannon divergence and missingness indicators
summarise distribution changes. The thresholds are investigation heuristics.
They cannot tell us that current fraud accuracy has deteriorated—or stayed
constant—without suitable new labels.”

Mention that relative time is not an actual calendar monitoring period. There
is no live bank feed, scheduled production monitor or automatic retraining.

## 4:05–5:05 — Show Model Explainability

Switch between **MDI** and **mean absolute grouped TreeSHAP**. Select a raw
feature and read its interpretation caveat. For a supported synthetic record,
show positive and negative local contributions and the reconciliation row.

“I verified the mapping from 813 transformed columns to 392 raw-feature groups.
Global TreeSHAP uses 512 label-free validation examples. Local explanations are
precomputed for 240 synthetic records, with only 120 approved identifiers exposed
publicly. The baseline plus signed contributions reconciles to the frozen
model's actual score; public requests do not compute SHAP.”

“A contribution explains the fitted model under the stated assumptions. It does
not identify a cause of fraud, validate a case outcome or calibrate the score.”

## 5:05–6:00 — Architecture, evidence and lessons

Open the [architecture diagrams](ARCHITECTURE.md) or explain them verbally:
Python research produces a frozen artifact and approved aggregates; FastAPI and
Uvicorn serve the public synthetic adapter; the browser renders Chart.js views
and holds local case state. Render uses minimal runtime dependencies, with no
challenger training or SHAP dependency added for public serving.

“The original Stage 10 fitted object was not saved. The serving artifact is a
verified reconstruction of the frozen procedure, so I keep its identity and the
historical evaluation record separate. I also learned that transformed-column
order matters for a Random Forest that samples features by position.”

Point to the [Stage 24 verification report](STAGE24_PORTFOLIO_DOCUMENTATION.md)
for fresh tests, preservation checks and documentation validation. Do not imply
that local test logs establish hosted CI, a production SLA or verified savings.

Close with the limitations that would govern a real pilot: prospective labels,
calibration, fairness evidence, authenticated case governance and verified
point-in-time features. These are future requirements, not capabilities already
built. The project demonstrates a reproducible decision-support workflow and
clear evidence boundaries rather than a claimed production banking system.

## Evidence for follow-up questions

- [Business framing](PROBLEM_STATEMENT.md) and [data validation](DATA_VALIDATION_REPORT.md).
- [Original model selection](STAGE9_TUNING_DESIGN_REPORT.md) and [historical evaluation](STAGE10_EVALUATION_PROTOCOL.md).
- [Artifact reconstruction and ordered schema](PRODUCTION_PIPELINE_REPORT.md).
- [Capacity](STAGE19_CAPACITY_SIMULATOR.md), [benchmark](STAGE20_CHALLENGER_BENCHMARK.md), [case management](STAGE21_CASE_MANAGEMENT.md).
- [Historical monitoring](STAGE22_MODEL_MONITORING.md) and [explainability](STAGE23_MODEL_EXPLAINABILITY.md).
- [Responsible AI assessment](RESPONSIBLE_AI_ASSESSMENT.md).
