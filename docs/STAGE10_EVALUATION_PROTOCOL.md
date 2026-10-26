# Stage 10 — Model Evaluation: Readiness Gate Report

**Status: the readiness gate is built and tested. `final_test` has NOT been accessed. The one-time evaluation has NOT been approved or run.** This report documents the frozen evaluation protocol (specified before any `final_test` row is ever read), the preflight rehearsal that proves it works, the access-control mechanism, and the remaining scientific limitations.

## 1. The frozen evaluation protocol (declared before any `final_test` access)

**Model — frozen, not re-tunable this stage:**
```python
RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1, random_state=42, class_weight=None,
)
```
This is Stage 9's audited provisional candidate (`candidate_index=4`), unchanged. `src/model_evaluation.py`'s `verify_frozen_config_unchanged()` recomputes a fingerprint of the live config dict and checks it against a constant captured at import time — a code-level guard against silent drift, not just a comment promising not to touch it.

**Preprocessing:** the approved Stage 5 pipeline (`src/preprocessing.py`, `build_preprocessing_pipeline`), completely unmodified, fit on `train` only.

**Metrics:**
- Primary: Average Precision (`sklearn.metrics.average_precision_score`, exactly as defined since Stage 1).
- Secondary: ROC-AUC, Precision@1%, Recall@1%, Precision@5%, Recall@5% — the same `evaluate_predictions`/`K_FRACTIONS` convention used in every prior stage, reused unmodified.

**Segmentation, boundaries fixed before any `final_test` row is read:**
- `ProductCD` — used as-is (5 categories: C, H, R, S, W).
- `TransactionAmt` — quartile boundaries computed from `train` amounts ONLY (`compute_amount_segment_boundaries`). Measured in the preflight rehearsal: **[43.38, 69.00, 125.00]** — these are `train`-derived numbers, reported here for transparency, but the function recomputes them fresh each run rather than hardcoding them, so a future re-run is never silently stale.
- Minimum reporting floor: **100 samples AND 10 positives** per segment, or the segment's metric is withheld (count still reported) rather than shown with a misleadingly precise number from too little data.

**Bootstrap confidence interval:** percentile bootstrap on AP, **1,000 resamples, fixed seed 42, 95% CI** — declared once, reused identically for the real evaluation. Known limitation stated here, not hidden: this resamples rows i.i.d., a simplification given the project's own carried-forward "potential dependence between transactions" caveat (e.g. repeated `card1` entities) — a block-bootstrap by entity would be more conservative but was not built this round.

**Cost-sensitive threshold analysis — Stage 1's own framework, made computable honestly:** Stage 1 (`docs/PROBLEM_STATEMENT.md` §2) defined 3 labelled scenarios (A/B/C) with the false-negative cost as a fraction of `TransactionAmt`, but the false-positive cost as an UNDEFINED number of "analyst-review units" — no dollar value was ever pinned down for one unit. Rather than inventing one (which Stage 1 explicitly forbids — "this project cannot state 'AP must exceed X' today... inventing one would violate Stage 1's own standard"), **the dollar-value-per-unit is itself SWEPT across 4 labelled [HYPOTHETICAL] candidates ($5/$10/$20/$50)**, giving a 3×4 sensitivity table of cost-minimizing thresholds instead of one invented number.

**Stage 1 success criteria, explicit treatment (not invented, found by reading `docs/PROBLEM_STATEMENT.md` §1 directly):**
1. *"A trivial baseline is established... reported alongside every subsequent model."* — Done since Stage 7 (no-skill AP 0.03463); `compare_to_no_skill_baseline()` recomputes this comparison against the frozen model's real result.
2. *"Any candidate model must be shown to beat that baseline on BOTH AP and recall@K, with the margin reported honestly."* — `compare_to_no_skill_baseline()` reports each margin independently (never collapsed into one pass/fail that could hide a partial failure).
3. *"The final chosen threshold is justified by the cost-sensitive analysis... reported across sensitivity scenarios A–C."* — the cost-threshold sweep above.
4. *"The evaluation is leakage-free and reproducible."* — addressed by this entire readiness gate (fit-on-train-only, fixed seeds, registry provenance).

No numerical AP/business-success threshold is invented anywhere in this protocol — none was ever specified in Stage 1, and this report does not manufacture one now.

## 2. Preflight rehearsal — complete results (on `validation`, NOT `final_test`)

`scripts/run_stage10_readiness_preflight.py` ran the **entire** pipeline for real, end to end, fitting on `train` and scoring on `validation` — proving correctness without touching `final_test`. Explicitly labelled a rehearsal throughout (`validation_consulted` / `validation_purpose` fields on its own registry record say so) — its numbers are **not** the project's Stage 10 result.

| Check | Result |
|---|---|
| `final_test` refused | ✅ (confirmed via `FinalTestAccessError`) |
| Frozen-config fingerprint | ✅ unchanged |
| Feature schema: train-fitted transform's output width vs. its own validation-transform output width | ✅ 813 = 813 (exact match, asserted) |
| Label alignment | ✅ via `align_labels_to_partition_df`'s own `validate="one_to_one"` merge check |
| Fit/preprocess/predict | preprocessing 14.9s, fit 41.5s, predict 0.26s |
| Artifact serialization round-trip (joblib) | ✅ within tolerance (max abs diff 3.33e-16 — the same `n_jobs=-1` cross-thread floating-point artifact already diagnosed in Stage 7, not a new bug; exact equality was the wrong check, `allclose` is the right one, same fix Stage 7 already applied) |
| AP vs. no-skill baseline (rehearsal numbers) | Model AP 0.447 vs. no-skill 0.03463 — beats on AP, Recall@1%, Recall@5% (all `True`) |
| Segment report (`ProductCD`) | All 5 categories had sufficient sample; AP ranged **0.176 (ProductCD=W) to 0.725 (ProductCD=R)** — a large, genuine heterogeneity worth carrying into the real evaluation's interpretation |
| Bootstrap CI | AP 0.447, 95% CI [0.428, 0.466] |
| Cost-threshold sweep | Produced a full 3×4 sensitivity table, best thresholds ranging 0.04–0.64 depending on scenario/assumption |
| Peak memory | ~16-17GB (deeper trees than Stage 7's `max_depth=10` reference cost somewhat more, as expected) |

**One real failure occurred and was handled correctly, not hidden:** the first rehearsal attempt used exact array equality for the serialization check and failed (a real, if harmless, floating-point artifact). `track_run()` logged it as `status="failed"`; the fix (switch to `allclose`, matching Stage 7's own precedent) was applied and the rehearsal re-run cleanly. Both the failed and the successful attempt are preserved in the registry.

Full suite: **214 passed, 0 failed, 0 skipped** (188 unchanged + 26 new in `tests/test_model_evaluation.py`).

## 3. Feature-schema and label-alignment safeguards, confirmed

- Feature schema: the train-fitted `ColumnTransformer`'s output width is asserted equal between its own `fit_transform(train)` and `transform(validation)` calls — the SAME fitted transformer object scores both, so a width mismatch would mean a real bug, caught immediately (`assert` in the rehearsal script, not a silent pass).
- Label alignment: `align_labels_to_partition_df` (Stage 7's own, already-tested function) is reused unmodified — explicit ID-based merge with `validate="one_to_one"`, raising on any unmatched row, never assuming file order matches label order.

## 4. The `final_test` access-control mechanism

Two independent gates, in this exact order, before any import capable of reading a dataset:
1. **`require_unlock_token(token)`** — the caller must supply the literal string `I_APPROVE_THE_ONE_TIME_FINAL_TEST_EVALUATION` as a CLI argument. No default, no partial match.
2. **`check_final_test_not_already_evaluated()`** — queries the (dataset-free) experiment registry for a `status="completed"` record under `search_id="stage10-frozen-rf-final-test-evaluation"`; refuses if one already exists.

Only past BOTH gates does `scripts/run_stage10_final_evaluation.py` import anything from `src.leakage_boundaries`/`src.preprocessing`/`src.model_selection`. Verified, not just designed:
- A live run with a wrong token was attempted this round and refused immediately, before any import — confirmed zero new registry records, zero data access.
- A static test (`test_final_evaluation_script_checks_gates_before_any_dataset_import`) parses the script's own AST and asserts both gate calls appear, by line number, before the first dataset-capable import — a reordering that broke this would fail CI, not just a review.
- Another static test confirms `allow_final_test=True` appears as an actual call site ONLY in this one script (3 call sites: `get_partition`, `load_partition_df`, `load_labels`) and nowhere else under `src/`.
- A refused duplicate-evaluation attempt and a refused failed-vs-completed distinction are both unit-tested against synthetic registries (a prior `status="failed"` record must NOT block a genuine retry; a prior `status="completed"` record must).

## 5. Proposed report structure for the real evaluation (once approved)

1. Exact frozen model + preprocessing recap (as above, unchanged).
2. Primary + secondary metrics on `final_test`, once.
3. Stage 1 success-criteria treatment (baseline comparison, margins).
4. Segment breakdown (`ProductCD`, `TransactionAmt`), with sufficiency flags.
5. Bootstrap 95% CI on AP.
6. Cost-sensitive threshold sensitivity table (3 scenarios × 4 unit-dollar assumptions).
7. Comparison to Stage 7's RF benchmark (0.41101 on `validation`) and Stage 9's CV mean (0.51236) — stated as different populations/signals, never averaged or treated as replicates.
8. Explicit "ran successfully" vs. "scientifically defensible" statement, per the authoritative workflow's own Stage 10 acceptance criterion.
9. Full evidence trail: registry run_id, evidence file path + checksum.

## 6. Scientific limitations carried forward

- `validation` was already consulted 15 times during Stage 7 (6 influencing a decision) — not a concern for the `final_test` evaluation itself, but relevant to how any informal `validation`-based sanity check (like this round's rehearsal) must be labelled.
- Stage 9's hyperparameter search was budget-constrained (18 of 20 candidates; no claim of global optimality).
- Temporal distribution shift: `final_test` is the chronologically latest slice of this dataset — if the real underlying fraud patterns evolved across the collection window, `final_test` performance reflects the END of that window, not a time-invariant average.
- Potential dependence between transactions (e.g. repeated `card1` entities) means the bootstrap CI's i.i.d.-row assumption is a simplification, not exact.
- No model-calibration check has been performed — `predict_proba` outputs are used for ranking (AP, P@K/R@K), not validated as calibrated probabilities.
- No explainability work has been done yet (Stage 11) — this evaluation says nothing about *why* the model ranks a transaction as fraud, only how well it ranks.
- Anonymised-feature prediction-time availability remains unverified (carried forward since Stage 6/7).
- This is an **offline ranking evaluation**, not a demonstrated live fraud-prevention outcome — the gap between "ranks well on historical data" and "prevents fraud in production" is not closed by Stage 10, and is explicitly out of this stage's scope (Stages 13+ territory, if ever pursued, consistent with the project's stated scope — a professional, reusable portfolio artifact, not a commercial MVP).

## 7. What has NOT happened

- `final_test` has not been read.
- No prediction has been generated against `final_test`.
- Stage 10 is not marked complete.
- No hyperparameter, feature, or preprocessing decision was made or changed this round.
