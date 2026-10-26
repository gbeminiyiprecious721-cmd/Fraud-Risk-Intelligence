# Stage 10 — Model Evaluation: Readiness Gate Report

**Status: the ONE-TIME `final_test` evaluation has been executed, approved by the user, exactly once, and its interpretation has been independently audited (§9) and corrected where it overclaimed.** This report documents the frozen evaluation protocol (specified before any `final_test` row was read), the preflight rehearsal, the access-control mechanism, the real results (§7), the interpretation audit (§9), and the remaining scientific limitations. **Stage 10 is NOT marked complete** — the user's final approval for formal closure is still pending, per explicit instruction.

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

## 7. THE real final_test evaluation (executed once, approved)

**Run_id `stage10-c2f0f3fc017a4db3b495e5a6f62084fb`, `search_id="stage10-frozen-rf-final-test-evaluation"`, status `completed`.** `final_test`: 76,855 rows, 2,682 positives, prevalence 3.49% (close to `train`'s ~3.5% and Stage 7's `validation` prevalence 3.463% — unlike Stage 9's CV folds, which varied 2.57%–4.16% by chronological window, `final_test` is not an unusually different population in base rate). Fitted feature count 813 (matches every prior full-scale run). Preprocessing 13.9s, fit 40.0s, predict 0.24s, peak memory ~17.2GB.

**Primary/secondary metrics:**

| Metric | Value |
|---|---|
| Average Precision | **0.42784** |
| ROC-AUC | 0.86637 |
| Precision@1% / Recall@1% | 0.8583 / 0.2461 |
| Precision@5% / Recall@5% | 0.3037 / 0.4351 |
| No-skill baseline AP | 0.0349 |

**Bootstrap 95% CI on AP:** **[0.40842, 0.44534]** (1,000 resamples, seed 42) — a tight interval given the sample size, but see §6's i.i.d.-row caveat.

**Stage 1 success-criterion #2 (beat the no-skill baseline, margin reported honestly):** AP margin **+0.39294** (True), Recall@1% margin **+0.2398** (True), Recall@5% margin **+0.4094** (True). All three beat the baseline, by a wide margin — Stage 1's criterion #2 is satisfied for this frozen configuration on `final_test`.

**`ProductCD` segment breakdown — large, important heterogeneity:**

| Segment | n | n_pos | AP | ROC-AUC |
|---|---|---|---|---|
| C | 8,378 | 1,117 | 0.713 | 0.908 |
| H | 2,289 | 130 | 0.525 | 0.884 |
| R | 3,785 | 179 | 0.735 | 0.945 |
| S | 2,757 | 123 | 0.368 | 0.611 |
| **W** | **59,646 (78% of all rows)** | **1,133** | **0.096** | **0.800** |

**The model performs dramatically worse on `ProductCD=W` (AP 0.096) than on any other category — and `W` is 78% of the entire `final_test` partition.** This single finding dominates the aggregate 0.428 AP: the strong-performing categories (C/H/R, AP 0.52–0.74) are a small minority of volume. `W`'s own AP is still ~5x its own no-skill floor (0.019 prevalence), so the model is not useless there, but this is the clearest, most consequential subgroup-variability finding in this evaluation.

**`TransactionAmt` segment breakdown (boundaries from `train` only: 43.38 / 69.00 / 125.00) — non-monotonic:**

| Segment | Range | n | AP |
|---|---|---|---|
| 0 | ≤43.38 | 19,241 | 0.616 |
| 1 | 43.38–69.00 | 19,580 | 0.341 |
| 2 | 69.00–125.00 | 19,449 | 0.340 |
| 3 | >125.00 | 18,585 | 0.353 |

The lowest-amount segment performs notably BETTER than the other three, which are roughly flat — not a simple "bigger transactions are harder" story.

**Cost-sensitive threshold sweep — genuinely sensitive, not robust (an honest finding, not glossed over):**

| Scenario \ unit-$ | $5 | $10 | $20 | $50 |
|---|---|---|---|---|
| A (conservative loss) | 0.05 | 0.11 | 0.32 | 0.53 |
| B (full transaction value) | 0.04 | 0.05 | 0.11 | 0.40 |
| C (capacity-constrained) | 0.11 | 0.40 | 0.53 | 0.71 |

The cost-minimizing threshold ranges from **0.04 to 0.71** depending entirely on which labelled assumption is used. Per Stage 1's own framing ("robust across scenarios A–C" or "sensitive to assumption X"), this result is clearly the latter: **sensitive, not robust**. No single recommended threshold is asserted; the sensitivity itself is the finding.

## 9. Scientific interpretation audit (read-only — corrects §7's framing, no retraining/rescoring)

This audit used ONLY already-saved evidence (`docs/_stage10_final_test_evaluation_raw.json`, `docs/experiment_registry.jsonl`). `final_test` was not reopened.

**§7's measured numbers are preserved exactly as reported: AP 0.42784, ROC-AUC 0.86637, P@1%=0.8583/R@1%=0.2461, P@5%=0.3037/R@5%=0.4351, 95% bootstrap CI [0.40842, 0.44534]. Unchanged.**

**ProductCD segment AP, correctly compared against each segment's OWN prevalence floor (available in saved evidence — `n_samples`, `n_positive`, `actual_prevalence` are all recorded per segment):**

| Segment | n | Prevalence | AP | **AP ÷ own prevalence** |
|---|---|---|---|---|
| R | 3,785 | 4.73% | 0.735 | **15.54x** |
| H | 2,289 | 5.68% | 0.525 | **9.24x** |
| S | 2,757 | 4.46% | 0.368 | **8.24x** |
| C | 8,378 | 13.33% | 0.713 | **5.34x** |
| W | 59,646 | 1.90% | 0.096 | **5.05x** |

**Correction: ranking segments by raw AP (R≈C > H > S > W) is misleading, because AP is mechanically bounded by each segment's own prevalence — a segment with a much lower base rate cannot reach the same absolute AP as a high-prevalence segment even with equally good ranking.** Normalized against each segment's own no-skill floor, **W and C are actually tied for the weakest relative performance (≈5x), not W uniquely so** — R (15.5x) and H (9.2x) are the strongest relative performers, S (8.2x) is mid-table. W's raw AP (0.096) looking dramatically worse than C's (0.713) is substantially an artifact of W's much lower prevalence (1.9% vs. 13.3%), not solely evidence of a uniquely broken sub-model for `ProductCD=W`. A real, more modest finding survives: W and C are the two segments ranking least-well relative to their own difficulty, and W is also 78% of all `final_test` volume — operationally the most consequential segment by sheer count, even though it is not uniquely catastrophic by this normalized measure.

**Correction — the prior claim that `ProductCD=W` "caused" or "almost entirely explains" the aggregate AP (0.428) was unsupported and is withdrawn.** Average Precision is computed from ONE joint ranking across all `final_test` rows; each segment's reported AP is computed from a SEPARATE, segment-internal re-ranking (only that segment's own rows, re-sorted among themselves). These are not the same computation, and aggregate AP is **not** a sample-count-weighted (or any other weighted) average of the per-segment AP values — no such decomposition was computed in this evaluation, and attempting one (e.g. "what would aggregate AP be excluding `W`") would require the row-level scores, which were not persisted as evidence and cannot be reconstructed without re-scoring `final_test` — explicitly out of scope for this read-only audit. What CAN be said, supported by the saved evidence: `W` is both the largest segment (78% of rows) and among the weakest by the normalized measure above, which is *consistent with* `W` contributing substantially to pulling the joint ranking's aggregate AP down from what the stronger segments alone might suggest — but this is a plausible, unquantified contribution, not a proven causal/decomposed claim.

**Correction — the prior claim that the CV-to-`final_test` AP gap (0.51236 vs. 0.42784) "proves" model-selection (CV-search) optimism was a categorical overclaim and is revised.** At least three distinct, co-plausible, UNDECOMPOSED explanations exist, and this evaluation does not distinguish between them:
1. **Selection effects** — Stage 9's search ranked 18 candidates by CV mean AP and reported the best; a "best of 18" statistic is optimistic relative to a fresh draw, by the multiple-comparisons logic already on record since Stage 9.
2. **Training-window/size differences** — Stage 9's 5 CV folds trained on expanding-window SUBSETS of `train` (73,020 to 365,100 rows), never the complete 438,125-row partition; the real Stage 10 evaluation trained on the complete partition. More training data can change a model's generalization behavior independent of any selection effect.
3. **Temporal population change** — Stage 9's CV folds scored each fold's own chronologically-local validation window (all still inside `train`'s time range); `final_test` is a later, disjoint time period. If the underlying fraud patterns shifted across time, this alone could produce a gap with no selection bias involved at all.

All three remain plausible; this audit does not rule any out or assign relative weight, and no further analysis was performed to decompose them (doing so would require, at minimum, scoring the exact CV-fold-trained models on `final_test`, which was not part of the approved protocol and has not been done).

**Cost-sensitivity: re-confirmed, no single threshold recommended.** The sweep (§7) remains explicitly conditional on 4 labelled, un-validated dollar-value assumptions crossed with 3 labelled cost scenarios — a sensitivity table, not a recommendation. No credible real operational cost data exists in this project (Stage 1 never pinned one down), so no single deployment threshold is, or should be, asserted.

**Stage 1's 4 success criteria, re-verified with demonstrated/conditional status made explicit:**

| # | Criterion | Status |
|---|---|---|
| 1 | Trivial baseline established, reported alongside every model | **Demonstrated** (since Stage 7; reconfirmed on `final_test` this round) |
| 2 | Candidate beats baseline on BOTH AP and recall@K, margin reported honestly | **Demonstrated** (AP +0.393, R@1% +0.240, R@5% +0.409, all on `final_test`) |
| 3 | Final threshold justified by the cost-sensitive analysis across scenarios A–C | **Conditional / NOT fully met** — the sensitivity analysis exists and is complete, but no specific threshold has been chosen, because Stage 1 never pinned down a real dollar value for the FP cost unit; a "final chosen threshold" would require that real operational input, which does not currently exist |
| 4 | Leakage-free, reproducible evaluation | **Demonstrated** (the entire readiness-gate architecture, fingerprint guards, access-control gates, and this audit itself) |

**All known limitations preserved, verbatim, not weakened by anything in this audit:** unverified prediction-time availability of anonymised features; temporal distribution shift (now additionally implicated as one of three co-plausible explanations for the CV/`final_test` gap above, not merely a background caveat); potential dependence between transactions (the bootstrap CI's i.i.d.-row assumption remains a stated simplification); no model-calibration check has been performed; the offline-ranking-vs-live-fraud-prevention gap is not closed by this stage; no financial-loss reduction has been demonstrated, only ranking quality on historical, labelled data.

**Registry and evidence verification:** exactly **one** `status="completed"` record exists under `search_id="stage10-frozen-rf-final-test-evaluation"` (`run_id=stage10-c2f0f3fc017a4db3b495e5a6f62084fb`) — confirmed by direct query, no second record created by this audit. **A genuine gap, caught by this audit and reported honestly rather than silently fixed:** that record's `evidence_checksums` field was never populated at evaluation time (`scripts/run_stage10_final_evaluation.py` set `evidence_paths` but did not call `compute_evidence_checksums`, unlike the Stage 7/9 backfill scripts' convention). The evidence file's sha256, computed now, read-only, for the permanent record: `021726fc9454e0b7e60406b36fcb8ae796df35dbc311bae14a89802058e3f3f5`. Per this project's append-only immutability discipline, the existing registry record was NOT retroactively edited to add this — the gap and the independently-computed checksum are both documented here instead, the same discipline already applied to the two self-caught Stage 7 git-commit-message errors.

## 10. What has NOT happened

- `final_test` was read exactly once, for the one evaluation, and has not been re-accessed — including during this audit, which used only already-saved evidence.
- No hyperparameter, feature, or preprocessing decision was made or changed in response to these results.
- The trained model was not changed; no package was installed; no prediction was rerun.
- Stage 10 is not marked complete — awaiting the user's final approval for formal closure.
- No claim is made that this offline ranking evaluation demonstrates real-world fraud-loss reduction or production readiness.
