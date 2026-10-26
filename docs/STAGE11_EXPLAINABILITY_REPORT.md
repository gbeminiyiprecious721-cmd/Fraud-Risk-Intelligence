# Stage 11 — Explainability Report

**Status: implementation complete, not yet marked done.** Explains the frozen Stage 9/10 `RandomForestClassifier` via SHAP (`TreeExplainer`) and a sklearn-native permutation-importance cross-check. Train-only throughout; `final_test` was not accessed.

## 1. SHAP installation and compatibility

Explicitly approved by the user before installation. `pip install --dry-run shap` was run first: SHAP 0.54.1 and its only new transitive dependency, `slicer` 0.0.8, installed cleanly against the EXISTING environment with **zero version change** to any already-installed package.

| Package | Before | After |
|---|---|---|
| Python | 3.12.15 | 3.12.15 |
| numpy | 2.5.4 | 2.5.4 |
| pandas | 3.1.0 | 3.1.0 |
| scikit-learn | 1.9.1 | 1.9.1 |
| scipy | 1.18.1 | 1.18.1 |
| shap | — | **0.54.1 (new)** |
| slicer | — | **0.0.8 (new)** |

No disruptive version change occurred, so the install proceeded (per the approval's own condition: "if SHAP requires disruptive version changes, stop and report the conflict instead of forcing it" — this never applied). `requirements.txt` updated with both new entries and a dated note explaining the approval and the clean dry-run.

## 2. Methodology

- **Explanation dataset**: a fixed, deterministic, stratified sample — 100 fraud + 100 non-fraud rows from `train` only, selected by sorting on `TransactionID` and sampling with `random_state=42` (`src.explainability.select_explanation_sample`). Selected BEFORE any SHAP value was computed; never chosen for how a plot looked. Sample fingerprint recorded for reproducibility.
- **Explainer**: `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")`. **Explicit, justified choice over `"interventional"`**: `"interventional"` SHAP independently resamples each feature from a background set, deliberately breaking real feature correlation to isolate a "pure" marginal effect — exactly the wrong choice for a dataset this project has repeatedly shown to have severe feature correlation (157+ `V`-column pairs, Stage 7). `"tree_path_dependent"` instead respects the training data's actual joint distribution via the trees' own internal path-weighting, at the cost of still splitting credit ambiguously *within* a correlated group (a stated limitation, not a free win — see §6).
- **Output space, stated explicitly**: SHAP's modern API returns a 3-D `Explanation` array `(n_samples, n_features, n_classes)` for this binary classifier; **class index 1 (fraud probability) was extracted** — the same output every metric in this project has ever reported.
- **Feasibility probe before the full run**: 10 rows in 0.396s (0.0396s/row) → estimated 7.9s for the full 200-row sample; the real run took 7.641s — the estimate held.
- **Additivity verified**: `sum(SHAP values) + expected_value` matches the model's actual output to `max_abs_diff ≈ 1.2e-15` (machine precision) — not assumed, checked.
- **Feature lineage**: every transformed column name (`numeric_plain__`, `numeric_indicator__missingindicator_`, `categorical__<col>_<category>`) is parsed back to its original raw column via `src.explainability.build_feature_lineage`, verified against the pipeline's real, empirically-inspected naming convention (not assumed).

## 3. Global explanation results

Top original columns by summed mean |SHAP| (one-hot categories collapsed to their source column):

| Rank | Column | Sum mean \|SHAP\| |
|---|---|---|
| 1 | C13 | 0.00814 |
| 2 | C1 | 0.00703 |
| 3 | C14 | 0.00609 |
| 4 | C8 | 0.00572 |
| 5 | V258 | 0.00488 |
| 6 | C4 | 0.00431 |
| 7 | card6 | 0.00415 |
| 8 | C7 | 0.00396 |
| 9 | V257 | 0.00383 |
| 10 | V45 | 0.00372 |

**The anonymised `C`-columns (counting features) dominate the top of the global ranking**, ahead of a cluster of `V`-columns and the one categorical feature (`card6`) that cracks the top 10. Global summary plot saved to `notebooks/figures/stage11_global_shap_importance.png` and reproduced in the notebook.

**Explicit limitation**: `C1`/`C4`/`C7`/`C8`/`C10`–`C14` and every `V`-column remain anonymised — unresolved since Stage 2/6/7. SHAP shows the model relies heavily on them; it cannot say what they represent in the real world.

## 4. Local explanation results

Three fixed-criteria examples (never selected for appearance): the highest-confidence true positive, the lowest-scored missed fraud case (false negative), and a borderline (~0.5) case.

| Case | TransactionID | Predicted P(fraud) | Actual | Top contributor |
|---|---|---|---|---|
| Highest-confidence TP | 3155871 | 0.9969 | 1 (fraud) | V200 (+0.0425) |
| Missed fraud (FN) | 3000769 | 0.0099 | 1 (fraud) | TransactionAmt (+0.0026, small) |
| Borderline | 3276924 | 0.4676 | 1 (fraud) | C1 (+0.0629) |

The missed-fraud case is instructive *because* it fails: its top contributors are small and mixed in sign — nothing locally compelling pushed the score up, and it was scored near the no-skill floor. A genuine, visible model limitation, not explained away.

## 5. Permutation-importance cross-check

Per explicit instruction, evaluated on a **predefined train-only chronological holdout** — reusing Stage 9's own fold-0 split (`src.hyperparameter_tuning.build_expanding_window_folds`), never `validation`/`final_test`. An **auxiliary** frozen-config fit on fold-0's 73,020 training rows was scored on fold-0's own 73,025 validation rows (held-out AP: **0.47545** — this is a separate, smaller-scale measurement for the cross-check only, NOT a Stage 10 result, and must never be reported as one).

Top permutation-importance results (mean AP drop, top 25 SHAP-ranked features, bounded to avoid scoring all 728+ columns):

| Feature | Mean AP drop |
|---|---|
| numeric_plain\_\_C13 | 0.00872 |
| numeric_plain\_\_C14 | 0.00625 |
| numeric_plain\_\_C1 | 0.00491 |
| numeric_plain\_\_C7 | 0.00428 |
| numeric_indicator\_\_V45 | 0.00352 |

**Correlated-group permutation importance** (whole correlated `V`-column groups permuted together, not one column at a time — individual-column permutation under-measures a redundant group's collective contribution): all 3 tested groups showed a real, positive AP drop when permuted as a group (0.00008–0.00130), confirming they carry genuine joint signal even though any single member's own individual permutation score could look small on its own.

**SHAP vs. permutation importance — agreement stated carefully, not oversold**: both methods rank the `C`-columns highest, which is reassuring, but **this agreement does not mean one method validates the other** — both can be fooled by the same correlation structure in different ways (SHAP can split credit ambiguously within a correlated group; naive single-column permutation can understate a redundant group's true importance, which is exactly why the group-level check exists).

## 6. Correlation and stability findings

- **Correlated `V`-column groups, recomputed fresh on the complete `train` partition**: 5 groups (threshold 0.99) among 339 `V`-columns. This is fewer than Stage 7's own finding (157+ pairs / 64–66 distinct columns) from its 15,000-row chronological pilot — **a different sample, not a contradiction**, and not reconciled further this round (stated honestly, not silently harmonized).
- **Stability under small, realistic perturbation** (±1% relative Gaussian noise, numeric columns ONLY — never on 0/1 indicator/one-hot columns, which would be an unrealistic perturbation): the high-confidence case was extremely stable (cosine similarity ≈0.99996, proba barely moved); the borderline case was slightly less stable (≈0.9995) — consistent with the intuitive expectation that predictions near a decision boundary are more sensitive to small input changes. **This is a narrow check — one noise scale, two examples — not a general robustness guarantee.**
- Changes in **prediction** (probability) and changes in **attribution** (SHAP vector) were measured and reported SEPARATELY throughout, per explicit instruction, never conflated into one "stability" number.

## 7. Scientific limitations (carried forward, none weakened)

- Anonymised-feature interpretation: unresolved since Stage 2.
- Correlation-related attribution ambiguity: `tree_path_dependent` SHAP respects correlation structure but cannot cleanly separate credit within a correlated group.
- SHAP values are explicitly **not** claimed to identify the causes of fraud — they describe how this model used this data, nothing more.
- Stability was checked narrowly (2 examples, 1 noise scale) — not a guarantee across the full population.
- This analysis used `train` only; whether these same explanations hold on `final_test` was not tested, and `final_test` must not be reopened to test it.
- No explainability finding was used to retune, reselect, or otherwise modify the frozen model or its evaluation.

## 8. Deliverables

- `src/explainability.py` (11 functions), `tests/test_explainability.py` (21 tests).
- `scripts/run_stage11_explainability.py` — the real analysis driver.
- `notebooks/02_stage11_explainability.ipynb` — executed, 0 errors, reads only the already-saved evidence (no refit, no final_test).
- `notebooks/figures/stage11_global_shap_importance.png`.
- `docs/_stage11_explainability_raw.json` (gitignored, generated evidence).
- This report.

## 9. Runtime and memory

Main analysis (full-train fit + preprocessing + SHAP on 200 rows): wall time 113.1s, peak memory ~16.7GB (consistent with every prior full-train fit of this pipeline since Stage 5/7/10). Auxiliary fold-0 fit + permutation importance: included in the same run, bounded to the top 25 SHAP-ranked features (not all 728+ columns) to keep the repeated-scoring cost justified.

## 10. Tests and git status

**235 tests passing project-wide before this round's additions; 21 new tests added for `src/explainability.py` (256 total)** — all passing, including static AST checks that the module imports nothing capable of reading a dataset and never calls anything with the final-partition override enabled. Two new registry records logged (`search_id="stage11-explainability"`): the main analysis and the clearly-labeled auxiliary fit, neither overwriting any historical Stage 7–10 result.
