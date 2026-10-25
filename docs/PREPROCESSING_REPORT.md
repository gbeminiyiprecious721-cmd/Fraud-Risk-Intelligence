# Preprocessing Report — Stage 5

**Status:** Implementation complete — **awaiting review and approval**. Do not treat Stage 5 as done until the user approves these findings.

**Scope boundary respected:** fit-on-train-only imputation and categorical encoding, built from the approved feature register and partition interface. No model training, hyperparameter tuning, final-test evaluation, or dashboard/API work.

**New dependency installed, flagged explicitly:** `scikit-learn` 1.9.1 (pulling in `scipy` 1.18.1, `joblib`, `threadpoolctl` as transitive dependencies) — installed without a separate approval round-trip because the approved Stage 5 requirements explicitly named `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, and `Pipeline` by class name, which only exist in this library.

---

## 1. Research-Stage Bias — Documented, Per Explicit Instruction

**Stage 3's EDA was computed over the full, unsplit `train_transaction.csv` (590,540 rows) — before Stage 4's partition existed.** This is now documented explicitly as a potential research-stage bias: any Stage 3 finding (missingness percentages, effect sizes, Cramér's V) technically reflects rows that are now in `validation`/`final_test`, not `train` alone.

**The line drawn this stage, and followed in practice:** using a Stage 3 finding to choose a **strategy** (e.g., "the D/V-column block has structurally high missingness, worth a missingness indicator") is ordinary problem understanding, not leakage. Using one to fit an actual **parameter** (a specific imputation value, a specific category list) would be. Every fitted parameter in this stage's pipeline comes from `train`-only data, verified below (§7) — the strategy choices are informed by Stage 3, the numbers are not.

---

## 2. Column Scope — From the Register, Not Hand-Maintained

**[VERIFIED]** `src/preprocessing.py::load_included_columns()` reads `docs/feature_availability_register.csv`, filters to `include`/`include_with_caveat`, and classifies by **actual dtype** (not the register's `family` column, since e.g. `card` mixes numeric and categorical) — reusing `classify_columns()` from Stage 2, which already handles pandas 3.1's `StringDtype` default.

| | Count |
|---|---|
| Numeric (included) | 378 |
| Categorical (included) | 14 |
| **Total** | **392** — matches the register's `include` (23) + `include_with_caveat` (369) exactly |

No `exclude_*` column reaches this pipeline — provably, since the function only reads the register's `include`/`include_with_caveat` rows.

---

## 3. Numeric Missingness — Measured on `train` Only

**[VERIFIED]** `measure_train_missingness()` restricts to `train` partition `TransactionID`s (via `get_partition()`) before computing missing-%, per the research-stage-bias correction in §1 — these are **not** Stage 3's full-dataset numbers, recomputed for this stage.

Highest-missingness numeric columns (train only): `D7` 93.56%, `dist2` 93.15%, `D13` 89.39%, `D14` 89.20%, `D12` 88.60%, `D6` 87.35%, `D8`/`D9` 86.69%, several `V15x` columns ~84.7% — consistent in shape with Stage 3's full-dataset findings (not contradicted), just independently re-measured on the correct partition.

### Indicator threshold decision

| Threshold | Columns getting an indicator |
|---|---|
| 0% (every column with any missingness) | 378 |
| **5% (chosen)** | **271** |
| 20% (more conservative) | 202 |

**271 of 378 numeric columns cross the 5% threshold** — far more than initially expected, because the anonymized `V`-column block (339 columns) is broadly moderate-to-severely missing, not just the handful Stage 3 highlighted as the most extreme cases.

---

## 4. Memory Implications of Indicator Columns — Investigated, Per Explicit Instruction

**[VERIFIED]** `estimate_indicator_memory_cost()`, measured at three thresholds on `train`'s row count (438,125):

| Scenario | Indicator columns | Estimated memory |
|---|---|---|
| Every column with any missingness (0%) | 378 | **1,324.9 MB** |
| Chosen (5%) | 271 | **949.9 MB** |
| Conservative (20%) | 202 | **708.0 MB** |

**This is a real cost, not a rounding error** — indicator columns alone, before any other feature, approach a gigabyte at the chosen threshold. The 5% threshold was kept anyway: it's the midpoint of the three tested, and Stage 3's own finding (missingness itself carries signal for the `R_emaildomain`/`V`-block) argues for not raising the threshold so high that most of that signal gets discarded. This is stated as a judgment call with a visible cost, not a free choice.

---

## 5. Categorical Encoding — Straightforward Approach, Alternatives Evaluated, Not Implemented

**[VERIFIED]** Cardinality per categorical column (train):

| Column | Categories | Column | Categories |
|---|---|---|---|
| `R_emaildomain` | **60** | `M4` | 3 |
| `P_emaildomain` | **59** | `M1`, `M2`, `M3`, `M5`–`M9` | 2 each |
| `ProductCD` | 5 | | |
| `card4`, `card6` | 4 each | | |

**Chosen approach: `OneHotEncoder(handle_unknown="ignore")` for all 14 columns**, after a constant `"missing"`-sentinel imputation. Estimated naive one-hot width (including the sentinel category per column): **165 columns** — manageable, not explosive; this is why starting straightforward is defensible here rather than reaching for a more complex encoder immediately.

**High-cardinality alternatives evaluated, not implemented (per explicit instruction):**
- **Frequency/count encoding** (replace each category with its occurrence count or rate) — would reduce `P_emaildomain`/`R_emaildomain` to 1 column each instead of ~60; candidate for Stage 6/7 if one-hot proves too wide or too sparse-information-poor for a chosen model family.
- **Hashing** (fixed-width hash buckets) — trades some category collisions for bounded width; not needed yet given 60 categories is not actually large by hashing standards.
- **Target/mean encoding** — explicitly **not implemented**, per instruction. Your `ENCODING-CATEGORICAL VARIABLES` notes already flag its overfitting risk; it also needs within-fold fitting discipline (the same leakage-safe pattern as everything else here) to be done correctly, which is a Stage 6/7-level design decision, not a Stage 5 default.

**Why `OneHotEncoder` over your documented `pd.get_dummies()` approach:** `get_dummies()` applied separately to `train` and `validation` can silently produce misaligned columns if the two sets of observed categories differ — a real, common bug. `handle_unknown="ignore"` avoids it by construction: an unseen category at transform time maps to an all-zero row instead of crashing or misaligning. Verified directly (§8).

---

## 6. Scaling — Deliberately Decoupled, Per Explicit Instruction

**`build_scaler()` is a separate function, not part of the default pipeline.** Tree/boosting model families (plausible Stage 7 candidates for this problem) don't need feature scaling; baking `StandardScaler` into the core pipeline would assume every future estimator does. Stage 7 composes it in only for model families that actually benefit (e.g. logistic regression), fit on `train` only — tested (§8) to confirm it fits/transforms correctly in isolation.

---

## 7. Fit-on-Train-Only — Enforced and Verified

**[VERIFIED]** `fit_on_train_only()` resolves `train` `TransactionID`s exclusively via `get_partition(["train"])` — never `allow_final_test=True`, never a raw read of `docs/_stage4_split_assignment.csv`. Fit wall time: **13.8s** (438,125 rows × 392 input columns).

### Non-trivial leakage-guard — two layers, per explicit instruction

1. **Synthetic, guaranteed-to-differ fixture** (`tests/test_preprocessing.py::test_leakage_guard_train_only_fit_differs_from_train_plus_validation_fit`): train data median 1.0, validation data median 100.0 by construction — fitting on `train` only gives 1.0; fitting on `train`+`validation` combined gives a contaminated value ≠ 1.0. This proves the *mechanism* the guard checks for is real and detectable, independent of whether this dataset happens to drift much.
2. **Real-data demonstration** (`scripts/run_stage5_preprocessing.py` §7a): `TransactionAmt`'s median — `train` only: **69.0**; `train`+`validation` combined: **68.95**. A small but genuinely nonzero, measured difference. Reported honestly: this dataset's `train`/`validation` distributions are similar enough (consistent with Stage 3/4's finding of stable fraud rate and amount distribution across the window) that a real violation here would be subtle in magnitude, not dramatic — which is exactly why the synthetic test (1) exists: a guard that only "works" when the data happens to drift a lot isn't a guard, it's a coincidence.

---

## 8. Train/Validation Consistency — Verified, Including Edge Cases

**[VERIFIED, synthetic]**
- `test_onehot_handles_unseen_category_without_crashing` — a category present only in a held-out set maps to an all-zero one-hot row, not a crash.
- `test_onehot_handles_missing_value_in_transform_not_seen_in_fit` — a missing value at transform time, for a column with no missing values in the fit data, is handled without error (the constant-fill imputer still has a learned fill value regardless of whether `train` happened to need it).
- `test_pipeline_output_has_no_missing_values_after_fit_transform` — no NaN survives the full pipeline.

**[VERIFIED, real data]** Transformed both `train` and `validation` with the single fitted object (no refitting):

| Partition | Shape | Sparse | Density | Memory | Transform time |
|---|---|---|---|---|---|
| `train` | (438,125, 813) | ✅ | 41.56% | **1,778.1 MB** | 5.53s |
| `validation` | (75,560, 813) | ✅ | 42.40% | 312.8 MB | 0.95s |

813 output columns = 107 plain-numeric + 271×2 (value + indicator) + ~163 realized one-hot columns (slightly under the 165 naive estimate, since not every imputed "missing" sentinel category appears in every column's actual fit data).

---

## 9. Sparse vs. Dense — Measured, Not Assumed (Per Explicit Instruction)

**[VERIFIED]** `ColumnTransformer`'s own default (`sparse_threshold=0.3`) returns **dense** once overall output density exceeds 30% — and this pipeline's actual density (41.56%) crosses that. Taking the default at face value would have meant dense storage. Tested empirically instead:

| | Memory (train) | Transform time (train) |
|---|---|---|
| Dense (sklearn's default behavior at this density) | 2,849.6 MB | 2.94s |
| **Sparse (forced, `sparse_threshold=1.0`)** | **1,778.1 MB** | 5.53s |

**Sparse wins on memory by ~38% even at 41.56% density**, because CSR's per-nonzero overhead (~12 bytes) is still cheaper than dense's per-cell cost (8 bytes × 100%) below roughly 67% density — the math and the measurement agree. **The pipeline's default was changed to force sparse** (`sparse_threshold=1.0`), accepting roughly double the transform time as the honest cost of that choice. Memory was judged the binding constraint for this dataset's scale; stated as a judgment call, not an automatic win either way.

**Caution for Stage 7+, stated plainly:** total process peak memory during this stage's run was **~16.2 GB** — substantially higher than the final 1.78 GB stored output. The gap is sklearn's internal intermediate arrays (notably `OneHotEncoder`/`ColumnTransformer` construction) plus holding the raw input DataFrame (~1.4 GB for 438K rows × 392 columns) simultaneously. This machine's 64 GB handles it; a smaller environment would not. Flagged as a real limitation, not glossed over because this machine happens to have the headroom.

---

## 10. Carried-Forward Stage 4 Requirements — Enforced

1. **Partition interface**: every partition access in this stage goes through `get_partition()` — verified by code inspection (`src/preprocessing.py` has no other read of `docs/_stage4_split_assignment.csv`) and by the §7b demonstration below.
2. **`final_test` protection**: never requested with `allow_final_test=True` anywhere in this stage's code or driver script.
3. **`include_with_caveat` stays experimental**: this pipeline treats all 369 such columns exactly like the 23 `include` columns procedurally — nothing in Stage 5 elevates their status; they remain not-confirmed-production-safe regardless of how they preprocess.
4. **Identity exclusion preserved**: `load_included_columns()` only ever sees `train_transaction` columns (confirmed in Stage 4: all `train_identity` columns are `exclude_conservative`/`exclude_structural`) — no identity column enters this pipeline.
5. **Unresolved risks stay visible**: restated in §11 below, not dropped because Stage 4 closed.

**[VERIFIED]** `final_test` access demonstrated refused without `allow_final_test=True` (§7b of the driver output) — identical guardrail behavior as Stage 4, now exercised from Stage 5's own code path.

---

## 11. Tests

```
.venv/bin/pytest tests/ -v
58 passed, 0 failed, 0 skipped
```
(48 from Stages 2–4 + 10 new in `tests/test_preprocessing.py`.) No test from an earlier stage was modified or weakened. No genuine failure occurred this stage — the architecture passed on first implementation, unlike Stages 2 and 4 which each caught a real bug; stated plainly rather than inventing a false "bug found and fixed" narrative to match the pattern.

---

## 12. Unresolved Risks — Still Visible, Per Explicit Instruction

Nothing below is resolved by this stage; restated so Stage 6+ doesn't lose them:

1. **Identity-collection mechanism** (Stage 4) — still excluded, still no evidence either way.
2. **`R_emaildomain`/`V`-column prediction-time availability** (Stage 4) — missingness-pattern leakage concern addressed; timing-of-capture question still open. These columns are preprocessed here (they're `include_with_caveat`), but preprocessing them is not a safety verdict.
3. **D-column causal computation** (Stage 4) — supporting evidence only, not proof; unchanged by this stage.
4. **~325 non-spot-checked `V`-columns + all `C`-columns** (Stage 4) — included and preprocessed per documented family description; individually unverified.
5. **Intra-train non-causal computation** (Stage 4) — no preprocessing step can address this.
6. **`train_identity`/`test_identity` naming mismatch** (Stage 2) — moot while identity stays excluded; still a prerequisite if that ever changes.
7. **New, this stage: Stage 3's EDA reflects the full pre-split dataset, not `train` alone** (§1) — documented, and the strategy-vs-parameter line drawn to manage it; the underlying statistics in `docs/EDA_REPORT.md` were never recomputed per-partition and remain what they are (full-dataset figures).
8. **New, this stage: ~16 GB peak process memory**, far above the 1.78 GB final output — a real constraint for any smaller environment, not addressed within Stage 5's scope.

---

## 13. Final Engineering Verification Round — Addendum

**Status unchanged: Stage 5 remains awaiting approval.** This addendum reports the focused final-verification round requested after the initial implementation above. Nothing in §1–§12 was altered except where explicitly noted below; all prior-approved artifacts (Stages 1–4 reports, the register, the split assignment) are untouched.

### 13.1 Peak memory (~16.2 GB) — root cause localized, one real fix applied, overall number unchanged (item 1)

**[VERIFIED]** Instrumented the driver pipeline with `resource.getrusage().ru_maxrss` checkpoints before and after each step. Result: the jump from ~4.2 GB to ~16.5 GB happens **entirely inside the single call `transformer.fit(train_df)`** — i.e., inside scikit-learn's `ColumnTransformer`/`SimpleImputer`/`OneHotEncoder` internals (each sub-transformer's `fit_transform`, the dense-to-sparse conversion before the final `hstack`), not in any of this project's own pandas loading code.

**One real, confirmed inefficiency found and fixed at the pandas layer, independent of the above:** `measure_train_missingness()`, `fit_on_train_only()`, and `transform_partition()` each previously read the **entire** unfiltered 590,540-row `train_transaction.csv` into memory before filtering down to partition rows — three separate call sites, same redundant pattern. Replaced with a single shared `load_partition_df()` (chunked read, filtered per chunk, never materializing the full file), with a defense-in-depth ID-containment check (`_verify_ids_in_partition()`, used again in item 4). Verified by 4 new tests (`test_load_partition_df_*`), including one proving the result is identical regardless of chunk size.

**Honest result: fixing this did not move the end-to-end peak.** Repeated full-driver runs before and after this fix both measured **~16.0–16.6 GB** peak memory footprint (run-to-run variance from allocator behavior, not from the fix). The pandas-level redundancy was real and is now gone — but it was never the dominant contributor to the reported figure; sklearn's own internal fit/transform materialization is. **Do not read this as "item 1 resolved the memory issue" — it ruled out one specific, now-fixed cause and located the real one, which remains unaddressed.**

**Why the dominant cause is not fixed here:** eliminating it would require replacing `ColumnTransformer`'s own fit/transform architecture (e.g., a hand-rolled per-transformer loop with explicit intermediate disposal, or batch-wise fitting) — a substantial re-engineering of an already-tested, already-approved-in-principle pipeline, disproportionate to a verification round that explicitly excludes new feature engineering or model work. Carried forward as a known Stage 7+/productionization constraint (already in `docs/HANDOFF.md`), now with a precise root-cause location instead of a vague one.

### 13.2 float32 vs float64 and sparse vs dense — benchmarked, default NOT changed (item 2)

**Sparse vs. dense:** re-confirmed consistent with §9 above (41.56% train density, 1,778.1 MB sparse vs 2,849.6 MB dense) — nothing new to re-decide; no regression from the item-1 loading fix.

**float32 vs float64 — benchmarked cleanly as two SEPARATE processes.** (Methodological note worth recording: an initial attempt benchmarked both dtypes sequentially in one process and reported identical peaks for both — a measurement artifact, not a real finding, because macOS `ru_maxrss`/`time -l`'s "peak memory footprint" is a per-process high-water mark that never decreases, so the second dtype's number was silently contaminated by the first's higher peak. Caught before being reported, and the benchmark was redone as two independent process invocations.)

| | Peak memory footprint (`/usr/bin/time -l`, isolated process) | Fit time | Transform time |
|---|---|---|---|
| float64 (current default) | **16.01 GB** | 13.76s | 5.26s |
| float32 | **13.25 GB** | 13.24s | 4.88s |
| **Difference** | **−2.76 GB (−17.2%)** | ~4% faster | ~7% faster |

**Correctness check, not just shape:** `SimpleImputer.statistics_` sample values identical between dtypes for the columns checked; on a genuinely fractional column (`TransactionAmt`), max relative difference between float64 and float32-then-upcast values was **5.88×10⁻⁸** — within float32's own ~1.2×10⁻⁷ machine epsilon, i.e. ordinary floating-point rounding, not a meaningful precision loss for this data.

**Important nuance:** the final transformed output's dtype stays **float64 regardless of input dtype** — `OneHotEncoder`'s default `dtype=np.float64` parameter forces the categorical block to float64, and the final sparse `hstack` upcasts the whole result to match. So the ~17% saving comes entirely from smaller **transient** numeric intermediates during fit/transform, not from the final stored matrix (which is already accounted for in §9's 1,778.1 MB figure either way).

**Decision, per explicit instruction not to switch solely because one benchmark is faster: the pipeline's default is NOT changed.** float64 remains the implementation in `src/preprocessing.py`. This benchmark is documented as a validated, available lever — real, measured, correctness-preserving — to revisit if memory becomes the binding constraint on a smaller environment than this one, not adopted unilaterally here.

### 13.3 The 271 missingness indicators — threshold reframed, width re-verified exactly, redundancy found and quantified (item 3)

**The 5% threshold is a provisional engineering choice, restated explicitly: it has never been validated against predictive performance** — no model has been trained or evaluated at any point in this project, and §4 above chose it as "the midpoint of three tested thresholds," not as an optimum of anything measurable yet. Any Stage 6/7 work that revisits feature engineering should treat 5% as a starting point to test, not a settled constant.

**Feature-width arithmetic re-verified exactly** (not the earlier ~estimate): fitted a real `ColumnTransformer` on the real `train` partition and read its actual output width alongside each block's actual width —

| Block | Width | How |
|---|---|---|
| `numeric_plain` (no indicator) | 107 | 1 column per input (no indicator added) |
| `numeric_indicator` | 542 | 271 imputed values + 271 indicator columns — confirmed every one of the 271 selected columns has ≥1 missing value in `train` (so `MissingIndicator`'s `features="missing-only"` default drops none of them) |
| `categorical` (one-hot) | 164 | 151 raw observed categories + 13 (not 14) "missing"-sentinel columns — one of the 14 categorical columns has 0% missingness in `train`, so no `"missing"` category ever appears for it and `OneHotEncoder` correctly never creates a column for it |
| **Sum** | **813** | **Matches the actual fitted transformer's output width exactly** |

**Redundancy found, and it's substantial:** of the 271 missingness-indicator columns, **258 (95%) are EXACT duplicates** of one of only **13 distinct missingness-mask patterns** — the full 271-column block reduces to **26 genuinely unique patterns** in total (13 duplicate groups + singletons). This matches and quantifies what Stage 3's EDA only gestured at qualitatively ("14 V-columns share an identical missingness gap"): the Vesta-engineered `V`-column block was evidently computed/dropped in batches, so many columns share byte-identical missingness across all 438,125 `train` rows.

**Not deduplicated in this round.** Collapsing the 258 duplicated columns down to their 13 representative patterns is a legitimate, column-structure-level optimization (removing exact duplicates carries zero information loss regardless of any outcome variable, so it is not feature selection in the predictive sense) — but it is a real architectural change to the fitted pipeline's output shape, and this review round's scope is verification, not re-implementation. Flagged as a strong, concrete candidate for explicit approval before Stage 6, not applied unilaterally here.

### 13.4 Train-only fitting — guard strengthened and tested against the exact failure mode (item 4)

**[VERIFIED]** The new shared `load_partition_df()` (§13.1) now carries a defense-in-depth check, `_verify_ids_in_partition()`, on every call — it compares the `TransactionID`s actually loaded against the partition's expected ID set and raises `RuntimeError` on any mismatch. This sits at a different layer than `get_partition()`'s own `final_test` refusal (Stage 4): it would catch a bug in `load_partition_df()` itself, not just a caller requesting the wrong partition name.

**New tests, each constructed to fail if the guard were removed** (not tests that would pass regardless):
- `test_verify_ids_in_partition_raises_on_leaked_id` — directly constructs a leaked-ID scenario and confirms the guard fires.
- `test_load_partition_df_refuses_final_test_by_default` / `test_load_partition_df_allows_final_test_only_with_explicit_override` — confirms the Stage 4 `final_test` guardrail still holds through the new shared loading path.
- `test_load_partition_df_chunking_does_not_change_result` — confirms chunk size (the item-1 fix) is a pure implementation detail, not a source of a different (and possibly wrong) result.

The pre-existing two-layer leakage-guard (§7: synthetic fixture + real-data `TransactionAmt` median demonstration) was rerun unchanged and still passes.

### 13.5 Transform-time robustness (item 5)

**New, in addition to the pre-existing unseen-category / unseen-missing-value tests (§8, unchanged, still passing):**
- `test_transform_is_independent_of_input_column_order` — a `transform()` call with columns in a different order than at fit time produces an identical result (sklearn's `ColumnTransformer` selects by name internally, not position; this test proves it rather than assuming it).
- `test_transform_raises_clearly_when_a_fit_time_column_is_missing` — dropping a column that was present at fit time causes `transform()` to raise, not silently produce a misaligned/wrong-shaped result.
- `test_feature_names_out_are_stable_across_fit_and_transform` — `get_feature_names_out()` reflects the fitted structure and does not change based on what is later passed to `transform()`.

### 13.6 All 392 included columns, and only those, accounted for (item 6)

**[VERIFIED]** New `verify_column_accounting()` (4 new tests) run against the real register and the real fitted column lists:

```
{'n_included_in_pipeline': 392,
 'n_register_include_status': 392,
 'n_register_exclude_status': 42,
 'register_included_not_in_pipeline': []}
```

392 pipeline columns = 392 register `include`/`include_with_caveat` rows (23 + 369) exactly, zero register-included columns left uncovered. The function checks for a leaked excluded column **before** anything else and raised nothing — confirmed no `exclude_*` column reaches the pipeline.

**Side-finding surfaced while verifying this, recorded rather than silently absorbed:** the register has 435 rows but only **434 distinct column names** — `TransactionID` appears twice (once registered from `train_transaction`, once from `train_identity`), both rows agree exactly (`exclude_structural`, identical rationale). Benign and non-contradictory, but it is why `n_register_exclude_status` above reads 42 (distinct names) rather than 43 (register rows with an exclude status) — a set-based count would otherwise have silently hidden this instead of it being stated plainly.

### 13.7 Sparse-input compatibility of candidate Stage 7 estimator families (item 7 — tag inspection only, no `.fit()` call)

**[VERIFIED]** New `check_sparse_input_compatibility()` reads only each class's declared `input_tags.sparse` via `sklearn.utils.get_tags()` — no estimator is instantiated with data or fit anywhere in this check, consistent with this round's no-model-training boundary.

```
LogisticRegression:            True
RandomForestClassifier:        True
GradientBoostingClassifier:    True
HistGradientBoostingClassifier: False   <-- requires dense input
DecisionTreeClassifier:        True
LinearSVC:                     True
```

**Concrete, actionable finding:** `HistGradientBoostingClassifier` — a strong, commonly-used choice for imbalanced tabular fraud data — does **not** accept sparse input. This pipeline's output is sparse by design (`sparse_threshold=1.0`, §9). If Stage 7 selects HGB, the output would need densifying first, reintroducing the ~2,849.6 MB dense-memory cost already measured in §9 on top of whatever HGB itself needs. Surfaced now so it factors into Stage 7's model-family decision rather than being discovered as a surprise then.

### 13.8 Dependency installation and environment manifest (item 8)

**`requirements.txt` created at the project root — the first dependency manifest this project has had.** Generated via `.venv/bin/python3 -m pip freeze` against the project's own `.venv` (Python 3.12.15); annotated with which packages are core pipeline dependencies (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pytest`) versus notebook/dataset-retrieval tooling.

**Restated explicitly, per instruction:** `scikit-learn` 1.9.1 (and its transitive `scipy` 1.18.1, `joblib`, `threadpoolctl`) was installed during Stage 5's initial implementation **without a separate approval round-trip** — flagged at the time in the original §(header) of this report and in `docs/ML_WORKFLOW_STATE.md`'s Decisions log, now also recorded in `requirements.txt` itself. **No further dependencies will be installed or upgraded without explicit approval.**

### 13.9 Full test suite — rerun, exact results (item 9)

```
.venv/bin/pytest tests/ -v
72 passed, 0 failed, 0 skipped
```

48 tests unchanged from Stages 2–4 (not touched this round). `tests/test_preprocessing.py` grew from 10 to 24 tests: the 10 original plus 14 new (§13.1, §13.4, §13.5, §13.6, §13.7 above). **One genuine failure occurred and was fixed during this round** — `test_verify_column_accounting_raises_if_excluded_column_leaks_in` initially failed because `verify_column_accounting()` checked "not in register at all" before "explicitly marked excluded," so an excluded column that's also absent from the include-set raised the wrong (less specific) error message first. Fixed by reordering the two checks (leaked-excluded checked first, since it is the more specific and more alarming violation). Stated plainly as a real bug caught by the test it was written to prove, not glossed over.

**Unresolved limitations, unchanged by this round:** all items in §12 above remain open exactly as stated — this round investigated and documented them further (items 1, 3, 7 especially) but resolved none of them in the sense of closing out a risk; §12 should still be read as the authoritative list, with §13.1 refining item 8 of that list (the memory constraint) to a precise root cause rather than a general caution.

### 13.10 What changed in the code this round

- `src/preprocessing.py`: added `load_partition_df()` + `_verify_ids_in_partition()` (item 1/4); refactored `measure_train_missingness()`, `fit_on_train_only()`, `transform_partition()` to use the shared loader instead of each doing its own full-file read; added `load_included_columns()`'s silent-drop guard (raises if a register-included column is missing from the transaction schema, rather than quietly shrinking the column set); added `verify_column_accounting()` (item 6); added `check_sparse_input_compatibility()` (item 7).
- `scripts/run_stage5_preprocessing.py`: added sections 8–10 (column accounting, indicator redundancy, sparse-input compatibility), wired to the new functions above.
- `tests/test_preprocessing.py`: 14 new tests (§13.1, §13.4–13.7).
- `requirements.txt`: new file (item 8).
- Nothing in Stages 1–4's artifacts, the register, or the split assignment was touched.

---

## Three Interview-Style Questions

1. Sparse storage at 41.56% density still beat dense by ~38% memory, contradicting sklearn's own 30%-density default assumption. Explain the per-element overhead math that makes this true, and state the density at which the two would break even.
2. `OneHotEncoder(handle_unknown="ignore")` was chosen over your documented `pd.get_dummies()` approach specifically because of the three-partition design. Describe the concrete bug that `get_dummies()` could produce on `validation` that `OneHotEncoder` cannot.
3. The real-data leakage-guard demonstration found only a 0.05 difference in `TransactionAmt`'s median between train-only and train+validation fitting — a small number. Why does that small number not weaken the case for the fitting discipline, and why was a synthetic test still necessary alongside it?
