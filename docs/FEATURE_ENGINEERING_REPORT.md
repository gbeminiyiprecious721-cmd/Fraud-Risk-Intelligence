# Feature Engineering Report — Stage 6

**Status:** Implementation complete AND a focused final scientific-verification round complete (§9 below) — **awaiting review and approval**. Do not treat Stage 6 as done until the user approves these findings.

**Scope boundary respected:** feature construction and structural/leakage verification only. No model was trained, scored, or compared anywhere in this stage — every number below is a *structural* property (coverage, distribution, memory/time cost), never a predictive-performance claim. No Stage 7 work (model selection, hyperparameter tuning, final-test evaluation, dashboard/API) was started.

**New dependency installed:** none. This stage uses only pandas/numpy (already present) and scikit-learn's `MissingIndicator`/`SimpleImputer`/`ColumnTransformer` (already installed and approved in Stage 5).

**Additive by construction:** `src/feature_engineering.py` never imports from or modifies `src/preprocessing.py`. The approved Stage 5 baseline pipeline (`build_preprocessing_pipeline`) is untouched — confirmed by inspection and by the full test suite (Stage 2–5's 72 tests all still pass unchanged, §4).

---

## 1. Scientific Requirements — Addressed in Order

### 1.1 Information legitimately available at prediction time (req. 1)

**[VERIFIED]** Every engineered feature is built only from columns already carrying an `include`/`include_with_caveat` status in the Stage 4 register (`card1`, `card2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`, `card4`, `TransactionAmt`, `TransactionDT`), or from `TransactionID`/`TransactionDT` used purely as sort/grouping keys. No `exclude_*` column was read by any function in `src/feature_engineering.py`. Each derived feature explicitly **inherits** its source columns' own availability caveats rather than being treated as newly "safe" — stated per-feature in `docs/engineered_features_register.csv` (§3).

### 1.2 Temporal features are relative phases, not calendar time (req. 2)

**[VERIFIED]** `compute_temporal_phase_features()` computes `txn_hour_phase = (TransactionDT // 3600) % 24` and `txn_dow_phase = (TransactionDT // 86400) % 7` — pure modular arithmetic on the relative offset. The function's docstring, the engineered-features register, and this report all state explicitly: **these are NOT claimed to be actual calendar hour-of-day or day-of-week.** `TransactionDT`'s absolute origin (epoch, timezone) remains unconfirmed (Stage 2 Known Limitations, still open). Real-data distribution (train+validation, 513,685 rows): hour-phase ranges from 0.41% (phase 9) to 7.17% (phase 19) of rows — a real, non-uniform cyclical pattern exists in the relative offset, which is evidence the modulus is picking up *some* genuine daily rhythm, but is not evidence of what absolute hour any phase corresponds to.

### 1.3 Causal, tie-safe historical entity features (req. 3)

**[VERIFIED, by construction and by test]** `add_entity_historical_features()` computes, per entity, `{entity}_prior_txn_count` and `{entity}_time_since_prior_txn` via `np.searchsorted(sorted_dts, sorted_dts, side="left")` within each entity group — this returns, for every row, the count of **strictly smaller** timestamps in that group, which:
- never counts a row with an **equal** timestamp as prior (ties share the same count, by construction of `searchsorted`'s `side="left"`), and
- never counts a row **after** it in time, and
- never counts the row **itself**.

Tested explicitly (not just asserted) by `tests/test_feature_engineering.py`:
- `test_prior_count_is_zero_for_a_rows_own_earliest_transaction_even_with_future_rows_present` — the exact "future leakage" failure mode.
- `test_equal_timestamps_never_count_each_other_as_prior` — the exact "equal timestamp" failure mode, including a 3-way tie.
- `test_a_single_row_entity_never_contributes_to_its_own_feature`.
- `test_historical_features_are_independent_per_entity`.

### 1.4 Train/validation historical-context definition (req. 4)

**[VERIFIED]** `build_entity_history_train_and_validation()` computes historical features over `train ∪ validation` **jointly** (via `get_partition(["train", "validation"])`), not train-only. This is a deliberate, stated design choice, not an oversight: a historical count/recency feature is a per-row, time-indexed quantity, not a single global statistic (unlike Stage 5's median/category parameters) — a `validation` row's correct history legitimately includes `train` rows, because `train` is chronologically before it. Fitting the historical feature "train-only" would make `validation`'s first rows artificially, incorrectly sparse, not more leakage-safe.

**What is NOT allowed, and is enforced:** a row's history reaching into `final_test` (never requested — `allow_final_test` stays at its default `False`) or into its own future (enforced row-by-row by the strictly-less-than construction in §1.3, unconditionally).

**Precise scientific caveat, stated rather than glossed over:** the guarantee "a `train` row's own feature value is unaffected by `validation`'s presence" holds **because** Stage 4's chronological split makes every `train` row's `TransactionDT` strictly earlier than every `validation` row's — not because the function enforces it generally for arbitrary interleaved timestamps. Documented in the function's own docstring and tested on a fixture that deliberately violated this invariant during development (§4.1 below — a real finding from this round, not hypothetical).

### 1.5 Frequency/count encoding, train-only, unseen-safe; target encoding deferred (req. 5)

**[VERIFIED]** `fit_frequency_encoder_on_train()` fits a category→count mapping on the `train` partition exclusively via `get_partition(["train"])`. `apply_frequency_encoder()` maps an unseen category (absent from the fitted mapping) to an explicit `0`, never `NaN`, never a raised exception — tested by `test_unseen_category_at_apply_time_maps_to_explicit_unseen_value_not_crash_or_nan`. **Target encoding was not implemented**, per instruction — no code in this module computes any encoding that reads `isFraud`.

Real-data result: `P_emaildomain` (60 train categories) and `R_emaildomain` (61 train categories) — **0.00% of real `validation` rows hit a category unseen in `train`** for either column. The unseen-category path is exercised and tested on synthetic data regardless, since "it didn't happen to occur on this particular split" is not evidence the code path works.

### 1.6 Interaction and amount-derived features — plausibility stated, not performance (req. 6)

**[VERIFIED]** `compute_interaction_and_amount_features()` adds `email_domain_match`, `productcd_card4_interaction`, `log1p_transaction_amt`, `transaction_amt_is_round`. **No claim of predictive usefulness is made anywhere in the code, tests, or this report** — each is documented in `docs/engineered_features_register.csv` as a stated hypothesis for Stage 7 to test, with its own measured structural property reported honestly:
- `email_domain_match`: only **21.85%** of train+validation rows have both domains present (consistent with `R_emaildomain`'s high missingness, Stage 3/4) — of that 21.85%, **81.10%** match. Most rows get `NaN` (unknown), not a match/mismatch verdict.
- `transaction_amt_is_round`: **51.57%** of rows have a whole-dollar amount — a high base rate that, if anything, argues against this being a strong standalone signal (if most transactions are "round," roundness doesn't distinguish much), but this is reported as a structural observation, not a performance conclusion.
- `log1p_transaction_amt`: reintroduced only as an interaction-ready column. Stage 3's finding (Cohen's d near-zero, sign-flipping between raw/log scales) is **not** re-litigated or reversed by this stage.

### 1.7 Duplicate-indicator reduction — prototyped, Stage 5 baseline untouched (req. 7)

**[VERIFIED]** `find_duplicate_indicator_groups()` and `build_collapsed_indicator_pipeline()` implement the reduction as a **separate, prototype** `ColumnTransformer`, recomputed fresh from real `train` data (not hardcoded from Stage 5's earlier scratch finding): **271 indicator columns → 26 unique missingness patterns**, exactly matching Stage 5's §13.3 finding independently. The prototype's output width for that block is **297** (271 imputed values + 26 shared indicators) vs. Stage 5's shipped **542** (271 + 271) — a real, measured ~45% reduction in that specific block's width, with every one of the 271 columns' own imputed *values* still fully preserved (only the redundant *indicator* columns are collapsed). `src/preprocessing.py::build_preprocessing_pipeline()` was not modified; the Stage 5 baseline remains exactly as approved. This prototype is explicitly **not adopted** — flagged in the register as `prototype_unvalidated_not_adopted`, pending the model-based regression test carried forward from Stage 5's approval (item 1).

### 1.8 Memory efficiency and computational overhead (req. 8)

See §5 below for full measurements. Headline: the complete Stage 6 driver (all 5 feature families, real data) measured **8.68 GB peak memory footprint**, **20.8s wall time** via `/usr/bin/time -l` on this machine (Apple Silicon, per the instruction's "MacBook M1" framing — the same `/usr/bin/time -l` tool used in Stages 2/4/5 works identically on Apple Silicon). This number is **not directly comparable** to Stage 5's ~16GB figure — the two scripts do structurally different work (Stage 6 never builds the full 813-column `ColumnTransformer` on all 392 source columns) — so no "Stage 6 is more efficient than Stage 5" claim is made; the number is reported on its own terms.

### 1.9 Feature register updated with lineage (req. 9)

**[VERIFIED]** `docs/engineered_features_register.csv` — a new, separate artifact (11 rows, one per engineered feature), **not** a mutation of the approved Stage 4/5 `feature_availability_register.csv` (which is left untouched, per "do not modify prior approved research findings without evidence"). Columns: `feature_name`, `source_columns` (explicit dependencies), `transformation_rule`, `availability_assumption`, `leakage_risk_classification`, `status`, `evidence_level`, `notes`. Full detail in §3.

### 1.10 Automated point-in-time correctness tests (req. 10)

**[VERIFIED]** `tests/test_feature_engineering.py` — 14 tests, all passing (§4). Specifically includes the three named failure-mode exposures:
- **Future-information leakage:** `test_prior_count_is_zero_for_a_rows_own_earliest_transaction_even_with_future_rows_present`.
- **Equal-timestamp errors:** `test_equal_timestamps_never_count_each_other_as_prior` (including a 3-way tie, and the row immediately after the tie).
- **Train/validation boundary violations:** `test_train_row_history_is_unaffected_by_a_later_validation_row`, plus the consistency check `test_train_validation_consistency_check_train_rows_identical_with_or_without_validation_present` and its real-data counterpart (§2b of the driver output, §4).

### 1.11 Stage 4/5 safeguards preserved (req. 11)

**[VERIFIED]** Every data-loading entry point in `src/feature_engineering.py` goes through `src.leakage_boundaries.get_partition()`; `final_test` is never requested with `allow_final_test=True` anywhere in this module, its tests, or its driver script. Real-data confirmation (`scripts/run_stage6_feature_engineering.py` §2c): an unguarded `final_test` request was refused with the same `FinalTestAccessError` demonstrated in Stages 4/5. `src/preprocessing.py` and its tests are byte-for-byte unmodified (confirmed by the full suite, §4).

---

## 2. Model Training Boundary — Explicitly Not Crossed

Per instruction, **no throwaway baseline, no model of any kind was trained, fit to a target, or scored in this stage.** Every "coverage"/"rate"/"distribution" number above and in §5 is a property of the feature's own structure (how often it's present, what values it takes), computed without reference to `isFraud`. The question "do any of these features actually help" is explicitly **not answered here** — it is Stage 7's job, under a controlled validation protocol, per instruction.

---

## 3. Feature Lineage — `docs/engineered_features_register.csv`

| Feature | Source columns | Status | Leakage-risk note |
|---|---|---|---|
| `txn_hour_phase` | `TransactionDT` | prototype_unvalidated | None beyond the phase/calendar-time caveat |
| `txn_dow_phase` | `TransactionDT` | prototype_unvalidated | Same |
| `card1_prior_txn_count` | `card1`, `TransactionDT`, `TransactionID` | prototype_unvalidated | Moderate — algorithm verified, `card1`'s entity-key quality uncertain (see §6) |
| `card1_time_since_prior_txn` | same | prototype_unvalidated | Same |
| `p_emaildomain_freq` | `P_emaildomain` | prototype_unvalidated | Low |
| `r_emaildomain_freq` | `R_emaildomain` | prototype_unvalidated | Inherits `R_emaildomain`'s open availability-timing caveat |
| `email_domain_match` | `P_emaildomain`, `R_emaildomain` | prototype_unvalidated | Inherits `R_emaildomain`'s caveat |
| `productcd_card4_interaction` | `ProductCD`, `card4` | prototype_unvalidated | None beyond source columns |
| `log1p_transaction_amt` | `TransactionAmt` | prototype_unvalidated | None; Stage 3's weak-signal finding stands |
| `transaction_amt_is_round` | `TransactionAmt` | prototype_unvalidated | None |
| `collapsed_missingness_indicators` | the 271 Stage 5 indicator columns | prototype_unvalidated_not_adopted | None beyond represented columns' own caveats; not regression-tested |

**Vocabulary note, stated to avoid conflating two different registers:** `status` here (`prototype_unvalidated`) is deliberately a different vocabulary from the raw-column register's `include`/`include_with_caveat`/`exclude_*`. The raw register answers "is this source column available and safe to use at all"; this one answers "has this derived feature been checked for structural/leakage correctness" (yes, extensively) vs. "has it been shown to help a model" (no — not in scope this stage).

---

## 4. Tests and Verification Results — Exact

```
.venv/bin/pytest tests/test_feature_engineering.py -v
14 passed, 0 failed, 0 skipped

.venv/bin/pytest tests/ -v
86 passed, 0 failed, 0 skipped
```
(72 unchanged from Stages 2–5 + 14 new in `tests/test_feature_engineering.py`.)

**One genuine issue found and fixed during this round, stated plainly:** the first version of `test_train_validation_consistency_check_train_rows_identical_with_or_without_validation_present` used a synthetic fixture where a `train` row's `TransactionDT` (25) was **later** than a `validation` row's (15) — a scenario that cannot occur in this project's real, approved chronological split (Stage 4 guarantees every `train` timestamp precedes every `validation` timestamp), but was a valid input to the general function. Under that unrealistic ordering, the function correctly treated the chronologically-earlier `validation` row as "prior" to the `train` row — which is the CORRECT point-in-time behavior for the timestamps given, but caused the test's assumption (train-alone should always match combined) to fail, because that assumption depends on the split invariant, which the fixture violated. **This was not a bug in the feature-engineering code** — it was a test fixture that didn't reflect the real data's guaranteed ordering. Fixed by rewriting the fixture with timestamps that respect the real invariant (both `train` timestamps earlier than the `validation` timestamp), and the function's docstring was updated to state this dependency precisely (§1.4) so a future reader doesn't assume a stronger guarantee than what's actually true.

### 4.1 Real-data verification (not just synthetic)

`scripts/run_stage6_feature_engineering.py` §2b: re-ran the train-row consistency check on the **real** split (438,125 `train` rows) — confirmed **all 438,125** rows' `card1_prior_txn_count` identical whether computed via `build_entity_history_train_and_validation` (combined with `validation`) or via `add_entity_historical_features` on `train` rows alone. This is the real-data analogue of the synthetic test above, run against the actual approved Stage 4 split rather than a constructed fixture.

---

## 5. Memory and Runtime Measurements

`/usr/bin/time -l .venv/bin/python3 scripts/run_stage6_feature_engineering.py` (Apple Silicon Mac, project's own `.venv`):

| Step | Rows | Wall time |
|---|---|---|
| Temporal phase features | 513,685 | 0.004s |
| Entity historical features (`card1`) | 513,685 | 1.150s |
| Train-row consistency check (real data) | 438,125 | included above |
| Frequency encoding (both email columns, fit + apply) | — | <0.1s each |
| Interaction/amount features | 513,685 | 0.115s |
| Duplicate-indicator-group detection (fresh) | 271 columns × 438,125 rows | 0.075s |
| Collapsed-indicator prototype pipeline fit+transform | 438,125 rows | 7.029s |
| **Whole script** | — | **20.82s real, 19.46s user** |

**Peak memory footprint: 8,677,858,576 bytes (~8.68 GB).** Not directly comparable to Stage 5's ~16GB figure (different operations; Stage 6 never materializes the full 813-column transformed output) — reported as its own measurement, not a before/after claim.

---

## 6. Outstanding Scientific Uncertainties

Nothing below is resolved by this stage; stated so Stage 7+ doesn't lose them:

1. **No engineered feature has been shown to help a model.** Explicitly out of scope this stage, per instruction — Stage 7 owns this question under a controlled validation protocol.
2. **`card1`'s quality as an entity key is unconfirmed and looks weak.** Real-data measurement: mean `prior_txn_count` of 1,112 and a maximum of 13,631 across 513,685 rows (only 2.53% of rows are a "first-seen" `card1` value) — far too high a repeat rate for `card1` alone to plausibly be a clean 1:1 card-or-customer identifier in this anonymized dataset. No true entity ID exists in the IEEE-CIS data (a limitation already known from Stage 4). The historical-feature *algorithm* is verified correct; whether `card1` (or a composite key like `card1+card2+addr1`) is the *right* grouping is a separate, open question for Stage 7 to weigh when deciding whether this feature family is worth using at all.
3. **`R_emaildomain`'s prediction-time-availability timing question (Stage 4) remains open**, and is inherited unchanged by `r_emaildomain_freq` and `email_domain_match` — neither feature resolves it; both would need re-evaluating if that caveat is ever resolved either way.
4. **The 26-unique-pattern indicator collapse is not yet validated as safe to adopt** — it is information-preserving by construction (verified: exact-duplicate basis, zero value loss) but has not been regression-tested against an actual model, per the explicit Stage 5 carried-forward requirement.
5. **Whether `card1_prior_txn_count`/`time_since_prior_txn` are computable in real time at serving (Stage 13+)** has not been investigated — this stage only verified they're computable correctly in an offline, batch, point-in-time-correct way.
6. **`TransactionDT`'s absolute origin remains unconfirmed** — `txn_hour_phase`/`txn_dow_phase` stay relative-phase features, not calendar features, until or unless that changes.

---

## 7. Three Interview-Style Questions

1. A `card1`-based "prior transaction count" feature has a maximum of 13,631 and a mean of 1,112 across ~514K rows. Why does this number, by itself, raise doubt about whether `card1` is a usable proxy for a true customer/card identity, and what would you check next to settle it?
2. The point-in-time historical-feature function uses `np.searchsorted(dts, dts, side="left")` within each entity group to get a tie-safe "strictly prior" count. Walk through why `side="left"` (rather than `side="right"`) is the correct choice, and what would go wrong with `side="right"` on a group containing a tie.
3. Why is it correct for a `validation` row's historical feature to include `train` rows, while it would be incorrect for a `train` row's historical feature to include `validation` rows — and why does that asymmetry depend specifically on Stage 4's chronological split, rather than being a universal rule?

---

## 8. What Changed in the Code This Round

- `src/feature_engineering.py` — new module (temporal phases, causal entity history, frequency encoding, interaction/amount features, duplicate-indicator-collapse prototype). Does not import from or modify `src/preprocessing.py`.
- `tests/test_feature_engineering.py` — new, 14 tests.
- `scripts/run_stage6_feature_engineering.py` — new driver, real-data run, memory/runtime measured.
- `docs/engineered_features_register.csv` — new lineage artifact (11 engineered features).
- `docs/_stage6_feature_engineering_raw.json` — generated evidence artifact (gitignored).
- Nothing in Stages 1–5's artifacts, the raw-column register, or the split assignment was touched.

---

## 9. Final Scientific Verification Round — Addendum

**Status unchanged: Stage 6 remains awaiting approval.** This addendum reports the focused final-verification round requested after the initial implementation above. No model was trained, `final_test` was not accessed, and nothing in Stages 1–5's artifacts was altered.

### 9.1 `card1` must not be described as a unique customer/card identifier (review item 1)

**[VERIFIED, corrected framing]** Real-data measurement (513,685 train+validation rows): `card1` has **12,992 distinct values**, mean group size **39.54**, median **4.0**, but a **maximum group size of 13,632** — one single `card1` value accounts for ~2.65% of all rows in the dataset. Only **0.656%** of rows are a "singleton" `card1` value (appearing exactly once).

**Correct interpretation, stated precisely:** a *moderate* repeat rate (median 4 transactions per `card1`) is expected and unremarkable — real physical cards are reused over time, so some reuse is not evidence of anything wrong. The **extreme long tail** (one value covering 13,632 transactions) is the actual red flag: no physical card or individual customer plausibly transacts that often in this dataset's timeframe. The most defensible reading, consistent with this being an anonymized competition dataset with no documented customer/account ID, is that `card1` is a **coarser code** (plausibly something like an issuer/BIN-level or hashed/binned value) shared across many distinct underlying cards or customers, not a 1:1 identifier. **This is our own structural conclusion from the measured distribution, not an appeal to external Kaggle community claims** — no outside source was relied on for this specific interpretation. `docs/engineered_features_register.csv` and `docs/FEATURE_ENGINEERING_REPORT.md` §1.3/§6 (original) already avoided calling `card1` a customer/card key; this round makes the evidence for that caution explicit and quantified rather than just asserted.

### 9.2 Composite entity key — structural comparison only, no target, no model (review item 2)

**[VERIFIED]** `compare_entity_key_candidates()` (new function, 2 new tests) measures candidate keys' grouping structure with **zero reference to `isFraud`** — proven behaviorally (not just by inspection): permuting `isFraud`'s values and rerunning produces byte-identical results (`test_compare_entity_key_candidates_reads_no_target_column`).

Real-data result (513,685 rows, train+validation):

| Candidate key | Distinct entities | Mean group size | Max group size | % singleton rows |
|---|---|---|---|---|
| `card1` | 12,992 | 39.54 | 13,632 | 0.656% |
| `card1+card2` | 13,728 | 37.42 | 13,618 | 0.737% |
| `card1`..`card6` (all 6) | 14,072 | 36.50 | 12,167 | 0.761% |
| `card1+card2+addr1` | **39,072** | **13.15** | **8,582** | **3.094%** |
| `card1+card2+addr1+addr2` | 39,118 | 13.13 | 8,582 | 3.102% |
| `card1`..`card6`+`addr1`+`addr2` | 40,396 | 12.72 | 8,582 | 3.208% |

**Finding, stated as a structural observation, not a predictive one:** adding `card3`–`card6` to `card1`/`card2` barely changes anything (14,072 vs. 13,728 distinct entities; max group size still over 12,000). Adding `addr1` is where the real structural shift happens — distinct entities roughly **triple** (13,728 → 39,072), mean group size drops by nearly two-thirds, and the maximum group shrinks by ~35%. `addr2` adds almost nothing once `addr1` is present (39,072 → 39,118), consistent with `addr2` being a coarser, highly correlated geographic field.

**Conclusion, carefully bounded:** `card1+card2+addr1` is **structurally more defensible** than `card1` alone as an entity-grouping key — it discriminates more distinct groups with less extreme concentration. It is **not** claimed to be close to a true 1:1 identity key either (max group size still 8,582; mean 13.15) — no such key is recoverable from this anonymized dataset by construction alone, and no target-informed or model-based selection was performed to make this comparison, per instruction. This is evidence for a future Stage 7 decision, not a decision itself.

### 9.3 Strictly-earlier-timestamp verification (review item 3)

**[VERIFIED, re-confirmed]** Re-inspected `add_entity_historical_features()`'s implementation and re-ran the existing causal-correctness tests (§1.3 of the original report, 4 tests) plus the two NEW mutation-invariance tests below (§9.4) — all pass. No change to the algorithm was needed; this item's purpose was to confirm the existing implementation under renewed scrutiny, which it withstood.

### 9.4 New non-trivial test: later-row mutation cannot alter an earlier row's feature (review item 4)

**[VERIFIED, new]** Added 3 new tests, each a strictly stronger property than the original "future rows present" test (which only checked existence-sensitivity):
- `test_earlier_rows_feature_value_is_invariant_to_mutating_a_later_rows_timestamp` — mutates a LATER row's own timestamp (pushes it further into the future) and proves an EARLIER row's feature value is bit-for-bit unchanged.
- `test_earlier_rows_feature_value_is_invariant_to_mutating_a_tied_later_row` — same property specifically for a tied-timestamp group: breaking one member's tie (changing its timestamp) must not perturb the earlier row OR the other still-tied rows.
- `test_real_data_validation_rows_are_mutation_invariant_to_a_later_validation_rows_timestamp` — the real-data analogue, using the actual leakage-safe entry point and a real (temporary) assignment/transaction file pair, mutating a later `validation` row's timestamp on disk and re-running the full pipeline.

All 3 pass. Combined with the 4 original causal-correctness tests, this module now has **7 dedicated point-in-time tests**.

### 9.5 Operational assumption for validation-window history — clarified (review item 5)

**Clarified explicitly, in both the code and this report (not previously stated this precisely):** `build_entity_history_train_and_validation()`'s design — letting a later `validation` row's history include earlier `validation` rows — is correct **only under a fixed-batch / post-hoc scoring assumption**: the entire `validation` window is treated as already fully occurred and recorded before any row in it is scored. This matches how Stage 10 actually intends to evaluate (`validation`/`final_test` scored as a completed, static file, not transaction-by-transaction in real time), and is NOT the same as proving a genuinely live, real-time, point-of-sale system would have identical history available at serving time.

**New, explicitly unresolved limitation surfaced by this clarification:** a real-time system additionally depends on **feature-store ingestion latency** — how quickly a just-happened transaction becomes readable by whatever aggregate computes the next transaction's historical feature. This offline, full-file recomputation cannot detect or rule out that kind of latency gap (it assumes zero ingestion lag, i.e., perfect and immediate availability of every strictly-prior transaction). This is a **new, separate** production-readiness question for Stage 13+, distinct from the intra-partition leakage question already tracked — recorded in Known Limitations (§10 of `docs/ML_WORKFLOW_STATE.md`) rather than conflated with it.

### 9.6 Frequency encoding — train-only fit and deterministic unseen-category handling reconfirmed (review item 6)

**[VERIFIED, re-confirmed + 1 new test]** Re-ran the existing train-only-fit and unseen-category tests (§1.5 of the original report) — unchanged, still passing. Added `test_frequency_encoding_is_deterministic_across_repeated_calls`: fitting the same `train` data twice produces an identical mapping (dict equality), applying it to the same series twice produces identical output (`pd.testing.assert_series_equal`), and applying it to a **shuffled** copy of the same values produces the same per-value result regardless of row order — proving the encoding is a pure, order-independent function of each category's own value, not something that could vary run-to-run.

### 9.7 Indicator-collapse prototype — status and language corrected (review item 7)

**[VERIFIED, language corrected]** Confirmed by inspection and the full test suite that `src/preprocessing.py::build_preprocessing_pipeline` is unmodified and remains the sole pipeline Stage 5 approved. **Corrected a phrasing risk this round, proactively:** the original report's and register's "zero information loss by construction" language (describing the missingness-MASK representation) could be misread as a claim of predictive equivalence to the Stage 5 baseline. Both `src/feature_engineering.py::build_collapsed_indicator_pipeline`'s docstring and `docs/engineered_features_register.csv`'s `collapsed_missingness_indicators` row now state explicitly: the "no information loss" claim is about the mask representation only (every one of the 271 original masks is exactly recoverable from its 26 representatives) — **not** a claim that a model trained on the collapsed representation would perform the same, better, or worse. That remains untested and unknown, exactly as the original report already said in its Known Limitations (§6, original), now stated at the point of the claim itself too, not only in the limitations section.

### 9.8 All 11 engineered features rechecked against the Stage 4 register (review item 8)

**[VERIFIED]** New driver section (`scripts/run_stage6_feature_engineering.py` §6) cross-references every engineered feature's `source_columns` against the current (434-row, post-maintenance) `docs/feature_availability_register.csv`:

```
txn_hour_phase:               {'TransactionDT': 'include'}
txn_dow_phase:                {'TransactionDT': 'include'}
card1_prior_txn_count:        {'card1': 'include', 'TransactionDT': 'include', 'TransactionID': 'exclude_structural'}
card1_time_since_prior_txn:   {'card1': 'include', 'TransactionDT': 'include', 'TransactionID': 'exclude_structural'}
p_emaildomain_freq:           {'P_emaildomain': 'include'}
r_emaildomain_freq:           {'R_emaildomain': 'include_with_caveat'}
email_domain_match:           {'P_emaildomain': 'include', 'R_emaildomain': 'include_with_caveat'}
productcd_card4_interaction:  {'ProductCD': 'include', 'card4': 'include'}
log1p_transaction_amt:        {'TransactionAmt': 'include'}
transaction_amt_is_round:     {'TransactionAmt': 'include'}
collapsed_missingness_indicators: {}  (inherits the 271 underlying columns' own statuses individually, already documented)

Any unexplained excluded-column dependency across all 11 engineered features: False
```

**One dependency on an `exclude_*`-status column was found and investigated, not silently waved through:** `card1_prior_txn_count`/`time_since_prior_txn` both depend on `TransactionID`, which is `exclude_structural` ("Row identifier; no predictive meaning"). **Resolution, stated precisely:** `TransactionID` is used ONLY as a deterministic sort tie-breaker when `TransactionDT` values are identical within the same `card1` group — its own value is never exposed as part of either feature's output (the outputs are a count and a time-delta, not an ID). `exclude_structural` governs use of a column **as a predictive feature value**, not internal bookkeeping/sort-key use — the same precedent already established for `TransactionDT`'s use as a sort/partition key throughout this project (Stages 2, 4, 5, 6). This resolution is now documented directly in `docs/engineered_features_register.csv`'s notes for both affected rows, not just asserted here. No other engineered feature depends on any `exclude_*` column.

### 9.9 Tests and full suite — exact results (review item 9)

```
.venv/bin/pytest tests/test_feature_engineering.py -v
20 passed, 0 failed, 0 skipped

.venv/bin/pytest tests/ -v
92 passed, 0 failed, 0 skipped
```
(72 unchanged from Stages 2–5 + 20 in `tests/test_feature_engineering.py`, up from 14 — 6 new this round: 3 mutation-invariance tests §9.4, 1 determinism test §9.6, 2 entity-key-comparison tests §9.2.) No genuine test failure occurred this round — all 6 new tests passed on first run.

### 9.10 What changed in the code this round

- `src/feature_engineering.py`: added `build_composite_entity_key()`, `describe_entity_key_structure()`, `compare_entity_key_candidates()` (review items 1/2); extended `build_entity_history_train_and_validation()`'s docstring with the operational-assumption clarification (review item 5); extended `build_collapsed_indicator_pipeline()`'s docstring with the predictive-equivalence disclaimer (review item 7).
- `tests/test_feature_engineering.py`: 6 new tests (14 → 20).
- `scripts/run_stage6_feature_engineering.py`: added §2d (entity-key structural comparison) and §6 (Stage 4 register recheck).
- `docs/engineered_features_register.csv`: notes clarified for `card1_prior_txn_count`/`card1_time_since_prior_txn` (TransactionID bookkeeping-use resolution) and `collapsed_missingness_indicators` (predictive-equivalence disclaimer) — no `status`, `availability_assumption`, or feature decision changed.
- Nothing in Stages 1–5's artifacts, the raw-column register, or the split assignment was touched.
