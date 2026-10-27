# Stage 16 — Automated Testing

**Status: implementation complete, not yet marked done.** Strengthens the existing 297-test suite with repeatable data-contract tests, one lightweight real-artifact integration test, and a genuine, demonstrated regression-catch — rather than rebuilding the suite or adding new infrastructure. `final_test` was not accessed; no model was retrained; the frozen Stage 13 artifact was not regenerated.

## 1. Test-coverage map (audit, before any new test was written)

| File | Tests | Category | Stage |
|---|---|---|---|
| `test_data_validation.py` | 9 | Unit (data validation functions) | 2 |
| `test_eda_analysis.py` | 7 | Unit (EDA functions) | 3 |
| `test_leakage_boundaries.py` | 32 | Unit (splitting/leakage) | 4 |
| `test_preprocessing.py` | 24 | Unit (pipeline building) | 5 |
| `test_feature_engineering.py` | 20 | Unit (feature functions) | 6 |
| `test_model_selection.py` | 48 | Unit (model-selection helpers) | 7 |
| `test_experiment_tracking.py` | 27 | Unit (registry) | 8 |
| `test_hyperparameter_tuning.py` | 21 | Unit (CV/search) | 9 |
| `test_model_evaluation.py` | 26 | Unit (evaluation/segment/CI) | 10 |
| `test_explainability.py` | 21 | Unit (SHAP/permutation) | 11 |
| `test_responsible_ai.py` | 11 | Unit (fairness/robustness checks) | 12 |
| `test_production_pipeline.py` | 21 | Unit + **1 heavyweight integration/regression** | 13 |
| `test_api.py` | 30 | **API contract tests** | 14/15 |
| **Total before Stage 16** | **297** | | |

**Genuine gaps identified before writing anything new** (per explicit instruction to audit first):
1. **No standing data-contract tests.** Stages 2–5 performed thorough *one-time* EDA/validation, but nothing in the suite would catch a *future* drift (a corrupted CSV, a changed label encoding, a renamed column) — the single biggest real gap.
2. **No test exercised the REAL, persisted Stage 13 artifact** (`artifacts/frozen_rf_fraud_risk_v1.joblib`) directly. `test_production_pipeline.py`'s fast tests use synthetic fixtures; its one real-data test (heavyweight) re-fits fresh rather than loading the saved file. Nothing confirmed the actual on-disk artifact, as shipped, behaves sanely.
3. **No test suite marker/organisation** existed to distinguish fast tests from the one existing ~2-minute test — it was identified only by name (`-k "not full_training_regression"`), an informal convention, not a documented, tool-enforced one.

**Avoided as duplicate, no-new-evidence additions**: re-testing `validate_raw_input`'s missing/duplicate/non-numeric paths (already covered, `test_production_pipeline.py`), re-testing CORS (already covered, `test_api.py`), and re-deriving Stage 5's full per-column missingness profile (would just reproduce Stage 5's own work rather than adding a new, distinct guard).

## 2. New data-contract tests (`tests/test_data_contract.py`, 12 tests)

Two tiers, split by the `data_contract_full` pytest marker (registered in the new `pytest.ini`):

**Fast tier (7 tests, no marker, 0.32s)** — register/header/assignment-file level only, no full-file read:
- Register has exactly the documented structural columns (`TransactionID`, `isFraud`) and no duplicate column names.
- Register's feature-column count (`include` + `include_with_caveat`) is exactly **392** — independently cross-checked against Stage 13's `FeatureSchema.raw_columns` count.
- The real `train_transaction.csv` header contains every required column exactly once (catches header-level drift without a 650MB read).
- `docs/_stage4_split_assignment.csv` has the documented partition row counts: **train 438,125 / validation 75,560 / final_test 76,855**.
- The assignment file's `TransactionID` is unique and non-null; its `isFraud` is strictly binary for every row *except* `final_test` (whose label this project never reads outside Stage 10's one authorised evaluation).

**Comprehensive tier (5 tests, `@pytest.mark.data_contract_full`, 7.89s)** — a full scan of the complete `train` partition via the project's own, unmodified `load_partition_df`, never `validation`/`final_test`:
- `isFraud` is strictly binary, no nulls, integer dtype, across all 438,125 real rows.
- `TransactionID` is unique, non-null, and the loaded partition has exactly the documented row count.
- `TransactionDT` has no nulls and is numeric.
- Every required feature column (all 392) is actually present when the full partition is loaded.
- **`test_legitimate_feature_missingness_is_not_a_contract_violation`** — the instruction's most specific requirement, made concrete: loads `D1`/`D4`/`D10`/`dist1` (already known from Stage 2/3/5 to carry real missingness) and asserts they **do** contain nulls. Real measured values: **D1 0.06%, D4 29.04%, D10 14.32%, dist1 61.44%** missing. If a future edit of this file ever tried to assert "no nulls anywhere," this test would fail — it is a structural guard against accidentally over-constraining the contract, not just documentation of intent.

No transaction-amount bound, fraud-rate limit, or demographic attribute is asserted anywhere in this file — only the already-approved register/schema/partition facts.

## 3. Lightweight integration test (`tests/test_production_pipeline.py`, 1 new test)

`test_lightweight_integration_real_artifact_full_path_is_sane` — loads the **real, already-built** `artifacts/frozen_rf_fraud_risk_v1.joblib` (skips, rather than regenerating it, if absent) and scores 5 schema-valid synthetic rows through the full raw-input → fitted preprocessing → frozen Random Forest → probability path. Distinct from Stage 13's heavyweight equivalence test (which fits two models and compares them) — this one fits nothing, it only confirms the shipped artifact behaves correctly.

Verified: output shape `(5,)`; all values finite; all values in `[0, 1]`; repeated calls agree (`np.allclose`, not exact equality — see §4); a scrambled raw-column order produces the identical result (name-based alignment, reusing the same contract Stage 13/16 §5 targets); the preprocessor's learned imputation statistics and the model's tree count are unchanged before vs. after every call in the test (no fitting, no mutation).

**Runtime: 1.09s.**

## 4. A genuine bug this round caught, in the new test itself

Writing this integration test's repeated-request-consistency check with `np.array_equal` (exact equality) **failed** on the real artifact:
```
assert False
 +  where False = array_equal(array([0.52019849, ...]), array([0.52019849, ...]))
```
This is the project's own, already-documented (Stage 7/10/13) macOS `RandomForestClassifier(n_jobs=-1)` cross-thread floating-point-summation artifact (~1e-15-level differences between calls) — not a defect in `src/production_pipeline.py`. Fixed by switching to `np.allclose(rtol=1e-9, atol=1e-12)`, the same tolerance this project has used for this exact phenomenon since Stage 7. Reported here plainly: this round's test-writing found a real, if already-understood, numerical behavior, not a hidden bug.

## 5. Deliberate regression demonstration — the stage's actual acceptance criterion

**Target**: the name-based column-alignment contract, `src.production_pipeline.align_raw_columns`, via the existing test `test_predict_fraud_probability_is_deterministic_under_raw_column_reordering` — chosen because the instruction explicitly preferred it, and because Stage 12's own closure already established this exact failure mode (column order silently changing model behavior) as the project's highest-value regression to guard against.

**Method, exactly as performed:**
```bash
# 1. Isolated environment -- a real git worktree, not a loose file copy.
git worktree add --detach /tmp/stage16_mutation_worktree HEAD

# 2. Baseline: the target test against the CLEAN worktree copy.
cd /tmp/stage16_mutation_worktree
.venv/bin/python3 -m pytest tests/test_production_pipeline.py::test_predict_fraud_probability_is_deterministic_under_raw_column_reordering -v
# -> PASSED
```
A genuine behavioral defect (not a syntax error, not a forced assertion) was then introduced **only in the worktree's copy**:
```python
# before (the real, approved implementation):
return raw_df[list(schema.raw_columns)]

# after (deliberately mutated, worktree only):
return raw_df.iloc[:, : len(schema.raw_columns)]
```
A realistic defect pattern: someone "optimizes" name-based selection into positional slicing, silently assuming the caller's columns already arrive in the right order.

```bash
# 3. Same test, same (now mutated) worktree.
.venv/bin/python3 -m pytest tests/test_production_pipeline.py::test_predict_fraud_probability_is_deterministic_under_raw_column_reordering -v
# -> FAILED
```
**Real failure output, unedited:**
```
ValueError: columns are missing: {'ProductCD', 'card6'}
```
The test's own reversed-column-order input caused the mutated positional slice to select the *wrong* columns entirely — sklearn's `ColumnTransformer` itself caught the resulting mismatch and named exactly the columns the mutation silently dropped. **This is the test failing for the intended reason, with a message that identifies the violated contract** — not a manufactured failure.

```bash
# 4. Revert, in the worktree only, and reconfirm.
cd /tmp/stage16_mutation_worktree && git checkout -- src/production_pipeline.py
.venv/bin/python3 -m pytest tests/test_production_pipeline.py::test_predict_fraud_probability_is_deterministic_under_raw_column_reordering -v
# -> PASSED

# 5. Full cleanup.
cd /Users/.../Fraud-Risk-Intelligence
git worktree remove /tmp/stage16_mutation_worktree --force
```
**Cleanup verification, performed and confirmed**: `git worktree list` shows only the main repository; `git status --short` in the main repository is clean; `src/production_pipeline.py`'s `align_raw_columns` in the main working tree was directly re-inspected and confirmed to still read `return raw_df[list(schema.raw_columns)]` — the real, unmutated implementation, unchanged throughout.

## 6. Test-suite organisation

`pytest.ini` (new, registers two markers — config only, no new dependency):
```ini
[pytest]
markers =
    heavy: full-training, multi-minute tests that fit a real model on the complete train partition
    data_contract_full: comprehensive full-train-partition data-contract scans
```

| Command | Runs | Excludes |
|---|---|---|
| `pytest tests/` | **everything** (310 tests) — nothing silently omitted | — |
| `pytest tests/ -m "not heavy and not data_contract_full"` | the fast subset (304 tests) | 1 heavyweight regression test, 5 comprehensive data-contract tests |
| `pytest tests/ -m heavy` | just the Stage 13 full-training regression test | everything else |
| `pytest tests/ -m data_contract_full` | just the 5 comprehensive data-contract tests | everything else |

**The bare, default `pytest tests/` command runs the complete suite, including both expensive categories** — nothing is silently excluded by default; the fast subset is the explicit opt-*in* (via `-m`), not the other way around. This was a deliberate choice, per the explicit instruction that "the standard test command must not silently omit important tests." The existing Stage 13 heavyweight test was marked `@pytest.mark.heavy` (one decorator line; its logic/assertions are untouched) purely for this organisational purpose.

## 7. Quality assurance

- **Fast suite**: `pytest tests/ -m "not heavy and not data_contract_full"` → **304 passed**, 6 deselected, **6.73s**.
- **Comprehensive suite**: `pytest tests/` → **310 passed**, 0 failed, **139.62s** (0:02:19), peak memory **~20.7GB** (`/usr/bin/time -l`) — dominated by the heavyweight test's two full-training fits, consistent with Stage 13's own prior measurement.
- **Artifact checksum**: recomputed and confirmed **identical** (`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`) to the recorded manifest value, before and after this round's work.
- **No `final_test` access**: the data-contract tests explicitly restrict `load_partition_df` calls to `"train"` only; the assignment-file test excludes `final_test` rows from its label check by construction.
- **No model fitting outside the pre-existing heavyweight test and the Stage 16 regression demonstration** (which ran entirely inside a disposable worktree, never the real project).
- **No dependency added**: `pytest.ini` is configuration, not a package; everything used (`pytest`, `pandas`, `numpy`) was already installed.
- **`git diff --stat`**: new files `pytest.ini`, `tests/test_data_contract.py`, `docs/STAGE16_AUTOMATED_TESTING_REPORT.md`; modified `tests/test_production_pipeline.py` (+79 lines: one marker decorator + the new lightweight integration test). **Zero Stage 1-15 source files touched.**

## 8. Remaining unverified risks / limitations

- Data-contract tests cover `train` only (by design, per "approved labelled training-data contract") — `validation`'s and `final_test`'s own structural integrity are not independently re-verified by this stage (an intentional scope boundary, not an oversight: re-touching `final_test` for a contract check would violate the project's own standing restriction).
- The comprehensive data-contract tier (7.89s) and the heavyweight regression test (~2 minutes) are both excluded from the fast default workflow a developer would run on every small change — by design, but this means a change that silently breaks a data-contract invariant would only be caught when someone explicitly runs the comprehensive suite, not automatically on every commit (no CI is configured — see §9).
- This stage did not re-examine whether the *existing* 297 tests have any coverage gaps beyond the three named in §1 — a full line/branch-coverage audit (e.g. via `pytest-cov`) was not performed, since that would be a new dependency requiring separate approval.
- The deliberate-regression demonstration covered exactly one invariant (name-based column alignment); it is illustrative of the suite's real bug-catching power, not proof that every contract in the codebase is equally well-guarded.

## 9. Why ML projects need both software tests and data-quality tests

A unit test proves the *code* does what it's supposed to, given some input — it says nothing about whether tomorrow's real input still looks like the input every prior test assumed. A pipeline that passes every unit test can still silently produce garbage predictions if the upstream data changes shape (a column renamed, a label re-encoded from `{0,1}` to `{"Y","N"}`, a new 40%-null column introduced) — none of that is a *code* bug, so no amount of unit testing catches it. Software testing assumes the contract between code and data is stable; data tests are what actually verify that assumption, continuously, rather than once at EDA time and never again. This project's own Stage 2–5 EDA was thorough but one-time; §2's new `tests/test_data_contract.py` is what turns that one-time finding into a standing, repeatable guard.

## 10. Running this suite under a future CI system (not configured now)

No CI service was set up this round (explicitly out of scope — "do not begin Stage 17 or set up deployment infrastructure"). If one were added later, it would need no new test code, only a workflow file invoking:
```bash
.venv/bin/python3 -m pytest tests/ -m "not heavy and not data_contract_full"  # on every push -- fast, ~7s
.venv/bin/python3 -m pytest tests/ -m "heavy or data_contract_full"          # on a schedule / release branch -- ~2min20s, needs the real data + artifact present
```
The second command requires the real `data/raw/ieee-fraud-detection/` files and the built `artifacts/frozen_rf_fraud_risk_v1.joblib` to be available to the runner — both are gitignored (per Stage 2/13's own decisions), so a real CI setup would need its own, separately-approved plan for sourcing them (e.g. a cached artifact, not re-downloading Kaggle data on every run) — named here as a real, un-solved prerequisite for anyone who takes this further, not glossed over.

## Three interview-defense questions

1. **"What kind of bug would your test suite NOT catch?"** — Anything that produces a *plausible-looking but wrong* prediction without violating any of this suite's structural invariants — e.g., a subtly miscalibrated preprocessing step that still returns the right shape, right dtype, right value range, but numerically different values than intended. The suite catches shape/contract/crash-class bugs very well (as §5 demonstrates for real); it does not catch "the math is quietly wrong but still looks like a probability."
2. **"How do you test something as inherently statistical as a model?"** — You don't test *that the model is right* with unit tests — Stages 7–12 already own that question, with real evaluation metrics, confidence intervals, and audits. Stage 16's tests check a different, narrower thing: that the *pipeline around* the model (data shape, column order, artifact identity, no silent refitting) behaves deterministically and contractually — the model's own statistical correctness is Stage 10's job, not this one's.
3. **"Why mark the heavyweight test instead of just leaving it in `-k "not full_training_regression"`?"** — A `-k` name filter is an informal, easy-to-forget convention that only works if everyone remembers the exact substring; a registered pytest marker is a documented, tool-enforced category that shows up in `pytest --markers`, survives test renames, and makes the deliberate choice ("this is slow, here's why, here's how to still run it") visible in the test file itself rather than only in a shell history.

**Stage 16 is not marked complete.** Stage 17 has not begun, no deployment infrastructure was set up, `final_test` was not accessed. Awaiting your scientific and engineering review.
