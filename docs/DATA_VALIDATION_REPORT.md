# Data Validation Report — Stage 2

**Status:** Round 2 verification complete — see `docs/ML_WORKFLOW_STATE.md` Approval Status for whether Stage 2 is itself marked done.

**Scope boundary respected:** structural/integrity validation only. No target-vs-feature relationships, no distributions beyond missingness %, no visualizations, no splitting, no preprocessing, no modeling. That is Stage 3+ work.

**Evidence tags** (same convention as `docs/PROBLEM_STATEMENT.md`, extended with one new tag now that code has actually run against the data):
- **[VERIFIED — Stage 0]** — confirmed by file/schema inspection only (`ls`, `wc -l`, `head -1`).
- **[VERIFIED — Stage 2]** — confirmed by actually reading data this stage, via the functions in `src/data_validation.py`, executed in `scripts/run_stage2_validation.py`.
- **[EXTERNAL, UNCONFIRMED]** — still not checked against this data copy.
- **[HYPOTHETICAL]** — a business-framing assumption from Stage 1, unaffected by this stage.

---

## Round 2 Verification Addendum

Per user review: pytest installed and run, the row-count discrepancy pinned down exactly, one new critical finding surfaced during that work, and the temporal-holdout conclusion refined to not overclaim. Original Stage 2 body (Sections 1–10 below) is left as written; this addendum supersedes/extends specific points, cross-referenced by section.

### A. pytest installed and executed (requested approval, granted)

`pytest` 9.1.1 installed into the existing `.venv` (no new environment). `tests/test_data_validation.py` run:

```
9 passed, 0 failed, 0 skipped in 1.02s
```

No genuine failures occurred, so no fixes were needed before this result — the "fix and rerun" step in the request had nothing to act on. All 9 tests (target presence on train/test, ID uniqueness, 3 parametrized `classify_columns` cases including the pandas-3.0 string-dtype case, schema-diff detection of dtype/missing/added columns, leakage-flag exclusion) passed against both the real data (target presence, ID uniqueness) and synthetic fixtures (the rest).

### B. Row-count discrepancy (141,907 vs. 141,908) — pinned down exactly

**[VERIFIED — Stage 2, Round 2]** `wc -l data/raw/ieee-fraud-detection/test_identity.csv` → **141,908** total lines. The file ends with a trailing newline (confirmed via `tail -c 1 | xxd` → `0a`, i.e. a clean final line, not a missing-final-newline edge case). Line 1 is the header (`TransactionID,id-01,id-02,...`). `141,908 − 1 (header) = 141,907`, exactly matching the pandas-parsed data-row count reported in Section 2/4 below. No ambiguity, no off-by-one bug in either number — they were always consistent, just counting different things (total lines vs. data rows).

### C. Critical finding surfaced during this verification round: `train_identity` / `test_identity` use different column-naming conventions for the *same* fields

**[VERIFIED — Stage 2, Round 2 — not previously reported]** While confirming (B) above, schema-diffing `train_identity` against `test_identity` with `diff_schema()` (the exact function built for "unexpected structural changes" detection) surfaced this:

```python
>>> diff_schema(train_identity_schema, test_identity_schema)
missing_columns: ['id_01', 'id_02', ..., 'id_38']   # in train, absent from test — by this exact name
added_columns:   ['id-01', 'id-02', ..., 'id-38']   # in test, absent from train — by this exact name
dtype_changes:   {}
```

**All 38 identity sub-fields use underscores in `train_identity.csv` (`id_01`…`id_38`) and hyphens in `test_identity.csv` (`id-01`…`id-38`).** Same fields, same order, same dtypes — only the separator differs, consistently, across all 38 columns. `DeviceType`/`DeviceInfo` and `TransactionID` are unaffected.

**Why this matters, concretely:** any code that references these columns by name (a feature-engineering function, a `ColumnTransformer` selector, a rename/cleanup step) and is written against one file's naming convention will **silently fail to find those columns in the other file** — not an error, just columns quietly treated as 100%-missing at inference/evaluation time if nobody normalizes the naming first. This is exactly the kind of schema-consistency problem Stage 2 exists to catch before it reaches Stage 5.

**What this does NOT do:** this is a naming/schema issue, not a leakage issue, and not something this stage fixes — no preprocessing or column-renaming code was written, per the stage boundary. It's carried forward as a **required prerequisite for Stage 5** (Preprocessing): any shared identity-handling code must normalize `id_XX`/`id-XX` to one convention before train and test identity data are both run through it. Recorded in `docs/ML_WORKFLOW_STATE.md`'s roadmap as a Stage 5 adaptation note, not resolved here.

### D. Refined conclusion on the chronological holdout (Section 5 below)

The original Section 5 conclusion is accurate but incomplete as stated, and is refined here per review: **the measured `TransactionDT` separation (test starts ~30 days after train ends) supports and is consistent with a chronological train/test holdout design — it does not, by itself, independently establish that every feature is available at prediction time, nor does it eliminate all leakage risk.** Two distinct things can both be true:

1. **Train/test separation** (confirmed): test transactions are chronologically later than train transactions as a *set*. This rules out one specific failure mode — using `train_test_split(shuffle=True)` or any split that mixes future rows into training relative to test.
2. **Intra-train feature availability** (NOT established by this check): the 30-day train/test gap says nothing about whether an individual anonymized feature, for a given training-set row, was computed using only information available *before that row's own* transaction time. If any `V`/`C`/`D` feature were computed as a rolling aggregate over a window that isn't strictly causal per-row (e.g. a window-based count computed over the full training period rather than the period up to that row), a training row early in the window could still encode information from later training rows — a real leakage mode that train/test chronological separation does nothing to rule out, because it's entirely internal to the training set.

Point 2 is not resolved here — it requires either documentation of how Vesta computed the `V`/`C`/`D` features (unavailable) or an empirical check in Stage 3/4 (e.g., checking whether an anonymized feature's values for a given row change if that row's position in the full historical sequence changes, which would indicate a non-causal aggregation window). This is now explicitly listed as unresolved (Section 10 / addendum E below), not implied to be covered by the train/test timing check.

### E. Unresolved leakage questions — preserved, not resolved

Per review: the anonymized `V`/`C`/`D` feature leakage risk (original Section 8) and the intra-train feature-availability question (addendum D, point 2 above) remain **open**. Nothing in this round of work clears either — both are explicitly carried into Stage 3/4 as prerequisites, alongside the new `id_`/`id-` naming-normalization requirement for Stage 5 (addendum C).

---

## 1. Environment & Memory Strategy

**[VERIFIED — Stage 2]**
- Machine: 64 GB RAM, 10 cores (checked via `sysctl hw.memsize`/`hw.ncpu`).
- Used the existing `.venv` (Python 3.12.15, pandas 3.1.0, numpy 2.5.4) as instructed — no new environment created.
- `.venv-1` (Python 3.15.1, no packages) and the missing dependency manifest remain unresolved from Stage 0 — not touched this stage.

**Why memory-efficient techniques were used anyway:** with 64 GB of RAM, a naive `pd.read_csv()` of all five files at once would almost certainly survive on *this* machine. That's not the point. The techniques below (chunking, column selection, avoiding simultaneous full loads) are what make this code portable to a smaller machine, a CI runner, or a production batch job — and "how would you do this with 1/8th the RAM" is exactly the kind of question this choice is meant to pre-empt in an interview.

**Measured result:** running the full validation suite (`/usr/bin/time -l .venv/bin/python3 scripts/run_stage2_validation.py`) against all five files:

| Metric | Value |
|---|---|
| Wall time | 9.5–9.8 s (two runs) |
| Peak resident set size | ~1.10–1.18 GB |
| Peak memory footprint | ~0.83–1.15 GB |

Peak memory stayed **below the size of a single large file** (train_transaction.csv alone is 683 MB on disk) even while validating all five files in one process — because no step ever materializes a full 394-column frame; each check reads only the columns or chunk it needs and discards it before the next check runs.

**Core technique — chunked, column-selective reading** (from `src/data_validation.py`):
```python
# Counting rows without loading the file: read only the first column, in chunks.
row_count = 0
for chunk in pd.read_csv(path, chunksize=100_000, usecols=[0]):
    row_count += len(chunk)

# Missingness across all ~394 columns of a 683MB file, without ever holding
# the whole file in memory at once — accumulate null counts chunk by chunk.
null_counts = None
total_rows = 0
for chunk in pd.read_csv(path, chunksize=100_000):
    total_rows += len(chunk)
    chunk_nulls = chunk.isna().sum()
    null_counts = chunk_nulls if null_counts is None else null_counts.add(chunk_nulls, fill_value=0)
missing_pct = (null_counts / total_rows * 100).round(2)
```
This gives the *exact same numbers* as loading the full file and calling `.isna().mean()` — chunking changes memory usage, not correctness, because sums are associative regardless of how the rows are grouped into chunks.

**"Optimized data types" — considered, not applied yet, and why:** downcasting numeric columns (e.g. `float64`→`float32`, or a smaller `int` width) could roughly halve memory for those columns. This was **not** applied in Stage 2 deliberately: choosing a narrower dtype is a decision that depends on each column's actual value range and tolerance for precision loss — that's a Stage 5 (Preprocessing) decision informed by Stage 3 (EDA) findings, not something to bake into a validation script that's supposed to report what the data *is*, not reshape it.

---

## 2. File Integrity & Row Counts

**[VERIFIED — Stage 2]** All five files exist, are readable by pandas end-to-end, and produced no parse errors:

| File | Rows | Columns (sampled) | Size on disk |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | 683.4 MB |
| `train_identity.csv` | 144,233 | 41 | 26.5 MB |
| `test_transaction.csv` | 506,691 | 393 | 613.2 MB |
| `test_identity.csv` | 141,907 | 41 | 25.8 MB |
| `sample_submission.csv` | 506,691 | 2 | 6.1 MB |

Row counts match Stage 0's `wc -l`-based counts (minus the header line each). `test_identity.csv` is 141,907 here vs. 141,908 reported in Stage 0 — confirmed precisely in the Round 2 Addendum (§B) as exactly the header line, not a new finding or an off-by-one bug.

**Why this check matters:** before trusting *any* downstream number, confirm the file is actually complete and parseable — a truncated download or an interrupted copy would show up here as a row-count mismatch or a parse error, not as a confusing bug three stages later.

---

## 3. Schema, Target Presence, and a dtype-inference finding

**[VERIFIED — Stage 2]**
- `isFraud` is present in `train_transaction.csv` and in `sample_submission.csv` (the submission template), and **absent** from `test_transaction.csv`, `train_identity.csv`, and `test_identity.csv` — exactly as Stage 0 inferred from the header row, now confirmed programmatically via `check_target_presence()`.

**Finding made during this stage's work, not anticipated going in:** the first version of the column-classification check reported **zero categorical columns** in `train_transaction.csv`, which was wrong on inspection — `ProductCD`, `card4`, `card6`, the email-domain columns, and `M1`–`M9` are clearly categorical. The cause: **pandas 3.1.0 (installed in `.venv`) infers plain string columns as the new `StringDtype` by default** (`str(dtype) == "str"`), not the legacy `object` dtype most pandas code (including the first draft of `classify_columns()`) checks for. This is a real, version-specific behavior change (pandas made string-type inference the default starting in pandas 3.0), not a one-off typo — confirmed directly:
```python
>>> import pandas as pd; pd.__version__
'3.1.0'
>>> df = pd.read_csv("train_transaction.csv", nrows=5000)
>>> df["ProductCD"].dtype
StringDtype(storage='python', na_value=nan)   # str(dtype) == "str", not "object"
```
`classify_columns()` in `src/data_validation.py` was corrected to also treat `"str"`/`"string"` as categorical, with a comment explaining why, and the validation suite was rerun to confirm the fix (see Section 5). **This is the kind of check that's supposed to catch exactly this class of bug** — a column classification silently wrong because of a library-version default — before it reached Stage 5 preprocessing and produced a `ColumnTransformer` that mishandled 14+ columns.

---

## 4. Identifier Integrity & Join Coverage

**[VERIFIED — Stage 2]**

**TransactionID uniqueness** (every file, full scan, ID column only):

| File | Total | Unique | Duplicates |
|---|---|---|---|
| `train_transaction` | 590,540 | 590,540 | 0 |
| `test_transaction` | 506,691 | 506,691 | 0 |
| `train_identity` | 144,233 | 144,233 | 0 |
| `test_identity` | 141,907 | 141,907 | 0 |

No duplicate transaction IDs anywhere — the join key is clean on both sides.

**Join coverage** (`train_transaction` ⋈ `train_identity`, and the test equivalent):

| Split | Transactions | Identity rows | Matched | Coverage | Orphan identity rows |
|---|---|---|---|---|---|
| train | 590,540 | 144,233 | 144,233 | **24.42%** | 0 |
| test | 506,691 | 141,907 | 141,907 | **28.01%** | 0 |

Every identity row matches a real transaction (zero orphans) in both splits — this is a clean, well-formed one-to-zero-or-one relationship (a transaction has at most one identity row), not a data quality defect. Test's slightly higher coverage (28.01% vs. 24.42%) is a real, measured difference worth carrying into Stage 3, not something to explain away here.

**Why both checks matter:** a duplicate ID would silently corrupt any row-level evaluation (double-counting one transaction's outcome); an orphan identity row (one with no matching transaction) would indicate either a join-key mismatch or a genuine data integrity problem — this dataset has neither.
```python
# Join coverage, reading only the ID column from each file (cheap even at 650MB):
txn_ids = set()
for chunk in pd.read_csv(transaction_path, usecols=["TransactionID"], chunksize=100_000):
    txn_ids.update(chunk["TransactionID"].tolist())
identity_ids = set()
for chunk in pd.read_csv(identity_path, usecols=["TransactionID"], chunksize=100_000):
    identity_ids.update(chunk["TransactionID"].tolist())
coverage_pct = 100 * len(txn_ids & identity_ids) / len(txn_ids)
```

---

## 5. TransactionDT Structure — the most consequential finding this stage

**[VERIFIED — Stage 2, upgraded from EXTERNAL/UNCONFIRMED]**

| | `TransactionDT` range (seconds) | Span |
|---|---|---|
| `train_transaction` | 86,400 → 15,811,131 | ≈ 182 days |
| `test_transaction` | 18,403,224 → 34,214,345 | ≈ 183 days |

**`test`'s minimum `TransactionDT` (18,403,224) is greater than `train`'s maximum (15,811,131)** — a gap of 2,592,093 seconds, which is almost exactly **30 days**. This is not an assumption anymore: it's a measured fact from this data copy.

**Why this matters (Stage 4 implication, confirmed not hypothesized):** this structurally confirms the Stage 0/1 hypothesis that `train` and `test` are a **chronological holdout**, not a random split — `test` is entirely "the future" relative to `train`, with a clean 30-day gap in between (plausibly a deliberate buffer to reduce near-boundary leakage, though *why* Kaggle chose a gap is outside what this data can tell us — noted as a question, not a claim). Stage 4 **must** replicate this structure (sort-then-split by `TransactionDT`, likely with its own internal gap) rather than defaulting to `train_test_split(shuffle=True)`, which would scramble this order and leak future information into training.

**Refined conclusion (Round 2 review — see Addendum §D for full reasoning):** the separation above supports and is consistent with a chronological holdout design. **It does not, by itself, independently establish that every feature is available at prediction time, nor does it rule out all leakage risk** — specifically, it says nothing about whether an anonymized feature was computed causally *within* the training period (a non-causal rolling aggregate could still leak information from later training rows into earlier ones, entirely inside `train`, regardless of how clean the train/test boundary is). Treat "confirmed chronological holdout" and "confirmed leakage-free" as two different claims — only the first is established here.

**What this does NOT tell us:** `TransactionDT`'s *absolute* meaning (is `0` an actual timestamp, or an arbitrary reference point?) is still **[EXTERNAL, UNCONFIRMED]** — we only know it's a consistent relative offset in seconds. Nothing here implies a real calendar date, consistent with Stage 1's explicit caution against treating `K` (investigation capacity) as tied to a real schedule.

```python
# Confirming the chronological-holdout structure — read only the DT column from each file:
def check_transactiondt_ordering(path, dt_col="TransactionDT", chunksize=100_000):
    overall_min = overall_max = None
    for chunk in pd.read_csv(path, usecols=[dt_col], chunksize=chunksize):
        c_min, c_max = chunk[dt_col].min(), chunk[dt_col].max()
        overall_min = c_min if overall_min is None else min(overall_min, c_min)
        overall_max = c_max if overall_max is None else max(overall_max, c_max)
    return overall_min, overall_max

train_min, train_max = check_transactiondt_ordering("train_transaction.csv")
test_min, test_max = check_transactiondt_ordering("test_transaction.csv")
assert test_min >= train_max  # True on this data — test is strictly after train
```

---

## 6. Missingness

**[VERIFIED — Stage 2]** Computed via the chunked accumulate-then-divide pattern (Section 1), so these percentages are exact, not sampled.

**`train_identity.csv`** — ten columns are 96%+ missing (`id_07`, `id_08`, `id_18`, `id_21`–`id_27`), and these are on top of the table only covering 24.42% of transactions in the first place — meaning in the *full* `train_transaction` population, these particular identity sub-fields are populated for well under 2% of all rows. This is a severe-missingness pattern that will need an explicit decision in Stage 5 (likely: drop, or encode "missing" itself as a signal) rather than a default imputer.

**`train_transaction.csv`** (full chunked scan, all ~394 columns) — the most-missing columns are `dist2` (93.63%), several `D`-family columns (`D7`, `D13`, `D14`, `D12`, `D6`, `D9`, `D8`, all 87–94%), and a block of `V`-family columns around 86%. The `V`/`C`/`D`/`id_` columns are Vesta's anonymized engineered features — their *names* don't reveal what they measure, so **why** they're missing (not computed for certain transaction types? sentinel for "not applicable"? genuinely absent?) is not determinable from missingness percentages alone. Flagged as an open question for Stage 3/5, not resolved here.

**Why the distinction between "missing" and "data-quality defect" matters:** nothing in this section implies these columns are broken. High, structured missingness in anonymized features is extremely common in this kind of engineered-feature dataset (several are likely "count of X in the last N days," which is undefined/missing when there's no prior history) — a real pattern to characterize in Stage 3, not a bug to "fix" by dropping columns reflexively.

---

## 7. Numeric vs. Categorical Classification (schema-level, corrected)

**[VERIFIED — Stage 2, corrected mid-stage — see Section 3]**

| File | Numeric | Categorical | Other |
|---|---|---|---|
| `train_transaction` | 380 | 14 | 0 |
| `train_identity` | 24 | 17 | 0 |

The 14 categorical columns in `train_transaction` are `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, and `M1`–`M9`. This is a **schema-level** classification (what dtype a column parses as) — it does not yet say what encoding each one needs, or how high-cardinality `P_emaildomain`/`R_emaildomain` actually are (that's a Stage 3 question). `card1`, `card2`, `card3`, `card5` remain numeric by dtype despite being identifier-like in meaning (card network/issuer codes) — worth flagging for Stage 5/6: a numeric dtype doesn't mean "treat as a continuous quantity," and this is a common place preprocessing goes wrong by default.

---

## 8. Preliminary Leakage / Feature-Availability Risk Flags

**[VERIFIED — Stage 2, structural only]**

- **Excluded from features, unconditionally:** `TransactionID` (row identifier, no predictive meaning) and `isFraud` (the target itself). Trivial to state, embarrassing to miss — now enforced by `flag_potential_leakage_columns()`, not just a comment.
- **368 anonymized `V`/`C`/`D` columns are "unverifiable from schema alone."** Vesta never published what these engineer, so a column name cannot tell us whether a given feature is computed using only information available *before* the transaction's outcome was known, or whether it could inadvertently summarize post-outcome information (e.g. a "number of chargebacks on this card" counter that's only updated after a chargeback is filed would be a real leakage risk, and nothing in the column name would reveal that). **This is reported as an open risk, not cleared** — resolving it requires either external documentation (none found) or empirical checks in Stage 3/4 (e.g., checking whether a feature's distribution shifts suspiciously at the `isFraud` boundary in a way inconsistent with being known pre-outcome).
- No column name in this schema suggests an obviously post-outcome field (no `chargeback_date`, `investigation_result`, etc.) — a structural scan, not an exhaustive semantic audit of 368 anonymized columns.

---

## 9. Reusable Validation Code

- `src/data_validation.py` — 11 pure, read-only functions (file integrity, schema sampling, target presence, ID uniqueness, join coverage, chunked missingness, `TransactionDT` ordering/range comparison, column classification, schema diffing, leakage flagging). No function fits, learns from, or transforms the data.
- `scripts/run_stage2_validation.py` — driver that executed all of the above against the real files and produced every number in this report; rerunnable (`.venv/bin/python3 scripts/run_stage2_validation.py`) and writes a structured JSON (`docs/_stage2_validation_raw.json`, gitignored) for programmatic reuse.
- `tests/test_data_validation.py` — pytest-style tests covering the functions above (target presence, ID uniqueness, column classification including the pandas-3.0 string-dtype case, schema diffing, leakage flagging) — **written but not executed**, see Unresolved Issues.
- `docs/_dataset_schema_snapshot.json` — a saved schema snapshot (all 5 files) for `diff_schema()` to compare future reruns against, so a silent schema change (re-downloaded data, a renamed column, a dtype flip) is caught automatically rather than discovered downstream.

---

## 10. Unresolved Issues

1. ~~pytest not installed~~ — **Resolved, Round 2:** installed into `.venv`; `tests/test_data_validation.py` run, 9/9 passed. See Addendum §A.
2. **Two virtual environments still exist** (`.venv`, functional; `.venv-1`, empty) — unresolved since Stage 0, not blocking Stage 2 (which only needed pandas + pytest), but will need a decision before Stage 5+ needs scikit-learn and friends.
3. **No dependency manifest** (`requirements.txt`/`pyproject.toml`) exists yet — everything run this stage used whatever happens to be in `.venv`; nothing is pinned. (`pytest==9.1.1` should be recorded once a manifest exists.)
4. **368 anonymized `V`/`C`/`D` columns' leakage risk is structurally unverifiable** from schema alone (Section 8) — **preserved as open per Round 2 review**, carried forward as a prerequisite for Stage 3/4, not resolved by the chronological-split finding (see Addendum §D).
5. **`TransactionDT`'s absolute real-world meaning is still unconfirmed** (Section 5) — we know it's a consistent relative offset, not whether `0` corresponds to any real calendar date. Doesn't block Stage 4 (which only needs relative ordering), but should stay flagged so it's never casually treated as a real timestamp later.
6. **Schema sampling uses the first 5,000 rows**, not a full-file scan, for dtype inference (`get_schema_sample`) — a rare mixed-type value appearing only later in a 590K-row file could change a column's true dtype without this check catching it.
7. **New, Round 2: `train_identity`/`test_identity` use inconsistent column naming** (`id_XX` vs. `id-XX` for all 38 identity sub-fields) — see Addendum §C. **Required prerequisite for Stage 5**: normalize before any shared code touches both files' identity columns. Not fixed here — out of Stage 2's scope (no preprocessing), but it would be a mistake to let this go unrecorded until Stage 5 discovers it the hard way.
8. **New, Round 2: intra-train feature-availability is not established**, only the train/test boundary timing is (Addendum §D) — a distinct, still-open leakage question from #4 above, since it concerns potential leakage *within* the training set rather than between train and test.

---

## Dataset Validation Summary (for quick reference)

| Area | Result |
|---|---|
| File integrity | All 5 files present, readable, row/column counts verified |
| Target (`isFraud`) | Present in train only, as expected; test correctly lacks it |
| TransactionID | Unique in all 4 files with an ID column; zero duplicates |
| Join coverage | train 24.42%, test 28.01%; zero orphan identity rows either side |
| TransactionDT | Chronological holdout **confirmed** for the train/test boundary; does **not** establish intra-train feature availability or clear all leakage risk (refined, §D) |
| Missingness | Severe (86–97%) in specific identity and D/V-family columns — structural, not yet explained |
| Column types | 380 numeric / 14 categorical (train_transaction); corrected mid-stage for pandas 3.1's string dtype |
| **Naming mismatch (new)** | `train_identity` uses `id_XX`, `test_identity` uses `id-XX` for all 38 fields — must be normalized before Stage 5 shared code |
| Leakage flags | Target + ID excluded; 368 anonymized columns + intra-train availability flagged unverifiable, not cleared |
| Memory | Peak ~1.1 GB RSS for the full 5-file suite; never loaded all files at once |
| Testing | pytest 9.1.1 installed; 9/9 tests passed, 0 failed, 0 skipped |
| Unresolved | 2 venvs; no dependency manifest; anonymized-column + intra-train leakage risk; TransactionDT's real-world meaning; id-naming normalization needed before Stage 5 |

## Three Interview-Style Questions

1. The join coverage between `train_transaction` and `train_identity` is 24.42%, with **zero** orphan identity rows. What does the "zero orphans" part specifically rule out that the coverage percentage alone wouldn't tell you?
2. `TransactionDT`'s range shows `test` starting exactly ~30 days after `train` ends. This confirms a chronological train/test boundary — but why does it **not** by itself prove every feature is leakage-free or available at prediction time? What separate thing would you need to check to close that gap?
3. `train_identity` and `test_identity` turned out to use `id_XX` vs. `id-XX` for the same 38 fields. If this had gone unnoticed until Stage 5, what's the most likely *symptom* you'd see (not the cause) — and why would that symptom be easy to misdiagnose as something else?
