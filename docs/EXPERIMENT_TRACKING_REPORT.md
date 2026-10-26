# Stage 8 — Experiment Tracking Report

**Status:** implementation complete, awaiting explicit approval to mark Stage 8 Done (same pattern as every prior stage — see `docs/ML_WORKFLOW_STATE.md`).

## 1. Why experiments need identifiable configurations

By the end of Stage 7, this project had produced roughly a dozen *files* of saved evidence (`docs/_stage7_*_raw.json`), but the actual number of distinct *experiments* inside those files — individual model fits, each with its own settings and result — turned out to be **28**, not 12. That gap is exactly the problem Stage 8 exists to close: a results file tells you *what happened*, but not, on its own, a reliable way to ask "which exact configuration produced this number, and is it still comparable to that other number over there?"

Two numbers that look alike are not necessarily comparable. The RF pilot and the RF full-train benchmark are **both** "Random Forest, same hyperparameters, same preprocessing pipeline" — but one is trained on 15,000 rows with 2.72% fraud prevalence and 790 fitted features, the other on 438,125 rows with 3.507% prevalence and 813 fitted features (this session's own feature-count reconciliation found the exact, verified reason for that last difference: `OneHotEncoder` sees more distinct categories with more training rows — see the Decisions log). Without a structured record of *which* config, *which* partition, *which* feature count each number came from, "AP went from 0.327 to 0.411" would read as a clean improvement when it is actually three things changing at once (dataset size, fraud prevalence, and feature count).

## 2. Reproducibility is more than `random_state`

Your Notion journal's `Introduction to Multiple Linear Regression` page covers `random_state` as "the seed used by the random number generator" — correct, but only one ingredient. A run is reproducible only if **everything that can change the result** is pinned and recorded, not just the seed:

| What can silently change a result | Where this project pins it | What the registry records |
|---|---|---|
| Random seed | `RANDOM_STATE = 42` throughout `src/model_selection.py` | `random_seed` |
| Model hyperparameters | `build_bounded_random_forest()` etc. | `hyperparameters` |
| Which rows were used to fit | `build_chronological_pilot_subset()` / `load_partition_df()` | `partitions` (ids + row counts + prevalence) |
| Which columns the preprocessing pipeline actually produced | `build_preprocessing_pipeline()` is fit-time-dependent (see §1) | `feature_count`, `feature_schema_fingerprint` |
| The code itself | git | `git_commit`, `git_dirty` |
| The library versions doing the computing | `requirements.txt` pins the project's environment, but a specific run can still have been executed under a different installed version over time | `dependency_versions` |

A number with a seed but no git commit, no feature count, and no partition identifier is not reproducible — it is just a number with a seed attached. That's the gap this registry closes.

## 3. Git, environment, and data fingerprints working together

Three independent "versions" have to agree for a result to be re-derivable:

1. **Code version** (`git_commit` + `git_dirty`) — *which lines of `src/` produced this?* `git_dirty=True` would mean the working tree had uncommitted changes at run time — a real reproducibility risk, since the commit hash alone would then describe code that wasn't actually what ran.
2. **Environment version** (`python_version` + `dependency_versions`) — *which installed library behavior produced this?* scikit-learn's own solver internals have changed between minor versions; this project's own `FutureWarning` about the deprecated `penalty` parameter (visible in `pytest`'s warning output) is a live example of a library surface that can shift under you.
3. **Data version** (`partitions`, `feature_schema_fingerprint`, `evidence_checksums`) — *which actual rows and columns produced this?* `feature_schema_fingerprint` is a `sha256` over the sorted column-name set, so two fits over the logically same columns always fingerprint identically regardless of the order columns happened to be listed in; `evidence_checksums` lets a reviewer verify that a cited evidence file hasn't silently changed since the number was reported.

All three have to line up for "reproduce my best run exactly" (the Stage 8 interview-defense question from the workflow's own reference) to have a real answer rather than a shrug.

## 4. Validation reuse and selection bias — made auditable, not eliminated

This is the most important caveat in this report, stated plainly: **this registry does not remove selection bias — it makes it visible.**

Across Stage 7's 28 backfilled runs, `validation_consulted=True` appears on 15 of them, and `influenced_subsequent_decisions=True` on 6 — meaning the `validation` partition's score was looked at, and in those 6 cases, that look is on record as having shaped what was tried next (e.g. the pilot's two SAGA class-weight arms were compared on validation AP specifically to recommend which one to carry into the next investigation round; RF's pilot AP being stronger than GBM's pilot AP was part of why RF was the one scaled to the full partition). Every one of those 15 `validation`-touching experiments drew on the *same* 75,560-row `validation` partition. That repetition is exactly the "multiple comparisons" risk the workflow's own Stage 9 documentation warns about: a score that has already been looked at several times is not the same thing, statistically, as a score being seen for the first time.

Logging this honestly does two things:
- It lets Stage 9 (Hyperparameter Tuning) see, per prior run, whether a `validation` number is "fresh" or "already-consulted" — informing the carried-forward requirement that Stage 9 must use a time-respecting, training-only validation scheme (e.g. a chronological inner split of `train` itself) rather than treating this stage's already-repeated `validation` scores as an unbiased signal for tuning.
- It does **not** retroactively make any of Stage 7's already-reported numbers "unbiased." The bias was already there; this registry's contribution is that it is now a traceable fact instead of an untracked one.

## 5. How another engineer finds the evidence behind a reported number

Given any reported Stage 7 number (e.g. "full-train Random Forest AP = 0.41101"):

1. Look it up: `query_registry(experiment_name="rf_full_train_benchmark")` (or `grep` the JSONL file for `"average_precision":0.41101`) — gives the full record: hyperparameters, partitions, feature count, timing, peak memory, evidence path.
2. Open the cited `evidence_paths` entry (`docs/_stage7_rf_full_train_raw.json`) and compare its `sha256` against the record's `evidence_checksums` entry — if they match, the evidence file hasn't been altered since this record was written.
3. Check `git_commit` (for a *live*, directly-verified run) against the project's git history to see the exact code that produced it. Backfilled historical records deliberately leave `git_commit=None` rather than guess which of the five Stage 7 commits (`a0672b2`…`4b398d1`) a given experiment belongs to — see §6.
4. Check `provenance` and `confidence` — a `"reconstructed_from_evidence"` / `"high"` record means every populated field traces to a saved JSON file; `missing_fields` lists exactly what could not be recovered (mostly `git_commit`/`python_version`/`dependency_versions` for every backfilled record, and `metrics` for the handful of `"diagnostic"`-status structural/exploratory entries that never produced a model metric in the first place).

## 6. Reproducibility limitations, stated honestly

- **No historical `git_commit` mapping.** Every backfilled record has `git_commit=None`. Mapping each experiment to the exact commit under which it ran would require inferring from file-modification timestamps against commit timestamps — an inference, not a measurement, and this registry's rule is to leave a field `None` rather than guess. A future, *live* run (via `track_run()`) records this automatically and exactly.
- **`experiment_timestamp` for backfilled records is the evidence file's filesystem modification time**, not an independently-logged execution timestamp (none existed before this stage) — labeled explicitly via `experiment_timestamp_provenance` on every such record, never presented as a verified fact.
- **Two `numeric_conditioning` records' `loss_trajectory` linkage is `confidence: medium`.** The original evidence file does not state outright whether the per-iteration loss trace was captured from the exact same fit as the arm's headline `fit_seconds`, or a separate warm-start pass with an identical configuration — both are plausible from the saved numbers, and the backfill does not resolve this ambiguity by guessing.
- **No cross-platform file locking.** `track_run`/`append_run_record` use POSIX `fcntl.flock`, which works on this project's macOS/Linux targets but not on Windows — stated in the module docstring, not silently assumed universal.
- **This registry is not a substitute for CI-grade reproducibility.** It records enough to trace a number back to its config and evidence; it does not re-execute anything to verify a number still reproduces bit-for-bit under today's environment.

## 7. Connecting to your Notion Data Science Journal

- **Python `logging`** (Software Engineering for ML/AI Engineers): `track_run()` is the same instinct as the `logging` module — "record what happened, with enough context to make sense of it later" — generalized from one script's console output to a whole project's run history, and made structured (JSON) and queryable instead of free-text.
- **Git workflows** (ML/AL Continue — Git version control): this registry treats the git commit hash the same way Git itself treats a commit — as the one unambiguous pointer to "exactly this code." `git_dirty` is the direct analogue of checking `git status` before trusting a result.
- **Testing** (your journal's unit-testing material, and this project's own `pytest` discipline across Stages 2–7): the 25 tests in `tests/test_experiment_tracking.py` apply the same idea you've already practiced — a round-trip test (write then read, get back what you wrote), a negative test (a duplicate run ID must be refused), and a static-analysis test (the module's own source is parsed to prove it never imports anything capable of reading `final_test`).
- **Model evaluation** (Intermediate ML — cross-validation, `RandomizedSearchCV`): this stage is the bookkeeping layer *underneath* evaluation — before Stage 9 can systematically search hyperparameters without accidentally re-using an already-biased validation signal, there has to be a record of which signal has already been touched. That record is this registry.

## 8. One example record

```json
{
  "run_id": "hist-7807aa826aa22e77",
  "workflow_stage": 7,
  "experiment_name": "rf_full_train_benchmark",
  "experiment_purpose": "Full 438,125-row train partition, identical settings to the verified pilot -- the strongest, most-verified Stage 7 result.",
  "status": "completed",
  "provenance": "reconstructed_from_evidence",
  "confidence": "high",
  "model_family": "RandomForestClassifier",
  "hyperparameters": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42},
  "partitions": {"train": {"id": "train (complete)", "n_rows": 438125, "fraud_prevalence": 0.03507}, "validation": {"id": "validation", "n_rows": 75560}},
  "feature_count": 813,
  "metrics": {"average_precision": 0.41101, "roc_auc": 0.85596, "...": "..."},
  "training_time_seconds": 15.871,
  "peak_memory_bytes": 16360797024,
  "evidence_paths": ["docs/_stage7_rf_full_train_raw.json"],
  "evidence_checksums": {"docs/_stage7_rf_full_train_raw.json": "<sha256>"},
  "validation_consulted": true,
  "influenced_subsequent_decisions": true,
  "git_commit": null,
  "missing_fields": ["dependency_versions", "git_commit", "git_dirty", "python_version", "..."]
}
```
