# Stage 13 — Production Pipeline

**Status: implementation complete, not yet marked done.** Packages the approved Stage 5 preprocessing + the frozen Stage 9/10 Random Forest into one importable, versioned, testable module — decoupled from any driver script. `final_test` was not accessed; the frozen configuration was not retuned, reselected, or changed.

## 1. Pipeline module architecture

`src/production_pipeline.py` exposes a small public interface, delegating entirely to the approved Stage 5/Stage 10 building blocks (`build_preprocessing_pipeline`, `build_frozen_model`) rather than reimplementing them:

| Function | Role |
|---|---|
| `fit_production_pipeline(train_df, ...)` | Fits the approved preprocessing + frozen RF, returns a `PipelineArtifact` |
| `validate_raw_input(schema, raw_df)` | Raises `SchemaValidationError` on a missing column, duplicate column name, or non-numeric value in a numeric column |
| `align_raw_columns(schema, raw_df)` | Reorders raw input columns **by name**, never by position |
| `predict_fraud_probability(artifact, raw_df)` | The main inference entry point: validate → align → `.transform` (never `.fit`) → `.predict_proba` |
| `predict_from_transformed(artifact, X, feature_names)` | A lower-level entry point for a caller already holding a transformed matrix; **requires** the caller's own `feature_names` and refuses to guess a reordering if they don't match the artifact's schema exactly |
| `save_artifact` / `load_artifact` | joblib persistence with a sidecar manifest and checksum verification |
| `compute_ordered_schema_fingerprint` | A deliberately **order-sensitive** fingerprint of the transformed feature-name sequence |

`FeatureSchema` (a frozen dataclass) is the artifact's identity: the exact ordered raw columns it expects, which of those are numeric, and the exact ordered transformed feature names its fitted model was trained on.

## 2. Frozen model and feature-schema details

Frozen config (unchanged since Stage 9): `RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42, class_weight=None)`. Fit on the complete `train` partition (438,125 rows) — the same effective training procedure Stage 10 used.

- **392 raw input columns** required (numeric + categorical, register-approved).
- **813 transformed feature columns**, in a fixed order that is now part of the artifact's identity.

**Why order matters, precisely (the direct lesson from Stage 12's audit):** `RandomForestClassifier(max_features="sqrt")` subsamples candidate features **by column position** at every split. Two preprocessing paths that produce the identical 812/813-feature *set* in a different *order* therefore fit genuinely different trees from identical data — which is exactly what Stage 12 found between Stage 9's per-fold-recomputed column order and Stage 11/12's whole-train-derived order (AP 0.47255 vs. 0.47545). `compute_ordered_schema_fingerprint` is deliberately the opposite of `experiment_tracking.compute_feature_schema_fingerprint` (which sorts before hashing, by design, for a different purpose) — this one does **not** sort, so it changes if the sequence changes even when the set doesn't.

## 3. Artifact metadata, size, and checksum

`artifacts/frozen_rf_fraud_risk_v1.joblib` (**16,799,089 bytes ≈ 16.0 MiB**) + `artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json` (git-tracked, ~41KB).

| Field | Value |
|---|---|
| Artifact ID / version | `frozen-rf-fraud-risk` / `1.0.0` |
| Training partition | `train`, 438,125 rows, row-fingerprint `a39d2eee...` |
| Git commit (at build time) | `52c24481...` (working tree dirty — this stage's own new files not yet committed) |
| Environment | numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, Python 3.12.15 |
| Artifact SHA-256 | `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04` |
| Ordered-schema fingerprint | `cba2e8082c5e96e60bcc87ed45b5231aebf023ff9357ebde95a66f5257a70ea8` |
| Stage 9 reference | candidate_index=4, mean CV AP 0.51236 (context only) |
| Stage 10 reference | AP 0.42784, ROC-AUC 0.86637, 95% CI [0.408, 0.445] on `final_test` (context only — **not re-measured by this artifact**) |

**This is a reconstruction, not a recovery.** Stage 10's original fitted `RandomForestClassifier` object was never serialized anywhere and no longer exists in memory — it cannot be recovered. This artifact is a **fresh fit of the identical frozen configuration on the identical training partition**. Its equivalence to an independently-built reference is *verified*, not assumed (§4). The historical Stage 9/10 metrics above are cited for context and must never be read as a new evaluation of this specific joblib object.

**16MB is large enough that it is NOT git-tracked** (`.gitignore`'s pre-existing `*.joblib` rule now actually applies, for the first time) — only the manifest is. The artifact itself is reproducible locally with one command: `.venv/bin/python3 scripts/build_stage13_artifact.py`.

## 4. Independent regression comparison results

`tests/test_production_pipeline.py::test_full_training_regression_against_independent_reference_procedure` — the stage's real completion evidence. Fits the frozen configuration **twice** on the complete `train` partition (438,125 rows): once via an **independent reference procedure** calling `build_preprocessing_pipeline`/`build_frozen_model` directly (never `fit_production_pipeline` — deliberately avoiding comparing the new wrapper against itself), and once via the new packaged `fit_production_pipeline`.

**Result: PASSED.**
- Model configuration identity: `model_reference.get_params() == artifact.model.get_params()` ✓, and both match every `FROZEN_RF_CONFIG` key exactly.
- Ordered feature-schema identity: `artifact.schema.transformed_feature_names == ct_reference.get_feature_names_out()` (all 813, in order) ✓.
- Prediction equivalence: `np.allclose(proba_reference, proba_new, rtol=1e-9, atol=1e-12)` ✓ — a tolerance justified by this project's own documented macOS `n_jobs=-1` parallel-float-summation artifact (Stage 7/10 precedent: ~1e-15-level differences between otherwise-identical fits), not loosened to force a pass.
- Timing: reference fit 50.7s, new-path fit 51.0s — consistent with each other, as expected for two independent fits of the same configuration on the same data.

**If this test had failed, the instruction was to stop and investigate rather than adjust the tolerance or the model — it did not fail, so no such investigation was needed this round.**

## 5. Save/load prediction-equivalence evidence

From the real artifact build (`scripts/build_stage13_artifact.py`): fit 51.150s → save 0.028s → load 0.027s → inference on 1,000 rows 0.060s. **`predictions_match_after_reload=True`** (`np.allclose` on the same 1,000 rows before and after a real joblib save/load round-trip). A second, synthetic-data version of this same check (`tests/test_production_pipeline.py::test_save_and_load_artifact_round_trip_predictions_match`) passed identically on first run.

## 6. Error-handling and schema-validation tests

| Case | Behavior |
|---|---|
| Missing required raw column | `SchemaValidationError` ("Missing required raw column(s): [...]") |
| Duplicate column name | `SchemaValidationError` ("Duplicate column name(s) in input: [...]") |
| Non-numeric value in a numeric column | `SchemaValidationError` ("...contains non-numeric value(s)...") — caught **before** reaching `SimpleImputer`, with a clearer message than Stage 12's equivalent deep-pipeline `ValueError` |
| Permitted missing (`NaN`) numeric value | Does **not** raise — the approved pipeline's imputation handles it, by design |
| Extra, unexpected raw column | Does **not** raise — ignored by `align_raw_columns`, a documented, deliberate contract choice |
| Raw input columns in scrambled order | Predictions identical to canonical order (`align_raw_columns` reorders by name, never position) |
| Unseen categorical value (e.g. `ProductCD="ZZ_never_seen"`) | Handled cleanly via `OneHotEncoder(handle_unknown="ignore")`, re-confirmed end-to-end through this module |
| `predict_from_transformed` called with mismatched/reordered `feature_names` | `SchemaValidationError` — refuses to guess a reordering of an already-numeric matrix |
| Loaded artifact file tampered with after saving | `ArtifactIntegrityError` on the next `load_artifact` call (checksum mismatch) |
| Repeated `predict_fraud_probability` calls | Confirmed **no refitting**: `SimpleImputer.statistics_` and the model's tree count are identical before and after |

## 7. Runtime and memory measurements

| Operation | Time | Notes |
|---|---|---|
| Real artifact fit (438,125 rows, 813 features) | 51.15s | `scripts/build_stage13_artifact.py` |
| Save (joblib dump) | 0.028s | |
| Load (joblib load + checksum verify) | 0.027s | |
| Inference, 1,000 rows | 0.060s | |
| Full-training regression test (2 fits + comparison) | 118.3s total | reference 50.7s + new path 51.0s + overhead |
| Peak memory, artifact build | ~16.7GB (`/usr/bin/time -l`) | consistent with every prior full-train fit since Stage 5 |
| Peak memory, regression test (2 concurrent fits in one process) | ~16.76GB | roughly the same order as one fit — the two `X` matrices and models do not both stay fully resident at peak the way sequential GC would suggest, measured not assumed |

All within the existing MacBook's demonstrated capacity — no new resource ceiling was approached.

## 8. Reproducibility and security limitations

- **Reproducibility**: fitting is deterministic given the frozen config + `random_state=42` + this exact row/column order — verified by the regression test, not merely assumed. The macOS `n_jobs=-1` parallel-summation artifact (~1e-15) remains a known, bounded source of non-bit-exactness, carried forward from Stage 7/10.
- **A fixed random seed alone is not sufficient for reproducibility** — demonstrated directly by this stage's own central finding: Stage 9 and Stage 11/12 used the identical `random_state=42` and still produced different fitted models, because column *order* (not the seed) was the actual variable that changed. Determinism requires fixing the ENTIRE input (data, config, order), not the seed in isolation.
- **Manifest's `artifact_path` field stores an absolute, machine-specific path** (set at build time) — a minor portability wart in the metadata only; `load_artifact` does not read this field to decide where to load from (the caller always supplies its own `path`), so it does not affect correctness, only informational completeness on a different machine.
- **Security**: joblib/pickle deserialization executes arbitrary code for an arbitrary pickled object. `load_artifact` verifies the loaded file's sha256 against its sidecar manifest — but this **only proves the file has not changed since this project's own `save_artifact` wrote it**. It does **not** establish that the artifact's original content was trustworthy; a checksum recorded by an attacker alongside their own malicious file would "match" too. The only real safeguard here is operational, stated explicitly rather than engineered: **never call `load_artifact` on a path outside this project's own `artifacts/` directory, and never on a file obtained from an untrusted source.** No cryptographic signing was added (would require new infrastructure this stage's scope excludes).
- No new dependency, no paid service, no external model registry — `joblib` is already a scikit-learn dependency.

## 9. Complete test results and git status

**265 tests passing** (246 prior + 19 new in `tests/test_production_pipeline.py`, zero regressions). **Two genuine bugs were caught and fixed during the fast unit tests** (not forced to pass, not hidden): (1) a test fixture tried to assign a string into an already-`float64` pandas column, which modern pandas now rejects at assignment time rather than silently upcasting — fixed by casting the column to `object` dtype first, in the test, not the source; (2) a test referenced the `ColumnTransformer`'s numeric branch by the wrong internal name (`"numeric"` instead of the pipeline's actual `"numeric_plain"`) — fixed to match the real, already-approved pipeline's naming. The heavyweight full-training regression test (§4) passed on first run.

`git diff --stat` for this round: new files `src/production_pipeline.py`, `tests/test_production_pipeline.py`, `scripts/build_stage13_artifact.py`, this report, `artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json`; modified `.gitignore` (clarifying comment; the `*.joblib` rule already existed and now actually applies) and `docs/experiment_registry.jsonl` (+1 record, `status=completed`, 139→140, with the field name `final_test_refused` deliberately NOT recorded in the registry — it tripped the registry's own forbidden-content guard for field names containing `"final_test"`, caught and fixed by removing the field from the tracked record rather than working around the guard). **Zero Stage 1-12 source files touched, zero approved modelling/preprocessing code modified, the frozen configuration unchanged, `final_test` not accessed, zero new dependency installed.**

## 10. Documentation and learning

**1. Why must a model and its fitted preprocessing be packaged together?** Because a `RandomForestClassifier` has no idea what a "ProductCD" or a missing `D4` value *means* — that knowledge lives entirely in the fitted `ColumnTransformer` (its learned medians, its learned one-hot categories). Serving the model without its exact fitted preprocessor is like keeping a translated book's pages but losing the translator — the words are there, the meaning is gone.

**2. Why can't feature count alone guarantee prediction correctness?** Because two column arrangements can share the same *count* (and even the same *set*) while differing in *order* — and for this specific model family, order changes which features compete for each split. "813 features" is necessary but not sufficient information; "813 features, in exactly this sequence" is the actual contract.

**3. How did transformed feature order cause the Stage 12 discrepancy?** Stage 9's per-fold search recomputed the missingness-indicator column order from each fold's own training rows; Stage 11/12's auxiliary fit instead reused an order computed once over the whole `train` partition. Same 812 columns, different sequence. `RandomForestClassifier(max_features="sqrt")` samples candidate features by position at every split — different sequence, different candidate pools, different trees, different AP (0.47255 vs. 0.47545) from identical row data.

**4. What does model serialization actually preserve?** The fitted *parameters* — every tree's learned splits, every imputer's learned medians, every encoder's learned categories — exactly as they existed in memory at save time. It does **not** preserve the training *process* (the specific rows, the exact library versions, the exact code that produced those parameters) unless that provenance is recorded separately, which is exactly what this stage's manifest does.

**5. Why is a fixed random seed alone insufficient for reproducibility?** Because a seed only controls the *randomized choices inside* a fixed procedure (which rows a tree bootstraps, which features it samples at a split) — it says nothing about the inputs the procedure is applied to. Fix the seed but change the column order, and you've changed *which* features index 3 and index 7 refer to at the moment the seeded RNG picks "feature index 3, then 7" — a different, but equally "seeded," result. Reproducibility requires fixing the whole input, not just the randomness.

**6. How does reconstruction equivalence differ from recovering an identical historical model?** Recovery would mean retrieving the literal Stage 10 Python object — impossible, since it was never saved. Reconstruction means building a *new* object from the same recipe and *proving*, through an independent test, that it behaves the same way (same config, same ordered schema, matching predictions within a justified tolerance). This project can only ever offer the second — and that's an honest, useful, verifiable claim, not a weaker fiction.

**7. Why must joblib artifacts come from trusted sources only?** `joblib.load` (like `pickle.load`) can execute arbitrary Python code embedded in a malicious file during deserialization — it is not a passive data format like JSON. A valid-looking checksum doesn't help if the attacker controls the file the checksum was computed from. The only real protection is provenance: only load files this project itself built, from its own `artifacts/` directory.

**Notion connections:** *Python software engineering* — the `FeatureSchema` frozen dataclass and the `SchemaValidationError`/`ArtifactIntegrityError` custom exception types are a direct application of "fail with a specific, named error type" over letting a generic `KeyError` or `ValueError` leak from deep inside sklearn. *ML pipelines* — this stage is the concrete answer to "how do you keep a notebook/script model and a served model from drifting apart," a question your journal's pipeline notes raise in the abstract. *Testing* — the fast-unit-vs-heavyweight-regression split (§9) is the same test-pyramid idea from your testing notes, applied to ML instead of application code: cheap synthetic checks for logic, one expensive real-data check for the thing that actually matters (does the real artifact match an independent real fit). *Git* — the decision to track the manifest but not the 16MB binary is a direct, concrete instance of "don't commit generated/large binaries," reasoned through rather than applied as a reflex rule.

### Worked example: loading the artifact and scoring a synthetic transaction

```python
from pathlib import Path
import pandas as pd
from src.production_pipeline import load_artifact, predict_fraud_probability

artifact, manifest = load_artifact(Path("artifacts/frozen_rf_fraud_risk_v1.joblib"))

# A single synthetic, schema-valid transaction (values made up for
# illustration only -- never a real transaction from this dataset).
synthetic_row = pd.DataFrame([{col: 0.0 for col in artifact.schema.numeric_columns}])
synthetic_row["TransactionAmt"] = 125.0
for cat_col in [c for c in artifact.schema.raw_columns if c not in artifact.schema.numeric_columns]:
    synthetic_row[cat_col] = "W" if cat_col == "ProductCD" else "debit" if cat_col == "card6" else "unknown"

risk_score = predict_fraud_probability(artifact, synthetic_row)[0]
print(f"Predicted fraud probability: {risk_score:.4f}")
```

This example is illustrative only — it is not run against `final_test` or any real transaction, and its output is not reported as a model-quality claim anywhere in this document.

## Three interview-defense questions

1. **"How do you guarantee the served model matches what you evaluated?"** — Not by trusting that the code "looks the same," but by an independent regression test that fits the exact same configuration twice, through two different code paths, on the exact same data, and checks model-parameter identity, exact ordered feature-schema identity, and `np.allclose` predictions — the test that would have caught Stage 12's own column-order bug if it had existed in this stage's code.
2. **"What's versioned together, and what breaks if they get out of sync?"** — The fitted preprocessor and the fitted model are saved as ONE joblib object, never two separately-versioned files; if they ever drifted apart (e.g., a preprocessing code change without a corresponding re-fit), the model would silently receive a different feature distribution than it was trained on — exactly the notebook/production-drift failure this stage exists to prevent.
3. **"Why isn't a fixed random seed enough to guarantee two fits are the same?"** — Because this project's own Stage 12 audit is a counterexample: identical seed, identical row split, identical hyperparameters, two different fitted models — because the input's column order differed and `max_features="sqrt"` samples by position, not by name.

**Stage 13 is not marked complete.** Stage 14 has not begun. No API, dashboard, or deployment was built. `final_test` was not accessed. Awaiting your scientific review.
