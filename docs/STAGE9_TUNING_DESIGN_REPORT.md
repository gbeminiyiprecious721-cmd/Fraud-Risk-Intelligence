# Stage 9 — Chronological Hyperparameter Tuning: Design and Feasibility Report

**Status: design, feasibility, AND the approved 20-candidate search are all complete.** This report presents the chronological CV design, the Random Forest search space, a bounded feasibility probe, a real full-scale five-fold feasibility experiment (§8), the tracking-registry extension, and the approved search's real results (§9) — all built against `train` only. `validation` and `final_test` were not accessed anywhere, in any round.

## 1. Chronological cross-validation design

**Why not plain `sklearn.model_selection.TimeSeriesSplit`, assumed safe:** it divides chronologically-sorted rows into equal-size chunks by row count, with no awareness of tied timestamps. Measured directly on the real `train` partition:

| Finding | Value |
|---|---|
| Unique `TransactionDT` values in `train` | 424,401 |
| Timestamps shared by more than one row | 13,332 |
| Rows involved in a tie | 27,056 (6.2% of `train`) |
| Max rows sharing one timestamp | 8 |

`assess_timeseries_split_compatibility()` checked `TimeSeriesSplit`'s actual row-count boundaries against this real tie structure across several candidate `n_splits`:

| `n_splits` | Tie straddled? |
|---|---|
| 3, 4, 5, 6, 7, 8 | No |
| 10 | **Yes** |

**This is the key empirical justification for not trusting `TimeSeriesSplit` by default:** it happens to avoid splitting a tied group at `n_splits=5` (the proposed value) on THIS dataset, but demonstrably does not at `n_splits=10` — a coincidence of where this dataset's ties happen to fall, not a property `TimeSeriesSplit` guarantees for any `n_splits`. Relying on it without checking would have been exactly the kind of unverified assumption this project has avoided at every prior stage.

**The deterministic alternative actually used:** `build_expanding_window_folds()` (`src/hyperparameter_tuning.py`) computes the same expanding-window boundaries, then calls `snap_boundary_forward()` to advance any boundary that would split a tied-timestamp group — always forward (into train, out of validation), never backward, so a boundary adjustment can only make a fold's training data larger, never pull a future row into it.

**Proposed `n_splits=5`, measured on the complete 438,125-row `train` partition:**

| Fold | Train rows | Val rows | Val date range (raw `TransactionDT`) | Val positives | Val prevalence | All checks passed |
|---|---|---|---|---|---|---|
| 0 | 73,020 | 73,020 | 1,624,994 – 3,007,352 | 1,878 | 2.57% | ✅ |
| 1 | 146,040 | 73,020 | 3,007,361 – 5,134,060 | 2,912 | 3.99% | ✅ |
| 2 | 219,060 | 73,020 | 5,134,603 – 7,210,402 | 3,036 | 4.16% | ✅ |
| 3 | 292,080 | 73,020 | 7,210,556 – 9,064,152 | 2,537 | 3.47% | ✅ |
| 4 | 365,100 | 73,025 | 9,064,164 – 11,093,269 | 3,041 | 4.16% | ✅ |

Every fold passed all three requirements: chronology (`train_dt.max() < val_dt.min()`), no tie-straddle, and sufficient validation positives (floor: 20; actual minimum: 1,878 — comfortably above). Fraud prevalence varies meaningfully across folds (2.57%–4.16%), itself worth noting as a real property of this data, not a modeling artifact — consistent with the pilot-vs-full-train prevalence difference already documented in Stage 7.

**No embargo/gap between a fold's train and validation chunks.** Stated explicitly, per instruction not to choose one arbitrarily: no label-maturity delay (e.g. "a fraud label isn't confirmed for N days after the transaction") was established in this project's Stage 2/3 EDA. Inventing a specific gap size without evidence would itself be an arbitrary choice, not a more careful one. If a real label-maturity assumption is confirmed later, an embargo should be added then, with its size traceable to that evidence — not before.

**The approved Stage 4 split is not replaced.** This CV scheme operates entirely *inside* the `train` partition that Stage 4 already carved out; `validation` and `final_test` remain exactly as approved.

## 2. Preventing preprocessing leakage across folds

`fit_transform_fold()` fits the APPROVED, UNMODIFIED `build_preprocessing_pipeline` (`src/preprocessing.py`, untouched) separately on each fold's own training rows — including re-deciding, per fold, which numeric columns cross the missingness-indicator threshold (`select_indicator_columns`, also untouched). This matters because that decision is itself a statistic: using the Stage-5-approved indicator split (computed once, over the *entire* `train` partition) inside an early CV fold would let that fold "see" the missingness pattern of rows belonging to *later* folds — a real, if subtle, leakage path. A dedicated test (`test_fit_transform_fold_indicator_split_can_differ_between_folds`) constructs a column whose missingness is concentrated in the later half of synthetic data and confirms the early and late folds' indicator splits genuinely differ, proving the split is recomputed per fold rather than reused.

The raw numeric/categorical **column set** itself (which raw columns are eligible at all) stays fixed from the approved Stage 4/5 register — that is a structural classification, not a fitted statistic, and correctly does not need to vary per fold.

## 3. Random Forest search space

| Hyperparameter | Candidate values | Justification |
|---|---|---|
| `n_estimators` | 50, 100, 150 | Stage 7 used 100 fixed; this brackets it. More trees reduce variance with diminishing returns past ~100 for this problem's scale. |
| `max_depth` | 6, 8, 10, 12, 15 | Stage 7's `max_depth=10` was explicitly a **bounded, not-yet-empirically-justified** choice (stated in Stage 7's own docstring) — this is where it actually gets tested, on both sides. |
| `min_samples_leaf` | 5, 10, 20, 40 | Stage 7 used 20. Controls overfitting to noise on a ~3.5%-positive problem; too small risks fitting individual minority-class points, too large risks washing out real fraud patterns. |
| `max_features` | `"sqrt"`, `"log2"`, `None` | Not varied in Stage 7 at all (sklearn's default was implicit). A real, previously-unexamined knob trading split quality against fit time. |

Sampled via sklearn's own `ParameterSampler` (`sample_rf_candidates()`) — the same mechanism `RandomizedSearchCV` uses internally, with a fixed `random_state` for determinism. **Proposed budget: 20 candidates × 5 folds = 100 fits** — modest and explicit, not yet executed.

## 4. Feasibility findings (measured, not assumed)

A bounded probe fit the Stage 7 reference RF config (`n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42`) across 3 expanding-window folds of the existing 15,000-row chronological pilot (reusing an already-approved-pattern subset, no new approval needed):

| Fold | Train rows | Val rows | Preprocessing (s) | Fit (s) | Total (s) |
|---|---|---|---|---|---|
| 0 | 3,750 | 3,750 | 0.194 | 0.151 | 0.345 |
| 1 | 7,500 | 3,750 | 0.311 | 0.158 | 0.469 |
| 2 | 11,250 | 3,750 | 0.385 | 0.238 | 0.623 |

Average: **0.479s/fold.** Structural extrapolation for the proposed 20×5=100-fit search: **47.9s naive, 71.9s with a 1.5× safety margin.**

**Explicit lower-bound caveat:** this estimate is built from per-fold costs measured on a 15,000-row bounded subset. The real proposed search's folds would run on up to the complete 438,125-row `train` partition (its final expanding-window fold alone has 365,100 training rows — ~32× this probe's largest fold). Fit and preprocessing time both grow with row count (confirmed directly by this probe's own fold-0→fold-2 trend, 0.345s→0.623s), so **the real search would cost substantially more than 71.9s** — this number is a feasibility floor, not a runtime promise, exactly as instructed. A full-scale timing estimate would need either a probe run at a larger bounded scale (e.g. reusing Stage 7's 60,000-row GBM-feasibility-probe size) or accepting the real search's actual measured cost once approved.

**Resource measurements for this round's probe:** peak memory 5.50GB, wall time 7.43s, single process (no parallelism multiplier risk).

## 5. Tracking-registry extension

`src/experiment_tracking.py`'s schema gained three backward-compatible, additive-only fields: `search_id`, `candidate_index`, `fold_index` — `None` for every pre-existing Stage 7/8 record (verified: all 27 prior tests still pass unchanged). This lets a future real search's many trials be reconstructed exactly: one record per (candidate × fold) fit, linked by a shared `search_id`, never a fabricated "fold-level" number backed only by an aggregate CV mean. `scripts/verify_stage9_tracker_end_to_end.py` confirmed, against a temporary registry: a 2-fold successful candidate, a candidate failing mid-fold (exception correctly logged as `status="failed"`, the fold it never reached correctly has no record at all — not a fabricated placeholder), a refused duplicate run ID, and a verifiable evidence-checksum round trip. The real probe's 4 genuine records (3 fold fits + 1 feasibility decision) are logged in `docs/experiment_registry.jsonl` under `search_id="stage9-rf-cv-design-feasibility"`.

## 6. Scientific review

- **Why chronological CV differs from the ordinary k-fold CV in your Notion `INTERMEDIATE ML` / `Introduction to Multiple Linear Regression` material:** those pages' k-fold CV treats every row as exchangeable — shuffled into folds, assuming no fold's rows depend on another's order. This project's entire premise since Stage 4 is the opposite: a transaction's valid "history" features depend on what happened strictly *before* it. Plain k-fold CV would let a fold trained on rows from later in time evaluate on rows from earlier in time — exactly the kind of future-leaks-into-past error Stage 4's chronological split, and Stage 6's point-in-time entity-history tests, were built to prevent. Chronological CV is that same discipline, applied recursively *inside* `train`.
- **How repeated CV search introduces selection bias:** trying many hyperparameter candidates and keeping the best CV score is a multiple-comparisons problem — that best score is optimistic by construction, because it was the winner of many looks, not a single unbiased estimate (the authoritative workflow's own Stage 9 scientific-validation requirement). This compounds the selection-bias risk Stage 8 already made auditable for Stage 7's repeated `validation` consultations — Stage 9 adds its own, separate source of the same risk, entirely *within* `train`'s inner CV.
- **Why `validation` is not used to select the winner:** it has already been consulted 15 times across Stage 7 (6 of those explicitly influencing a decision, per the experiment registry). Using it again to *choose* a hyperparameter configuration would be a third layer of the same risk; the design here keeps `validation` completely unread until a single, final, disclosed, non-selecting comparison (deferred, pending approval of the actual search).

## 7. Limitations

- The §4 feasibility estimate (71.9s) was explicitly labeled a LOWER BOUND from a small, bounded subset, not a full-scale measurement — and §8's real full-scale five-fold run confirms it badly understated the real cost (by roughly 15-30x), exactly the risk that label was meant to flag. §8's revised estimate (18.6-37.1 minutes) supersedes it for planning purposes.
- No embargo/gap is applied, for the explicit reason given in §1 — this could change if a label-maturity assumption is later established.
- `TimeSeriesSplit`'s safety was checked only for the specific `n_splits` values tested (3–8, 10); a different `n_splits` not tested here would need its own check before use, via `assess_timeseries_split_compatibility()`.
- No real hyperparameter search has been run — every number in §3/§4 above is a proposal or an extrapolation, not a result.

## 8. Real, full-scale five-fold feasibility experiment (second approval round)

**Approved scope: ONE fixed Random Forest configuration (the Stage 7 reference, unchanged) executed across all 5 real, full-scale expanding-window folds — a resource-feasibility measurement, not a hyperparameter search.** Each fold ran as its own process invocation (`scripts/run_stage9_five_fold_feasibility.py --fold_index N`), under its own `/usr/bin/time -l` measurement and the harness's 600-second-per-fold timeout, sequentially, stopping-on-failure rather than continuing automatically.

**One real failure occurred and was handled exactly as designed, not glossed over:** fold 0's first attempt crashed with `KeyError: 'TransactionDT'` — a genuine bug in the new driver script (it tried to re-merge a column, `TransactionDT`, that turns out to already be a register-`include`-status column and was therefore already present in the loaded frame). `track_run()` correctly logged this as `status="failed"` with the exact exception before the process exited; the fold it never reached was not fabricated. The bug was fixed (removed the redundant merge) and fold 0 was re-run successfully. Both the failed attempt and the 5 successful runs are preserved in `docs/experiment_registry.jsonl` under `search_id="stage9-five-fold-full-scale-feasibility"` — nothing was deleted or silently corrected away.

**Results, all 5 folds, fixed config (`n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42`), evaluated on each fold's OWN held-out rows (never the external `validation` partition):**

| Fold | Train rows | Val rows | Preproc (s) | Fit (s) | Predict (s) | Total (s) | Peak RSS (GB) | Features | Train prev. | Val prev. | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73,020 | 73,020 | 2.994 | 1.662 | 0.203 | 4.859 | 6.16 | 812 | 2.684% | 2.572% | 0.42833 | 0.86513 |
| 1 | 146,040 | 73,020 | 5.408 | 3.599 | 0.214 | 9.221 | 8.24 | 813 | 2.628% | 3.988% | 0.48054 | 0.85224 |
| 2 | 219,060 | 73,020 | 7.810 | 6.473 | 0.220 | 14.502 | 10.77 | 813 | 3.081% | 4.158% | 0.51400 | 0.87677 |
| 3 | 292,080 | 73,020 | 10.147 | 9.149 | 0.211 | 19.507 | 13.47 | 813 | 3.350% | 3.474% | 0.41720 | 0.84225 |
| 4 | 365,100 | 73,025 | 12.502 | 13.374 | 0.212 | 26.088 | 16.24 | 813 | 3.375% | 4.164% | 0.53229 | 0.87631 |

**Total measured cost, all 5 folds: 74.177s of actual compute (sum of preprocessing+fit+predict); 102.9s of wall-clock time including per-invocation process/import overhead (~5.6-5.9s per invocation, an artifact of deliberately running each fold as a separate process for clean memory measurement, not a cost a real single-process search would repeat 100 times).**

**Feature-count difference confirmed and explained, exactly as predicted:** fold 0 (73,020 training rows, the smallest) fits 812 columns; every larger fold fits 813. This is the identical mechanism already reconciled in Stage 7 (the 790-vs-813 pilot-vs-full-train discrepancy): `OneHotEncoder` observed one fewer distinct category in fold 0's smaller training set. This is correct train-only-fit behavior, not a bug — each fold's preprocessing is refit from scratch, never reused from a global fit (confirmed directly: no code path in `fit_transform_fold` persists a fitted transformer across folds).

**Peak memory scales with fold size, as expected, topping out at 16.24GB for fold 4 (the largest, 365,100 training rows — 83% of the complete 438,125-row `train` partition) — consistent with, and slightly below, Stage 5/7's historical ~16.2-16.6GB peak for fitting this same pipeline on the complete partition.** Since folds ran sequentially in separate processes, the experiment's peak memory exposure at any instant was bounded by the single largest fold, never a sum across folds. No timeout, no memory pressure, no leakage-safeguard violation occurred on any successful fold.

### Revised, defensible resource estimate for the proposed 20-candidate × 5-fold search

Two scenarios, both using these REAL full-scale per-fold measurements (not the earlier bounded-probe lower bound, which is now superseded):

| Scenario | Basis | Raw estimate | With 1.5× safety margin |
|---|---|---|---|
| **Naive** (preprocessing refit for every candidate × fold — sklearn's `RandomizedSearchCV(Pipeline(...))` default behavior, no explicit caching) | 20 × 74.177s | 1,483.5s (≈24.7 min) | **2,225.3s (≈37.1 min)** |
| **Optimized** (preprocessing fit ONCE per fold, cached; only the RF step refit per candidate — requires explicit `memory=` caching or an equivalent manual cache, not yet implemented) | 38.861s (preprocessing, once×5 folds) + 20×35.317s (RF fit+predict) | 745.2s (≈12.4 min) | **1,117.8s (≈18.6 min)** |

**Uncertainty, stated explicitly:** both scenarios assume RF fit/predict time scales linearly with the search's own row counts the way these 5 folds did (a reasonable but unverified extrapolation across 20 *different* hyperparameter configurations — some candidates in the proposed search space, e.g. `n_estimators=150, max_depth=15`, will cost measurably more per fit than this round's fixed `n_estimators=100, max_depth=10` reference; others, e.g. `n_estimators=50`, less). Neither scenario includes the one-time cost of implementing and testing the caching optimization itself.

**Proposed hard total search-time budget: 3,600 seconds (60 minutes),** comfortably above the naive scenario's margin-adjusted estimate (37.1 min), providing headroom for both scenario uncertainty and the slower, more-expensive candidates in the declared search space.

**Proposed recovery strategy:** log every (candidate × fold) trial to `docs/experiment_registry.jsonl` via `track_run()` immediately on completion (append-only, crash-safe — already demonstrated working correctly by this round's own real fold-0 failure). If the search is interrupted (timeout, crash, or manual stop), resume by querying the registry for already-completed `(search_id, candidate_index, fold_index)` combinations and skipping them — never redoing completed work, never fabricating a result for a trial that didn't finish. This is the same pattern already proven in §4/§7 and re-confirmed by the real failure-then-recovery sequence in this section.

## 9. Approved 20-candidate hyperparameter search — real results (third approval round)

**Approved scope: ONE controlled RandomizedSearchCV-style search — 20 candidates × 5 chronological folds = 100 planned fits, selecting by mean fold Average Precision on `train`-internal folds only.** Candidates were declared and written to disk (`docs/_stage9_search_declaration_raw.json`) BEFORE any fit ran. The Stage 7 reference configuration (`n_estimators=100, max_depth=10, min_samples_leaf=20, max_features="sqrt"`) was **not** among the 20 sampled candidates (checked programmatically, not assumed).

### Outcome: the hard 3,600-second budget was reached and the search stopped safely, exactly as designed

| | Count |
|---|---|
| Candidates fully completed (all 5 folds) | **18** (indices 0–17) |
| Candidates incomplete | **2** (index 18: 4 of 5 folds; index 19: never started) |
| Individual fits completed successfully | **94** |
| Individual fits that failed (exception) | **0** |
| Individual fits not attempted (budget reached) | **6** |
| Total wall-clock time | **3,602.8s** (stopped just past the 3,600s mark — the fit already in progress when the deadline was crossed was allowed to finish, never hard-killed, per the stated design) |

**No fit-level exception occurred anywhere in the search** — every incomplete trial is incomplete purely because the budget was reached, not because anything crashed. This is a cleaner outcome than either feasibility round anticipated (which had each seen at least one real failure).

**Why the budget bound the search, not a bug:** a few candidates proved dramatically more expensive than the Stage 7 reference config used for feasibility planning. Candidate 13 (`n_estimators=100, max_depth=12, min_samples_leaf=40, max_features=None`) took **379.57s for its fold-4 fit alone**; candidate 7 (`max_features=None`, 150 estimators) took 319.99s. `max_features=None` (considering every feature at every split, instead of a `sqrt`/`log2` subset) was consistently the dominant cost driver — confirmed directly: all 5 `max_features=None` candidates (6, 7, 12, 13, 18) rank among the slowest fits in the log. This is exactly the uncertainty §8's revised estimate flagged ("some candidates... will cost measurably more... others... less") — realized in practice, not merely theoretical.

### Candidate ranking (mean fold AP, among the 18 candidates with all 5 folds complete)

| Rank | Candidate | Config | Mean AP | Std AP |
|---|---|---|---|---|
| 0 | 4 | `n_estimators=100, max_depth=15, min_samples_leaf=5, max_features=sqrt` | **0.51236** | 0.04434 |
| 1 | 8 | `n_estimators=150, max_depth=12, min_samples_leaf=10, max_features=sqrt` | 0.49451 | 0.04438 |
| 2 | 13 | `n_estimators=100, max_depth=12, min_samples_leaf=40, max_features=None` | 0.48410 | 0.05094 |
| 3 | 2 | `n_estimators=50, max_depth=15, min_samples_leaf=40, max_features=sqrt` | 0.47993 | 0.05096 |
| 4 | 9 | `n_estimators=100, max_depth=12, min_samples_leaf=40, max_features=sqrt` | 0.47728 | 0.04962 |
| ⋮ | | (13 more completed candidates, 0.409–0.475) | | |
| — | 18 | `n_estimators=50, max_depth=10, min_samples_leaf=5, max_features=None` | **incomplete** (4/5 folds) — excluded from ranking | — |
| — | 19 | *(never started — no record exists, nothing fabricated)* | **incomplete** (0/5 folds) — excluded from ranking | — |

Full per-candidate, per-fold detail (all 94 real fits) is in `docs/experiment_registry.jsonl` under `search_id="stage9-rf-randomized-search-20x5"`, and in `docs/_stage9_hyperparameter_search_raw.json`.

### Fold fraud prevalence and temporal variation (same 5 folds as §8, confirmed identical)

| Fold | Val date range | Val fraud prevalence |
|---|---|---|
| 0 | 1,624,994–3,007,352 | 2.572% |
| 1 | 3,007,361–5,134,060 | 3.988% |
| 2 | 5,134,603–7,210,402 | 4.158% |
| 3 | 7,210,556–9,064,152 | 3.474% |
| 4 | 9,064,164–11,093,269 | 4.164% |

Every candidate's fold-level AP tracks this same prevalence pattern (lowest on fold 0, highest on fold 2 or 4) regardless of hyperparameters — e.g. candidate 4's per-fold APs were approximately 0.47/0.52/0.55/0.45/0.57 across folds 0–4, the same low-fold-0/high-fold-2-and-4 shape every other candidate shows. **This means a meaningful share of the variance in reported AP is which chronological window a fold happens to cover, not which hyperparameters were used** — directly motivating why this report ranks by *mean* fold AP rather than any single fold's score.

### Feature-representation differences (confirmed, same mechanism as every prior round)

Feature counts matched §8's five-fold run exactly across every candidate (812 for fold 0, 813 for folds 1–4) — expected, since the fold definitions and the 15,000-fewer-row fold 0 are identical regardless of which RF hyperparameters are being fit; `OneHotEncoder`'s category count depends on the fold's row set, never on the model's hyperparameters.

### Search-selection optimism, stated plainly

The gap between rank 0 (0.51236) and rank 1 (0.49451) — 0.0178 — is **smaller than a single candidate's own fold-to-fold standard deviation (~0.044–0.051)**. This is the multiple-comparisons risk named in §6, now visible in real numbers: among the top several candidates, the ranking is not sharply distinguishable from fold-to-fold noise alone. The selected "best" configuration should be read as "among the strongest of 18 tried," not as a precisely-identified optimum.

### Explicit non-claims

- **The CV mean AP (0.51236 for the top candidate) is NOT comparable to Stage 7's external-`validation` AP (0.41101).** They are measured on different populations (chronological `train`-internal folds vs. the true held-out `validation` partition) with different prevalence — exactly the same non-comparability already noted for the §8 feasibility folds. The apparent "improvement" is not evidence this configuration would outperform the Stage 7 benchmark on `validation` or `final_test`.
- **No claim of global optimality.** This search tried 18 complete, randomly-sampled configurations out of a much larger space; a configuration outside the declared `RF_SEARCH_SPACE` (§3) was never considered, and 2 declared candidates were never evaluated.
- **No external-validation or final-test performance is claimed anywhere in this round** — `validation`/`final_test` were not accessed (confirmed: `validation_consulted=False` on every logged record, and the loading code path reads `train` exclusively).
- **The Stage 7 full-training RF benchmark's evidence is unchanged and not overwritten** — `docs/_stage7_rf_full_train_raw.json` and its registry record are untouched; this search's results are additive, new records under a new `search_id`.

### Connecting to your Notion Intermediate ML material

Your `RandomizedSearchCV`/cross-validation notes describe exactly this mechanism — sample N random hyperparameter combinations, score each by CV, pick the best mean score — but assume ordinary k-fold CV over exchangeable rows. This search is that same algorithm with the CV *scheme* swapped for the time-respecting one Stage 9 built (§1), and with an explicit multiple-comparisons caveat your journal's material doesn't carry (since plain k-fold tutorials rarely dwell on "the winner's margin is smaller than the noise" the way this report's ranking table does above).

## 10. Independent scientific audit and provisional selection (fourth round)

**Read-only audit, no model retrained.** `scripts/audit_stage9_search_results.py` independently recomputed every claim in §9 directly from `docs/experiment_registry.jsonl` (the authoritative record, per Stage 8 requirement 1) rather than trusting the search's own printed console log — the same discipline Stage 7 applied when independently re-verifying the RF pilot.

**1. Candidate/rank reconciliation — CONFIRMED.** Exactly one candidate's independently-recomputed mean AP matches 0.51236: **candidate_index=4**, rank 0. Exact hyperparameters, as actually passed to `RandomForestClassifier` (confirmed from the registry's own `hyperparameters` field, not reconstructed): `{n_estimators=100, max_depth=15, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42, class_weight=None}`.

**2. Completeness/consistency — CONFIRMED.** All 18 eligible candidates have exactly 5 `status="completed"` fold records, each with matching hyperparameters across that candidate's own 5 folds (verified programmatically, not by inspection), and each with a non-empty `evidence_paths`. **Every fold's `(train_fold, val_fold)` row fingerprint is byte-identical across all 18-19 candidates that reached it** — direct, cryptographic-hash-based proof that the 5 chronological folds never drifted between candidates; every candidate really was scored on the exact same 5 train/validation splits.

**3. Fit totals — CONFIRMED.** Recomputed directly from the registry: 94 completed, 0 failed, 6 unattempted (out of 100 planned) — matches §9's reported figures exactly.

**4. Paired fold-by-fold comparison, rank 0 (candidate 4) vs. rank 1 (candidate 8) — a finding that CORRECTS §9's earlier framing:**

| Fold | Candidate 4 AP | Candidate 8 AP | Paired difference |
|---|---|---|---|
| 0 | 0.47255 | 0.45130 | +0.02125 |
| 1 | 0.51617 | 0.49993 | +0.01624 |
| 2 | 0.55416 | 0.53682 | +0.01734 |
| 3 | 0.45258 | 0.43678 | +0.01580 |
| 4 | 0.56632 | 0.54771 | +0.01861 |

**Candidate 4 beat candidate 8 on all 5 of 5 folds**, with a remarkably stable margin (mean +0.01785, standard deviation of the *paired differences* only ≈0.00196). §9 compared the gap between the two candidates' means (0.0178) against each candidate's own *unpaired* across-fold standard deviation (~0.044–0.051) and concluded the ranking "is not sharply distinguishable from fold-to-fold noise." **That comparison used the wrong baseline.** Because both candidates were scored on the identical 5 folds, the correct comparison is *paired*: fold-to-fold swings driven by shared factors (each fold's own fraud prevalence and difficulty, §9's temporal-variation finding) affect both candidates almost equally and cancel out in the paired difference — which is why the paired-difference spread (≈0.002) is more than 20x smaller than the unpaired spread (≈0.044–0.051) used for the earlier, overly cautious framing. **Correction: candidate 4's advantage over candidate 8 is real and consistent across every fold tried, not noise-level** — though this says nothing about candidates 18/19, which were never evaluated at all (see point 5).

**5. Scope of the winner, stated precisely:** Candidate 4 is the highest-mean-AP configuration **among the 18 candidates that completed all 5 folds within the approved search's budget.** It is explicitly **not** established as the winner among all 20 originally-declared candidates — candidates 18 and 19 were never fully evaluated (candidate 18 reached 4 of 5 folds; candidate 19, zero), their true standing relative to candidate 4 is genuinely unknown, and no recovery run was approved to resolve that.

**6. `validation`/`final_test` access — CONFIRMED never touched.** Every one of the 95 records for this search (94 fits + 1 summary) has `validation_consulted=False`; independently cross-checked against the module's own code paths (`src/hyperparameter_tuning.py`/`scripts/run_stage9_hyperparameter_search.py` read `train` exclusively).

**7. Provisional tuned candidate, conditional on this audit (which passed in full):**

```
RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1, random_state=42, class_weight=None,
)
```
**Selection rule:** highest mean Average Precision across 5 train-internal, time-respecting, expanding-window chronological folds, among candidates with all 5 folds successfully completed within the approved search's resource budget.

**This is PROVISIONAL, not final.** Stage 9 is **not** marked complete by this audit. Carried-forward limitations, restated for the record:
- **Incomplete-search limitation:** 2 of 20 declared candidates (18, 19) were never evaluated; this ranking covers only the 18 that were.
- **Temporal-variation limitation:** fold-level AP is driven substantially by which chronological window a fold covers (§9), not purely by hyperparameters — every candidate shows the same low-fold-0/high-fold-2-and-4 pattern.
- **Validation-selection caveat:** this candidate was selected using `train`-internal CV only; it has never been scored against the external `validation` partition, and that score (whenever it happens) would be the FIRST and only legitimate use of `validation` for this candidate — not a confirmation of something already known, and not interchangeable with the CV mean AP reported here.

## 11. Interview-defense questions

1. "Your `TimeSeriesSplit` check passed for `n_splits=5` but failed for `n_splits=10` — why would adding more folds break something that worked with fewer?"
2. "Why do you refit the missingness-indicator selection separately for every CV fold instead of reusing the one Stage 5 already computed over all of `train`?"
3. "Your first feasibility estimate was 71.9 seconds from a bounded probe, but the real five-fold run's own numbers imply 20-37 minutes for the full search — which numbers should someone trust, and why did the first one undershoot so badly?"
4. "Fold 0 failed with a `KeyError` on your first real attempt — walk me through what happened to that failed run's record, and why didn't it block folds 1-4 from running?"
5. "Your top two candidates' mean AP differ by 0.018, but each candidate's own fold-to-fold standard deviation is around 0.045 — does that mean you actually found a better configuration, or not?"
6. "Two of your 20 declared candidates never got evaluated at all — how does that affect whether you can trust the search's ranking?"
7. "You first said the top-two gap wasn't distinguishable from noise, then a paired comparison said the opposite — which is right, and what was wrong with the first analysis?"
