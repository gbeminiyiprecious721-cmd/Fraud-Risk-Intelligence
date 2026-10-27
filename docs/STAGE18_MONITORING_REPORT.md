# Stage 18 — Local Monitoring Design and Implementation

Status: **implementation complete, awaiting scientific and engineering review.**
A separate, concise runbook lives at
[`docs/STAGE18_MONITORING_RUNBOOK.md`](STAGE18_MONITORING_RUNBOOK.md).

Scope reminder: this is local, free/local portfolio monitoring — batch,
not live telemetry; no paid service, no cloud infrastructure, no
commercial dashboard. It demonstrates the *mechanism* of drift and
failure detection, not a production observability stack.

---

## 1. Monitoring components implemented

| File | Purpose |
|---|---|
| `src/monitoring.py` | Input-distribution monitor (PSI), prediction-score-distribution monitor (PSI), operational log monitor — all pure functions, no change to `src/api.py`. |
| `monitoring/config.json` | Versioned, hand-authored alert-rule configuration (7 rules — see §4). |
| `monitoring/reference_distributions.json` | Generated reference decile distributions for `TransactionAmt`/`C1`/`dist1`, built from the real `train` partition. |
| `monitoring/reference_score_distribution.json` | Generated reference `fraud_probability` decile distribution, built by scoring a real `train` sample with the trusted Stage 13 artifact (no refit). |
| `scripts/run_stage18_monitoring_demo.py` | The real orchestration script that built the references above and ran the control-vs-drift and operational demonstrations in §5. |
| `tests/test_monitoring.py` | 30 unit tests (see §9). |
| `docs/_stage18_monitoring_demo_raw.json` | Full structured, real evidence from the demonstration run (gitignored, like every other stage's raw-evidence file). |
| `docs/STAGE18_MONITORING_REPORT.md` | This report. |
| `docs/STAGE18_MONITORING_RUNBOOK.md` | The separate response runbook. |

**Not touched:** `src/api.py`, `src/production_pipeline.py`, the frozen artifact, `deploy/start_release.sh`/`stop_release.sh`/`health_check.sh` (reused exactly as-is for the operational control demonstration, not modified or duplicated into a new deployment pathway).

---

## 2. Observability audit — what was actually available before any monitor was designed

Per the explicit instruction to audit first, `src/api.py`'s `_log_event` calls and the real Stage 17 log files (`deploy/run/version_a.log`, `deploy/run/version_b.log`) were inspected directly (not assumed).

**Available in the existing structured JSON log lines:**
- `request_id`, `api_contract_version`, `status` (HTTP status code), `duration_seconds` — present on **every** predict-path log line, success or error.
- `artifact_version`, `artifact_sha256` — present **only on the success (200) path**.
- `error_category` — present **only on the error path**.
- A `startup` event line (`artifact_loaded`, `startup_seconds` or `error_category`).

**Confirmed NOT present (genuine gaps, not assumptions):**
- **No wall-clock timestamp anywhere.** `src/api.py`'s log handler formatter is `logging.Formatter("%(message)s")` only — no `%(asctime)s`. This means "an observation window" cannot be a calendar-time range from these logs; it can only be "the set of records in one bounded run/log file." Stated plainly here rather than worked around by inventing timestamps.
- **No `fraud_probability` or any raw feature value in any log line** — by original Stage 14 design (privacy-conscious logging), confirmed unchanged. Prediction-distribution monitoring therefore cannot be built from logs at all — it is built by scoring a fresh batch directly with the trusted artifact instead (§4 below), which is the only way to get real score data without changing the API.
- **No artifact identity on error responses** — `error_response()` never logs `artifact_version`/`artifact_sha256`. A run of pure failures could theoretically be served by an unapproved artifact and the identity-mismatch check would never see it. Named as a real, permanent limitation (§8), not fixed (fixing it would mean editing `src/api.py`, outside this stage's preference for a separate monitoring layer).

**Consequence for design:** per instruction, a separate local monitoring layer (`src/monitoring.py`) was built rather than editing the approved API. Operational monitoring treats each log *file* as one bounded window (not a calendar window), and prediction-distribution monitoring scores batches directly with the artifact rather than trying to recover scores from logs that were never designed to carry them.

---

## 3. Reference datasets and provenance

| Reference | Source | Real sample size | Provenance recorded |
|---|---|---|---|
| `TransactionAmt` distribution | `train` partition (`src.preprocessing.load_partition_df`, never `validation`/`final_test`) | 438,125 finite values, 0 missing | `population_fingerprint` (sha256 of partition/feature/n/sum/mean), bin edges, bin proportions |
| `C1` distribution | `train` partition | 438,125 finite values, 0 missing | same as above. Chosen because Stage 11 ranked `C1` among the top SHAP features — the single most consequential input-drift signal available given existing project evidence. |
| `dist1` distribution | `train` partition | 168,943 finite values, **269,182 missing** (61.44% — matches Stage 16's already-documented real missingness exactly) | same as above. Chosen deliberately to exercise missing-value handling against real data, not a synthetic fixture. |
| Prediction-score distribution | 1,000 real `train` rows (`random_state=10`), scored once by the trusted Stage 13 artifact (sha256 `2fb0cbc4...`), no refit | 1,000 scores, mean 0.0382 | `artifact_sha256` recorded and checked on every later comparison — a mismatch refuses the comparison outright (see §4) |

All four references are real, generated by `scripts/run_stage18_monitoring_demo.py`, and persisted to `monitoring/reference_distributions.json` / `monitoring/reference_score_distribution.json` (both small, git-tracked, like Stage 13's manifest — not like the gitignored raw-evidence dumps).

---

## 4. Monitoring signals, calculations, and thresholds

Full detail for all 7 rules, including each one's threshold rationale and expected false-alert limitations, is in `monitoring/config.json` (git-tracked, versioned `1.0.0`) — summarized here:

| Rule | Signal | Threshold | Rationale (summary) |
|---|---|---|---|
| `input_drift_TransactionAmt` | PSI vs. train reference | 0.25 | Conventional PSI literature heuristic — a statistical-adequacy convention, explicitly **not** derived from Stage 1's cost scenarios and **not** claimed economically optimal. |
| `input_drift_C1` | PSI vs. train reference | 0.25 | Same convention; chosen for its Stage 11 SHAP importance. |
| `input_drift_dist1` | PSI vs. train reference (finite values only) | 0.25 | Same convention; missingness tracked as a separate count, never imputed or folded into PSI. |
| `prediction_score_drift` | PSI of `fraud_probability` vs. reference scores | 0.25 | Same PSI convention, applied to scores; refuses comparison on artifact-checksum mismatch. |
| `operational_server_error` | any 5xx (count, not rate) | **0** (any) | With this API's low expected local traffic, a rate threshold would hide the first real outage — deliberately more sensitive than a production-scale rule. |
| `operational_artifact_identity_mismatch` | unapproved `artifact_sha256` seen on a successful response | any non-empty set | Serving from an unapproved artifact is always significant per Stage 13's permanent checksum-identity contract. |
| `operational_latency_p95` | p95 `duration_seconds`, min. sample 20 | 1.0s (~20x Stage 14's measured warm p95 of 0.0462s) | Deliberately loose so it fires only on a large real regression, not routine local jitter. |

**Explicit, per-instruction constraint honored:** Stage 1's cost scenarios and Stage 10's evaluation results are cited as *business/scientific context* in `monitoring/config.json`'s rationale fields, never as an automatic mathematical derivation of a drift threshold — no claim of an economically optimal threshold is made anywhere.

---

## 5. Genuine control-vs-drift alert evidence (real, not described)

All numbers below are from the real run captured in `docs/_stage18_monitoring_demo_raw.json`.

**Control batch** (500 real, independent `train` rows, `random_state=99` — a genuinely different sample from the reference, not a copy):

| Feature | PSI | Status |
|---|---|---|
| `TransactionAmt` | 0.023096 | `ok` — no alert |
| `C1` | 0.013835 | `ok` — no alert |
| `dist1` | 0.050392 | `ok` — no alert |
| prediction score | 0.041026 | `ok` — no alert |

**Deliberately shifted batch** (500 real `train` rows, `random_state=7`, with `TransactionAmt` multiplied by 20 — the one deliberate, documented defect):

- `input_drift::TransactionAmt` → **`drift_alert`**, PSI **7.421445** (>> the 0.25 threshold), 500-sample batch, 2 values landed outside the reference's historical range (captured in the batch's own extreme bin, not dropped).
- The real alert record emitted includes every required field: `alert_id` (`input_drift::TransactionAmt`), `signal`, `calculation`, `observed_statistic` (7.421445), `threshold` (0.25), `sample_size` (500), `reference_provenance` (partition + population fingerprint), `batch_provenance`, `severity` (`moderate`), `recommended_action`, and a real `detected_at` wall-clock timestamp (`2026-10-27T08:37:27.489+00:00`, generated live by the monitor — not reconstructed from the un-timestamped historical API logs, a different and narrower claim).
- The same shifted batch, scored by the real artifact, also produced a **prediction-score drift alert** (PSI 0.696394) — reported because it genuinely occurred, not forced; the config's own documented limitation (§4, `prediction_score_drift` row) says this cross-trigger is expected and should be read together with the input-drift alert, not independently.

This is the actual mechanism firing on actual data — not a manually-inserted alert record. Verified by: the real control batch (same distribution) produced *no* alert while the real shifted batch did, with both run through the identical code path.

---

## 6. Operational failure monitoring evidence (real)

**Control window** — a REAL, fresh local deployment using Stage 17's own, unmodified `deploy/start_release.sh`/`stop_release.sh` (no new deployment pathway created): 25 real `POST /v1/predict` requests + 3 `GET /health` requests against a freshly started Version A, then a clean stop.

```
n_records_with_status_and_duration: 25, n_2xx: 25, n_5xx: 0, error_rate_5xx: 0.0
latency_median_seconds: 0.0414, latency_p95_seconds: 0.0613 (n=25)
distinct_artifact_sha256_seen: ["2fb0cbc4..."] -- matches the approved artifact exactly
```
→ **0 alerts fired.** Correct: a healthy window should not alert.

**Failure window** — the REAL, already-approved Stage 17 evidence, parsed directly from `deploy/run/version_b.log` (not regenerated, not a synthetic fixture — this file was still present on disk from the Stage 17 session; the script falls back to a verbatim reproduction of that exact, already-reported content only if the file is ever absent):

```
n_records_with_status_and_duration: 1, n_5xx: 1, error_rate_5xx: 1.0
latency_insufficient_sample: true (n=1 — correctly flagged as too small for a latency claim)
```
→ **1 alert fired: `operational_server_error`**, `observed_statistic=1.0` (100% of the 1 observed request), `threshold=0.0` (any), `sample_size=1`, `severity=critical`, recommending the exact check-health-then-roll-back sequence from `docs/STAGE17_ROLLBACK_RUNBOOK.md`. The artifact-identity-mismatch rule correctly did **not** fire here, because — per the audited gap in §2 — error responses never log artifact identity at all, so there was nothing to compare; this is reported honestly as a blind spot, not hidden.

---

## 7. Privacy and data-retention safeguards

- **No new telemetry pathway**: the operational monitor only reads log files the existing, approved API already writes (or, for the input/score monitors, scores *already-approved* `train`-derived or synthetic batches directly with the artifact — never real customer data, since none exists).
- **Data minimisation**: reference distributions store only bin edges, proportions, counts, and a population fingerprint — never raw row-level values. Alert records store only aggregate statistics (PSI, counts, means), never individual predictions or feature values.
- **Bounded retention**: `monitoring/reference_distributions.json` and `monitoring/reference_score_distribution.json` are the only files this stage persists durably (git-tracked, small, like a manifest); the demonstration's raw evidence (`docs/_stage18_monitoring_demo_raw.json`) and the operational log files under `deploy/run/` are gitignored and ephemeral, exactly like Stage 17's own evidence.
- **Local-only storage**: nothing in this stage sends data anywhere — no external logging/telemetry/monitoring service of any kind.

---

## 8. Known limitations (honest, not hidden)

1. **No wall-clock timestamp exists in the current structured logs** (§2) — "observation window" in this stage means "one bounded log file/run," never a calendar-time range. A future stage that needs real time-windowed monitoring would need to add a timestamp to `src/api.py`'s logging — a change this stage deliberately did not make.
2. **Artifact identity is never logged on error responses** (§2, §6) — the identity-mismatch check has a real blind spot during any run consisting entirely of failures.
3. **`fraud_probability` remains an uncalibrated ranking signal** (Stage 10/14) — the prediction-score monitor tracks distribution *shape* only; it makes no accuracy, calibration, or decay claim, and no classification threshold is invented anywhere in this stage.
4. **No realized-performance (ground-truth) monitoring exists or can exist** — this project has no live transactions and no chargeback-label feed; "performance decay" in the sense of *actual accuracy* dropping over time cannot be measured locally and is not claimed to be.
5. **PSI cannot distinguish a malicious/erroneous data change from a genuine, legitimate population shift** (e.g. a real seasonal spending change) — stated per-rule in `monitoring/config.json`, not just here.
6. **The `dist1` drift rule only monitors finite values** — a change in the *missingness rate itself* (e.g. a new upstream source that always populates a previously-sparse column) would not be caught by this PSI calculation.
7. **Thresholds are statistical conventions, not economically optimal figures** — no operational evidence (false-alert cost, investigation cost) exists yet to derive a cost-optimal threshold; Stage 1/10 are cited as context, never as an automatic derivation.
8. **This is batch, pull-based monitoring, run on demand** — there is no scheduler, no paging, no dashboard. The runbook explicitly names the solo project operator as the only responsible party; there is no staffed on-call function.
9. **Only two failure classes have real demonstrated evidence**: a feature-level distribution shift (synthetic amplitude, real underlying data) and a 503-serving-failure (Stage 17's real evidence). Other operational failure modes (e.g. a slow memory leak, a dependency version mismatch) were not exercised.

---

## 9. Tests and resource usage

**Tests:** `tests/test_monitoring.py` — **30 tests, all passing**, covering: PSI correctness (identical distributions, a known shift, a zero-proportion bin), reference-distribution construction (normal sample, too-small sample, missing values, constant feature, non-finite values, fingerprint determinism), batch scoring (matching batch, empty batch, too-small batch, unusable reference, out-of-range values), end-to-end input-drift evaluation (control batch → no alert; deliberately shifted batch → real alert with every required field), prediction-score monitoring (no-drift, drift, provenance-mismatch refusal, empty batch), operational log parsing (Uvicorn plain-text lines correctly skipped, a real Stage 17 log file if present), operational summarization (error rate, latency, insufficient-sample flagging, artifact-mismatch detection, empty-list safety), operational alerting (fires on any server error, fires on artifact mismatch, stays silent on a clean window), and config-file structural validation.

**One real bug caught and fixed while writing these tests:** the original `score_batch_distribution` used a plain `np.histogram` against the reference's bin edges, which silently **drops** any batch value outside `[edges[0], edges[-1]]`. A batch shifted far enough to fall *entirely* outside the historical range was therefore reported as `malformed_input` instead of the extreme drift alert it actually represents — exactly backwards for a monitoring system. Fixed by replacing the binning with an open-ended `_bin_counts_open_ended()` helper (`np.digitize` against interior edges only), so out-of-range values land in the extreme bins instead of being discarded. Verified by `test_score_batch_distribution_out_of_range_values_are_counted` and by the real demonstration itself (§5: 2 of the shifted batch's 500 `TransactionAmt` values fell outside the reference range and were correctly counted, not dropped).

**Full project suite:** `pytest tests/ -m "not heavy and not data_contract_full"` → **334 passed, 6 deselected**, 6.50s (304 pre-existing + 30 new, zero regressions).

**Resource usage of the demonstration run:** dominated by the existing, already-measured costs of loading the `train` partition (Stage 5/16's own documented figures) and a single real artifact load/score — no new expensive computation was introduced. The operational control sub-demonstration reused Stage 17's own already-measured ~1.5s cold-start cost (unchanged). The full `run_stage18_monitoring_demo.py` script completed in well under a minute on this machine.

The original fitted artifact's checksum was reconfirmed unchanged before and after this stage: `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.

---

## 10. Three interview-defense questions

**Q1: Why Population Stability Index instead of a KS test or a simple mean/variance comparison?**
PSI was chosen because it is bin-based and symmetric, works on both continuous and count-like anonymised features (`C1`) without a parametric assumption, and is the most common, well-understood convention for this exact use case in industry practice — easy to defend and explain to a non-statistician reviewer. It was deliberately **not** computed from a single summary statistic like an imputer median (explicitly forbidden by this stage's own instruction) — every reference distribution here is built from the real, finite values of the actual `train` partition. Its main weakness, also stated in the report: PSI tells you *that* a distribution moved, never *why*, and it is sensitive to bin choice and small-sample noise, which is why a minimum sample size is enforced before any PSI number is trusted.

**Q2: Your control run showed 0% errors and the failure run showed 100% — isn't a 1-request sample meaningless?**
Yes, and the report says so explicitly (`latency_insufficient_sample: true`, and §8 names this directly) — a single-request window cannot support a *statistical* conclusion about a long-run error rate. But that is exactly why the `operational_server_error` rule is deliberately NOT rate-based: on a low-traffic local API, any observed 5xx is actionable on its own, regardless of how many total requests preceded it — waiting for a larger sample before alerting would mean missing the very first real outage. The rule is honest about trading statistical rigor for operational sensitivity, and says exactly that in `monitoring/config.json`'s own rationale field.

**Q3: This whole system has no real ground-truth fraud labels flowing back in. What does "monitoring" actually mean here?**
It means detecting *proxies* for decay — a shift in what's coming IN (feature distributions) and a shift in what the model is predicting OUT (score distribution) — plus basic operational health (is the service actually answering, correctly, from the approved artifact). It explicitly does **not** mean detecting *accuracy* decay, because that would require real-world fraud outcomes arriving later, which this project does not have and has never claimed to have. This is stated as a structural limitation (§8, item 4) rather than worked around with a fabricated or assumed label source — a real operating organisation would additionally need exactly that ground-truth feedback loop, which is also named explicitly as something this local demonstration does not provide.
