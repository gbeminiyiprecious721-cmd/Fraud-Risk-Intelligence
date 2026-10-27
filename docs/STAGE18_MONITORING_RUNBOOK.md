# Stage 18 — Local Monitoring Response Runbook

Companion to [`docs/STAGE18_MONITORING_REPORT.md`](STAGE18_MONITORING_REPORT.md) and
[`docs/STAGE17_ROLLBACK_RUNBOOK.md`](STAGE17_ROLLBACK_RUNBOOK.md). This is
operator-facing and deliberately short.

**Responsible party: the solo project operator (you).** This project has
**no staffed on-call function, no automated paging service, and no real
ground-truth fraud-label feedback loop.** Every response below is
something a single person does by hand, on demand — there is no
scheduler running these checks automatically.

---

## 0. Running the monitors

```bash
# Rebuild references + run the full control-vs-drift + operational demonstration:
.venv/bin/python3 scripts/run_stage18_monitoring_demo.py

# Or use the functions directly in src/monitoring.py against a real batch:
#   build_reference_distribution / score_batch_distribution / evaluate_input_drift
#   build_reference_score_distribution / evaluate_score_drift
#   parse_log_file / summarize_operational_records / evaluate_operational_health
```

All alert rules and their thresholds are in `monitoring/config.json`.

---

## 1. When an input-distribution or prediction-score alert fires

| Step | Action |
|---|---|
| **1. Observe and verify** | Re-read the alert record itself (`alert_id`, `observed_statistic`, `threshold`, `sample_size`, `reference_provenance`, `batch_provenance`, `detected_at`). Confirm the batch size was large enough (`sample_size` ≥ the rule's `required_sample_size` in `monitoring/config.json`) — a tiny batch can produce a noisy PSI. |
| **2. Investigate source/schema change** | Check whether the batch's feature definitions, units, or upstream source changed (e.g. a different currency, a new data provider). Cross-reference: did an `input_drift_*` alert AND `prediction_score_drift` both fire for the same batch? If only the score alert fired with no input alert, suspect the batch's *composition* (e.g. an unusually fraud-heavy real batch) rather than a data-quality problem. |
| **3. Check model/artifact provenance** | Run `GET /version` against the live API (or re-check `reference_provenance`/`batch_provenance` in the alert) — confirm the artifact actually serving requests still matches the approved `2fb0cbc4...` checksum (Stage 13/17). A `provenance_mismatch` status from `evaluate_score_drift` means someone compared scores from two different artifacts — treat this as a process error, not a drift finding, until re-run with matching provenance. |
| **4. Check API/deployment health** | `GET /health`, `GET /version` — is the service actually up and serving the approved artifact? If not, this is really an operational failure (§2 below), not (only) a drift finding. |
| **5. Review whether prior validated assumptions still apply** | Re-read the relevant Stage 1-12 assumption (e.g. Stage 1's cost scenarios, Stage 10's evaluation population) — a real, legitimate population shift (not a bug) may mean those assumptions no longer hold and deserve a fresh look, not a silent continuation. |
| **6. Escalate (hypothetical)** | In a real operating organisation, this is the point a drift finding would go to a model-governance process for a retrain/recalibration decision. **This project does not do that automatically, and never will without a separate, explicit approval** — no auto-retrain, no auto-replace, no auto-reject-transaction in response to any alert here. |

---

## 2. When an operational alert fires

| Alert | First action |
|---|---|
| `operational_server_error` (any 5xx observed) | Immediately `GET /health` and `GET /version`. If the release is unready, follow [`docs/STAGE17_ROLLBACK_RUNBOOK.md`](STAGE17_ROLLBACK_RUNBOOK.md) step by step — stop the bad release, redeploy the last known-good Version A, reconfirm health + a real prediction + checksum match. |
| `operational_artifact_identity_mismatch` | Verify which artifact is actually loaded (`GET /version`'s `artifact_sha256`) against `monitoring/config.json`'s `approved_artifact_sha256`. If it does not match, treat this as equivalent to a bad deployment — roll back per Stage 17's runbook. |
| `operational_latency_p95` | Check for resource contention on this machine (other processes, thermal throttling). This is **not** evidence of model correctness decay — do not treat a latency alert as a reason to question prediction quality. |

**A single transient 5xx during an intentional, operator-initiated redeploy (e.g. while deliberately testing a new release, as in Stage 17's own exercise) will always fire `operational_server_error`.** Step 1 ("observe and verify") exists specifically to make that distinction by hand — this monitor cannot tell "a real incident" apart from "an expected deployment transition" on its own.

---

## 3. What this runbook explicitly does NOT cover

- Real-time/streaming monitoring (everything here is batch, run on demand).
- Any action triggered automatically — every row above is a *recommended next step for a human*, never code that executes it.
- Ground-truth/accuracy-decay monitoring (no label feed exists — see the report §8, item 4).
- Paging, ticketing, or any multi-person workflow — there is exactly one operator.

If this project were ever operated by a real organisation instead of as a
portfolio demonstration, it would additionally need: a scheduler to run
these checks automatically, a paging/ticketing integration, a staffed
rotation, and a real feedback loop for ground-truth labels — none of
which exist here, and none of which this runbook pretends to provide.
