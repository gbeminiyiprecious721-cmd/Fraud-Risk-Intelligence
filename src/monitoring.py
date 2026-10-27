"""Stage 18 -- local, free/local batch monitoring for the frozen Stage 13
model and the Stage 14 API, built as a SEPARATE module (never a change to
`src/api.py`) per explicit instruction preferring a separate, local,
privacy-conscious batch-monitoring script over changes to approved API
code.

Three monitor families, all local, all batch (no live telemetry, no
external service):

1. **Input distribution monitoring** -- compares a batch's feature
   distribution against a reference distribution built from the
   approved `train` partition, using the Population Stability Index
   (PSI). Never attempted from imputer medians alone -- reference bins
   are built from the real underlying training data.
2. **Prediction (score) distribution monitoring** -- scores a batch
   with the existing, trusted Stage 13 artifact (no refit) and compares
   the resulting `fraud_probability` distribution against a reference
   score distribution built the same way, with an explicit provenance
   check (same artifact checksum) before any comparison is trusted.
3. **Operational monitoring** -- parses the EXISTING Stage 14/17
   structured JSON log lines (one real log event per line, interleaved
   with Uvicorn's own plain-text access-log lines, which are skipped)
   to compute request count, status-code breakdown, error rate,
   latency percentiles, and artifact-identity consistency.

**Audited, not assumed (Stage 18 Section 2):** the existing Stage 14
`_log_event` lines do NOT contain `fraud_probability`, raw feature
values, or a wall-clock timestamp (the logging handler's formatter is
`"%(message)s"` only -- see `src/api.py`). Error-path log lines do NOT
carry `artifact_version`/`artifact_sha256` (only the success path
does). These are real, structural gaps in the existing observability,
documented in `docs/STAGE18_MONITORING_REPORT.md` -- this module works
within them rather than silently assuming data that isn't there.

Every alert record this module produces carries its own wall-clock
`detected_at` (generated live, by the monitor, at evaluation time --
NOT reconstructed from the un-timestamped historical logs, which is a
different and narrower claim).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = ROOT / "monitoring"
MONITORING_CONFIG_PATH = MONITORING_DIR / "config.json"
REFERENCE_DISTRIBUTIONS_PATH = MONITORING_DIR / "reference_distributions.json"
REFERENCE_SCORE_DISTRIBUTION_PATH = MONITORING_DIR / "reference_score_distribution.json"

MONITORED_INPUT_FEATURES = ("TransactionAmt", "C1", "dist1")  # dist1 chosen deliberately for its
# real, legitimate ~61.44% missingness (Stage 16 finding) -- exercises missing-value handling
# with real data rather than a synthetic fixture alone.
N_DISTRIBUTION_BINS = 10
MIN_REFERENCE_SAMPLE_SIZE = 200  # ~20 expected observations/bin at 10 bins -- the conventional
# rule of thumb for stable quantile-bin proportions; a statistical adequacy threshold, not a
# cost-derived one.
MIN_BATCH_SAMPLE_SIZE = 50
PSI_ALERT_THRESHOLD = 0.25  # conventional literature heuristic (Population Stability Index):
# <0.1 no significant shift, 0.1-0.25 moderate, >0.25 significant -- a widely used statistical
# convention, NOT derived from Stage 1's cost scenarios and NOT claimed to be economically optimal.
MIN_OPERATIONAL_LATENCY_SAMPLE_SIZE = 20
LATENCY_P95_ALERT_SECONDS = 1.0  # generous relative to Stage 14's own measured p95 (~0.046s) --
# deliberately loose so this alert fires only on a real, large regression, not routine jitter.


class MonitoringInputError(ValueError):
    """Raised when an operation is given data that cannot be monitored
    at all (e.g. a feature that does not exist) -- distinct from a
    `malformed_input`/`insufficient_sample` STATUS, which is a normal,
    expected monitoring outcome, not a programming error."""


# ---------------------------------------------------------------------------
# Provenance / fingerprinting
# ---------------------------------------------------------------------------

def fingerprint_reference_population(values: pd.Series, partition: str, feature: str) -> str:
    """A cheap, deterministic fingerprint of a reference population --
    NOT a cryptographic guarantee, only enough to detect "this
    reference was rebuilt from different underlying data than it was
    before." Reuses only descriptive statistics, never raw values."""
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    payload = (
        f"partition={partition}|feature={feature}|n={len(values)}|"
        f"n_finite={len(finite)}|sum={finite.sum():.6f}|mean={finite.mean():.6f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Input distribution monitoring
# ---------------------------------------------------------------------------

def _clean_numeric(values: pd.Series) -> tuple[np.ndarray, int]:
    """Returns (finite_values, n_non_finite_or_missing). Never silently
    drops a non-finite value without counting it."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    finite_mask = np.isfinite(numeric)
    return numeric[finite_mask], int((~finite_mask).sum())


def build_reference_distribution(values: pd.Series, feature: str, partition: str, n_bins: int = N_DISTRIBUTION_BINS) -> dict:
    """Builds a reference decile-style distribution from REAL data
    (never from a single summary statistic like an imputer median).
    Persists everything a later comparison needs: bin edges,
    proportions, sample sizes, missing-value handling, and a
    population fingerprint."""
    finite, n_missing = _clean_numeric(values)
    n_total = len(values)
    result: dict[str, Any] = {
        "feature": feature,
        "partition": partition,
        "reference_sample_size_total": n_total,
        "reference_sample_size_finite": len(finite),
        "n_missing_or_non_finite": n_missing,
        "n_bins_requested": n_bins,
        "population_fingerprint": fingerprint_reference_population(values, partition, feature),
    }
    if len(finite) < MIN_REFERENCE_SAMPLE_SIZE:
        result.update(status="insufficient_reference_sample", bin_edges=None, bin_proportions=None, constant_feature=None)
        return result

    n_unique = len(np.unique(finite))
    if n_unique <= 1:
        result.update(
            status="constant_feature",
            constant_feature=True,
            constant_value=float(finite[0]),
            bin_edges=[float(finite[0]), float(finite[0])],
            bin_proportions=[1.0],
        )
        return result

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(finite, quantiles))
    if len(edges) < 2:
        edges = np.array([finite.min(), finite.max()])
    counts, edges = np.histogram(finite, bins=edges)
    proportions = (counts / counts.sum()).tolist()
    result.update(
        status="ok",
        constant_feature=False,
        bin_edges=edges.tolist(),
        bin_proportions=proportions,
    )
    return result


def _bin_counts_open_ended(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bins `values` into `len(edges) - 1` bins using the INTERIOR edges
    only, so the first and last bins are open-ended (<edges[1] and
    >=edges[-2] respectively) rather than clipping values outside
    [edges[0], edges[-1]]. This matters: a batch that has drifted far
    enough to fall entirely outside the reference's historical range is
    the single most extreme drift signal possible -- dropping it (as a
    plain `np.histogram` call would) would hide exactly the failure
    this monitor exists to catch."""
    interior = edges[1:-1]
    n_bins = len(edges) - 1
    if len(interior) == 0:
        return np.array([len(values)])
    idx = np.digitize(values, interior)
    counts = np.bincount(idx, minlength=n_bins)
    return counts[:n_bins]


def score_batch_distribution(values: pd.Series, reference: dict) -> dict:
    """Bins a NEW batch against an EXISTING reference's bin edges.
    Returns a status distinct from a drift finding when the batch
    itself cannot be meaningfully evaluated (empty, too small, or the
    reference itself was never usable)."""
    n_total = len(values)
    finite, n_missing = _clean_numeric(values)

    out: dict[str, Any] = {
        "feature": reference.get("feature"),
        "batch_sample_size_total": n_total,
        "batch_sample_size_finite": len(finite),
        "n_missing_or_non_finite": n_missing,
    }
    if n_total == 0:
        out.update(status="malformed_input", reason="empty_batch")
        return out
    if reference.get("status") not in ("ok", "constant_feature"):
        out.update(status="reference_unusable", reason=reference.get("status"))
        return out
    if len(finite) < MIN_BATCH_SAMPLE_SIZE:
        out.update(status="insufficient_batch_sample")
        return out

    if reference.get("constant_feature"):
        constant_value = reference["constant_value"]
        matches = float(np.mean(np.isclose(finite, constant_value)))
        out.update(status="ok", bin_proportions=[matches, 1.0 - matches], note="reference feature was constant")
        return out

    edges = np.array(reference["bin_edges"])
    counts = _bin_counts_open_ended(finite, edges)
    n_out_of_reference_range = int(np.sum((finite < edges[0]) | (finite > edges[-1])))  # informational
    # only -- these values ARE included in the extreme bins' counts above, never dropped.
    proportions = (counts / counts.sum()).tolist() if counts.sum() > 0 else None
    out.update(
        status="ok" if proportions is not None else "malformed_input",
        bin_proportions=proportions,
        n_out_of_reference_range=n_out_of_reference_range,
    )
    return out


def population_stability_index(reference_proportions: list[float], batch_proportions: list[float], epsilon: float = 1e-4) -> float:
    """Standard PSI: sum((batch - ref) * ln(batch / ref)) per bin, with
    additive smoothing so a zero-proportion bin never causes log(0) or
    a divide-by-zero."""
    ref = np.array(reference_proportions, dtype="float64") + epsilon
    batch = np.array(batch_proportions, dtype="float64") + epsilon
    ref = ref / ref.sum()
    batch = batch / batch.sum()
    return float(np.sum((batch - ref) * np.log(batch / ref)))


@dataclass
class AlertRecord:
    """A single, fully self-contained alert -- every field the Stage 18
    instruction required is present by construction, never optional."""
    alert_id: str
    signal: str
    calculation: str
    observed_statistic: float | None
    threshold: float | None
    sample_size: int
    reference_provenance: dict
    batch_provenance: dict
    severity: str
    recommended_action: str
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "signal": self.signal,
            "calculation": self.calculation,
            "observed_statistic": self.observed_statistic,
            "threshold": self.threshold,
            "sample_size": self.sample_size,
            "reference_provenance": self.reference_provenance,
            "batch_provenance": self.batch_provenance,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "detected_at": self.detected_at,
        }


def evaluate_input_drift(reference: dict, batch_values: pd.Series, threshold: float = PSI_ALERT_THRESHOLD) -> dict:
    """The end-to-end input-drift check for one feature: bin the batch
    against the reference, compute PSI, and return either a non-alert
    'ok'/'insufficient_*'/'malformed_input' result or a full
    `AlertRecord` dict when PSI exceeds the threshold."""
    feature = reference.get("feature")
    batch_dist = score_batch_distribution(batch_values, reference)
    if batch_dist["status"] != "ok" or reference.get("status") not in ("ok", "constant_feature"):
        return {"status": batch_dist["status"], "feature": feature, "batch_distribution": batch_dist}

    psi = population_stability_index(reference["bin_proportions"], batch_dist["bin_proportions"])
    result = {
        "status": "drift_alert" if psi > threshold else "ok",
        "feature": feature,
        "psi": round(psi, 6),
        "threshold": threshold,
        "batch_sample_size": batch_dist["batch_sample_size_finite"],
    }
    if result["status"] == "drift_alert":
        result["alert"] = AlertRecord(
            alert_id=f"input_drift::{feature}",
            signal=f"input_distribution::{feature}",
            calculation="Population Stability Index (PSI) vs. the train-partition reference decile distribution",
            observed_statistic=round(psi, 6),
            threshold=threshold,
            sample_size=batch_dist["batch_sample_size_finite"],
            reference_provenance={"partition": reference.get("partition"), "population_fingerprint": reference.get("population_fingerprint")},
            batch_provenance={"n_missing_or_non_finite": batch_dist["n_missing_or_non_finite"], "n_out_of_reference_range": batch_dist.get("n_out_of_reference_range")},
            severity="moderate",
            recommended_action="Investigate upstream feature source/schema change for this feature before trusting new predictions; see the monitoring runbook.",
        ).to_dict()
    return result


# ---------------------------------------------------------------------------
# Prediction (score) distribution monitoring
# ---------------------------------------------------------------------------

def build_reference_score_distribution(scores: np.ndarray, artifact_sha256: str, n_bins: int = N_DISTRIBUTION_BINS) -> dict:
    """Builds a reference `fraud_probability` distribution from scores
    already produced by the real, trusted artifact (never refit here).
    Carries the artifact's checksum so a later comparison can refuse to
    trust a mismatched-provenance comparison rather than silently
    comparing scores from two different model versions."""
    scores = np.asarray(scores, dtype="float64")
    finite = scores[np.isfinite(scores)]
    result: dict[str, Any] = {
        "artifact_sha256": artifact_sha256,
        "reference_sample_size": len(finite),
        "n_non_finite": int(len(scores) - len(finite)),
    }
    if len(finite) < MIN_REFERENCE_SAMPLE_SIZE:
        result.update(status="insufficient_reference_sample", bin_edges=None, bin_proportions=None)
        return result
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([finite.min(), finite.max()])
    counts, edges = np.histogram(finite, bins=edges)
    result.update(status="ok", bin_edges=edges.tolist(), bin_proportions=(counts / counts.sum()).tolist(),
                   mean=float(finite.mean()), median=float(np.median(finite)), std=float(finite.std()))
    return result


def evaluate_score_drift(reference: dict, batch_scores: np.ndarray, current_artifact_sha256: str, threshold: float = PSI_ALERT_THRESHOLD) -> dict:
    """Never claims `fraud_probability` is a calibrated real-world
    probability, and never derives a classification threshold here --
    this compares SCORE DISTRIBUTION SHAPE only, exactly the same
    uncalibrated-ranking-signal framing established in Stage 10/14."""
    if reference.get("artifact_sha256") != current_artifact_sha256:
        return {
            "status": "provenance_mismatch",
            "reason": "Reference scores and the current batch's scores come from different artifact checksums -- "
            "refusing to compare distributions that are not like-for-like.",
            "reference_artifact_sha256": reference.get("artifact_sha256"),
            "current_artifact_sha256": current_artifact_sha256,
        }
    if reference.get("status") != "ok":
        return {"status": reference.get("status")}

    batch_scores = np.asarray(batch_scores, dtype="float64")
    finite = batch_scores[np.isfinite(batch_scores)]
    if len(batch_scores) == 0:
        return {"status": "malformed_input", "reason": "empty_batch"}
    if len(finite) < MIN_BATCH_SAMPLE_SIZE:
        return {"status": "insufficient_batch_sample", "batch_sample_size": len(finite)}

    edges = np.array(reference["bin_edges"])
    counts = _bin_counts_open_ended(finite, edges)
    if counts.sum() == 0:
        return {"status": "malformed_input", "reason": "all_values_out_of_reference_range"}
    batch_props = (counts / counts.sum()).tolist()
    psi = population_stability_index(reference["bin_proportions"], batch_props)
    result = {
        "status": "drift_alert" if psi > threshold else "ok",
        "psi": round(psi, 6),
        "threshold": threshold,
        "batch_sample_size": len(finite),
        "batch_mean": float(finite.mean()),
        "reference_mean": reference.get("mean"),
    }
    if result["status"] == "drift_alert":
        result["alert"] = AlertRecord(
            alert_id="prediction_score_drift",
            signal="prediction_score_distribution::fraud_probability",
            calculation="PSI of the batch's fraud_probability distribution vs. the reference score distribution "
            "(both produced by the same trusted artifact; fraud_probability is an uncalibrated ranking signal, not "
            "a real-world probability -- see Stage 10/14).",
            observed_statistic=round(psi, 6),
            threshold=threshold,
            sample_size=len(finite),
            reference_provenance={"artifact_sha256": reference.get("artifact_sha256"), "reference_mean": reference.get("mean")},
            batch_provenance={"artifact_sha256": current_artifact_sha256, "batch_mean": float(finite.mean())},
            severity="moderate",
            recommended_action="Investigate whether an upstream input-distribution alert also fired for the same batch; "
            "if not, treat as a possible model-behavior change and review before trusting new scores operationally.",
        ).to_dict()
    return result


# ---------------------------------------------------------------------------
# Operational monitoring (parses the EXISTING Stage 14/17 structured logs)
# ---------------------------------------------------------------------------

def parse_log_file(path: str | Path) -> dict:
    """Parses the project's own one-JSON-object-per-line `_log_event`
    output, skipping Uvicorn's interleaved plain-text access-log lines
    (which are not valid JSON) rather than erroring on them. Returns
    both the parsed records and a count of skipped lines, so a caller
    can see exactly how much of the file was -- and was not --
    machine-readable, instead of assuming every line is a structured event."""
    records: list[dict] = []
    n_skipped_non_json = 0
    text = Path(path).read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            n_skipped_non_json += 1
    return {"records": records, "n_lines_total": len([l for l in text.splitlines() if l.strip()]), "n_skipped_non_json": n_skipped_non_json}


def summarize_operational_records(records: list[dict], expected_artifact_sha256: str | None = None) -> dict:
    """Computes request count, status-code breakdown, error rate, and
    latency percentiles from the EXISTING log fields only -- never
    invents a timestamp or a window the logs don't actually carry (see
    this module's docstring and the Stage 18 report's audit section)."""
    predict_like = [r for r in records if "status" in r and "duration_seconds" in r]
    n_total = len(predict_like)
    statuses = [int(r["status"]) for r in predict_like]
    n_2xx = sum(1 for s in statuses if 200 <= s < 300)
    n_4xx = sum(1 for s in statuses if 400 <= s < 500)
    n_5xx = sum(1 for s in statuses if 500 <= s < 600)
    durations = sorted(float(r["duration_seconds"]) for r in predict_like)

    artifact_shas = {r.get("artifact_sha256") for r in predict_like if r.get("artifact_sha256")}
    mismatched_artifacts = (
        sorted(artifact_shas - {expected_artifact_sha256}) if expected_artifact_sha256 else sorted(artifact_shas)
    )

    summary: dict[str, Any] = {
        "n_records_with_status_and_duration": n_total,
        "n_2xx": n_2xx,
        "n_4xx_client_error": n_4xx,
        "n_5xx_server_error": n_5xx,
        "error_rate_5xx": (n_5xx / n_total) if n_total else None,
        "distinct_artifact_sha256_seen": sorted(artifact_shas),
        "artifact_sha256_mismatches": mismatched_artifacts,
        "n_records_missing_artifact_identity": sum(1 for r in predict_like if r.get("status") == 200 and not r.get("artifact_sha256")),
    }
    if durations and n_total >= MIN_OPERATIONAL_LATENCY_SAMPLE_SIZE:
        summary["latency_median_seconds"] = round(durations[len(durations) // 2], 4)
        summary["latency_p95_seconds"] = round(durations[int(len(durations) * 0.95) if len(durations) > 1 else 0], 4)
        summary["latency_sample_size"] = len(durations)
    else:
        summary["latency_median_seconds"] = None
        summary["latency_p95_seconds"] = None
        summary["latency_sample_size"] = len(durations)
        summary["latency_insufficient_sample"] = True
    return summary


def evaluate_operational_health(summary: dict, expected_artifact_sha256: str) -> list[dict]:
    """Turns an operational summary into zero or more real alert
    records. Any observed 5xx is flagged regardless of sample size --
    with a low-traffic local API, waiting for a RATE threshold to
    accumulate would hide the very first real outage; this is stated
    explicitly as the rationale, not left implicit."""
    alerts: list[dict] = []
    n_total = summary["n_records_with_status_and_duration"]

    if summary["n_5xx_server_error"] > 0:
        alerts.append(AlertRecord(
            alert_id="operational_server_error",
            signal="operational::5xx_server_error_rate",
            calculation="count(status in [500,599]) over all log records carrying a status+duration_seconds field",
            observed_statistic=summary["error_rate_5xx"],
            threshold=0.0,
            sample_size=n_total,
            reference_provenance={"expected_artifact_sha256": expected_artifact_sha256},
            batch_provenance={"n_5xx": summary["n_5xx_server_error"], "n_total": n_total},
            severity="critical",
            recommended_action="Check API/deployment health immediately (GET /health, GET /version); "
            "if the release is unready, roll back per docs/STAGE17_ROLLBACK_RUNBOOK.md.",
        ).to_dict())

    if summary["artifact_sha256_mismatches"]:
        alerts.append(AlertRecord(
            alert_id="operational_artifact_identity_mismatch",
            signal="operational::artifact_identity",
            calculation="set(artifact_sha256 seen in successful-response log records) - {expected_artifact_sha256}",
            observed_statistic=None,
            threshold=None,
            sample_size=n_total,
            reference_provenance={"expected_artifact_sha256": expected_artifact_sha256},
            batch_provenance={"unexpected_artifact_sha256_values": summary["artifact_sha256_mismatches"]},
            severity="critical",
            recommended_action="A response was served by an artifact other than the approved one -- verify which "
            "release is actually running and roll back if it is not the approved Version A.",
        ).to_dict())

    if summary.get("latency_p95_seconds") is not None and summary["latency_p95_seconds"] > LATENCY_P95_ALERT_SECONDS:
        alerts.append(AlertRecord(
            alert_id="operational_latency_p95",
            signal="operational::latency_p95",
            calculation="95th percentile of duration_seconds over records with a sufficient latency sample",
            observed_statistic=summary["latency_p95_seconds"],
            threshold=LATENCY_P95_ALERT_SECONDS,
            sample_size=summary["latency_sample_size"],
            reference_provenance={"stage14_measured_p95_seconds": 0.0462},
            batch_provenance={},
            severity="moderate",
            recommended_action="Investigate for resource contention or an unusually large/slow request; "
            "not evidence of model correctness decay.",
        ).to_dict())

    return alerts


# ---------------------------------------------------------------------------
# Config / persistence helpers
# ---------------------------------------------------------------------------

def load_monitoring_config(path: str | Path = MONITORING_CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
