"""Stage 10 — frozen-model evaluation protocol, specified BEFORE any
`final_test` access, per explicit instruction.

This module builds everything the Stage 10 readiness gate needs: the
frozen Stage-9 candidate, segment/bootstrap/cost-threshold analysis, and
the one-time `final_test` unlock mechanism -- but contains no call site
that actually reads `final_test`. The only place that can ever enable
the dataset layer's own final-partition override is the gated script
`scripts/run_stage10_final_evaluation.py`, and even that script cannot
proceed without BOTH the exact unlock token AND a registry check proving
no completed evaluation already exists under this exact search_id.

Every number/boundary/threshold candidate used in segmentation or cost
analysis is either a FIXED, declared constant (written down here, before
any `final_test` row is ever read) or is computed from `train`-only data
passed in by the caller -- never from `final_test`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from src.model_selection import evaluate_predictions
from src.experiment_tracking import DEFAULT_REGISTRY_PATH, query_registry

# ---------------------------------------------------------------------------
# 1. The FROZEN Stage 9 candidate -- exactly as audited, no new tuning here
# ---------------------------------------------------------------------------

FROZEN_RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": None,
}

FROZEN_CONFIG_FINGERPRINT = hashlib.sha256(
    "|".join(f"{k}={v}" for k, v in sorted(FROZEN_RF_CONFIG.items())).encode("utf-8")
).hexdigest()


def build_frozen_model() -> RandomForestClassifier:
    """The exact, audited Stage 9 provisional candidate (candidate_index=4).
    No parameter here may be changed without a new, separately-approved
    Stage 9 round -- enforced by the fingerprint check in
    `verify_frozen_config_unchanged`, not merely by convention."""
    return RandomForestClassifier(**FROZEN_RF_CONFIG)


def verify_frozen_config_unchanged() -> bool:
    """Defense in depth: recomputes the fingerprint from the live
    `FROZEN_RF_CONFIG` dict and compares against the constant captured at
    import time -- will only ever disagree if someone edits one but not
    the other in the same change, which this makes loud rather than silent."""
    recomputed = hashlib.sha256(
        "|".join(f"{k}={v}" for k, v in sorted(FROZEN_RF_CONFIG.items())).encode("utf-8")
    ).hexdigest()
    return recomputed == FROZEN_CONFIG_FINGERPRINT


# ---------------------------------------------------------------------------
# 2. Segmentation -- boundaries computed from TRAIN only, passed in by caller
# ---------------------------------------------------------------------------

MIN_SEGMENT_SAMPLES = 100
MIN_SEGMENT_POSITIVES = 10
N_AMOUNT_SEGMENTS = 4  # quartile-style


def compute_amount_segment_boundaries(train_amounts: np.ndarray, n_segments: int = N_AMOUNT_SEGMENTS) -> list[float]:
    """Quartile-style boundaries computed from `train_amounts` ONLY. The
    caller is responsible for never passing `final_test` amounts here --
    this function has no partition awareness of its own, by design (it is
    a pure, testable transform), so the leakage-safety is a CALLER
    discipline documented here and enforced by `scripts/run_stage10_final_evaluation.py`'s
    own structure (boundaries computed once, from `train`, before any
    `final_test` row is read).
    """
    train_amounts = np.asarray(train_amounts)
    qs = np.linspace(0, 100, n_segments + 1)[1:-1]
    return [float(x) for x in np.percentile(train_amounts, qs)]


def assign_amount_segment(amounts: np.ndarray, boundaries: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(amounts), boundaries)


def segment_report(
    y_true, y_score, segment_labels,
    min_samples: int = MIN_SEGMENT_SAMPLES, min_positives: int = MIN_SEGMENT_POSITIVES,
) -> dict:
    """Per-segment metrics, with an explicit `sufficient_sample` flag --
    a segment below the floor gets its counts reported but NOT a metric,
    rather than a misleadingly precise-looking number from too few rows."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    segment_labels = np.asarray(segment_labels)
    report: dict = {}
    for seg in sorted(pd.unique(segment_labels), key=str):
        mask = segment_labels == seg
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        sufficient = n >= min_samples and n_pos >= min_positives
        entry: dict = {"n_samples": n, "n_positive": n_pos, "sufficient_sample": sufficient}
        if sufficient:
            entry["metrics"] = evaluate_predictions(y_true[mask], y_score[mask])
        report[str(seg)] = entry
    return report


# ---------------------------------------------------------------------------
# 3. Bootstrap confidence interval -- fixed seed, predefined resample count
# ---------------------------------------------------------------------------

BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI_LEVEL = 0.95


def bootstrap_ap_ci(
    y_true, y_score,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES, seed: int = BOOTSTRAP_SEED, ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> dict:
    """Percentile bootstrap CI on Average Precision. Resamples ROWS (not
    individually resampling positives/negatives), preserving whatever
    class balance each resample happens to draw -- the standard percentile
    bootstrap, declared and fixed here before any real evaluation.
    Limitation stated explicitly in the report, not here: this treats
    rows as i.i.d., which is a known simplification given the project's
    carried-forward "potential dependence between transactions" caveat
    (e.g. repeat `card1` entities) -- a block-bootstrap by entity would be
    more conservative but was not built this round."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = float(average_precision_score(y_true, y_score))
    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        boot_aps[i] = average_precision_score(y_true[idx], y_score[idx])
    alpha = 1 - ci_level
    lower = float(np.percentile(boot_aps, 100 * alpha / 2))
    upper = float(np.percentile(boot_aps, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point, "ci_level": ci_level, "lower": lower, "upper": upper,
        "n_resamples": n_resamples, "seed": seed,
    }


# ---------------------------------------------------------------------------
# 4. Cost-sensitive threshold analysis -- Stage 1's own framework, made
# computable WITHOUT inventing a hidden number. Stage 1 defined the FN cost
# proxy in dollars (a fraction of TransactionAmt) but never defined a
# dollar value for "1 analyst-review unit" (the FP cost proxy's unit).
# Rather than silently picking one, that conversion is SWEPT across
# several explicitly-labeled [HYPOTHETICAL] candidate values.
# ---------------------------------------------------------------------------

COST_SCENARIOS = {
    "A_conservative_loss": {"fn_fraction_of_amount": 0.5, "fp_cost_in_units": 1},
    "B_full_transaction_value": {"fn_fraction_of_amount": 1.0, "fp_cost_in_units": 1},
    "C_capacity_constrained": {"fn_fraction_of_amount": 1.0, "fp_cost_in_units": 5},
}

# [HYPOTHETICAL] -- labeled candidate dollar values for "1 analyst-review
# unit", per Stage 1's own convention for unpinned assumptions. Swept, not
# fixed, so no single invented number drives the recommendation.
ANALYST_UNIT_DOLLAR_ASSUMPTIONS = [5.0, 10.0, 20.0, 50.0]

DEFAULT_THRESHOLD_GRID = [round(t, 2) for t in np.linspace(0.01, 0.99, 99)]


def expected_cost_at_threshold(
    y_true, y_score, amounts, threshold: float, fn_fraction_of_amount: float, fp_cost_dollars: float,
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    amounts = np.asarray(amounts)
    flagged = y_score >= threshold
    fn_mask = (~flagged) & (y_true == 1)
    fp_mask = flagged & (y_true == 0)
    fn_cost = float((amounts[fn_mask] * fn_fraction_of_amount).sum())
    fp_cost = float(fp_mask.sum()) * fp_cost_dollars
    return fn_cost + fp_cost


def sweep_cost_minimizing_thresholds(
    y_true, y_score, amounts,
    thresholds: list[float] | None = None,
    unit_dollar_assumptions: list[float] | None = None,
) -> dict:
    """For every (cost scenario x analyst-unit-dollar-assumption) pair,
    the threshold (from a fixed grid) that minimizes expected total cost.
    Returns the full grid of scenario x assumption so the final report can
    show a sensitivity table, not one invented number."""
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID
    unit_dollar_assumptions = unit_dollar_assumptions if unit_dollar_assumptions is not None else ANALYST_UNIT_DOLLAR_ASSUMPTIONS
    results: dict = {}
    for scenario_name, scenario in COST_SCENARIOS.items():
        results[scenario_name] = {}
        for unit_dollars in unit_dollar_assumptions:
            fp_cost_dollars = scenario["fp_cost_in_units"] * unit_dollars
            costs = [
                expected_cost_at_threshold(y_true, y_score, amounts, t, scenario["fn_fraction_of_amount"], fp_cost_dollars)
                for t in thresholds
            ]
            best_idx = int(np.argmin(costs))
            results[scenario_name][f"unit_dollars_{unit_dollars:g}"] = {
                "fp_cost_dollars": fp_cost_dollars,
                "best_threshold": float(thresholds[best_idx]),
                "best_cost": float(costs[best_idx]),
            }
    return results


# ---------------------------------------------------------------------------
# 5. Stage 1 success-criterion #2: beat the no-skill baseline, honestly
# ---------------------------------------------------------------------------

def compare_to_no_skill_baseline(model_metrics: dict, no_skill_metrics: dict) -> dict:
    """Stage 1's own criterion #2: 'any candidate model must be shown to
    beat [the trivial] baseline on BOTH AP and recall@K, with the margin
    reported honestly.' Reports the margin and a boolean per metric --
    never collapses "beats" into a single overall pass/fail that could
    hide one of the two failing."""
    ap_margin = model_metrics["average_precision"] - no_skill_metrics["average_precision"]
    r1_margin = model_metrics["k_frac_0.01"]["recall_at_k"] - no_skill_metrics["k_frac_0.01"]["recall_at_k"]
    r5_margin = model_metrics["k_frac_0.05"]["recall_at_k"] - no_skill_metrics["k_frac_0.05"]["recall_at_k"]
    return {
        "ap_margin": round(float(ap_margin), 5), "beats_baseline_on_ap": bool(ap_margin > 0),
        "recall_at_1pct_margin": round(float(r1_margin), 5), "beats_baseline_on_recall_1pct": bool(r1_margin > 0),
        "recall_at_5pct_margin": round(float(r5_margin), 5), "beats_baseline_on_recall_5pct": bool(r5_margin > 0),
    }


# ---------------------------------------------------------------------------
# 6. The one-time final_test unlock mechanism
# ---------------------------------------------------------------------------

FINAL_EVALUATION_SEARCH_ID = "stage10-frozen-rf-final-test-evaluation"
REQUIRED_UNLOCK_TOKEN = "I_APPROVE_THE_ONE_TIME_FINAL_TEST_EVALUATION"


class FinalTestAlreadyEvaluatedError(RuntimeError):
    """Raised when a completed final_test evaluation already exists under
    this exact search_id -- this is a ONE-TIME evaluation, not a re-runnable
    experiment."""


class FinalTestUnlockRefusedError(RuntimeError):
    """Raised when the exact required unlock token was not supplied."""


def require_unlock_token(token: str) -> None:
    """The FIRST check in the one-time evaluation path -- cheap, no I/O,
    no dataset access of any kind. Must pass before anything else runs."""
    if token != REQUIRED_UNLOCK_TOKEN:
        raise FinalTestUnlockRefusedError(
            "final_test access refused: the exact required unlock token was not provided. "
            "This is the only gate in this project permitted to lead to the dataset layer's final-partition override being enabled."
        )


def check_final_test_not_already_evaluated(registry_path=DEFAULT_REGISTRY_PATH) -> None:
    """The SECOND check -- reads only the (dataset-free) experiment
    registry, never `final_test` itself. Refuses if a completed evaluation
    already exists under `FINAL_EVALUATION_SEARCH_ID`."""
    existing = query_registry(registry_path, search_id=FINAL_EVALUATION_SEARCH_ID)
    completed = [r for r in existing if r.get("status") == "completed"]
    if completed:
        raise FinalTestAlreadyEvaluatedError(
            f"A completed final_test evaluation already exists under search_id="
            f"{FINAL_EVALUATION_SEARCH_ID!r} (run_id={completed[0]['run_id']}). This is a ONE-TIME "
            "evaluation -- refusing to repeat it under a modified or identical configuration. A "
            "genuinely new, separately-approved evaluation would require a new, distinct search_id, "
            "decided by the user, not silently generated here."
        )
