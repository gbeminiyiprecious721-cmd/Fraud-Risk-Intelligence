"""Stage 12 -- Responsible AI assessment helpers.

Pure functions operating on already-loaded DataFrames/arrays and an
already-fitted pipeline + model. This module never reads a dataset file
itself and never requests `final_test` -- the driver script owns all
data access, same separation as `src/explainability.py`.

Two distinct kinds of check are kept deliberately separate throughout:

  - INPUT VALIDATION: does the fitted pipeline + model accept an
    unusual input without crashing, and return a finite probability in
    [0, 1]? This says nothing about whether that probability is a
    *correct* assessment of fraud risk for an unusual transaction --
    only that the system behaves predictably rather than crashing or
    returning garbage.
  - PREDICTIVE ROBUSTNESS (early-vs-late chronological performance,
    reported from already-saved Stage 9 evidence in the driver script,
    not computed here): whether the model's actual AP holds up as time
    moves forward within `train`. This module does not claim an
    edge-case input's predicted probability represents genuine fraud
    behaviour -- that would require labelled examples of exactly this
    kind, which do not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Subgroup lift -- a thin addition on top of `model_evaluation.segment_report`,
# which already reports per-segment AP, prevalence and sufficiency. This adds
# the AP-to-prevalence ratio (Stage 10 audit's "prevalence-normalised" idea,
# generalised to any segmentation) without modifying the approved Stage 10 module.
# ---------------------------------------------------------------------------

def segment_report_with_lift(segment_report_result: dict) -> pd.DataFrame:
    """Flatten a `segment_report()` dict into a DataFrame with an added
    `ap_to_prevalence_lift` column (AP / prevalence) for every segment
    that had a sufficient sample. Segments below the sufficiency floor
    keep their counts but get `NaN` for every metric column, never a
    fabricated one."""
    rows = []
    for segment, entry in segment_report_result.items():
        row = {
            "segment": segment,
            "n_samples": entry["n_samples"],
            "n_positive": entry["n_positive"],
            "sufficient_sample": entry["sufficient_sample"],
        }
        if entry["sufficient_sample"]:
            metrics = entry["metrics"]
            prevalence = metrics["actual_prevalence"]
            ap = metrics["average_precision"]
            row.update(
                {
                    "prevalence": prevalence,
                    "average_precision": ap,
                    "ap_to_prevalence_lift": ap / prevalence if prevalence > 0 else float("nan"),
                    "precision_at_1pct": metrics["k_frac_0.01"]["precision_at_k"],
                    "recall_at_1pct": metrics["k_frac_0.01"]["recall_at_k"],
                    "precision_at_5pct": metrics["k_frac_0.05"]["precision_at_k"],
                    "recall_at_5pct": metrics["k_frac_0.05"]["recall_at_k"],
                }
            )
        else:
            row.update(
                {
                    "prevalence": float("nan"), "average_precision": float("nan"), "ap_to_prevalence_lift": float("nan"),
                    "precision_at_1pct": float("nan"), "recall_at_1pct": float("nan"),
                    "precision_at_5pct": float("nan"), "recall_at_5pct": float("nan"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


# ---------------------------------------------------------------------------
# Edge-case construction -- derived from a single REAL row (never fabricated
# from scratch), with exactly one field changed per case, so any crash or
# odd output can be attributed to that one change.
# ---------------------------------------------------------------------------

def build_edge_case_rows(template_row: pd.Series, numeric_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Build named single-row DataFrames, each a copy of `template_row`
    with exactly one realistic-or-deliberately-unrealistic change."""
    def one(**changes) -> pd.DataFrame:
        row = template_row.copy()
        for col, val in changes.items():
            row[col] = val
        return row.to_frame().T

    cases = {
        "amount_zero": one(TransactionAmt=0.0),
        "amount_tiny_positive": one(TransactionAmt=0.01),
        "amount_negative": one(TransactionAmt=-50.0),
        "amount_tail_p999_seen_in_data": one(TransactionAmt=2769.81),
        "amount_far_out_of_distribution": one(TransactionAmt=1_000_000.0),
        "unseen_productcd": one(ProductCD="ZZ"),
        "unseen_card6": one(card6="prepaid_unknown_type"),
        "all_numeric_missing": one(**{c: np.nan for c in numeric_columns}),
    }
    return cases


def run_edge_case_predictions(model, column_transformer, edge_cases: dict[str, pd.DataFrame], all_cols: list[str]) -> dict[str, dict]:
    """For each named edge case, attempt `transform` + `predict_proba` and
    report what actually happened -- never assumed. A caught exception is
    reported with its type and message, not silently swallowed."""
    results: dict[str, dict] = {}
    for name, row_df in edge_cases.items():
        entry: dict = {"raised": False, "exception_type": None, "exception_message": None, "predicted_proba": None, "is_finite": None, "in_valid_range": None}
        try:
            X = column_transformer.transform(row_df[all_cols])
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            proba = float(model.predict_proba(X_dense)[0, 1])
            entry["predicted_proba"] = proba
            entry["is_finite"] = bool(np.isfinite(proba))
            entry["in_valid_range"] = bool(0.0 <= proba <= 1.0)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: we want to OBSERVE whatever happens
            entry["raised"] = True
            entry["exception_type"] = type(exc).__name__
            entry["exception_message"] = str(exc)
        results[name] = entry
    return results


def schema_mismatch_case(column_transformer, row_df: pd.DataFrame, all_cols: list[str], drop_column: str) -> dict:
    """Drop one required column entirely from the input and attempt
    `transform` -- reports whether this raises a clear, catchable error
    (the desired behaviour for a future production integration) rather
    than silently producing a wrong-shape or wrong-value output."""
    reduced_cols = [c for c in all_cols if c != drop_column]
    entry: dict = {"raised": False, "exception_type": None, "exception_message": None}
    try:
        column_transformer.transform(row_df[reduced_cols])
    except Exception as exc:  # noqa: BLE001
        entry["raised"] = True
        entry["exception_type"] = type(exc).__name__
        entry["exception_message"] = str(exc)
    return entry


def malformed_dtype_case(template_row: pd.Series, column: str, bad_value: str = "not_a_number") -> pd.DataFrame:
    """A copy of `template_row` with one numeric column set to a string
    that cannot be interpreted as a number -- a plausible real-world
    integration bug (e.g. a numeric field arriving as free text from an
    upstream system), not a hypothetical one."""
    row = template_row.copy()
    row[column] = bad_value
    return row.to_frame().T
