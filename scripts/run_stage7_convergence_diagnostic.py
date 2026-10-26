"""Stage 7 — focused convergence investigation.

Reuses the EXACT SAME 15,000-row chronological `train` subset as
scripts/run_stage7_pilot.py (same function, same n_rows, same
determinism) so this round's findings are directly comparable to that
pilot's, not a new, uncontrolled slice.

TRAIN-ONLY throughout: no validation label is used to select anything in
this script. Validation AP is computed once at the end, for continuity
with the prior round's report, and is explicitly NOT used to choose
between the configurations investigated here. `final_test` is never
requested.

No model is "selected" by this script -- it diagnoses WHY the previous
round's saga fits did not converge, and compares saga against
SGDClassifier(loss="log_loss") under matched, non-tuned settings.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.feature_engineering import collapsed_indicator_representatives, find_duplicate_indicator_groups  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    RANDOM_STATE,
    build_chronological_pilot_subset,
    build_scaled_pipeline_with_collapsed_indicators,
    evaluate_predictions,
    find_duplicate_value_columns,
    find_near_constant_columns,
    load_labels,
    max_pairwise_correlation,
    sparse_magnitude_summary,
    track_saga_training_loss,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000  # identical to scripts/run_stage7_pilot.py, for direct comparability


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("0. Rebuild the SAME 15,000-row chronological pilot subset (not a new slice)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    print(f"Pilot subset: {len(pilot_df)} rows, fraud prevalence {pilot_df['isFraud'].mean():.4%} "
          f"(matches scripts/run_stage7_pilot.py's subset exactly -- same deterministic selection)")

    section("0b. Confirm final_test still refused from this script too")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. Zero-variance / near-constant columns (post-impute, pre-scale) -- MEASURED")
    from sklearn.impute import SimpleImputer

    numeric_value_cols = numeric_no_indicator + numeric_with_indicator
    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(pilot_df[numeric_value_cols])
    near_constant = find_near_constant_columns(imputed_values, numeric_value_cols, std_threshold=1e-6)
    print(f"Near-constant columns (std < 1e-6) among {len(numeric_value_cols)} numeric columns: {len(near_constant)}")
    if near_constant:
        print(f"  {near_constant[:10]}{'...' if len(near_constant) > 10 else ''}")
    report["near_constant_columns"] = {"count": len(near_constant), "examples": near_constant[:10]}

    section("2. Duplicate/highly-correlated VALUE columns (distinct from the already-collapsed indicator MASKS) -- MEASURED")
    dup_value_groups = find_duplicate_value_columns(imputed_values, numeric_value_cols)
    n_dup_value_cols = sum(len(v) for v in dup_value_groups.values())
    print(f"Exact-duplicate VALUE columns: {len(dup_value_groups)} groups covering {n_dup_value_cols} columns")
    corr_result = max_pairwise_correlation(imputed_values, numeric_value_cols)
    print(f"Max pairwise |correlation| among distinct columns: {corr_result['max_abs_correlation']} "
          f"({corr_result['column_a']} vs {corr_result['column_b']}); "
          f"{corr_result['n_pairs_above_0.99']} pairs exceed 0.99")
    report["duplicate_value_columns"] = {"n_groups": len(dup_value_groups), "n_columns_involved": n_dup_value_cols}
    report["max_pairwise_correlation"] = corr_result

    section("3. Sparse-matrix magnitude/density of the pilot's actual training matrix -- MEASURED")
    groups = find_duplicate_indicator_groups(pilot_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(groups)
    ct = build_scaled_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)
    t0 = time.perf_counter()
    X_train = ct.fit_transform(pilot_df[all_cols])
    pipeline_fit_s = time.perf_counter() - t0
    mag = sparse_magnitude_summary(X_train)
    print(f"Pipeline fit+transform: {pipeline_fit_s:.3f}s, shape {mag['shape']}, density {mag['density']}")
    print(f"Nonzero magnitude: min={mag['min_abs_nonzero']:.4g} max={mag['max_abs_nonzero']:.4g} "
          f"median={mag['median_abs_nonzero']:.4g}  dynamic range: {mag['dynamic_range_orders_of_magnitude']} orders of magnitude")
    report["pipeline_fit_seconds"] = round(pipeline_fit_s, 3)
    report["sparse_magnitude"] = mag

    section("4. SAGA training-loss trajectory (TRAIN-ONLY, warm_start, both class_weight configs) -- MEASURED, not hypothesized")
    y_train = pilot_df["isFraud"].to_numpy()
    traces = {}
    for cw_name, cw in [("balanced", "balanced"), ("unweighted", None)]:
        print(f"\n--- class_weight={cw_name} ---")
        t0 = time.perf_counter()
        trace = track_saga_training_loss(X_train, y_train, class_weight=cw, n_steps=10, iters_per_step=100)
        trace_s = time.perf_counter() - t0
        for point in trace:
            print(f"  iters={point['cumulative_iters']:5d}  train_log_loss={point['train_log_loss']:.6f}  "
                  f"coef_change={point['coef_l2_change_from_previous_step']}  "
                  f"still_warning={point['convergence_warning_raised']}")
        traces[cw_name] = {"trace": trace, "total_seconds": round(trace_s, 3)}
        first_loss, last_loss = trace[0]["train_log_loss"], trace[-1]["train_log_loss"]
        loss_drop_last_half = trace[4]["train_log_loss"] - trace[-1]["train_log_loss"]
        print(f"  Loss at iter 100: {first_loss:.6f} -> loss at iter 1000: {last_loss:.6f} "
              f"(drop over last 500 iters: {loss_drop_last_half:.6f})")
    report["saga_loss_trajectory"] = traces

    section("5. Bounded solver comparison: SAGA vs SGDClassifier(loss='log_loss') -- SAME subset, matched regularization")
    print("Objective-function difference, stated explicitly (not glossed over):")
    print("  - LogisticRegression(solver='saga'): exact finite-sum L2-regularized logistic MLE, variance-reduced")
    print("    proximal gradient -- processes the full dataset's gradient information each epoch.")
    print("  - SGDClassifier(loss='log_loss'): the SAME logistic loss, but optimized by plain (non-variance-reduced)")
    print("    stochastic gradient descent with a decaying learning-rate schedule -- noisier updates, different")
    print("    convergence guarantees, typically MORE sensitive to feature scale, not less.")
    n_train = X_train.shape[0]
    alpha_matched = 1.0 / (n_train * 1.0)  # matches LogisticRegression's C=1.0 under sklearn's own stated equivalence

    solver_results = {}
    for cw_name, cw in [("balanced", "balanced"), ("unweighted", None)]:
        for solver_name, make_model in [
            ("saga", lambda: LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight=cw)),
            ("sgd_log_loss", lambda: SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha_matched, max_iter=1000, random_state=RANDOM_STATE, class_weight=cw, tol=1e-3)),
        ]:
            model = make_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                fit_s = time.perf_counter() - t0
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            train_loss = log_loss(y_train, model.predict_proba(X_train) if hasattr(model, "predict_proba") else None) if solver_name == "saga" else None
            n_iter = int(model.n_iter_[0]) if hasattr(model.n_iter_, "__len__") else int(model.n_iter_)
            key = f"{solver_name}_{cw_name}"
            print(f"{key}: fit_seconds={fit_s:.3f} n_iter_={n_iter} converged={converged}")
            solver_results[key] = {
                "solver": solver_name,
                "class_weight": cw_name,
                "fit_seconds": round(fit_s, 3),
                "n_iter": n_iter,
                "converged": converged,
                "train_log_loss": round(float(train_loss), 6) if train_loss is not None else None,
            }
    report["solver_comparison"] = solver_results

    section("6. Validation AP -- reported for CONTINUITY only, NOT used to select anything in this script")
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    X_val = ct.transform(val_df[all_cols])
    for key in list(solver_results.keys()):
        cw = "balanced" if "balanced" in key else None
        if "saga" in key:
            model = LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight=cw).fit(X_train, y_train)
        else:
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha_matched, max_iter=1000, random_state=RANDOM_STATE, class_weight=cw, tol=1e-3).fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        solver_results[key]["validation_average_precision_for_continuity_only"] = metrics["average_precision"]
        print(f"{key}: validation AP (continuity only, not used for selection) = {metrics['average_precision']}")

    out_path = DOCS_DIR / "_stage7_convergence_diagnostic_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
