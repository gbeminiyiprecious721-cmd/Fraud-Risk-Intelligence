"""Stage 8 — backfill Stage 7's already-completed experiments into the
experiment registry (`docs/experiment_registry.jsonl`).

This script does NOT re-run anything and does NOT access `final_test` --
it only reads the already-saved evidence (`docs/_stage7_*_raw.json`, all
gitignored but still on disk) and the already-written Decisions log in
`docs/ML_WORKFLOW_STATE.md`, and writes one historical record per
genuinely distinct, identifiable experiment found in that evidence.

Granularity rule applied throughout: one record per distinct EXECUTION
(its own timing/config/result numbers), not one record per source file --
several files bundle more than one experiment (e.g. two baselines in one
file, two SAGA class-weight arms in another). The exact count is derived
from what is actually in the evidence, not assumed in advance.

Provenance discipline: every record below is `provenance="reconstructed_from_evidence"`
(or `"reconstructed_partial"` where genuinely incomplete), with
`confidence` and `source_description` set per record, and `git_commit`/
`git_dirty` deliberately left `None` -- this backfill does NOT guess which
historical commit each experiment corresponds to; that mapping was never
recorded at the time and inferring it from file-modification time vs.
commit timestamps would be a fabrication, not a measurement.

Idempotent: every record uses `deterministic_historical_run_id()`, so
re-running this script after it has already populated the registry
re-derives the same run_ids and skips them (prints "SKIP"), rather than
creating near-duplicate entries.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_tracking import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    append_run_record,
    build_run_record,
    compute_evidence_checksums,
    deterministic_historical_run_id,
    run_id_exists,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATASET_PROVENANCE = (
    "IEEE-CIS Fraud Detection (Kaggle), data/raw/ieee-fraud-detection/train_transaction.csv -- "
    "train/validation partitions only, via src.leakage_boundaries.get_partition() (final_test never accessed in Stage 7)."
)


def file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def load_json(relative_path: str) -> dict:
    return json.loads((DOCS / relative_path).read_text())


LOGGED = 0
SKIPPED = 0


def log_historical(registry_path: Path, *, source_file: str, sub_key: str | None, **kwargs) -> None:
    global LOGGED, SKIPPED
    source_description = f"{source_file} :: {sub_key}" if sub_key else source_file
    run_id = deterministic_historical_run_id(source_description)
    if run_id_exists(run_id, registry_path):
        print(f"SKIP (already backfilled): {source_description}")
        SKIPPED += 1
        return
    evidence_paths = kwargs.get("evidence_paths")
    if evidence_paths:
        absolute_to_relative = {str(ROOT / p): p for p in evidence_paths}
        checksums_by_absolute = compute_evidence_checksums(list(absolute_to_relative))
        kwargs["evidence_checksums"] = {absolute_to_relative[abs_p]: digest for abs_p, digest in checksums_by_absolute.items()}
    record = build_run_record(
        run_id=run_id,
        source_description=source_description,
        dataset_provenance=DATASET_PROVENANCE,
        git_commit=None,  # see module docstring: not reconstructed, left genuinely unknown
        git_dirty=None,
        include_environment=False,
        **kwargs,
    )
    append_run_record(record, registry_path)
    print(f"LOGGED: {run_id}  <- {source_description}")
    LOGGED += 1


# ---------------------------------------------------------------------------
# 1. docs/_stage7_baselines_raw.json -- 2 experiments
# ---------------------------------------------------------------------------

def backfill_baselines(registry_path: Path) -> None:
    source_file = "_stage7_baselines_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)

    log_historical(
        registry_path, source_file=source_file, sub_key="no_skill",
        workflow_stage=7, experiment_name="no_skill_baseline",
        experiment_purpose="Constant-score (train prevalence) baseline -- the AP floor every candidate must beat.",
        status="completed", provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts,
        experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime, used as a proxy for actual execution time (no separate run log existed before Stage 8).",
        model_family="constant_score_baseline", hyperparameters=None, random_seed=None,
        partitions={"validation": {"id": "validation", "n_rows": data["no_skill"]["n_rows"]}},
        metrics={k: v for k, v in data["no_skill"].items() if k not in ("note",)},
        metric_definitions={"average_precision": "No-skill AP equals validation's own positive-class prevalence, by construction -- verified numerically in this run, not just asserted."},
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=True, validation_purpose="the baseline floor every later candidate is compared against",
        influenced_subsequent_decisions=False,
        field_provenance={"metrics": "verified (from saved JSON)", "model_family": "verified (no_skill_baseline() function)"},
    )
    log_historical(
        registry_path, source_file=source_file, sub_key="simple_logreg",
        workflow_stage=7, experiment_name="simple_logreg_baseline",
        experiment_purpose="Small 3-feature (TransactionAmt, ProductCD, card4) Logistic Regression baseline.",
        status="completed", provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts,
        experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="LogisticRegression", hyperparameters={"features": ["TransactionAmt", "ProductCD", "card4"]}, random_seed=None,
        preprocessing_config={"pipeline": "build_simple_baseline_pipeline"},
        partitions={"validation": {"id": "validation", "n_rows": data["simple_logreg"]["n_rows"]}},
        training_time_seconds=data["simple_logreg"]["fit_seconds"], inference_time_seconds=data["simple_logreg"]["predict_seconds"],
        metrics={k: v for k, v in data["simple_logreg"].items() if k not in ("fit_seconds", "predict_seconds")},
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=True, validation_purpose="second baseline point, compared against the no-skill floor",
        influenced_subsequent_decisions=False,
        field_provenance={"metrics": "verified (from saved JSON)"},
    )


# ---------------------------------------------------------------------------
# 2. docs/_stage7_pilot_raw.json -- 2 experiments (saga_balanced, saga_unweighted)
# ---------------------------------------------------------------------------

def backfill_pilot(registry_path: Path) -> None:
    source_file = "_stage7_pilot_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    pilot = data["pilot_subset"]
    partitions = {
        "train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": pilot["n_rows_actual"], "fraud_prevalence": pilot["fraud_prevalence"]},
        "validation": {"id": "validation", "n_rows": 75560},
    }
    for key, label in (("saga_balanced", "class_weight=balanced"), ("saga_unweighted", "class_weight=None")):
        cfg = data["saga_configs"][key]
        log_historical(
            registry_path, source_file=source_file, sub_key=key,
            workflow_stage=7, experiment_name=f"logreg_pilot_{key}",
            experiment_purpose=f"Bounded 15,000-row chronological pilot, solver=saga, {label} -- first SAGA attempt after the liblinear hang diagnosis.",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="LogisticRegression",
            hyperparameters={"solver": cfg["solver"], "penalty": cfg["penalty"], "C": cfg["C"], "max_iter": cfg["max_iter"], "class_weight": cfg["class_weight"]},
            random_seed=42,
            preprocessing_config={"pipeline": "build_scaled_pipeline_with_collapsed_indicators", "output_shape": data["pilot_pipeline"]["output_shape"]},
            partitions=partitions, feature_count=data["pilot_pipeline"]["output_shape"][1],
            training_time_seconds=cfg["fit_seconds"],
            metrics={"converged": cfg["converged"], "n_iter": cfg["n_iter"], "average_precision": cfg["validation_average_precision"], "roc_auc": cfg["validation_roc_auc"]},
            status_reason=None if cfg["converged"] else cfg["convergence_warning"],
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=True,
            validation_purpose="compared across the two class_weight arms to recommend which configuration to carry forward into the next investigation round",
            influenced_subsequent_decisions=True,
            field_provenance={"metrics": "verified (from saved JSON)", "hyperparameters": "verified (from saved JSON)"},
        )


# ---------------------------------------------------------------------------
# 3. docs/_stage7_convergence_diagnostic_raw.json -- 1 structural bundle +
#    2 loss-trajectory runs + 4 solver-comparison runs = 7 experiments
# ---------------------------------------------------------------------------

def backfill_convergence_diagnostic(registry_path: Path) -> None:
    source_file = "_stage7_convergence_diagnostic_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    partitions = {"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": 15000}}

    log_historical(
        registry_path, source_file=source_file, sub_key="structural_diagnostics",
        workflow_stage=7, experiment_name="convergence_structural_diagnostics",
        experiment_purpose="Near-constant columns, exact-duplicate-value columns, max pairwise correlation, sparse-matrix magnitude summary -- data-structure diagnostics, no model fit.",
        status="diagnostic", status_reason="Structural/data-conditioning analysis, not a model fit -- 'diagnostic' is the correct status, not 'completed' used for model runs.",
        provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        preprocessing_config={"pipeline": "build_scaled_pipeline_with_collapsed_indicators", "fit_seconds": data["pipeline_fit_seconds"]},
        partitions=partitions,
        metrics={
            "near_constant_columns_count": data["near_constant_columns"]["count"],
            "duplicate_value_n_groups": data["duplicate_value_columns"]["n_groups"],
            "duplicate_value_n_columns_involved": data["duplicate_value_columns"]["n_columns_involved"],
            "max_pairwise_correlation": data["max_pairwise_correlation"]["max_abs_correlation"],
            "n_pairs_above_0.99": data["max_pairwise_correlation"]["n_pairs_above_0.99"],
            "sparse_density": data["sparse_magnitude"]["density"],
        },
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=False,
        field_provenance={"metrics": "verified (from saved JSON)"},
    )

    for weighting in ("balanced", "unweighted"):
        trace = data["saga_loss_trajectory"][weighting]
        log_historical(
            registry_path, source_file=source_file, sub_key=f"saga_loss_trajectory_{weighting}",
            workflow_stage=7, experiment_name=f"saga_loss_trajectory_{weighting}",
            experiment_purpose=f"Warm-start, per-iteration SAGA training-loss trajectory, class_weight={weighting} -- used to investigate the (later partially-corrected) 'oscillating loss' claim.",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="LogisticRegression",
            hyperparameters={"solver": "saga", "class_weight": weighting, "max_iter": 1000},
            random_seed=42, partitions=partitions,
            training_time_seconds=trace["total_seconds"],
            metrics={"final_train_log_loss": trace["trace"][-1]["train_log_loss"], "loss_trace": trace["trace"]},
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=False,
            validation_purpose="this specific trace does not report a validation AP -- see the sibling solver_comparison entry, which does",
            field_provenance={"metrics": "verified (from saved JSON)"},
        )

    for key, cfg in data["solver_comparison"].items():
        log_historical(
            registry_path, source_file=source_file, sub_key=f"solver_comparison_{key}",
            workflow_stage=7, experiment_name=f"solver_comparison_{key}",
            experiment_purpose="Bounded SAGA-vs-SGDClassifier(log_loss) comparison, same 15,000-row pilot, isolating solver choice from class_weight.",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="LogisticRegression" if "sgd" not in key else "SGDClassifier",
            hyperparameters={"solver": cfg["solver"], "class_weight": cfg["class_weight"]},
            random_seed=42, partitions=partitions,
            training_time_seconds=cfg["fit_seconds"],
            metrics={
                "n_iter": cfg["n_iter"], "converged": cfg["converged"], "train_log_loss": cfg["train_log_loss"],
                "average_precision": cfg["validation_average_precision_for_continuity_only"],
            },
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=True,
            validation_purpose="explicitly labeled 'for_continuity_only' in the original evidence -- reported alongside the comparison, not used to pick a configuration",
            influenced_subsequent_decisions=False,
            field_provenance={"metrics": "verified (from saved JSON)"},
        )


# ---------------------------------------------------------------------------
# 4. docs/_stage7_numeric_conditioning_raw.json -- 4 experiments
#    (arm_a, arm_b, and 2 exploratory C-value checks)
# ---------------------------------------------------------------------------

def backfill_numeric_conditioning(registry_path: Path) -> None:
    source_file = "_stage7_numeric_conditioning_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    partitions = {"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": 15000}}

    for arm_key, label in (("arm_a", "unchanged 542-column representation"), ("arm_b", "494-column, 48 redundant numeric columns dropped")):
        arm = data[arm_key]
        log_historical(
            registry_path, source_file=source_file, sub_key=arm_key,
            workflow_stage=7, experiment_name=f"numeric_conditioning_{arm_key}",
            experiment_purpose=f"Controlled A/B: {label}. Same solver/seed/class_weight=None, isolating numeric-redundancy reduction as the only variable.",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="LogisticRegression",
            hyperparameters={"solver": "saga", "class_weight": None, "max_iter": 1000},
            random_seed=42,
            preprocessing_config={"pipeline": "build_scaled_pipeline_with_collapsed_indicators_and_value_reduction" if arm_key == "arm_b" else "build_scaled_pipeline_with_collapsed_indicators", "fit_seconds": arm["pipeline_fit_seconds"]},
            partitions=partitions, feature_count=arm["sparse_magnitude"]["shape"][1],
            training_time_seconds=arm["fit_seconds"],
            metrics={"n_iter": arm["n_iter"], "converged": arm["converged"], "average_precision": arm["validation_ap_continuity_only"], "sparse_density": arm["sparse_magnitude"]["density"]},
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=True,
            validation_purpose="reported once per instruction, explicitly 'continuity only' -- the decision not to adopt numeric reduction was based on fit-time/convergence behavior, not this AP",
            influenced_subsequent_decisions=False,
            field_provenance={"metrics": "verified (from saved JSON)", "loss_trajectory": "linked from the sibling loss_trajectories.{arm} entry in the same file -- confidence medium on this specific linkage, since the source JSON does not state outright whether the trace was captured from this exact fit or a separate warm-start pass with identical config"},
        )

    for c_key, cfg in data["exploratory_regularization_check_arm_b"].items():
        c_value = c_key.split("_")[1]
        log_historical(
            registry_path, source_file=source_file, sub_key=f"exploratory_regularization_{c_key}",
            workflow_stage=7, experiment_name=f"numeric_conditioning_arm_b_C_{c_value}",
            experiment_purpose=f"Small, explicitly-exploratory (not a search) check of C={c_value} on Arm B's 494-column representation.",
            status="diagnostic", status_reason="Explicitly labeled 'exploratory_regularization_check' in the original evidence -- a single predefined sanity check, not a result used for selection.",
            provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="LogisticRegression",
            hyperparameters={"solver": "saga", "class_weight": None, "C": float(c_value), "max_iter": 1000},
            random_seed=42, partitions=partitions,
            training_time_seconds=cfg["fit_seconds"],
            metrics={"n_iter": cfg["n_iter"], "converged": cfg["converged"]},
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=False,
            field_provenance={"metrics": "verified (from saved JSON)"},
        )


# ---------------------------------------------------------------------------
# 5. docs/_stage7_final_saga_verification_raw.json -- 1 experiment
# ---------------------------------------------------------------------------

def backfill_final_saga_verification(registry_path: Path) -> None:
    source_file = "_stage7_final_saga_verification_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    log_historical(
        registry_path, source_file=source_file, sub_key=None,
        workflow_stage=7, experiment_name="logreg_final_saga_verification",
        experiment_purpose="ONE definitive check: max_iter raised to 3000 (3x) on the unchanged 542-feature representation -- ruling out 'just needs a few more iterations' as a practical fix.",
        status="completed", provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="LogisticRegression", hyperparameters=data["config"], random_seed=data["config"]["random_state"],
        preprocessing_config={"pipeline": "build_scaled_pipeline_with_collapsed_indicators"},
        partitions={"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": data["pilot_n_rows"], "fraud_prevalence": data["pilot_fraud_prevalence"]}},
        feature_count=data["matrix_shape"][1],
        training_time_seconds=data["fit_seconds"],
        metrics={"n_iter": data["n_iter"], "converged": data["converged"], "final_train_log_loss": data["final_train_log_loss"]},
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=False,
        validation_purpose="validation was structurally never loaded by this script at all -- not just unused, absent by construction.",
        influenced_subsequent_decisions=False,
        field_provenance={"metrics": "verified (from saved JSON)", "hyperparameters": "verified (from saved JSON config block)"},
    )


# ---------------------------------------------------------------------------
# 6/7. Tree pilots -- 1 experiment each
# ---------------------------------------------------------------------------

def backfill_tree_pilot(registry_path: Path, source_file: str, model_label: str) -> None:
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    log_historical(
        registry_path, source_file=source_file, sub_key=None,
        workflow_stage=7, experiment_name=f"{model_label}_pilot",
        experiment_purpose=f"{model_label.upper()} pilot on the 15,000-row chronological subset, approved (unscaled) Stage 5 pipeline.",
        status="completed", provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="RandomForestClassifier" if model_label == "rf" else "GradientBoostingClassifier",
        hyperparameters=data["config"], random_seed=data["config"]["random_state"],
        preprocessing_config={"pipeline": "build_preprocessing_pipeline (approved, unscaled)"},
        partitions={"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": data["pilot_n_rows"], "fraud_prevalence": data["pilot_fraud_prevalence"]}, "validation": {"id": "validation", "n_rows": data["metrics"]["n_rows"]}},
        feature_count=data["matrix_shape"][1],
        training_time_seconds=data["fit_seconds"], inference_time_seconds=data["predict_seconds"],
        metrics=data["metrics"],
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=True, validation_purpose="compare RF vs GBM pilot performance",
        influenced_subsequent_decisions=True,
        field_provenance={"metrics": "verified (from saved JSON)", "hyperparameters": "verified (from saved JSON config block)"},
    )


# ---------------------------------------------------------------------------
# 8. RF pilot independent verification -- 1 experiment
# ---------------------------------------------------------------------------

def backfill_rf_pilot_verification(registry_path: Path) -> None:
    source_file = "_stage7_rf_pilot_verification_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    log_historical(
        registry_path, source_file=source_file, sub_key=None,
        workflow_stage=7, experiment_name="rf_pilot_independent_verification",
        experiment_purpose="Independent 8-point re-derivation of the RF pilot (row count, label alignment, pipeline structure, tree-fit completeness, depth/leaf plausibility, param match, no hidden repeated work, validation-only eval).",
        status="completed" if data.get("all_checks_passed") else "failed",
        status_reason=None if data.get("all_checks_passed") else "one or more of the 8 verification checks failed",
        provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="RandomForestClassifier",
        preprocessing_config={"pipeline": "build_preprocessing_pipeline (approved, unscaled)", "source_column_count": data["check3_source_column_count"]},
        feature_count=data["fitted_output_shape"][1],
        training_time_seconds=data["check7_isolated_refit_seconds"],
        metrics={k: v for k, v in data.items() if k.startswith("check")},
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=True, validation_purpose="re-verified that evaluation used exactly the validation partition with zero final_test overlap",
        influenced_subsequent_decisions=False,
        field_provenance={"metrics": "verified (from saved JSON)"},
    )


# ---------------------------------------------------------------------------
# 9. RF full-train benchmark -- 1 experiment
# ---------------------------------------------------------------------------

def backfill_rf_full_train(registry_path: Path) -> None:
    source_file = "_stage7_rf_full_train_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    log_historical(
        registry_path, source_file=source_file, sub_key=None,
        workflow_stage=7, experiment_name="rf_full_train_benchmark",
        experiment_purpose="Full 438,125-row train partition, identical settings to the verified pilot -- the strongest, most-verified Stage 7 result.",
        status="completed", provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="RandomForestClassifier",
        hyperparameters={"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42},
        random_seed=42,
        preprocessing_config={"pipeline": "build_preprocessing_pipeline (approved, unscaled)"},
        partitions={"train": {"id": "train (complete)", "n_rows": data["train_n_rows"], "fraud_prevalence": data["train_fraud_prevalence"]}, "validation": {"id": "validation", "n_rows": data["metrics"]["n_rows"]}},
        feature_count=data["fitted_feature_count"],
        feature_schema_fingerprint=None,
        runtime_settings={"timeout_budget_seconds": 600, "isolated_subprocess": True, "note": "ulimit -v confirmed non-functional on macOS; wall-clock timeout was the real enforced safety net"},
        training_time_seconds=data["fit_seconds"], inference_time_seconds=data["predict_seconds"],
        peak_memory_bytes=16_360_797_024,  # from /usr/bin/time -l, recorded in ML_WORKFLOW_STATE.md Decisions log; not itself in this JSON file
        metrics=data["metrics"],
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=True,
        validation_purpose="the headline reported Stage 7 result",
        influenced_subsequent_decisions=True,
        field_provenance={
            "metrics": "verified (from saved JSON)",
            "peak_memory_bytes": "reconstructed from ML_WORKFLOW_STATE.md Decisions log prose (/usr/bin/time -l output quoted there), not present in this JSON file itself -- confidence high since it was a directly-reported measured number, just saved in a different file",
        },
    )


# ---------------------------------------------------------------------------
# 10. GBM feasibility probe -- 3 measured sub-runs + 1 infeasible-not-run decision
# ---------------------------------------------------------------------------

def backfill_gb_feasibility(registry_path: Path) -> None:
    source_file = "_stage7_gb_feasibility_probe_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    for m in data["measured"]:
        log_historical(
            registry_path, source_file=source_file, sub_key=f"probe_{m['n_rows']}_rows",
            workflow_stage=7, experiment_name=f"gb_feasibility_probe_{m['n_rows']}_rows",
            experiment_purpose="One measured point of the GBM full-data feasibility power-law fit (NOT assuming linear scaling).",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="GradientBoostingClassifier",
            hyperparameters={"n_estimators": 100, "max_depth": 3, "random_state": 42},
            random_seed=42,
            preprocessing_config={"pipeline": "build_preprocessing_pipeline (approved, unscaled)"},
            partitions={"train_pilot": {"id": f"train (earliest {m['n_rows']} rows, chronological)", "n_rows": m["n_rows"]}},
            feature_count=m["shape"][1],
            training_time_seconds=m["fit_seconds"],
            metrics=None,
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=False,
            validation_purpose="this probe measures fit time only -- no validation scoring occurs",
            field_provenance={"training_time_seconds": "verified (from saved JSON)"},
        )
    log_historical(
        registry_path, source_file=source_file, sub_key="full_data_decision",
        workflow_stage=7, experiment_name="gb_full_data_training",
        experiment_purpose="The full-data (438,125-row) Gradient Boosting benchmark itself -- never attempted.",
        status="infeasible_not_run",
        status_reason=(
            f"Power-law extrapolation (exponent k={data['fitted_exponent_k']}, worse than linear) with a 1.5x safety margin "
            f"estimated {data['conservative_estimate_seconds']}s against a {data['practical_budget_seconds']}s practical budget -- "
            "exceeds the budget, so the run was deliberately not launched, per explicit instruction not to launch an expensive run "
            "once infeasibility is documented. This is a resource-budget decision, not a predictive-performance judgment."
        ),
        provenance="reconstructed_from_evidence", confidence="high",
        experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
        model_family="GradientBoostingClassifier",
        hyperparameters={"n_estimators": 100, "max_depth": 3, "random_state": 42},
        partitions={"train": {"id": "train (complete)", "n_rows": data["full_train_n_rows"]}},
        runtime_settings={"practical_budget_seconds": data["practical_budget_seconds"], "conservative_estimate_seconds": data["conservative_estimate_seconds"], "fitted_scaling_exponent_k": data["fitted_exponent_k"]},
        metrics=None,
        evidence_paths=[f"docs/{source_file}"],
        validation_consulted=False,
        field_provenance={"status_reason": "verified (derived from saved JSON's own extrapolation fields)"},
    )


# ---------------------------------------------------------------------------
# 11/12. RF ablations -- 2 experiments each
# ---------------------------------------------------------------------------

def backfill_feature_ablation(registry_path: Path) -> None:
    source_file = "_stage7_rf_feature_ablation_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    for arm_key, label, purpose_extra in (
        ("arm_a", "Stage 5 preprocessing alone", "the reference arm"),
        ("arm_b", "Stage 5 + Stage 6 engineered features", "point-in-time-correct entity-history computed per the module docstring's reasoning"),
    ):
        arm = data[arm_key]
        log_historical(
            registry_path, source_file=source_file, sub_key=arm_key,
            workflow_stage=7, experiment_name=f"rf_feature_ablation_{arm_key}",
            experiment_purpose=f"Feature-engineering ablation, Random Forest, {label} ({purpose_extra}).",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="RandomForestClassifier",
            hyperparameters={"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42},
            random_seed=42,
            preprocessing_config={"pipeline": "build_preprocessing_pipeline" + (" + build_engineered_feature_block (Stage 6)" if arm_key == "arm_b" else "")},
            partitions={"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": 15000}, "validation": {"id": "validation", "n_rows": arm["metrics"]["n_rows"]}},
            feature_count=arm["shape"][1],
            training_time_seconds=arm["fit_seconds"], inference_time_seconds=arm["predict_seconds"],
            metrics=arm["metrics"],
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=True,
            validation_purpose="compare whether Stage 6 engineered features add measurable value",
            influenced_subsequent_decisions=False,
            field_provenance={"metrics": "verified (from saved JSON)"},
        )


def backfill_indicator_ablation(registry_path: Path) -> None:
    source_file = "_stage7_rf_indicator_ablation_raw.json"
    data = load_json(source_file)
    ts = file_mtime_iso(DOCS / source_file)
    for arm_key, label in (("original", "approved full indicator representation"), ("collapsed", "Stage 6 collapsed-indicator prototype")):
        arm = data[arm_key]
        log_historical(
            registry_path, source_file=source_file, sub_key=arm_key,
            workflow_stage=7, experiment_name=f"rf_indicator_ablation_{arm_key}",
            experiment_purpose=f"Missingness-indicator ablation, Random Forest, {label}.",
            status="completed", provenance="reconstructed_from_evidence", confidence="high",
            experiment_timestamp=ts, experiment_timestamp_provenance="reconstructed: evidence file's filesystem mtime.",
            model_family="RandomForestClassifier",
            hyperparameters={"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42},
            random_seed=42,
            preprocessing_config={"pipeline": "build_preprocessing_pipeline" if arm_key == "original" else "build_preprocessing_pipeline_with_collapsed_indicators"},
            partitions={"train_pilot": {"id": "train (earliest 15,000 rows, chronological)", "n_rows": 15000}, "validation": {"id": "validation", "n_rows": arm["metrics"]["n_rows"]}},
            feature_count=arm["shape"][1],
            training_time_seconds=arm["fit_seconds"], inference_time_seconds=arm["predict_seconds"],
            metrics=arm["metrics"],
            evidence_paths=[f"docs/{source_file}"],
            validation_consulted=True,
            validation_purpose="compare the approved full-indicator representation against the collapsed prototype",
            influenced_subsequent_decisions=False,
            field_provenance={"metrics": "verified (from saved JSON)"},
        )


def main() -> None:
    registry_path = DEFAULT_REGISTRY_PATH
    backfill_baselines(registry_path)
    backfill_pilot(registry_path)
    backfill_convergence_diagnostic(registry_path)
    backfill_numeric_conditioning(registry_path)
    backfill_final_saga_verification(registry_path)
    backfill_tree_pilot(registry_path, "_stage7_tree_pilot_rf_raw.json", "rf")
    backfill_tree_pilot(registry_path, "_stage7_tree_pilot_gb_raw.json", "gb")
    backfill_rf_pilot_verification(registry_path)
    backfill_rf_full_train(registry_path)
    backfill_gb_feasibility(registry_path)
    backfill_feature_ablation(registry_path)
    backfill_indicator_ablation(registry_path)
    print(f"\nTotal: {LOGGED} logged, {SKIPPED} already present (skipped). Registry: {registry_path}")


if __name__ == "__main__":
    main()
