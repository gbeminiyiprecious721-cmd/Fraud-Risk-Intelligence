import ast
from pathlib import Path

import numpy as np
import pytest

from src.model_evaluation import (
    FROZEN_RF_CONFIG,
    FinalTestAlreadyEvaluatedError,
    FinalTestUnlockRefusedError,
    REQUIRED_UNLOCK_TOKEN,
    FINAL_EVALUATION_SEARCH_ID,
    assign_amount_segment,
    bootstrap_ap_ci,
    build_frozen_model,
    check_final_test_not_already_evaluated,
    compare_to_no_skill_baseline,
    compute_amount_segment_boundaries,
    expected_cost_at_threshold,
    require_unlock_token,
    segment_report,
    sweep_cost_minimizing_thresholds,
    verify_frozen_config_unchanged,
)
from src.experiment_tracking import append_run_record, build_run_record

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "model_evaluation.py"


# ---------------------------------------------------------------------------
# Frozen config
# ---------------------------------------------------------------------------

def test_frozen_config_matches_audited_stage9_candidate():
    assert FROZEN_RF_CONFIG == {
        "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5,
        "max_features": "sqrt", "n_jobs": -1, "random_state": 42, "class_weight": None,
    }


def test_build_frozen_model_uses_exact_config():
    model = build_frozen_model()
    params = model.get_params()
    for k, v in FROZEN_RF_CONFIG.items():
        assert params[k] == v


def test_verify_frozen_config_unchanged_detects_no_drift():
    assert verify_frozen_config_unchanged() is True


# ---------------------------------------------------------------------------
# Amount segmentation -- boundaries from train-like data only
# ---------------------------------------------------------------------------

def test_compute_amount_segment_boundaries_is_quartile_like():
    train_amounts = np.arange(1, 101)  # 1..100
    boundaries = compute_amount_segment_boundaries(train_amounts, n_segments=4)
    assert len(boundaries) == 3
    assert boundaries == sorted(boundaries)


def test_assign_amount_segment_buckets_correctly():
    boundaries = [25.0, 50.0, 75.0]
    amounts = np.array([1.0, 25.0, 50.0, 99.0])
    segments = assign_amount_segment(amounts, boundaries)
    assert list(segments) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Segment report
# ---------------------------------------------------------------------------

def test_segment_report_flags_insufficient_sample():
    y_true = np.array([0, 1] * 5 + [0] * 5)  # 15 rows, small segment
    y_score = np.random.default_rng(0).random(15)
    segment_labels = np.array(["tiny"] * 15)
    report = segment_report(y_true, y_score, segment_labels, min_samples=100, min_positives=10)
    assert report["tiny"]["sufficient_sample"] is False
    assert "metrics" not in report["tiny"]
    assert report["tiny"]["n_samples"] == 15


def test_segment_report_computes_metrics_when_sufficient():
    rng = np.random.default_rng(1)
    n = 500
    y_true = (rng.random(n) < 0.1).astype(int)
    y_score = rng.random(n)
    segment_labels = np.array(["big"] * n)
    report = segment_report(y_true, y_score, segment_labels, min_samples=100, min_positives=10)
    assert report["big"]["sufficient_sample"] is True
    assert "metrics" in report["big"]
    assert "average_precision" in report["big"]["metrics"]


def test_segment_report_covers_every_distinct_segment():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    segment_labels = np.array(["a", "a", "b", "b", "c", "c"])
    report = segment_report(y_true, y_score, segment_labels, min_samples=1, min_positives=0)
    assert set(report.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ap_ci_is_deterministic_with_fixed_seed():
    rng = np.random.default_rng(2)
    y_true = (rng.random(300) < 0.2).astype(int)
    y_score = rng.random(300)
    a = bootstrap_ap_ci(y_true, y_score, n_resamples=200, seed=42)
    b = bootstrap_ap_ci(y_true, y_score, n_resamples=200, seed=42)
    assert a == b


def test_bootstrap_ap_ci_bounds_sensible():
    rng = np.random.default_rng(3)
    y_true = (rng.random(300) < 0.2).astype(int)
    y_score = rng.random(300)
    result = bootstrap_ap_ci(y_true, y_score, n_resamples=200, seed=42)
    assert 0.0 <= result["lower"] <= result["point_estimate"] <= result["upper"] <= 1.0


# ---------------------------------------------------------------------------
# Cost-sensitive threshold sweep
# ---------------------------------------------------------------------------

def test_expected_cost_at_threshold_zero_when_perfect_and_threshold_matches():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.0, 1.0, 1.0])
    amounts = np.array([10.0, 10.0, 100.0, 100.0])
    cost = expected_cost_at_threshold(y_true, y_score, amounts, threshold=0.5, fn_fraction_of_amount=1.0, fp_cost_dollars=5.0)
    assert cost == 0.0


def test_expected_cost_at_threshold_counts_false_negatives_and_positives():
    y_true = np.array([0, 1])
    y_score = np.array([0.9, 0.1])  # both wrong at threshold 0.5
    amounts = np.array([20.0, 50.0])
    cost = expected_cost_at_threshold(y_true, y_score, amounts, threshold=0.5, fn_fraction_of_amount=1.0, fp_cost_dollars=5.0)
    # FN: the fraud row (amount 50) missed -> cost 50*1.0 = 50; FP: the non-fraud row flagged -> cost 5
    assert cost == pytest.approx(55.0)


def test_sweep_cost_minimizing_thresholds_covers_all_scenarios_and_assumptions():
    rng = np.random.default_rng(4)
    n = 200
    y_true = (rng.random(n) < 0.1).astype(int)
    y_score = rng.random(n)
    amounts = rng.uniform(5, 500, n)
    result = sweep_cost_minimizing_thresholds(y_true, y_score, amounts, thresholds=[0.1, 0.5, 0.9], unit_dollar_assumptions=[5.0, 10.0])
    assert set(result.keys()) == {"A_conservative_loss", "B_full_transaction_value", "C_capacity_constrained"}
    for scenario_result in result.values():
        assert set(scenario_result.keys()) == {"unit_dollars_5", "unit_dollars_10"}
        for assumption_result in scenario_result.values():
            assert assumption_result["best_threshold"] in [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Stage 1 success criterion #2 -- beat the no-skill baseline
# ---------------------------------------------------------------------------

def test_compare_to_no_skill_baseline_reports_each_margin_independently():
    model_metrics = {"average_precision": 0.5, "k_frac_0.01": {"recall_at_k": 0.3}, "k_frac_0.05": {"recall_at_k": 0.6}}
    no_skill_metrics = {"average_precision": 0.03, "k_frac_0.01": {"recall_at_k": 0.01}, "k_frac_0.05": {"recall_at_k": 0.05}}
    result = compare_to_no_skill_baseline(model_metrics, no_skill_metrics)
    assert result["beats_baseline_on_ap"] is True
    assert result["beats_baseline_on_recall_1pct"] is True
    assert result["beats_baseline_on_recall_5pct"] is True
    assert result["ap_margin"] == pytest.approx(0.47)


def test_compare_to_no_skill_baseline_can_report_a_failure_honestly():
    model_metrics = {"average_precision": 0.02, "k_frac_0.01": {"recall_at_k": 0.3}, "k_frac_0.05": {"recall_at_k": 0.6}}
    no_skill_metrics = {"average_precision": 0.03, "k_frac_0.01": {"recall_at_k": 0.01}, "k_frac_0.05": {"recall_at_k": 0.05}}
    result = compare_to_no_skill_baseline(model_metrics, no_skill_metrics)
    assert result["beats_baseline_on_ap"] is False
    assert result["ap_margin"] < 0


# ---------------------------------------------------------------------------
# One-time final_test unlock mechanism
# ---------------------------------------------------------------------------

def test_require_unlock_token_accepts_exact_token():
    require_unlock_token(REQUIRED_UNLOCK_TOKEN)  # must not raise


def test_require_unlock_token_rejects_wrong_token():
    with pytest.raises(FinalTestUnlockRefusedError):
        require_unlock_token("wrong-token")


def test_require_unlock_token_rejects_empty_or_none_like_values():
    with pytest.raises(FinalTestUnlockRefusedError):
        require_unlock_token("")


def test_check_final_test_not_already_evaluated_passes_when_registry_empty(tmp_path):
    registry = tmp_path / "registry.jsonl"
    check_final_test_not_already_evaluated(registry_path=registry)  # must not raise


def test_check_final_test_not_already_evaluated_refuses_duplicate(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = build_run_record(
        run_id="final-test-eval-1", workflow_stage=10, experiment_name="stage10_final_test_evaluation",
        experiment_purpose="the one-time final evaluation", status="completed", provenance="directly_verified",
        search_id=FINAL_EVALUATION_SEARCH_ID,
    )
    append_run_record(record, registry)
    with pytest.raises(FinalTestAlreadyEvaluatedError):
        check_final_test_not_already_evaluated(registry_path=registry)


def test_check_final_test_not_already_evaluated_ignores_non_completed_records(tmp_path):
    """A FAILED prior attempt must NOT block a genuine retry -- only a
    COMPLETED evaluation counts as 'already done'."""
    registry = tmp_path / "registry.jsonl"
    record = build_run_record(
        run_id="final-test-eval-failed", workflow_stage=10, experiment_name="stage10_final_test_evaluation",
        experiment_purpose="a failed attempt", status="failed", status_reason="simulated crash",
        provenance="directly_verified", search_id=FINAL_EVALUATION_SEARCH_ID,
    )
    append_run_record(record, registry)
    check_final_test_not_already_evaluated(registry_path=registry)  # must not raise


def test_module_source_never_references_allow_final_test_literally():
    """Mechanically verifiable, not just a promise: `src/model_evaluation.py`
    itself never calls anything with `allow_final_test=True` -- that call
    exists ONLY in the separately-gated script, never in this module."""
    source = MODULE_SOURCE_PATH.read_text()
    assert "allow_final_test=True" not in source
    assert "allow_final_test = True" not in source


def test_final_evaluation_script_checks_gates_before_any_dataset_import():
    """Static proof, not just a convention: in
    `scripts/run_stage10_final_evaluation.py`'s `main()`, the calls to
    `require_unlock_token` and `check_final_test_not_already_evaluated`
    appear BEFORE any `import` statement that could reach `final_test`
    (the deferred `src.leakage_boundaries`/`src.preprocessing`/
    `src.model_selection`/`src.experiment_tracking` imports). A reordering
    that moved a dataset-touching import above the gates would fail this
    test.
    """
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_stage10_final_evaluation.py"
    tree = ast.parse(script_path.read_text())
    main_func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")

    require_token_line = None
    check_duplicate_line = None
    first_deferred_import_line = None
    for node in ast.walk(main_func):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require_unlock_token":
            require_token_line = node.lineno
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "check_final_test_not_already_evaluated":
            check_duplicate_line = node.lineno
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if first_deferred_import_line is None or node.lineno < first_deferred_import_line:
                first_deferred_import_line = node.lineno

    assert require_token_line is not None and check_duplicate_line is not None and first_deferred_import_line is not None
    assert require_token_line < first_deferred_import_line
    assert check_duplicate_line < first_deferred_import_line
    assert require_token_line < check_duplicate_line  # token check must come first (cheapest, fastest-failing gate)


def _find_allow_final_test_true_calls(py_file: Path) -> list[int]:
    """AST-based (not string-matching, which would false-positive on the
    many legitimate docstring/error-message PROSE mentions of this
    mechanism): finds actual call sites passing `allow_final_test=True`
    as a keyword argument."""
    tree = ast.parse(py_file.read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "allow_final_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    hits.append(node.lineno)
    return hits


def test_src_modules_never_actually_call_with_allow_final_test_true():
    """Every file under `src/` may MENTION `allow_final_test=True` in a
    docstring/error message (explaining the mechanism, as several already
    do) but must never actually CALL anything with it set -- that call
    site exists only in the separately-gated script, checked in the next
    test."""
    root = Path(__file__).resolve().parent.parent
    offending = {}
    for py_file in (root / "src").rglob("*.py"):
        hits = _find_allow_final_test_true_calls(py_file)
        if hits:
            offending[str(py_file)] = hits
    assert offending == {}, f"Found actual allow_final_test=True call site(s) under src/: {offending}"


def test_final_evaluation_script_is_the_only_allow_final_test_call_site():
    """The gated script is expected to have exactly the 3 call sites it
    needs (get_partition, load_partition_df, load_labels) -- confirms the
    override exists where intended, not just that it's absent elsewhere."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_stage10_final_evaluation.py"
    hits = _find_allow_final_test_true_calls(script_path)
    assert len(hits) == 3


def test_module_imports_nothing_that_reads_datasets():
    """Parses the module's own imports -- it must never import
    `src.preprocessing`/`src.leakage_boundaries`/`pandas`-level CSV access
    beyond what's already vetted (pandas itself is used for the segment
    report's pd.unique, not for reading any file)."""
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    assert "src.preprocessing" not in imported_names
    assert "src.leakage_boundaries" not in imported_names
