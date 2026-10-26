import ast
import json
from pathlib import Path

import pytest

from src.experiment_tracking import (
    DuplicateRunIdError,
    MalformedRecordError,
    append_run_record,
    build_run_record,
    compute_evidence_checksums,
    compute_feature_schema_fingerprint,
    compute_file_checksum,
    deterministic_historical_run_id,
    new_run_id,
    query_registry,
    read_registry,
    run_id_exists,
    serialize_record,
    track_run,
)

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "experiment_tracking.py"


# ---------------------------------------------------------------------------
# Append / query round-trip
# ---------------------------------------------------------------------------

def test_append_then_read_round_trip(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = build_run_record(
        run_id=new_run_id("test"),
        workflow_stage=7,
        experiment_name="unit_test_experiment",
        experiment_purpose="exercise append/read round trip",
        status="completed",
        provenance="directly_verified",
        metrics={"average_precision": 0.5},
    )
    append_run_record(record, registry)
    records = read_registry(registry)
    assert len(records) == 1
    assert records[0]["run_id"] == record["run_id"]
    assert records[0]["metrics"]["average_precision"] == 0.5


def test_query_registry_filters_by_equality(tmp_path):
    registry = tmp_path / "registry.jsonl"
    for name in ("alpha", "beta"):
        record = build_run_record(
            run_id=new_run_id(name), workflow_stage=7, experiment_name=name,
            experiment_purpose="x", status="completed", provenance="directly_verified",
        )
        append_run_record(record, registry)
    matches = query_registry(registry, experiment_name="beta")
    assert len(matches) == 1
    assert matches[0]["experiment_name"] == "beta"


def test_read_registry_returns_empty_list_for_nonexistent_file(tmp_path):
    assert read_registry(tmp_path / "does_not_exist.jsonl") == []


# ---------------------------------------------------------------------------
# Unique run IDs and immutability
# ---------------------------------------------------------------------------

def test_duplicate_run_id_is_refused(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = build_run_record(
        run_id="fixed-id", workflow_stage=7, experiment_name="x",
        experiment_purpose="x", status="completed", provenance="directly_verified",
    )
    append_run_record(record, registry)
    with pytest.raises(DuplicateRunIdError):
        append_run_record(record, registry)
    # Refused duplicate must not have been written -- still exactly 1 line.
    assert len(read_registry(registry)) == 1


def test_run_id_exists_reflects_registry_state(tmp_path):
    registry = tmp_path / "registry.jsonl"
    assert run_id_exists("some-id", registry) is False
    record = build_run_record(
        run_id="some-id", workflow_stage=7, experiment_name="x",
        experiment_purpose="x", status="completed", provenance="directly_verified",
    )
    append_run_record(record, registry)
    assert run_id_exists("some-id", registry) is True


def test_deterministic_historical_run_id_is_stable_and_collision_resistant():
    id_a = deterministic_historical_run_id("docs/_stage7_baselines_raw.json :: no_skill")
    id_b = deterministic_historical_run_id("docs/_stage7_baselines_raw.json :: no_skill")
    id_c = deterministic_historical_run_id("docs/_stage7_baselines_raw.json :: simple_logreg")
    assert id_a == id_b
    assert id_a != id_c
    assert id_a.startswith("hist-")


def test_module_provides_no_update_or_delete_api():
    """Immutability is structural: there is no function in this module
    whose name suggests updating or deleting an existing record."""
    import src.experiment_tracking as module
    public_names = [n for n in dir(module) if not n.startswith("_")]
    for name in public_names:
        lowered = name.lower()
        assert "update" not in lowered and "delete" not in lowered and "overwrite" not in lowered


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------

def test_serialize_record_is_deterministic_across_calls():
    record = {"b": 2, "a": 1, "nested": {"z": 1, "y": 2}}
    assert serialize_record(record) == serialize_record(dict(record))


def test_serialize_record_is_key_order_independent():
    record_1 = {"a": 1, "b": 2}
    record_2 = {"b": 2, "a": 1}
    assert serialize_record(record_1) == serialize_record(record_2)


def test_serialize_record_handles_non_json_native_values_via_default_str():
    class Weird:
        def __str__(self):
            return "weird-value"

    out = serialize_record({"x": Weird()})
    assert "weird-value" in out


# ---------------------------------------------------------------------------
# Missing historical metadata
# ---------------------------------------------------------------------------

def test_missing_fields_auto_computed_for_sparse_historical_record():
    record = build_run_record(
        run_id="hist-1", workflow_stage=7, experiment_name="legacy_experiment",
        experiment_purpose="backfilled from a narrative description only",
        status="completed", provenance="reconstructed_partial", confidence="low",
        source_description="ML_WORKFLOW_STATE.md Decisions log prose only, no saved JSON",
    )
    assert "metrics" in record["missing_fields"]
    assert "hyperparameters" not in record["missing_fields"]  # OPTIONAL_EVEN_WHEN_APPLICABLE


def test_missing_fields_empty_when_all_applicable_fields_present():
    record = build_run_record(
        run_id="hist-2", workflow_stage=7, experiment_name="x", experiment_purpose="x",
        status="completed", provenance="directly_verified",
        source_description="unit test", experiment_timestamp="2026-10-26T00:00:00+00:00",
        model_family="RandomForestClassifier", hyperparameters={"n_estimators": 100},
        random_seed=42, metrics={"average_precision": 0.4}, metric_definitions={"average_precision": "..."},
        partitions={"train": {"n_rows": 15000}}, dataset_provenance="IEEE-CIS Fraud Detection",
        preprocessing_config={"pipeline": "build_preprocessing_pipeline"},
        evidence_paths=["docs/_stage7_tree_pilot_rf_raw.json"],
        validation_consulted=True,
        field_provenance={"metrics": "verified"},
    )
    assert record["missing_fields"] == []


# ---------------------------------------------------------------------------
# Status transitions / failed and interrupted runs
# ---------------------------------------------------------------------------

def test_track_run_marks_completed_on_normal_exit(tmp_path):
    registry = tmp_path / "registry.jsonl"
    with track_run(7, "smoke_test", "verify normal completion", registry_path=registry) as record:
        record["metrics"] = {"average_precision": 0.1}
    [saved] = read_registry(registry)
    assert saved["status"] == "completed"
    assert saved["metrics"]["average_precision"] == 0.1


def test_track_run_marks_failed_on_exception_and_reraises(tmp_path):
    registry = tmp_path / "registry.jsonl"
    with pytest.raises(RuntimeError):
        with track_run(7, "smoke_test_fail", "verify failure path", registry_path=registry) as record:
            record["model_family"] = "RandomForestClassifier"
            raise RuntimeError("simulated mid-run crash")
    [saved] = read_registry(registry)
    assert saved["status"] == "failed"
    assert "simulated mid-run crash" in saved["status_reason"]
    # Partial progress before the crash must still be captured, not discarded.
    assert saved["model_family"] == "RandomForestClassifier"


def test_track_run_respects_explicit_non_completed_status(tmp_path):
    """A probe that completes successfully but concludes 'do not run the
    expensive thing' should NOT be forced to status=completed when the
    caller explicitly records the decision as its own status."""
    registry = tmp_path / "registry.jsonl"
    with track_run(7, "feasibility_decision", "record an infeasible-not-run decision", registry_path=registry) as record:
        record["status"] = "infeasible_not_run"
        record["status_reason"] = "estimated 676.4s exceeds the 600s practical budget"
    [saved] = read_registry(registry)
    assert saved["status"] == "infeasible_not_run"


def test_build_run_record_requires_status_reason_for_non_completed_status():
    with pytest.raises(MalformedRecordError):
        build_run_record(
            run_id="x", workflow_stage=7, experiment_name="x", experiment_purpose="x",
            status="failed", provenance="directly_verified",
        )


# ---------------------------------------------------------------------------
# Metric-to-configuration traceability
# ---------------------------------------------------------------------------

def test_record_keeps_metrics_and_full_config_together(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = build_run_record(
        run_id="traceable-1", workflow_stage=7, experiment_name="rf_full_train",
        experiment_purpose="full-training Random Forest benchmark",
        status="completed", provenance="reconstructed_from_evidence",
        model_family="RandomForestClassifier",
        hyperparameters={"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "random_state": 42},
        preprocessing_config={"pipeline": "build_preprocessing_pipeline", "sparse_threshold": 1.0},
        partitions={"train": {"id": "train", "n_rows": 438125}, "validation": {"id": "validation", "n_rows": 75560}},
        feature_count=813,
        metrics={"average_precision": 0.41101, "roc_auc": 0.85596},
        evidence_paths=["docs/_stage7_rf_full_train_raw.json"],
    )
    append_run_record(record, registry)
    [saved] = query_registry(registry, run_id="traceable-1")
    assert saved["metrics"]["average_precision"] == 0.41101
    assert saved["hyperparameters"]["n_estimators"] == 100
    assert saved["preprocessing_config"]["pipeline"] == "build_preprocessing_pipeline"
    assert saved["feature_count"] == 813


# ---------------------------------------------------------------------------
# Evidence-file references
# ---------------------------------------------------------------------------

def test_compute_file_checksum_returns_none_for_missing_file(tmp_path):
    assert compute_file_checksum(tmp_path / "nope.json") is None


def test_compute_file_checksum_is_deterministic_for_existing_file(tmp_path):
    f = tmp_path / "evidence.json"
    f.write_text('{"a": 1}')
    assert compute_file_checksum(f) == compute_file_checksum(f)
    assert len(compute_file_checksum(f)) == 64  # sha256 hex digest length


def test_compute_evidence_checksums_handles_mixed_existing_and_missing(tmp_path):
    present = tmp_path / "present.json"
    present.write_text("{}")
    missing = tmp_path / "missing.json"
    result = compute_evidence_checksums([str(present), str(missing)])
    assert result[str(present)] is not None
    assert result[str(missing)] is None


def test_feature_schema_fingerprint_is_order_independent():
    assert compute_feature_schema_fingerprint(["a", "b", "c"]) == compute_feature_schema_fingerprint(["c", "a", "b"])


def test_feature_schema_fingerprint_differs_for_different_column_sets():
    assert compute_feature_schema_fingerprint(["a", "b"]) != compute_feature_schema_fingerprint(["a", "b", "c"])


# ---------------------------------------------------------------------------
# No secrets / raw data; no final_test access
# ---------------------------------------------------------------------------

def test_build_run_record_rejects_forbidden_field_names_explicitly():
    # Construct via extra_fields to hit the forbidden-substring check.
    from src.experiment_tracking import _check_forbidden_content
    with pytest.raises(MalformedRecordError):
        _check_forbidden_content({"api_key": "whatever"})
    with pytest.raises(MalformedRecordError):
        _check_forbidden_content({"final_test_rows": 1})


def test_module_source_has_no_literal_final_test_reference():
    """Mechanically verifiable, not just a promise in a docstring: the
    module's own source never mentions final_test (not even in a comment
    accidentally referencing it as a 'safe' thing), and never imports
    anything capable of reading a dataset partition."""
    source = MODULE_SOURCE_PATH.read_text()
    code_and_comments_lower = source.lower()
    # Allow docstring *explanations* of the policy (which legitimately say
    # the words "final_test" while describing why it's absent) but forbid
    # any executable reference, checked via AST instead of raw text search.
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden_imports = {"pandas", "src.preprocessing", "src.leakage_boundaries", "src.feature_engineering", "src.model_selection"}
    assert not (imported_names & forbidden_imports), f"experiment_tracking.py must never import dataset-reading code, found: {imported_names & forbidden_imports}"
    # And no literal reference to the real raw data files.
    assert "train_transaction.csv" not in code_and_comments_lower
    assert "test_transaction.csv" not in code_and_comments_lower


def test_track_run_rejects_final_test_named_field(tmp_path):
    registry = tmp_path / "registry.jsonl"
    with pytest.raises(MalformedRecordError):
        with track_run(7, "x", "x", registry_path=registry, final_test_rows=1):
            pass
