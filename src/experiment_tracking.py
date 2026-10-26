"""Stage 8 — a lightweight, dependency-free, append-only experiment registry.

Design, approved explicitly (no MLflow/Weights & Biases/new dependency):
one immutable JSON record per line in a JSONL file (`docs/experiment_registry.jsonl`
by default), stdlib only. Unlike the per-script `docs/_stage7_*_raw.json`
evidence dumps (gitignored, regenerable by re-running a script), this
registry file IS tracked in git — it is the durable, curated audit trail
itself, not a regenerable byproduct.

Two kinds of records this module produces:
- "directly_verified" records, written automatically by `track_run()`
  around a script's own execution, at the moment that execution happens.
- "reconstructed_from_evidence" / "reconstructed_partial" / "unknown"
  records, written by a backfill script from already-saved evidence
  (JSON dumps, the Decisions log) for experiments that ran before this
  registry existed. These NEVER claim a field that cannot be traced to
  real saved evidence -- an untraceable field is left `None` and listed
  in `missing_fields`, never guessed.

What "status" means, precisely: whether the RUN executed to completion
without crashing, timing out, or being interrupted -- NOT whether the
result was scientifically favorable. A Logistic Regression run that
hits its iteration cap without converging is `status="completed"` (it
ran to completion; the model just didn't converge) with the
non-convergence recorded in `metrics`/`status_reason` as the actual
finding, not disguised as a failed run. A run that was never attempted
at all (e.g. full-data Gradient Boosting, assessed infeasible and
deliberately not launched) gets its own record with
`status="infeasible_not_run"` and no execution-derived fields.

Final-test protection (review item 6): this module never imports from
`src.preprocessing`, `src.leakage_boundaries`, or `src.feature_engineering`,
never calls `pandas.read_csv`, and never opens any file under `data/raw/`.
It cannot read `final_test` (or any partition) because it has no code
path that reads any dataset at all -- enforced structurally, verified by
`tests/test_experiment_tracking.py`'s static-analysis test, not merely by
omitting a field from the schema.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "docs" / "experiment_registry.jsonl"

# Every field this schema recognizes. Used to (a) auto-compute `missing_fields`
# for a record (any of these whose value is None), and (b) reject attempts to
# smuggle in an unrecognized field name via `build_run_record`'s **extra_fields.
KNOWN_FIELDS = [
    "run_id", "created_at", "experiment_timestamp", "experiment_timestamp_provenance",
    "workflow_stage", "experiment_name", "experiment_purpose",
    "status", "status_reason",
    "model_family", "hyperparameters", "random_seed",
    "preprocessing_config", "feature_engineering_config",
    "partitions", "feature_count", "feature_schema_fingerprint",
    "git_commit", "git_dirty", "python_version", "dependency_versions",
    "dataset_provenance", "runtime_settings",
    "metrics", "metric_definitions",
    "training_time_seconds", "inference_time_seconds", "peak_memory_bytes",
    "evidence_paths", "evidence_checksums",
    "validation_consulted", "validation_purpose", "influenced_subsequent_decisions",
    "provenance", "field_provenance", "missing_fields", "confidence", "source_description",
    "schema_version",
    # Stage 9 extension (backward-compatible, additive only -- every Stage 7/8
    # record predates these and correctly has them as None/missing): links a
    # single hyperparameter-search trial to its parent search and CV fold,
    # so a "search" (many candidates x many folds) is reconstructable from
    # individual per-fold records without fabricating fold-level detail from
    # an aggregate.
    "search_id", "candidate_index", "fold_index",
]

# Fields that are never expected to be populated for every record (e.g. a
# structural-diagnostic run genuinely has no "model_family") -- excluded from
# the auto-computed `missing_fields` list so that list stays meaningful
# ("genuinely unknown/not captured"), not noisy with "not applicable".
OPTIONAL_EVEN_WHEN_APPLICABLE = {
    "status_reason", "model_family", "hyperparameters", "random_seed",
    "feature_engineering_config", "feature_count", "feature_schema_fingerprint",
    "runtime_settings", "training_time_seconds", "inference_time_seconds",
    "peak_memory_bytes", "evidence_checksums", "validation_purpose",
    "influenced_subsequent_decisions", "experiment_timestamp_provenance",
    "search_id", "candidate_index", "fold_index",
}

VALID_STATUSES = {"completed", "failed", "interrupted", "infeasible_not_run", "diagnostic"}
VALID_PROVENANCE = {"directly_verified", "reconstructed_from_evidence", "reconstructed_partial", "unknown"}

# Defense in depth (review item 1's "do not record ... secrets ... or
# unnecessary sensitive data" and item 6's final_test protection): a record
# or any extra field name containing one of these is refused outright,
# rather than relying only on nobody ever passing one in.
FORBIDDEN_FIELD_SUBSTRINGS = (
    "password", "secret", "token", "credential", "api_key", "apikey",
    "raw_transaction", "card_number", "ssn", "final_test",
)

DEFAULT_METRIC_DEFINITIONS = {
    "average_precision": "Area under the precision-recall curve; the primary Stage 7 metric. No-skill baseline equals the evaluated set's positive-class prevalence.",
    "roc_auc": "Area under the ROC curve; secondary metric. No-skill baseline is 0.5 regardless of class balance.",
    "precision_at_k": "True positives in the top-K ranked predictions, divided by K, for a FIXED, pre-decided K (never searched).",
    "recall_at_k": "True positives in the top-K ranked predictions, divided by the total number of actual positives, for a FIXED K.",
}


class DuplicateRunIdError(ValueError):
    """Raised when a run_id already exists in the registry -- records are
    immutable and append-only; this module provides no update/overwrite API
    at all, by design."""


class MalformedRecordError(ValueError):
    """Raised when a record fails validation before being written -- a loud
    failure beats a silently malformed line in the registry."""


# ---------------------------------------------------------------------------
# Identity, environment, and fingerprint helpers
# ---------------------------------------------------------------------------

def new_run_id(prefix: str = "run") -> str:
    """A fresh, globally-unique run ID for a run being logged live."""
    return f"{prefix}-{uuid.uuid4().hex}"


def deterministic_historical_run_id(source_description: str) -> str:
    """A STABLE run ID for a backfilled historical record, derived from a
    hash of its own source description. Re-running the backfill script is
    therefore idempotent: the same historical experiment always gets the
    same run_id, so `append_run_record`'s duplicate check naturally skips
    (or loudly flags) re-insertion rather than creating a near-duplicate
    with a fresh random ID each time.
    """
    digest = hashlib.sha256(source_description.encode("utf-8")).hexdigest()[:16]
    return f"hist-{digest}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit_and_dirty(repo_root: str | Path | None = None) -> tuple[str | None, bool | None]:
    """(commit_hash, is_dirty) for `repo_root` (default: this file's repo).
    Returns (None, None) if git itself is unavailable or the call fails --
    never raises, since this is provenance metadata, not a correctness gate.
    """
    repo_root = str(repo_root) if repo_root else str(Path(__file__).resolve().parent.parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        if commit.returncode != 0:
            return None, None
        commit_hash = commit.stdout.strip() or None
        is_dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit_hash, is_dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def get_environment_versions() -> dict:
    """Best-effort `{package: version}` for the packages this project
    actually depends on. Missing packages are simply omitted, not guessed."""
    versions: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "sklearn", "scipy"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def get_python_version() -> str:
    return platform.python_version() if hasattr(platform, "python_version") else sys.version


def compute_feature_schema_fingerprint(columns: list[str]) -> str:
    """A deterministic sha256 hex digest of a column-name set, order-independent
    (sorted before hashing) -- so two fits over the same logical column set
    always fingerprint identically regardless of the order columns happened
    to be listed in."""
    joined = "\x1f".join(sorted(columns))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_file_checksum(path: str | Path) -> str | None:
    """sha256 of a file's bytes, or None if the file does not exist (e.g. a
    gitignored evidence file that was since cleaned up) -- absence is
    reported honestly, not treated as an error."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_evidence_checksums(paths: list[str]) -> dict[str, str | None]:
    """`{path: sha256_or_None}` for each path -- None means the referenced
    evidence file was not found at checksum time, reported per-path rather
    than silently dropped or crashing the whole record."""
    return {p: compute_file_checksum(p) for p in paths}


# ---------------------------------------------------------------------------
# Record construction and validation
# ---------------------------------------------------------------------------

def _check_forbidden_content(record: dict) -> None:
    for key in record:
        lowered = key.lower()
        if any(bad in lowered for bad in FORBIDDEN_FIELD_SUBSTRINGS):
            raise MalformedRecordError(
                f"Field name '{key}' matches a forbidden substring ({FORBIDDEN_FIELD_SUBSTRINGS}) -- "
                "this registry never records secrets, credentials, or final_test-related data."
            )


def build_run_record(
    *,
    run_id: str,
    workflow_stage: int,
    experiment_name: str,
    experiment_purpose: str,
    status: str,
    provenance: str,
    status_reason: str | None = None,
    confidence: str = "high",
    source_description: str | None = None,
    experiment_timestamp: str | None = None,
    experiment_timestamp_provenance: str | None = None,
    include_environment: bool = True,
    **extra_fields,
) -> dict:
    """Build and validate one immutable run record. Does NOT write it --
    pair with `append_run_record`. Raises `MalformedRecordError` on any
    unrecognized field name, forbidden content, or invalid status/provenance
    value -- validation happens before anything is ever written.
    """
    if status not in VALID_STATUSES:
        raise MalformedRecordError(f"status={status!r} not in {sorted(VALID_STATUSES)}")
    if provenance not in VALID_PROVENANCE:
        raise MalformedRecordError(f"provenance={provenance!r} not in {sorted(VALID_PROVENANCE)}")
    if status != "completed" and not status_reason:
        raise MalformedRecordError(f"status={status!r} requires a non-empty status_reason explaining why.")

    unknown = set(extra_fields) - set(KNOWN_FIELDS)
    if unknown:
        raise MalformedRecordError(f"Unrecognized field(s), not part of the documented schema: {sorted(unknown)}")

    record: dict = {
        "run_id": run_id,
        "created_at": utcnow_iso(),
        "experiment_timestamp": experiment_timestamp,
        "experiment_timestamp_provenance": experiment_timestamp_provenance,
        "workflow_stage": workflow_stage,
        "experiment_name": experiment_name,
        "experiment_purpose": experiment_purpose,
        "status": status,
        "status_reason": status_reason,
        "provenance": provenance,
        "confidence": confidence,
        "source_description": source_description,
        "schema_version": SCHEMA_VERSION,
    }
    record.update(extra_fields)

    if include_environment and provenance == "directly_verified":
        commit, dirty = get_git_commit_and_dirty()
        record.setdefault("git_commit", commit)
        record.setdefault("git_dirty", dirty)
        record.setdefault("python_version", get_python_version())
        record.setdefault("dependency_versions", get_environment_versions())

    for field in KNOWN_FIELDS:
        record.setdefault(field, None)

    record["missing_fields"] = sorted(
        f for f in KNOWN_FIELDS
        if f not in OPTIONAL_EVEN_WHEN_APPLICABLE and f != "missing_fields" and record.get(f) is None
    )

    _check_forbidden_content(record)
    # Round-trip through JSON once here, at build time, so any
    # non-serializable value (e.g. a numpy scalar) fails loudly now rather
    # than producing a malformed line later inside the lock.
    try:
        serialize_record(record)
    except TypeError as exc:
        raise MalformedRecordError(f"Record is not JSON-serializable: {exc}") from exc

    return record


def serialize_record(record: dict) -> str:
    """Deterministic one-line JSON serialization: sorted keys, fixed
    separators, `default=str` as a last resort for anything unexpected
    (never silently drops a field). Calling this twice on equivalent data
    always produces byte-identical output."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Registry I/O -- append-only, immutable, lock-guarded
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _locked_registry_file(registry_path: Path):
    """Open the registry for append+read under an advisory exclusive lock
    (POSIX `fcntl.flock` -- macOS and Linux; this project does not target
    Windows). Guards the read-existing-run-ids-then-append sequence against
    a second concurrent writer process, which a single Python-level write()
    call alone would not. The lock is released automatically when the file
    closes, including on an exception.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.touch(exist_ok=True)
    fd = os.open(str(registry_path), os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict]:
    """All records currently in the registry. A malformed line (e.g. from a
    genuinely partial write, which this module's single-write-call append
    is designed to avoid, but a reviewer may have hand-edited the file) is
    skipped with a note printed to stderr, rather than crashing the read --
    one bad historical line should not make the rest of the registry
    unreadable.
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return []
    records = []
    with open(registry_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"experiment_tracking: skipping malformed line {line_no} in {registry_path}: {exc}", file=sys.stderr)
    return records


def run_id_exists(run_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> bool:
    return any(r.get("run_id") == run_id for r in read_registry(registry_path))


def append_run_record(record: dict, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
    """Append ONE validated record as one JSON line. Refuses (raises
    `DuplicateRunIdError`) if `record['run_id']` already exists -- this
    module never overwrites or updates a record; the registry is append-only
    and each line is meant to be immutable once written.

    Atomicity: the existing-IDs check and the append happen under one
    `flock`-held lock, and the write itself is a single `os.write()` call
    for a line far under the OS pipe-buffer size, so a second concurrent
    writer's append can never interleave bytes with this one on a POSIX
    filesystem.
    """
    registry_path = Path(registry_path)
    line = serialize_record(record) + "\n"
    with _locked_registry_file(registry_path) as fd:
        # Re-read under the lock (not using the earlier read_registry call,
        # if any) so the duplicate check sees any writer that appended
        # between an earlier unlocked check and now.
        os.lseek(fd, 0, os.SEEK_SET)
        existing_raw = os.read(fd, os.fstat(fd).st_size).decode("utf-8", errors="replace")
        for existing_line in existing_raw.splitlines():
            existing_line = existing_line.strip()
            if not existing_line:
                continue
            try:
                existing_record = json.loads(existing_line)
            except json.JSONDecodeError:
                continue
            if existing_record.get("run_id") == record["run_id"]:
                raise DuplicateRunIdError(f"run_id {record['run_id']!r} already exists in {registry_path} -- records are immutable.")
        os.lseek(fd, 0, os.SEEK_END)
        os.write(fd, line.encode("utf-8"))


def query_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH, **equality_filters) -> list[dict]:
    """Records matching every `field=value` filter given. Trivial, but
    enough for this project's scale -- see the module docstring for why a
    real database/tracking framework was deliberately not introduced."""
    records = read_registry(registry_path)
    return [r for r in records if all(r.get(k) == v for k, v in equality_filters.items())]


# ---------------------------------------------------------------------------
# Safe automatic logging for a live (not backfilled) run
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def track_run(
    workflow_stage: int,
    experiment_name: str,
    experiment_purpose: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    run_id: str | None = None,
    **initial_fields,
):
    """Wrap a script's actual experiment code. Yields a mutable `record`
    dict the caller fills in with real results as they are computed
    (model_family, hyperparameters, metrics, timings, ...) -- this context
    manager changes NOTHING about what the wrapped code computes.

    Status handling, so a crashed run can never be silently marked
    "completed": the record starts as `status="completed"` (optimistic);
    if the wrapped block raises, `status` is forced to `"failed"` with the
    exception recorded in `status_reason`, and the exception is then
    RE-RAISED (this context manager never swallows an error) -- only then
    is the record appended, in a `finally`, so even a crashed run is logged
    (as failed, with whatever partial fields were set before the crash),
    not silently dropped.
    """
    run_id = run_id or new_run_id(prefix=f"stage{workflow_stage}")
    unknown = set(initial_fields) - set(KNOWN_FIELDS)
    if unknown:
        raise MalformedRecordError(f"Unrecognized field(s) passed to track_run: {sorted(unknown)}")

    commit, dirty = get_git_commit_and_dirty()
    record: dict = {
        "run_id": run_id,
        "created_at": utcnow_iso(),
        "experiment_timestamp": None,
        "experiment_timestamp_provenance": "directly_verified: same as created_at (live run)",
        "workflow_stage": workflow_stage,
        "experiment_name": experiment_name,
        "experiment_purpose": experiment_purpose,
        "status": "completed",
        "status_reason": None,
        "provenance": "directly_verified",
        "confidence": "high",
        "source_description": f"Logged automatically by track_run() during its own execution of this script.",
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": get_python_version(),
        "dependency_versions": get_environment_versions(),
    }
    for field in KNOWN_FIELDS:
        record.setdefault(field, None)
    record.update(initial_fields)

    try:
        yield record
    except BaseException as exc:
        record["status"] = "failed"
        record["status_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if record["status"] not in VALID_STATUSES:
            raise MalformedRecordError(f"status={record['status']!r} not in {sorted(VALID_STATUSES)}")
        if record["status"] != "completed" and not record.get("status_reason"):
            raise MalformedRecordError(f"status={record['status']!r} requires a non-empty status_reason explaining why.")
        record["missing_fields"] = sorted(
            f for f in KNOWN_FIELDS
            if f not in OPTIONAL_EVEN_WHEN_APPLICABLE and f != "missing_fields" and record.get(f) is None
        )
        _check_forbidden_content(record)
        append_run_record(record, registry_path)
