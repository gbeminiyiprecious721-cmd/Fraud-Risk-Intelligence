"""Stage 4 leakage-boundary utilities — IEEE-CIS Fraud Detection data.

Three groups of functions, all read-only:
1. Empirical checks that investigate (not assume) specific leakage questions
   carried forward from Stage 3 — these look for evidence, not for
   confirmation of a pre-decided answer.
2. A feature-availability register builder — combines a per-family decision
   map (informed by Stage 0-3 findings + external research, see
   docs/LEAKAGE_BOUNDARIES_REPORT.md) with the actual schema so every column
   gets an individually traceable row.
3. A chronological split builder + verifier, operating on train_transaction
   only (test_transaction has no labels — Stage 2).

None of these fit, impute, encode, or select features by looking at the
target in a way a model would — that's Stage 5/6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CHUNKSIZE = 100_000


# ---------------------------------------------------------------------------
# 1. Empirical investigation of carried-forward leakage questions
# ---------------------------------------------------------------------------

def check_remaildomain_productcd_relationship(path: str | Path) -> pd.DataFrame:
    """Check whether R_emaildomain's presence tracks ProductCD rather than fraud outcome.

    Why this check, specifically: Stage 3 found R_emaildomain ~32pp less
    missing for fraud transactions. One legitimate (non-leakage) explanation
    is that certain product types always carry a receiver email (e.g.
    gift/digital-goods products shipped to a different recipient) — a
    transaction-time attribute, not a post-outcome one. If R_emaildomain's
    presence tracks ProductCD strongly, that's evidence *for* a legitimate
    explanation; if it doesn't, the missingness-by-target pattern remains
    unexplained and the leakage question stays fully open.
    """
    df = pd.read_csv(path, usecols=["ProductCD", "R_emaildomain", "isFraud"])
    df["has_r_email"] = df["R_emaildomain"].notna()
    by_product = df.groupby("ProductCD").agg(
        n=("has_r_email", "size"),
        pct_has_r_email=("has_r_email", "mean"),
        fraud_rate=("isFraud", "mean"),
    )
    by_product["pct_has_r_email"] = (by_product["pct_has_r_email"] * 100).round(2)
    by_product["fraud_rate"] = (by_product["fraud_rate"] * 100).round(3)
    return by_product.sort_values("pct_has_r_email", ascending=False)


def check_d_columns_sanity(
    path: str | Path,
    d_cols: Iterable[str],
    dt_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Sanity-check D-columns against the "time delta" description found in research.

    Two checks per column, neither of which assumes an answer:
    - `pct_negative`: a genuine time-since-past-event delta should never be
      negative. A non-trivial negative share would contradict the
      "time delta" description and is worth flagging, not assuming away.
    - `corr_with_dt`: a very high correlation with `TransactionDT` itself
      would suggest the column is (at least partly) a disguised absolute
      time-position feature rather than a genuine per-entity delta — not
      leakage by itself, but a different thing than its name suggests.
    """
    cols = list(d_cols)
    df = pd.read_csv(path, usecols=cols + [dt_col])
    rows = []
    for col in cols:
        series = df[col].dropna()
        n = len(series)
        pct_negative = round(100 * (series < 0).mean(), 3) if n else float("nan")
        corr = df[[col, dt_col]].dropna().corr().iloc[0, 1] if n else float("nan")
        rows.append(
            {
                "column": col,
                "n_non_null": n,
                "pct_negative": pct_negative,
                "corr_with_TransactionDT": round(float(corr), 3) if pd.notna(corr) else None,
            }
        )
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# 2. Feature-availability register
# ---------------------------------------------------------------------------

@dataclass
class FamilyDecision:
    status: str  # "include", "include_with_caveat", "exclude_conservative", "exclude_structural"
    rationale: str
    evidence_level: str  # "documented_primary", "documented_secondary", "empirical", "domain_logic", "absence_of_evidence"


def classify_family(column: str) -> str:
    """Map a column name to its family, for family-level decisions with per-column traceability."""
    if column in ("TransactionID",):
        return "identifier"
    if column == "isFraud":
        return "target"
    if column == "TransactionDT":
        return "transaction_time"
    if column == "TransactionAmt":
        return "transaction_amount"
    if column in ("ProductCD",):
        return "product"
    if column.startswith("card"):
        return "card"
    if column in ("addr1", "addr2"):
        return "address"
    if column in ("dist1", "dist2"):
        return "distance"
    if column == "P_emaildomain":
        return "purchaser_email"
    if column == "R_emaildomain":
        return "receiver_email"
    if column in ("DeviceType", "DeviceInfo"):
        return "identity_device"
    if column.startswith("id_") or column.startswith("id-"):
        return "identity_id"
    # Family-letter + digits only (e.g. "C7", "D11", "M3", "V250") — checked
    # after the identity_device/identity_id names above specifically because
    # "DeviceType"/"DeviceInfo" also start with "D" and would otherwise be
    # misclassified as D_timedelta (caught by test_classify_family).
    rest = column[1:]
    if column[:1] == "C" and rest.isdigit():
        return "C_count"
    if column[:1] == "D" and rest.isdigit():
        return "D_timedelta"
    if column[:1] == "M" and rest.isdigit():
        return "M_match"
    if column[:1] == "V" and rest.isdigit():
        return "V_engineered"
    return "unclassified"


def build_feature_availability_register(
    transaction_schema: dict[str, str],
    identity_schema: dict[str, str],
    family_decisions: dict[str, FamilyDecision],
) -> pd.DataFrame:
    """Build one traceable row per column from family-level decisions.

    Every column gets its own row (individually traceable); the status,
    rationale, and evidence level are applied per-family (justified,
    reviewable in one place in docs/LEAKAGE_BOUNDARIES_REPORT.md) rather
    than hand-typed 435 times.
    """
    rows = []
    for source, schema in (("train_transaction", transaction_schema), ("train_identity", identity_schema)):
        for column in schema:
            family = classify_family(column)
            decision = family_decisions.get(family)
            if decision is None:
                raise KeyError(f"No family decision registered for family '{family}' (column '{column}')")
            rows.append(
                {
                    "column": column,
                    "source_file": source,
                    "family": family,
                    "status": decision.status,
                    "rationale": decision.rationale,
                    "evidence_level": decision.evidence_level,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Chronological split + verification
# ---------------------------------------------------------------------------

@dataclass
class SplitCutoffs:
    train_end_dt: int
    validation_end_dt: int
    gap_seconds: int = 0


def chronological_split(
    path: str | Path,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    dt_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    target_col: str = "isFraud",
    gap_seconds: int = 0,
) -> tuple[pd.DataFrame, SplitCutoffs]:
    """Build a reproducible, time-proportional train/validation/final-test split.

    Split by *time span*, not row count — the cutoffs are fixed fractions of
    the observed [min, max] TransactionDT range, so the same input always
    produces the same cutoffs (deterministic, no randomness). `gap_seconds`
    reserves a buffer after each cutoff that belongs to neither partition —
    0 by default; see docs/LEAKAGE_BOUNDARIES_REPORT.md for why no gap was
    used here.
    """
    df = pd.read_csv(path, usecols=[id_col, dt_col, target_col])
    dt_min, dt_max = df[dt_col].min(), df[dt_col].max()
    span = dt_max - dt_min
    train_end = dt_min + int(span * train_frac)
    validation_end = train_end + gap_seconds + int(span * validation_frac)

    def assign(dt: int) -> str:
        if dt <= train_end:
            return "train"
        if dt <= validation_end:
            return "validation"
        return "final_test"

    df = df.sort_values(dt_col, kind="stable").reset_index(drop=True)
    df["partition"] = df[dt_col].apply(assign)
    cutoffs = SplitCutoffs(train_end_dt=train_end, validation_end_dt=validation_end, gap_seconds=gap_seconds)
    return df[[id_col, dt_col, target_col, "partition"]], cutoffs


def verify_split(assignment: pd.DataFrame, dt_col: str = "TransactionDT", id_col: str = "TransactionID", target_col: str = "isFraud") -> dict:
    """Verify chronological ordering, ID disjointness, and per-partition counts/fraud rate.

    Every check here either passes or fails explicitly — nothing is assumed.
    """
    results: dict = {}

    partitions = ["train", "validation", "final_test"]
    ranges = {}
    for part in partitions:
        sub = assignment.loc[assignment["partition"] == part]
        ranges[part] = (sub[dt_col].min(), sub[dt_col].max()) if len(sub) else (None, None)
        results[part] = {
            "n": int(len(sub)),
            "fraud_count": int(sub[target_col].sum()),
            "fraud_rate_pct": round(100 * sub[target_col].mean(), 3) if len(sub) else None,
            "dt_min": ranges[part][0],
            "dt_max": ranges[part][1],
        }

    ordering_ok = (
        ranges["train"][1] is not None
        and ranges["validation"][0] is not None
        and ranges["final_test"][0] is not None
        and ranges["train"][1] < ranges["validation"][0]
        and ranges["validation"][1] < ranges["final_test"][0]
    )
    results["chronological_ordering_ok"] = bool(ordering_ok)

    id_sets = {part: set(assignment.loc[assignment["partition"] == part, id_col]) for part in partitions}
    overlap = (
        (id_sets["train"] & id_sets["validation"])
        | (id_sets["train"] & id_sets["final_test"])
        | (id_sets["validation"] & id_sets["final_test"])
    )
    results["id_disjoint"] = len(overlap) == 0
    results["id_overlap_count"] = len(overlap)

    results["total_accounted_for"] = sum(results[p]["n"] for p in partitions) == len(assignment)
    results["all_partitions_have_fraud_cases"] = all(results[p]["fraud_count"] > 0 for p in partitions)

    return results


def check_cutoff_tie_handling(
    assignment: pd.DataFrame,
    cutoffs: SplitCutoffs,
    dt_col: str = "TransactionDT",
) -> dict:
    """Verify how rows at the exact cutoff timestamp are handled.

    `chronological_split()`'s assignment rule is `dt <= cutoff -> earlier
    partition`, applied uniformly to every row by its own TransactionDT value
    alone (never by index, ID, or any per-row tiebreaker). That means two
    rows sharing the exact same TransactionDT can never land in different
    partitions — the rule has no way to split them, since it only looks at
    the value. This check confirms that empirically, rather than asserting
    it from reading the code: it finds every row whose TransactionDT exactly
    equals a cutoff and confirms they're all in the same (earlier) partition.
    """
    results = {}
    for name, cutoff in (("train_end", cutoffs.train_end_dt), ("validation_end", cutoffs.validation_end_dt)):
        at_cutoff = assignment.loc[assignment[dt_col] == cutoff]
        partitions_present = sorted(at_cutoff["partition"].unique().tolist())
        results[name] = {
            "cutoff_value": cutoff,
            "n_rows_at_exact_cutoff": int(len(at_cutoff)),
            "partitions_present": partitions_present,
            "all_on_same_side": len(partitions_present) <= 1,
        }
    return results


class FinalTestAccessError(RuntimeError):
    """Raised when code outside Stage 10 tries to read final_test rows."""


def get_partition(
    assignment: pd.DataFrame,
    partitions: Iterable[str],
    allow_final_test: bool = False,
) -> pd.DataFrame:
    """Guarded accessor — the enforcement mechanism for the final-holdout boundary.

    Requesting "final_test" without `allow_final_test=True` raises
    `FinalTestAccessError` instead of silently returning the rows. This is
    the actual code-level guardrail (not just documentation) that Stage 5+
    code should route all partition access through, so that feature
    selection, threshold tuning, or hyperparameter search can't read
    final_test rows by accident — only a call that explicitly says
    `allow_final_test=True` can, and that call is meant to exist exactly
    once, in Stage 10.
    """
    partitions = list(partitions)
    if "final_test" in partitions and not allow_final_test:
        raise FinalTestAccessError(
            "Refused to return final_test rows without allow_final_test=True. "
            "final_test is reserved for the single, final Stage 10 evaluation — "
            "not for feature selection, threshold tuning, or hyperparameter search."
        )
    return assignment.loc[assignment["partition"].isin(partitions)]


def assert_no_final_test_contamination(
    used_ids: Iterable,
    assignment: pd.DataFrame,
    id_col: str = "TransactionID",
) -> None:
    """Raise if any ID used in a modeling step belongs to the final_test partition.

    A second, independent layer of the same guardrail: even if code bypasses
    `get_partition()` and reads the raw split assignment directly, this
    function can be called on whatever IDs a modeling step actually touched
    to confirm none of them belong to final_test — intended as an assertion
    Stage 7-9 test suites can call directly.
    """
    final_test_ids = set(assignment.loc[assignment["partition"] == "final_test", id_col])
    used_ids = set(used_ids)
    overlap = used_ids & final_test_ids
    if overlap:
        raise FinalTestAccessError(
            f"{len(overlap)} ID(s) used in this step belong to final_test — "
            f"e.g. {sorted(overlap)[:5]}. final_test must not be touched before Stage 10."
        )
