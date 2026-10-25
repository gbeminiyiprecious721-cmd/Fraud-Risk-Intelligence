# Leakage Boundaries & Data Splitting Report — Stage 4

**Status:** Reviewed per user request before final approval — several conclusions refined to not overclaim, a code-level enforcement mechanism added and tested, register and split-boundary completeness independently verified. Still **awaiting final approval**; see `docs/ML_WORKFLOW_STATE.md` Approval Status.

**Scope boundary respected:** feature-availability decisions and a chronological split only. No fitted imputers/encoders/scalers/feature-selectors, no models, no hyperparameter tuning, no dashboard.

**Evidence tags** (continuing the convention from Stages 1–3), with one addition for this stage's external-research work:
- **[VERIFIED]** — computed this stage from the actual data.
- **[DOCUMENTED — PRIMARY]** — stated on the Kaggle competition's own data description (via web search this stage; not re-fetched from the live page directly in this session, so treated as high-but-not-first-hand confidence).
- **[DOCUMENTED — SECONDARY]** — repeated consistently across independent third-party technical write-ups, but not confirmed against the primary Kaggle page text directly in this session.
- **[ABSENCE OF EVIDENCE]** — specifically searched for and not found; distinct from "unknown because untried."
- **[EXTERNAL, UNCONFIRMED]** / **[HYPOTHETICAL]** — carried from earlier stages, unaffected here.

---

## 1. The Prediction-Time Information Boundary

Per `docs/PROBLEM_STATEMENT.md`, the decision this model supports is **ranking transactions for human investigation** under a capacity-constrained queue (automated decline-at-checkout is explicitly out of scope). That fixes the boundary precisely:

**A feature is available if and only if its value is knowable at the moment a transaction completes checkout and is queued for scoring — before any investigation outcome, chargeback, or manual review exists.** Concretely:
- Fields supplied by the payment flow itself (amount, product, card, address, email domain) — available.
- Fields that are themselves **outputs of a risk-assessment or verification step that may have already run by the time scoring happens** — not automatically available, regardless of how "transaction-time" they sound, because *when in the checkout flow* they're populated matters as much as *what* they contain. This is the exact distinction Stage 3 flagged for identity data and is resolved per-family below.
- Nothing about this boundary changes if a feature "looks numeric" or "looks anonymized" — availability is a question about the data-generation process, not the column's name or dtype (this is the same distinction the Stage 3 review drew between *association*, *usefulness*, and *leakage*, applied here to *availability* specifically).

---

## 2. Authoritative Documentation — Research Findings

Searched specifically for official/host-level documentation before making any availability decision, per instruction. Findings, with confidence explicitly separated from interpretation:

| Claim | Evidence level | Source type |
|---|---|---|
| Not every transaction has a matching identity record | **[DOCUMENTED — PRIMARY]** | Kaggle competition data description (paraphrased via search) |
| `TransactionDT` is a time offset ("timedelta from a given reference datetime"), not a real timestamp | **[DOCUMENTED — PRIMARY]** | Kaggle competition data description |
| Identity (`id_*`, `DeviceType`, `DeviceInfo`) and Vesta-engineered (`V`) feature *names* are masked for anonymization | **[DOCUMENTED — PRIMARY]** | Kaggle competition data description |
| `V1`–`V339`: "Vesta engineered rich features including ranking, counting, and other entity relations" | **[DOCUMENTED — SECONDARY]** | Repeated consistently across independent Kaggle top-solution write-ups |
| `C1`–`C14`: counting features (e.g. number of cards/addresses associated with an entity) | **[DOCUMENTED — SECONDARY]** | Independent technical write-ups/papers |
| `D1`–`D15`: time-delta features (e.g. days since a previous transaction/action) | **[DOCUMENTED — SECONDARY]** | Independent technical write-ups/papers |
| `M1`–`M9`: match flags (e.g. does cardholder name/address match) | **[DOCUMENTED — SECONDARY]** | Independent technical write-ups/papers |
| **Why/when identity records are created** — automatic-but-sometimes-fails, vs. a secondary step triggered by an initial risk signal | **[ABSENCE OF EVIDENCE]** | Specifically searched; no source found addressing this |
| **Per-column lookback windows or computation boundaries** for any individual `V`/`C`/`D` column | **[ABSENCE OF EVIDENCE]** | Specifically searched; only family-level descriptions found, never per-column |
| **Why there is a gap between the competition's train and test periods**, or confirmation that the test period is chronologically later | **[ABSENCE OF EVIDENCE]** — Stage 2's ~30-day-gap finding stands as **[VERIFIED]** (measured directly from this data), independently of documentation; no external source corroborating or explaining it was found |

**What this means for the carried-forward questions:** two of the four (identity mechanism, V/C/D per-column computation) hit a genuine, searched-for absence of evidence — not resolved by documentation, and resolved below either empirically or by conservative default, per instruction.

---

## 3. Investigating the Four Carried-Forward Questions

### (a) Identity-data availability at prediction time — **A decision is resolved (exclude); the underlying question is not**

No documentation found on collection timing/trigger (Section 2). Stage 3's measured 3.75x fraud-rate gap (7.847% vs. 2.094%) is large enough that the conservative default applies: **all 41 identity-table columns (`id_01`–`id_38`/`id-01`–`id-38`, `DeviceType`, `DeviceInfo`) are excluded** from the available-feature set for now. Precision matters here: what's "resolved" is *what to do given the absence of evidence* (exclude), not *whether identity collection is actually risk-triggered* — that mechanism question remains genuinely unknown, not answered in either direction. This is not "excluded because anonymized" — it's excluded because the specific mechanism question has no evidence either way, and the effect size is too large to default to inclusion. Revisable the moment positive evidence of uniform, pre-decision collection surfaces.

### (b) `R_emaildomain` + 14 `V`-columns' missingness pattern — **Refined (review round 2): one failure mode ruled out; availability still not independently established**

**[VERIFIED, this stage]** Checked whether `R_emaildomain`'s presence tracks `ProductCD` rather than fraud outcome:

| ProductCD | n | % has `R_emaildomain` | Fraud rate |
|---|---|---|---|
| W | 439,670 | **0.00%** | 2.040% |
| H | 33,024 | 67.23% | 4.766% |
| C | 68,519 | 96.86% | **11.687%** |
| S | 11,628 | 94.93% | 5.900% |
| R | 37,699 | 99.97% | 3.783% |

This is a **near-deterministic structural split**: `R_emaildomain` is populated for essentially every non-`W` product and for none of the `W` product. This explains Stage 3's missingness-by-fraud-status finding as a product-type-driven pattern (fraud concentrates in non-`W` products, which happen to almost always carry a receiver email) — ruling out **one specific failure mode**: that presence directly tracks fraud outcome through some post-outcome process reacting to the fraud determination itself.

**What this does NOT establish (correction, review round 2):** the `ProductCD` relationship is a statement about *why the missingness pattern looks the way it does*, not a statement about *when in the checkout/fulfillment flow `R_emaildomain`'s value is actually captured*. "The field correlates with a legitimate, transaction-time attribute" and "the field's value is known by the moment a risk score needs to be computed" are two different claims — only the first is verified here. It remains entirely possible that for non-`W` products, the receiver email is captured at a fulfillment/shipping step that happens *after* the point this project's prediction-time boundary (Section 1) is drawn, in which case the structural explanation would be correct and the feature would still be unavailable at the moment that matters. This timing has not been checked and is not claimed to be resolved.

**Spot-checked, not assumed, that the missingness-pattern explanation extends to the 14 co-moving `V`-columns:** `V180` and `V197` (2 of the 14) show the same `ProductCD=W → ~0%` structural pattern. The remaining 12 are inferred — not individually re-verified — to share this explanation, since Stage 3 found all 14 move together with an identical missingness gap. The same availability-timing caveat above applies equally to these.

**Status: `include_with_caveat`, unchanged** — but the caveat's substance has shifted, not shrunk: the *missingness-is-outcome-driven* leakage concern is substantially addressed; a *separate*, still-open *availability-timing* concern replaces it. See Section 4a for what `include_with_caveat` does and doesn't license operationally.

### (c) Leakage risk within the 368 anonymized `V`/`C`/`D` columns generally — **Partially investigated; not blanket-resolved or blanket-dismissed**

Per instruction, did not infer leakage (or its absence) solely from association, missingness, or importance. Two concrete, scoped empirical checks were run instead:

**[VERIFIED, this stage]** `D`-column sanity check (does behavior match the documented "time-delta" description?):

| Check | Result |
|---|---|
| Share of negative values across all 15 `D`-columns | 0.000%–0.005% |
| Correlation with `TransactionDT` across all 15 `D`-columns | −0.07 to 0.10 (all low) |

**What this evidence does and does not show (correction, review round 2):** these two checks rule out two specific, narrow failure modes — a value inconsistent with any genuine past-event delta (ruled out by near-zero negativity), and a column that's really just a disguised absolute-time-position proxy (ruled out by low correlation with `TransactionDT`). **They are supporting evidence, not proof that the underlying feature generation is causal or safe at prediction time.** A column computed from a window that looks both backward and forward from each row in a symmetric way could pass both of these checks and still be non-causal — this check has no way to detect that case, because it only examines the column's own values and their relationship to `TransactionDT`, not the actual computation that produced them (which remains undocumented). The result is real and worth having, but it narrows the space of *undetected* failure modes; it does not eliminate them.

A few columns (`D4`/`D6`/`D11`/`D12`/`D14`/`D15`) show a tiny non-zero negative share (≤0.005%) — left as an open, unexplained minor anomaly, not resolved away by its small size.

**What remains unresolved:** the other ~325 `V`-columns and all 14 `C`-columns were not individually empirically checked this stage (scope/time) — included per the register below, with the caveat stated honestly rather than cleared. **This question is not closed.**

### (d) Temporal validation / information-availability boundaries beyond the train/test gap — **Addressed via split design (Section 5), one sub-question remains open**

The chronological split (below) directly operationalizes the train/test-boundary finding for our own validation/final-test partitions. The deeper sub-question — whether any anonymized feature's value for an early training row depends on later training rows (a non-causal aggregation window, entirely internal to `train`) — is **not** resolved by any split design, since all three of our partitions are still slices of the same already-finalized `train_transaction.csv`. This remains explicitly open; see Unresolved Risks.

---

## 4. Feature-Availability Register

**Complete, individually traceable register for all 435 columns** (394 `train_transaction` + 41 `train_identity`): `docs/feature_availability_register.csv`. Decisions are made at the family level (reviewable in one place, per instruction) and applied to each column programmatically — `src/leakage_boundaries.py::build_feature_availability_register()` — so every row is traceable to its exact column, not just its family.

| Status | Count | Families |
|---|---|---|
| `include` | 23 | transaction_time, transaction_amount, product, card, address, distance, purchaser_email, M_match |
| `include_with_caveat` | 369 | receiver_email, C_count, D_timedelta, V_engineered (caveats differ by family — see Section 3 and the CSV's `rationale` column) |
| `exclude_structural` | 3 | identifier (`TransactionID` ×2), target (`isFraud`) |
| `exclude_conservative` | 40 | identity_id, identity_device (all of `train_identity` except `TransactionID`) |

**No family was blanket-excluded for having an unclear name** — per instruction, `V_engineered` (339 columns) is `include_with_caveat`, not excluded, on the strength of its documented description and the 2-column empirical spot-check; only `identity_id`/`identity_device` are excluded, and that's driven by the specific, evidenced mechanism question in (a), not by anonymization alone.

### 4a. What `include_with_caveat` actually licenses (clarified, review round 2)

This status does **not** mean "confirmed safe" and was never meant to. Stated explicitly, since it governs 369 of 435 columns:

| | `include_with_caveat` means | `include_with_caveat` does NOT mean |
|---|---|---|
| **Scope** | Eligible for **controlled Stage 7+ experiments** — training a model, measuring whether the feature helps, under full tracking (Stage 8) so every result is traceable to exactly which features were used | Confirmed safe for an actual production decision made about a real transaction |
| **Basis** | A documented family-level description (plausible, by Section 2) plus, for some families, a narrow empirical check that passed | Individual, per-column verification of causal computation or exact prediction-time availability |
| **What a pass or fail in Stage 7+ would mean** | A feature that measurably helps is a candidate for *further* availability verification before it could ever be shipped — not evidence the availability question is resolved | Nothing about predictive performance resolves an availability or leakage question — see the Stage 3 review's "association ≠ usefulness ≠ leakage" framework, which still applies |
| **Who can downgrade it** | Any Stage 5+ finding that contradicts the documented description, or any later evidence on timing/mechanism | Nothing automatically upgrades it to `include` — that requires new evidence, not the absence of a problem so far |

Put plainly: **`include_with_caveat` is a license to experiment under controlled, reproducible conditions, not a safety certificate.** Nothing in this stage, or in any Stage 7-10 result built on top of it, should be read as "the availability question got resolved because the model worked."

### 4b. Register completeness — independently verified (review item 5)

**[VERIFIED, this stage]** Ran a dedicated completeness check (`scripts/run_stage4_analysis.py` §5b) rather than trusting the earlier row count by inspection alone:

| Check | Result |
|---|---|
| Expected total (394 `train_transaction` + 41 `train_identity`) | 435 |
| Actual register rows | **435** — matches exactly |
| Every status value is one of the 4 defined statuses | ✅ True |
| Duplicate `(column, source_file)` rows | **0** |
| Rows with an empty/missing rationale | **0** — every row has a non-empty rationale |
| Rows in the `unclassified` family (i.e. `classify_family()` couldn't place them) | **0** |
| Status counts sum to the total row count | ✅ True (`23 + 369 + 3 + 40 = 435`) |

No column is silently missing, double-counted, or carrying an invalid/placeholder status.

---

## 5. Chronological Split Design

**Source:** `train_transaction.csv` only (labelled data) — `test_transaction.csv` has no usable labels (Stage 2). `src/leakage_boundaries.py::chronological_split()`.

**Method:** time-proportional, not row-count-proportional — cutoffs are fixed fractions of the observed `[min, max]` `TransactionDT` range, so the same input always produces the same cutoffs deterministically (no randomness anywhere in this split).

| Partition | Fraction of time span | Purpose |
|---|---|---|
| `train` | 70% | Model fitting (Stage 7+) |
| `validation` | 15% | Model selection, threshold calibration (Stage 7–10) — **never used for the final evaluation claim** |
| `final_test` | 15% | Touched exactly once, in Stage 10, for the honest final evaluation |

**Why your notes' 80:20 train-test-validation split and k-fold CV don't apply here (per instruction, explained via your own Notion material):** `INTERMEDIATE ML` documents an 80:20 train-test split and k-fold cross-validation — both assume rows are *exchangeable* (any row could swap with any other without changing what the split represents). That assumption is exactly what Stage 0/2 ruled out for this dataset: a random 80:20 split, or k-fold CV with its random re-shuffling across folds, would let future transactions train a model that's then evaluated on *past* transactions in another fold — leaking information a production model would never have. A chronological split removes the randomness entirely so that every partition only ever sees data from its own time window, in order.

**Why no temporal gap was added (explicit decision, not a default):** a gap is standard in some time-series validation setups, but only when justified by something specific — most commonly either (i) a *label-maturity* lag (the true label isn't finalized until some days after the event, e.g. a chargeback filed weeks later) or (ii) a feature with a known forward-looking computation window. Neither applies here with evidence: (i) is a property of the *source* dataset's label-freeze process, which already happened before this data was ever downloaded — a gap between *our own* partitions, carved after the fact from the same frozen file, does nothing to fix or protect against that; it would only be relevant if we controlled when labels were finalized, which we don't. (ii) is exactly carried-forward question (c), which remains *open*, not resolved — adding a gap doesn't resolve a possible non-causal window either, because a backward-looking (causal) window never reaches across a gap in the first place, and a genuinely non-causal (forward-looking) one isn't fixed by a gap downstream of it. Given neither justification is evidenced for *this* split, no gap was added — stated as a reasoned decision, not an oversight, and explicitly not a default replication of the competition's ~30 days (which has no evidenced justification found either, per Section 2).

---

## 6. Split Verification

**[VERIFIED]** `src/leakage_boundaries.py::verify_split()`, run against the real split:

| Partition | n | Fraud count | Fraud rate | `TransactionDT` range |
|---|---|---|---|---|
| train | 438,125 | 15,364 | 3.507% | 86,400 – 11,093,269 |
| validation | 75,560 | 2,617 | 3.463% | 11,093,842 – 13,452,406 |
| final_test | 76,855 | 2,682 | 3.490% | 13,452,433 – 15,811,131 |

- **Chronological ordering:** ✅ confirmed (`train.max < validation.min`, `validation.max < final_test.min`).
- **Identifier disjointness:** ✅ confirmed (0 `TransactionID`s appear in more than one partition).
- **All rows accounted for:** ✅ `438,125 + 75,560 + 76,855 = 590,540`.
- **Every partition has fraud cases:** ✅ (2,617 and 2,682 in the smaller partitions — comfortably usable, not a handful of edge cases).
- **Fraud rate is stable across all three partitions** (3.46%–3.51%, tightly around the overall 3.499%) — no partition is anomalous, consistent with Stage 3's finding that daily fraud rate stays in a stable band.

The 70/15/15 **time** split produced a 74.2% / 12.8% / 13.0% **row** split — expected and fine, since transaction volume isn't uniform across the window (Stage 3, Figure C1); the split is proportional by time, not row count, by design.

### 6a. Determinism and ordering — re-confirmed this review round

**[VERIFIED]** Same cutoffs on re-run: `train_end_dt=11,093,711`, `validation_end_dt=13,452,420` — identical to the first run, confirming no hidden randomness. `tests/test_leakage_boundaries.py::test_chronological_split_is_deterministic` asserts this via `pd.testing.assert_frame_equal` on two independent calls, not just an eyeball comparison.

### 6b. Tie-handling at exact cutoff timestamps (review item 4)

**The rule:** `chronological_split()`'s assignment is `TransactionDT <= cutoff → earlier partition`, applied to each row by its own `TransactionDT` value alone — never by row index, insertion order, or `TransactionID`. Because the rule only looks at the value, **two rows sharing the exact same `TransactionDT` can never be split across two partitions** — there is no mechanism in the code that could put one on one side and the other on the other.

**Verified two ways, not just argued from reading the code:**
- **[VERIFIED, real data]** `check_cutoff_tie_handling()` run against the actual split: **zero rows** in `train_transaction.csv` have `TransactionDT` exactly equal to either cutoff (`11,093,711` or `13,452,420`). There is no literal tie to handle in this particular split — reported honestly as "no ties occurred," not as "ties are handled" (a claim about data that happens to be absent, not a claim about the mechanism).
- **[VERIFIED, synthetic fixture]** `tests/test_leakage_boundaries.py::test_check_cutoff_tie_handling_detects_a_real_tie` constructs a fixture where multiple rows **do** share the exact cutoff value, runs the same check, and confirms they all land on the same (earlier) side — this is where the *rule itself* is actually tested, independent of whether real ties happen to exist.

### 6c. Final-holdout enforcement mechanism (review item 3)

**The gap this closes:** until this review round, "don't touch `final_test` before Stage 10" was a documentation convention — nothing in the code actually stopped a later stage from reading those rows during feature selection, threshold tuning, or hyperparameter search. That's now a code-level guardrail, not only a written rule:

- **`get_partition(assignment, partitions, allow_final_test=False)`** — refuses to return `final_test` rows unless called with `allow_final_test=True`. Any Stage 5+ code that routes partition access through this function cannot read `final_test` by accident; only a call that explicitly says so can, and that call is meant to exist exactly once, in Stage 10.
- **`assert_no_final_test_contamination(used_ids, assignment)`** — a second, independent layer: given whatever `TransactionID`s a modeling step actually touched, raises if any belong to `final_test`. Usable even if some future code bypasses `get_partition()` entirely and reads the raw assignment directly.

**[VERIFIED, real data]** Both demonstrated against the actual split this review round: an unguarded request for `["train", "final_test"]` was refused (`FinalTestAccessError`); the same request with `allow_final_test=True` correctly returned all 76,855 `final_test` rows; a contamination check against the real `train` partition's IDs correctly passed (no false positive). **[VERIFIED, tests]** `tests/test_leakage_boundaries.py` covers both functions on synthetic fixtures, including the failure path (default refusal) and the success path (explicit override) for each.

**Honest limitation:** this guardrail only works if Stage 5+ code actually calls these functions instead of re-reading `docs/_stage4_split_assignment.csv` directly and filtering by hand. It's an enforcement mechanism available for use, not a physical impossibility — the convention still matters; this just makes violating it require a deliberate bypass rather than an easy oversight.

---

## 7. Tests

`tests/test_leakage_boundaries.py` — determinism, chronological-ordering, ID-disjointness (including a deliberately-broken fixture to confirm the check actually fires, not just that it passes on clean data), proportion sanity, the register builder's per-column coverage and family-decision-missing error path, and (added this review round) the final-holdout guardrail functions and cutoff tie-handling, including a synthetic fixture that constructs a real tie to prove the rule, not just the absence of one.

**A genuine failure was caught and fixed during this stage**, not retrofitted after the fact: `classify_family()`'s first draft matched `"DeviceType"`/`"DeviceInfo"` into the `D_timedelta` family (both start with `"D"`), which `test_classify_family[DeviceType-identity_device]` caught immediately. Fixed by checking identity names before the generic letter-prefix families and requiring a pure-digit suffix for `C`/`D`/`M`/`V` matches.

```
.venv/bin/pytest tests/ -v
48 passed, 0 failed, 0 skipped
```
(16 from Stages 2–3 + 25 from Stage 4's first pass + 7 new this review round — `get_partition`, `assert_no_final_test_contamination`, `check_cutoff_tie_handling`.) No test regressed; no existing test was weakened to make a new one pass.

---

## 8. Unresolved Risks (carried forward, explicitly not closed by this stage — corrected/strengthened this review round)

1. **Identity-collection mechanism** — still no evidence either way; excluded conservatively, not proven unsafe or safe.
2. **`R_emaildomain` + 14 `V`-columns' prediction-time availability — NOT established, corrected this round.** The missingness-pattern leakage concern is addressed (§3b); the separate question of *when* the field is captured relative to the prediction-time boundary is not, and was previously stated with more confidence than the evidence supported.
3. **D-column causal computation — NOT proven, corrected this round.** The non-negativity and low-correlation checks (§3c) are supporting evidence against two specific failure modes, not proof of causal, prediction-time-safe generation; a symmetric (backward-and-forward) window would pass both checks undetected.
4. **The ~325 non-spot-checked `V`-columns and all 14 `C`-columns** — included per documented family description, but not individually empirically verified. A real residual risk, stated plainly.
5. **Intra-train non-causal computation** — no split design can address this; it requires either documentation (unavailable) or a more involved empirical check (e.g. per-entity, per-row causality tracing) than this stage's scope covered.
6. **`train_identity`/`test_identity` naming mismatch** (Stage 2) — still a Stage 5 prerequisite, unrelated to the above.
7. **Label-maturity lag in the source data itself** — a plausible real-world phenomenon (chargebacks often post weeks after a transaction) that, if present in how Kaggle/Vesta froze labels, would already be baked into `train_transaction.csv` before we ever received it. Unverifiable from this data, and not something our own split can fix.
8. **`include_with_caveat` is not a safety certificate for any of its 369 columns** — restated explicitly (§4a) after this review round, to prevent a Stage 7+ result from being read as having resolved an availability question it didn't touch.
9. **The final-holdout guardrail (§6c) is a tool, not a guarantee** — it only protects `final_test` if Stage 5+ code actually routes through `get_partition()`/`assert_no_final_test_contamination()` rather than reading the raw assignment file directly.

---

## Three Interview-Style Questions

1. `R_emaildomain`'s presence is ~0% for `ProductCD=W` and ~95–100% for every other product code. This rules out one specific leakage mechanism. Name a *different* mechanism — one this check cannot rule out — by which `R_emaildomain` could still fail to be available at prediction time, even with this exact `ProductCD` relationship holding.
2. The D-column checks (non-negativity, low correlation with `TransactionDT`) both passed for all 15 columns. Describe a hypothetical feature-generation process that would pass both checks and still be non-causal (uses information from after the row's own transaction time).
3. `get_partition()` raises an error if `"final_test"` is requested without `allow_final_test=True`. What's the one concrete way a future stage could still read `final_test` rows and bypass this guardrail entirely — and what does that imply about treating this as "enforcement" versus "a safety net"?
