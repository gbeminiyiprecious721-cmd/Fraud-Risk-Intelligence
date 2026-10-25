# Exploratory Data Analysis Report — Stage 3

**Status:** Reviewed per user request before approval — one new conceptual section added, three existing sections (B, E, F) strengthened with verification/caveats. Still **awaiting final approval**; see `docs/ML_WORKFLOW_STATE.md` Approval Status.

**Source of truth:** every number below was copied directly from the executed outputs of `notebooks/01_exploratory_data_analysis.ipynb` (`.venv/bin/jupyter nbconvert --to notebook --execute --inplace`, 0 errors, **19** code cells — one verification cell added this review round). Nothing here was independently recomputed and could have drifted from the notebook — the notebook *is* the evidence; this file narrates it.

**Scope boundary respected:** structural/distributional EDA only. No fitted imputers/scalers/encoders, no target-based feature *selection*, no train/test split, no modeling.

**Evidence tags** (continuing the convention from `docs/PROBLEM_STATEMENT.md`/`docs/DATA_VALIDATION_REPORT.md`): **[VERIFIED]** = computed this stage from the actual data; **[EXTERNAL, UNCONFIRMED]** = still not checked against this data; **[HYPOTHETICAL]** = Stage 1 business framing, unaffected here.

---

## Learning Context (Notion)

Reviewed the "DATA SCIENCE JOURNAL" database. **Documented in notes** (not assumed to be mastered — per Learning Mode, notes are prior exposure, not proof of understanding): `ENCODING-CATEGORICAL VARIABLES` (fully reviewed) covers ordinal/label/one-hot/binary/hashing/target encoding and `.dt`-based date-time feature extraction; `INTERMEDIATE ML` structures EDA and Feature Engineering as explicit workflow steps (too large — 358K characters — to read in full this session, so treated as topic-coverage signal, not mined line-by-line); `Introduction to Feature Selection Methods` covers sequential backward selection, tree feature importances, and PCA. This stage's analysis builds on that foundation (effect sizes, chi-squared-based association, chunked processing) rather than limiting itself to it, per explicit instruction.

---

## A. Target Distribution and Class Imbalance

**[VERIFIED]**

| | Count | % |
|---|---|---|
| Legitimate | 569,877 | 96.501% |
| Fraud | 20,663 | 3.499% |
| **Total** | **590,540** | — |

**Imbalance ratio: 27.6 non-fraud transactions per fraud transaction.**

This is the first time this project has *measured* the fraud rate on this data copy — Stage 1 cited the ~3.5% figure as `[EXTERNAL, UNCONFIRMED]` (publicly known about the Kaggle competition, not yet checked here). **It's now `[VERIFIED]`: 3.499%, matching the external figure closely.**

**Why it matters:** confirms Stage 1's rejection of plain accuracy — a trivial "always legitimate" model would score 96.5% accuracy while catching zero fraud. Validates Average Precision, precision/recall@K, and cost-sensitive thresholding as the right metric family (Stage 10).

**Limitation:** this is the *train* file's rate only. Stage 2 confirmed a chronological train/test split; the (label-free) test period's actual rate is unverifiable without labels.

---

## B. Transaction Amount Analysis

**[VERIFIED]**

| | Overall | Legitimate | Fraud |
|---|---|---|---|
| n | 590,540 | 569,877 | 20,663 |
| Mean | $135.03 | $134.51 | $149.24 |
| Median | $68.77 | $68.50 | $75.00 |
| Std | 239.16 | 239.40 | 232.21 |
| Q1 / Q3 | 43.32 / 125.00 | 43.97 / 120.00 | 35.04 / 161.00 |
| IQR | 81.68 | 76.03 | 125.96 |
| IQR-outlier count (%) | 66,482 (11.26%) | 64,043 (11.24%) | 1,984 (9.60%) |
| Max | 31,937.39 | 31,937.39 | 5,191.00 |

**Effect size — Cohen's d:**
- Raw scale: **d = 0.062**
- log1p scale: **d = −0.010**

**The most important honest finding here: both effect sizes are near zero and disagree in sign.** Transaction amount is **not** a strong standalone fraud signal in this data, on either scale.

**Why the sign flips — verified, not asserted (added on review).** Re-derived `d_raw` and `d_log` independently from the group means/stds above: **0.0616** and **−0.0097**, matching the notebook's 0.062/−0.010 — not a calculation bug. The mechanism, checked directly against the data:

| | Legitimate | Fraud |
|---|---|---|
| Geometric-mean-like typical amount (back-transformed from the log-scale mean) | **$79.10** | **$78.37** |
| Q3/Q1 ratio (multiplicative spread) | 2.73 | **4.59** |
| Share of transactions ≤ $5 | 0.263% | **0.561%** |
| Share of transactions ≥ $1,000 | **1.297%** | 0.905% |

Fraud transactions aren't shifted to be "bigger" — their *typical* size (the geometric-mean-like figure, which is far less sensitive to a few large values than the arithmetic mean) is actually very slightly **lower** than legitimate transactions'. What's different is **dispersion**: fraud's Q1–Q3 range is nearly 60% wider in relative terms, with both a lower Q1 ($35.04 vs. $43.97) and a higher Q3 ($161 vs. $120) than legitimate transactions, and a higher share of very small transactions (consistent with a card-testing-style pattern — small verification charges before a larger attempt, a known real-world fraud tactic, though this dataset can't confirm that's literally what's happening, only that the pattern is consistent with it).

**This changes the interpretation, not just the number.** Cohen's d is a *location* (central-tendency) statistic — it's built to answer "is the center different," and here the honest answer is "barely, and the sign depends on which scale you ask on." The actual, verified difference is a **shape/dispersion** difference (the Q3/Q1 ratio gap), which neither d value is designed to capture. Reporting either d alone — or even both — without the quartile/spread comparison would describe the wrong thing as "the finding."

**Assumptions and limitations of Cohen's d (stated explicitly, as required):** it assumes roughly symmetric distributions with comparable variances. `TransactionAmt` is right-skewed (max $31,937 against a median of $69) and now *verified* to differ in spread, not just location, between groups — exactly the condition under which Cohen's d is the wrong tool, not just an imprecise one. It identifies magnitude of a location difference only, never which transactions drive it, and it is not a causal claim.

**Modeling implication:** don't over-weight transaction amount as a feature just because it's the most intuitive one, and don't expect a location-based transform (log, Box-Cox) to surface the real signal here — a **dispersion-based** feature (e.g. a flag for very-small or unusually-large-for-context amounts, or the amount's deviation from a per-card/per-product typical value) is better motivated by this verified finding than "transaction amount" as a single raw or log-transformed number.

---

## C. Temporal Analysis (relative time only)

**[VERIFIED]** 182 relative one-day bins (`TransactionDT // 86400`) — bin 0 is the first 86,400 seconds of the observed train window, **not** a calendar date.

| | Mean | Std | Min | Max |
|---|---|---|---|---|
| Transactions/day | 3,244.7 | 827.2 | 2,048 | 6,852 |
| Fraud rate %/day | 3.605% | 0.998pp | 1.097% | 6.994% |

**Finding:** daily fraud rate is noisy but stays in a plausible band around the overall 3.499% rate (full chart in the notebook, Figure C2) — no evidence of a dramatic trend or an obvious data-collection anomaly in the volume series (Figure C1), though a roughly 3.3x range between the min and max daily volume is visible and worth a visual check if revisited.

**Why it matters:** supports treating the observed window as reasonably homogeneous for Stage 4's split design, while flagging day-to-day fraud-rate variance (~1pp std) as something Stage 18's monitoring design should expect, not treat as drift the moment it's observed.

**Limitation:** daily bins smooth intra-day patterns; this chart flags *where* to look, not *why* any given day deviates.

---

## D. Missing Data: Structural vs. Quality, by Fraud Status

**[VERIFIED]** Full 394-column chunked scan, split by `isFraud`.

**The standout finding:** `R_emaildomain` and a block of ~14 `V`-columns are **~30 percentage points *less* missing for fraud transactions** than legitimate ones:

| Column | % missing (legit) | % missing (fraud) | Gap |
|---|---|---|---|
| `R_emaildomain` | 77.88% | 45.67% | 32.21pp |
| `V197, V210, V189, V194, V195, V198, V200, V201, V185, V184, V180, V208, V209, V188` (14 columns) | 77.41% | 46.43% | 30.98pp each |

These 14 `V`-columns move together with an almost identical gap — consistent with being populated by a shared upstream process, plausibly tied to whether a receiver/shipping email domain was captured at all.

**Why it matters:** this is not noise — missingness itself carries signal here. **Structural, not necessarily a quality problem:** consistent with Stage 2's expectation that anonymized engineered features can be undefined (not broken) when their inputs don't exist for a transaction.

**Limitation:** this shows *association* between missingness and target, not *why*. Whether `R_emaildomain`'s presence is itself determined by something only knowable after a transaction was already flagged is an open, specific question — see Section G.

**Modeling implication:** a `{col}_was_missing` indicator is a stronger Stage 6 candidate than blind imputation for this block of columns specifically.

---

## E. Categorical Features vs. Target

**[VERIFIED]** Cramér's V (computed from first principles via numpy — no scipy), ranked:

| Column | Cramér's V | χ² | n | Expected-cell-count assumption met? |
|---|---|---|---|---|
| `ProductCD` | **0.1684** | 16,742.17 | 590,540 | ✅ Yes |
| `R_emaildomain` | 0.1635 | 3,670.55 | 137,291 | ❌ No |
| `M4` | 0.1445 | 6,450.45 | 309,096 | ✅ Yes |
| `card6` | 0.1006 | 5,957.03 | 588,969 | ❌ No |
| `P_emaildomain` | 0.0840 | 3,497.81 | 496,084 | ❌ No |
| `M3`, `M2`, `M9`, `M5` | 0.032–0.039 | — | — | ✅ Yes (all) |
| `card4`, `M6`, `M8`, `M7` | 0.007–0.025 | — | — | mixed |
| `M1` | 0.0013 (≈ none) | 0.51 | 319,440 | ❌ No |

**`R_emaildomain` shows up twice** — both in the top missingness-gap list (Section D) and as the second-strongest categorical association here, **despite failing the expected-cell-count assumption** (many categories, some sparse — flagged honestly, not hidden). Its Cramér's V should be read with more caution than `ProductCD`'s or `M4`'s, both of which pass the assumption cleanly.

**Why the expected-cell-count failure matters, concretely (added on review):** `R_emaildomain` has dozens of distinct email domains, and most transactions cluster into a handful of common ones (gmail.com-equivalent, etc.) while many domains have only a handful of occurrences each. Cramér's V computed on a table with that many sparse cells is known to be biased **upward** — some of the 0.1635 is almost certainly inflated by categories too small to estimate a stable rate from, not real signal. This is exactly why `categorical_fraud_rates()` separately filters to `n ≥ 50` before reporting a per-category rate (Figure E1) — the *ranking* (0.1635) and the *per-category rates actually plotted* use different reliability safeguards, and only the latter should be trusted for "which specific domain is risky."

**Association strength is not proof of predictive usefulness — stated explicitly, as this review asked for.** Even a `R_emaildomain` association that *were* fully reliable would only establish that the variable moves with the target in this sample; whether a model can actually exploit it depends on factors EDA can't answer: how it interacts with other features (addressed by multivariate methods, not a univariate V), how it behaves on transactions a model hasn't seen, and whether its sparse categories generalize at all versus just memorizing training-set noise. See the new "Statistical Association vs. Predictive Usefulness vs. Data Leakage" section below for the full distinction, applied to every finding in this report, not just this one.

**Assumptions and limitations of Cramér's V (stated explicitly, as required):** the usual chi-squared rule of thumb wants expected cell counts ≥5; several columns above fail it (sparse categories relative to sample size), which biases V upward for those specific columns — reported per-column above, not applied as a blanket caveat. Cramér's V measures association strength only — not direction, not which category drives it (pair with per-category fraud rates in the notebook), and **not causation, predictive usefulness, or leakage** on its own.

**Modeling implication:** `ProductCD`, `R_emaildomain` (with caution — treat its ranking as a lead to test, not a confirmed signal), and `M4` are the strongest Stage 6/7 categorical candidates; final usefulness is a Stage 7/10 question, not a Stage 3 one.

---

## E (continued). Numeric Features vs. Target — Missingness-First Selection

**[VERIFIED]** Selection criterion decided *before* checking fraud association (to avoid post-hoc cherry-picking):

| Candidate | % missing | Selected (< 15%)? |
|---|---|---|
| `card1`, `C1`, `C2`, `C8`, `C13` | 0.00% | ✅ |
| `D1` | 0.21% | ✅ |
| `card2` | 1.51% | ✅ |
| `addr1` | 11.13% | ❌ (excluded by cap of 6) |
| `D10` | 12.87% | ❌ (excluded by cap of 6) |
| `D15` | 15.09% | ❌ |
| `D4` | 28.60% | ❌ |
| `dist1` | 59.65% | ❌ |

Cohen's d for the 6 selected: **`D1` = −0.366** (the standout — approaching Cohen's "medium" threshold of 0.5), `C2` = 0.203, `C8` = 0.175, `C1` = 0.167, `card1` = −0.074, `C13` = −0.061 (last two negligible).

**Limitation, stated plainly:** this selection criterion necessarily excludes higher-missingness columns (`D4`, `D10`, `dist1`) that Stage 2 already flagged as potentially informative — absence from this list is a limitation of *this chart*, not a finding that those columns don't matter.

**Modeling implication:** `D1` is the strongest "cheap" (near-zero missingness) numeric candidate found this stage.

---

## Memory Technique Demonstration: Optimised Data Types

**[VERIFIED]** Downcasting 4 columns (`TransactionAmt`→float32, `card1`→smallest sufficient int, `card2`→float32, `isFraud`→int8) on a 4-column working frame: **18.90 MB → 6.50 MB (65.6% reduction)**. Demonstration only — not applied to any pipeline; per-column dtype choices are a Stage 5 decision once Stage 4's leakage boundaries are fixed.

---

## F. Identity Coverage vs. Fraud Rate

**[VERIFIED]**

| Group | n | Fraud count | Fraud rate |
|---|---|---|---|
| Has identity data | 144,233 | 11,318 | **7.847%** |
| No identity data | 446,307 | 9,345 | **2.094%** |

**Identity-data presence correlates with a ~3.75x higher fraud rate.** This is the opposite direction a naive "missing identity looks suspicious" assumption would predict — confirming why the task explicitly said not to assume that direction, and showing exactly why checking empirically mattered.

**This finding is revised from how it was originally going to be framed:** before running this analysis, a `has_identity` flag looked like a clean, leakage-free candidate feature ("it's just whether a device record exists, known at transaction time"). **Given the actual size of this gap, that conclusion doesn't hold without more evidence.** If identity/device data collection is itself triggered by a prior risk signal at checkout (e.g., an extra verification step for transactions already flagged as risky), then "an identity record exists" could be downstream of a fraud-risk-aware process — making it leakage-adjacent, not neutral. **This is now a specific, answerable question for Stage 4**, not a resolved feature recommendation.

**What is observed, precisely (added on review, item-by-item):** a measured, full-population, well-powered fraud-rate gap between transactions with and without a matching identity row (7.847% vs. 2.094%, n=144,233 and n=446,307 respectively — both large enough that this is not a sparse-category reliability concern the way `R_emaildomain`'s V is). That is the entire extent of what this stage's data can establish.

**What remains unknown about feature availability (the actual gap in evidence):** this dataset is a static historical export — `train_identity.csv` was already fully populated when received. Nothing in it reveals the **timing** of identity-record creation relative to a production fraud-risk decision. Two mechanisms are both consistent with the exact same observed 7.847%/2.094% split, and they have opposite implications:
1. *Identity/device fingerprinting happens automatically and uniformly* at every checkout attempt, and "no record" simply means the attempt genuinely couldn't be fingerprinted (e.g. an older browser, a blocked script) — in this case `has_identity` is known at the moment a prediction is needed, and it's a legitimate feature.
2. *Identity/device verification is a secondary step triggered by an initial risk signal* (e.g. a 3-D-Secure-style step-up check) — in this case, by the time an identity record exists, part of the risk decision has already been made elsewhere in the business process, and using `has_identity` as a model input would mean partly predicting the outcome using a proxy for a decision that already happened.

**What Stage 4 must specifically investigate (not more EDA on this file — it can't resolve this):**
- Check the Kaggle competition's own documentation/description/forum discussion (external research) for any stated description of how and when identity data is captured, before assuming either mechanism.
- Absent clear documentation, default to treating `has_identity` as **provisionally excluded** from Stage 6 candidate features — the conservative choice given the size of the effect — rather than provisionally included, and require positive evidence of mechanism 1 before reversing that default.
- Record whichever conclusion is reached, and the evidence for it, in Stage 4's own output — this stage deliberately stops at "open question," not a recommendation either way.

---

## Statistical Association vs. Predictive Usefulness vs. Data Leakage

**Added on review, before Stage 3 approval.** These three concepts get conflated easily — this project's own findings (above) give concrete cases of each, and of why none of them implies the others.

**Statistical association** asks: *do two variables move together in the data we have?* Cramér's V and Cohen's d both answer this, and both are computed here on the **full population** (590,540 rows), not a sample — so there's no sampling-uncertainty question about whether the association is "real" in this dataset. `ProductCD` (Cramér's V = 0.1684) and the identity-coverage fraud-rate gap (7.847% vs. 2.094%) are both strong, reliably-measured associations by this definition.

**Predictive usefulness** asks a different question: *will a model, trained properly and evaluated on unseen data, actually benefit from this feature?* This is not established by EDA at all — it requires Stage 7 (Model Selection) and Stage 10 (Evaluation), because:
- A univariate association can vanish or reverse once other features are controlled for (the same thing your `Introduction to Feature Selection Methods` notes get at with L1-regularization-based and tree-importance selection — both *multivariate* methods, precisely because a feature's marginal association isn't the same question as its contribution alongside everything else). Cramér's V and Cohen's d here are a cheap univariate *screen*, not a usefulness verdict — your notes' F-statistic/p-value univariate screening approach has exactly the same limitation, for the same reason.
- A association built on sparse categories (see `R_emaildomain` below) can be partly noise that a model would overfit rather than usefully learn from.
- A feature can be associated with the target in a way a specific model family can't exploit (e.g. a non-monotonic relationship that hurts a linear model but not a tree).

**Data leakage** asks a third, independent question: *does this feature encode information that would not actually be available, or would be causally downstream of the outcome, at the moment a real prediction has to be made?* This is a question about **how and when the data was generated**, not about how strong the association looks. Two implications, both visible in this project's own findings:

1. **A strong association is neither necessary nor sufficient evidence of leakage.** `ProductCD` has a strong, clean association (V=0.1684, expected-cell-count assumption met) and no structural reason to suspect leakage — it's a transaction-time attribute, known before any outcome. The identity-coverage gap (Section F) is an even *stronger*, cleaner association (large n on both sides, no sparse-category problem) — and it's exactly the one case this stage flagged as a genuine open leakage question. **The size of the gap is what makes it worth checking, but the leakage concern itself comes from a hypothesis about the data-generation mechanism (is identity collection risk-triggered?), not from the gap's size.** A large, reliable association is, if anything, the case where it's *easiest* to mistake "the numbers look great" for "this is safe to use."
2. **The real leakage clue for `R_emaildomain` isn't its Cramér's V at all — it's the missingness pattern from Section D.** `R_emaildomain`'s association strength (0.1635) is actually *less* trustworthy than `ProductCD`'s, because it fails the expected-cell-count assumption (many sparse categories). What makes `R_emaildomain` worth investigating for Stage 4 is a completely separate piece of evidence: it's ~32 percentage points less missing for fraud transactions (Section D) — a clue about *when and why the field gets populated*, which is a mechanism question, not an association-strength question. Conflating "has a high Cramér's V" with "is suspected of leaking" would have pointed Stage 4 at the wrong piece of evidence.

**Summary table, using only this project's own findings:**

| Finding | Statistical association | Predictive usefulness | Leakage status |
|---|---|---|---|
| `ProductCD` vs. target | Strong (V=0.168), reliable | Candidate — unverified until Stage 7 | No structural concern; available at transaction time |
| `R_emaildomain` vs. target | Moderate (V=0.164), **unreliable** (sparse categories) | Candidate, but the association number itself should be trusted less | **Open — not from its V, but from its missingness-by-target pattern (Section D)** |
| Identity coverage vs. target | Very strong, reliable (large n both sides) | Looks promising, but usefulness is moot if it's excluded for leakage | **Open — mechanism (is collection risk-triggered?) unknown, regardless of how clean the number is** |
| `TransactionAmt` vs. target | Weak, near-zero, sign-flips by scale (Section B) | Low, on current evidence | No leakage concern — the issue here is weak signal, not leakage |

---

## G. Leakage Investigation — Open Questions for Stage 4

**Established vs. unresolved, stated explicitly (item 6 of this review):** everything in Sections A–F above (prevalence, amount dispersion, temporal stability, missingness percentages, association rankings, identity-coverage rates) is an **established data-quality/distributional finding** — measured, verified, not in dispute. What is listed below is a **separate, unresolved category**: questions about data-generation *mechanism* that no amount of re-analyzing this same static file can answer, per the distinction drawn above. Two **new, specific** findings from this stage, plus two carried over from Stage 2 — named directly, not left as generic placeholders:

1. **`R_emaildomain` + 14 `V`-columns, ~30pp missingness gap by fraud status** (Section D) — a concrete lead: check whether `R_emaildomain`'s presence is determined by something only knowable post-transaction.
2. **Identity-data presence ↔ 3.75x fraud-rate gap** (Section F) — check whether identity collection is itself risk-triggered.
3. **368 anonymized `V`/`C`/`D` columns remain uncleared** (Stage 2) — nothing this stage resolves whether any were computed non-causally.
4. **`train_identity`/`test_identity` naming mismatch** (Stage 2, `id_XX` vs. `id-XX`) — still a Stage 5 prerequisite, not a leakage issue.

**No Stage 1/2 finding was overturned.** Stage 3 upgraded Stage 1's fraud-prevalence figure from `[EXTERNAL, UNCONFIRMED]` to `[VERIFIED]` (3.499%, closely matching the externally-known ~3.5%) — an extension, not a contradiction.

---

## Memory and Performance Observations

- Every analysis read only 1–8 columns at a time; the one full-width (394-column) pass was the chunked missingness-by-group scan (Section D).
- **No row-sampling was used anywhere in this stage.** It was considered (per the explicit requirement to use "representative samples where appropriate") and deliberately not applied: at ~590K rows, a column-selective read is a few MB regardless of row count, which already bounds memory below what sampling would save, while staying exact. Sampling would have been the right call for a much higher row count or a many-columns-at-once analysis — neither applied here.
- Notebook executed end-to-end via `jupyter nbconvert --execute` in **~22 seconds**, 0 errors, all 7 figures saved to `notebooks/figures/`.

## Test Results

`.venv/bin/pytest tests/ -v` → **16 passed, 0 failed, 0 skipped** (9 from Stage 2's `test_data_validation.py`, 7 new in `tests/test_eda_analysis.py` covering `standardized_mean_difference`, `cramers_v`, and the IQR helper against known synthetic inputs). Re-confirmed after this review's notebook addition (verification cell added to Section B, notebook rebuilt and re-executed twice — once to add the cell, once more after a cosmetic string-escaping fix — 0 errors both times, test suite unaffected since no `src/` function was changed this round).

## Unresolved Issues Carried Forward

1. `R_emaildomain`/`V`-column missingness pattern (new, Section D/G) — needs Stage 4 investigation.
2. Identity-coverage fraud-rate gap (new, Section F/G) — needs Stage 4 investigation.
3. 368 anonymized columns' leakage risk (Stage 2) — still open.
4. `id_XX`/`id-XX` naming mismatch (Stage 2) — still a Stage 5 prerequisite.
5. Two virtual environments, no dependency manifest (Stage 0/2) — still unresolved, not yet blocking.
6. `TransactionDT`'s absolute real-world meaning (Stage 2) — still unconfirmed; all Section C analysis stayed in relative terms accordingly.

## Three Interview-Style Questions

1. Transaction amount's Cohen's d flips sign between the raw scale (+0.062) and the log1p scale (−0.010). What does that sign flip tell you about *what's driving* the raw-scale difference, and why would reporting only the raw-scale number have been misleading?
2. The identity-coverage fraud rate is 7.85% with identity data vs. 2.09% without. Why does this finding make `has_identity` a *question* for Stage 4 rather than a safe feature to add in Stage 6, even though "whether a record exists" sounds like it's obviously available at inference time?
3. `R_emaildomain`'s Cramér's V (0.1635) is the second-highest of all categorical columns, but it fails the expected-cell-count assumption. What does that failure mean practically, and would you trust a ranking that put `R_emaildomain` above `M4` (V=0.1445, assumption met) without checking anything else?
