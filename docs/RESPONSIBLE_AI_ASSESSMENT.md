# Stage 12 — Responsible AI Assessment

**Status: implementation complete, not yet marked done.** Checks fairness, robustness, and misuse potential for the frozen Stage 9/10 Random Forest before anything ships. Train-only throughout (an auxiliary fold-0 fit, identical in method to Stage 11's cross-check); `final_test` was not accessed; the model was not retrained, retuned, or reselected.

## 1. Responsible AI risk register

| # | Risk category | Description | Severity | Evidence available | Mitigation | Residual uncertainty |
|---|---|---|---|---|---|---|
| 1 | Observable subgroup performance differences | AP/lift varies substantially across `ProductCD`/`card4`/`card6` segments (§2) | Medium — affects prioritization fairness across transaction types | Measured directly, this stage | Report lift per segment; never recommend one fixed global threshold | Whether this translates to a real operational disparity depends on review-capacity policy, not measured here |
| 2 | Demographic fairness limitations | No protected attribute (age, gender, race, etc.) exists in this dataset | High — cannot be closed with this data | Confirmed absent (register, Stage 1-2) | State the limitation explicitly every time; never substitute a proxy as if it were a protected attribute | Permanently unresolvable without a different, consented dataset |
| 3 | Data quality / out-of-distribution inputs | Unseen categories, missing/malformed fields, extreme amounts | Low-Medium — could cause silent bad predictions in production | Tested directly, this stage (§3) | `OneHotEncoder(handle_unknown="ignore")`; `SimpleImputer`; schema validation raises loudly on a missing column | Only a few column types tested; not exhaustive over all 434 register columns |
| 4 | False positives / missed fraud | Flagging a legitimate transaction vs. missing real fraud | High — direct consumer and business harm | Quantified via Stage 10's P@K/R@K and this stage's cost-sensitivity carry-forward | Human-review framing (never auto-decline), capacity-based thresholding | No real financial-loss or customer-harm data exists to calibrate the true cost ratio (Stage 1's own limitation) |
| 5 | Automation misuse | Using the score to auto-decline or auto-restrict without a human | High — this is explicitly the scenario Stage 1 scoped OUT | Not implemented in this portfolio (scope boundary, not a safeguard against an external deployer) | Documented recommendation: human-in-the-loop only (§6) | Nothing in this codebase can prevent a future integrator from misusing the score this way — a policy/governance gap, not a technical one |
| 6 | Human oversight and recourse | No real investigator workflow exists yet | Medium — a design gap, not yet an operational one | This project is pre-deployment | Proposed process documented (§6), not implemented | A proposed process is not a tested one |
| 7 | Privacy and model governance | Transaction/card data is sensitive; anonymised `C`/`V`/`D` columns may still be re-identifiable in combination | Medium — this is a Kaggle research dataset, not live PII, but the *pattern* generalizes | No re-identification testing performed (out of scope) | `.gitignore` excludes raw data from git; no raw transaction data appears in any committed report | A real deployment would need a formal privacy/governance review this portfolio does not substitute for |

## 2. Subgroup-performance evidence

**Evaluated on an AUXILIARY fit of the frozen configuration** — fit on Stage 9's fold-0 training rows (73,020 rows, chronologically earliest ~1/5 of `train`), scored on fold-0's own held-out validation rows (73,020 rows) — the same train-only, genuinely-held-out mechanism Stage 11 used for its permutation-importance cross-check. **Held-out AP: 0.47545** (ROC-AUC 0.87858). This is explicitly a *different population* from Stage 10's `final_test` evaluation and from Stage 9's own cross-validation mean — all three remain separate, non-interchangeable measurements, per the project's standing discipline.

**These groupings are observed transaction and payment-method fields — `ProductCD` (the product/service category), `card4` (card network), `card6` (debit/credit) — NOT verified demographic or protected-characteristic groups.** No subgroup analysis in this section establishes or claims anything about demographic fairness (see §4).

### By `ProductCD`

| Segment | n | n fraud | Sufficient? | Prevalence | AP | AP÷prevalence lift |
|---|---|---|---|---|---|---|
| C | 10,203 | 791 | Yes | 7.753% | 0.68619 | 8.85x |
| H | 9,774 | 228 | Yes | 2.333% | 0.58762 | 25.19x |
| R | 13,891 | 112 | Yes | 0.806% | 0.30059 | 37.29x |
| S | 1,133 | 51 | Yes | 4.501% | 0.34347 | 7.63x |
| W | 38,019 | 696 | Yes | 1.831% | 0.28272 | 15.44x |

### By `card4` (network)

| Segment | n | n fraud | Sufficient? | Prevalence | AP | Lift |
|---|---|---|---|---|---|---|
| american express | 2,667 | 16 | Yes | 0.600% | 0.25379 | 42.30x |
| discover | 1,010 | 30 | Yes | 2.970% | 0.35749 | 12.04x |
| mastercard | 21,268 | 595 | Yes | 2.798% | 0.48963 | 17.50x |
| visa | 48,073 | 1,237 | Yes | 2.573% | 0.47865 | 18.60x |
| missing | 2 | 0 | **No** (below the 100-sample / 10-positive floor) | — | — | — |

### By `card6` (debit/credit)

| Segment | n | n fraud | Sufficient? | Prevalence | AP | Lift |
|---|---|---|---|---|---|---|
| credit | 25,200 | 873 | Yes | 3.464% | 0.52629 | 15.19x |
| debit | 47,810 | 1,005 | Yes | 2.102% | 0.42969 | 20.44x |
| charge card / debit-or-credit / missing | ≤4 each | 0 | **No** | — | — | — |

**Uncertainty and small-sample caveats, stated plainly, not glossed over:**
- `american express` (16 positives) and `ProductCD=S` (51 positives) are the thinnest *sufficient* segments — their AP and lift figures carry noticeably more sampling noise than `visa`'s 1,237 positives, even though both clear the 10-positive sufficiency floor. No bootstrap CI was computed per segment this round (keeping this stage proportional, per instruction); a segment-level CI is noted as a natural extension, not performed here.
- Segments below the 100-sample/10-positive floor (`card4`/`card6` rare categories) are reported with their raw counts and explicitly marked insufficient — never assigned a misleadingly precise AP.
- **This `ProductCD` breakdown is NOT the same measurement as Stage 10's `final_test` `ProductCD` breakdown** (R=15.54x, H=9.24x, S=8.24x, C=5.34x, W=5.05x, from `docs/STAGE10_EVALUATION_PROTOCOL.md`). The two use different populations (fold-0's held-out `train` rows here vs. the one authorised `final_test` evaluation there) and are not expected to match — reported side by side as two separate, honestly-labeled data points, not reconciled into one number.
- **Subgroup AP differences alone are not evidence of discrimination.** A segment with low AP/lift (e.g. `S`, `R` on raw AP; `american express` on raw AP before normalizing) can reflect genuinely different base rates and feature distributions for that transaction/card type, not disparate treatment of any protected group — because no protected group is even represented in this data (see §4).

## 3. Robustness and edge-case findings

**Distinguishing two different questions, explicitly, per instruction:** *input validation* (does the fitted pipeline + model accept an unusual row without crashing, and return a finite probability in [0,1]?) is tested directly below. *Predictive robustness* (does the model's actual accuracy hold up moving forward in time?) is a different question, answered only from already-saved evidence in §3.6 — no synthetic edge case below is interpreted as telling us anything about genuine fraud behaviour.

### 3.1 Transaction-amount extremes (input validation only)

Using one real row from fold-0's validation set as a template, changing only `TransactionAmt`:

| Case | TransactionAmt | Raised? | Predicted P(fraud) |
|---|---|---|---|
| Zero | 0.00 | No | 0.3280 |
| Tiny positive | 0.01 | No | 0.3280 |
| Negative (implausible) | -50.00 | No | 0.3280 |
| 99.9th-percentile value seen in real data | 2,769.81 | No | 0.3206 |
| Far out-of-distribution | 1,000,000.00 | No | 0.3206 |

No case crashed; every probability stayed finite and in [0,1]. **Zero/tiny/negative all produced an identical probability**, and the two large values also matched each other — consistent with how a tree ensemble works (a continuous feature only changes a prediction when it crosses one of the trees' learned split thresholds for *this specific row's* other feature values; once a value is large enough to clear every relevant threshold, further increases change nothing). This is reported as an observation about this one row's decision path, not a general claim about the model's amount-sensitivity. **The negative-amount case is itself a mild input-validation gap**: a real pipeline integration should reject a negative `TransactionAmt` before it ever reaches the model, since no real transaction has one — this system does not currently do that (a concrete, named finding for Stage 13+, not fixed here per the no-retraining/no-pipeline-change instruction).

### 3.2 Unseen categorical values (input validation only)

| Case | Raised? | Predicted P(fraud) |
|---|---|---|
| `ProductCD="ZZ"` (never seen) | No | 0.2822 |
| `card6="prepaid_unknown_type"` (never seen) | No | 0.3216 |

Both handled cleanly via `OneHotEncoder(handle_unknown="ignore")` (an all-zero encoding for the unseen category) — the same mechanism verified at the `ColumnTransformer` level since Stage 5, now re-confirmed end-to-end through the fitted model.

### 3.3 Missing/malformed numeric input

| Case | Raised? | Detail |
|---|---|---|
| Every numeric column set to missing | No | `SimpleImputer` fills with the fold's training median; P(fraud)=0.1984 — noticeably different from the real row's 0.328, since the imputed values differ substantially from this row's actual ones |
| `TransactionAmt` set to the string `"not_a_number"` | **Yes** — `ValueError` | `"Cannot use median strategy with non-numeric data: could not convert string to float: 'not_a_number'"` |

The malformed-dtype case is a genuine, useful finding: **the pipeline fails loudly and specifically, rather than silently coercing or crashing opaquely** — exactly the behaviour a future production integration would want, since it gives an unambiguous, catchable error rather than a wrong prediction.

### 3.4 Feature-schema mismatch

Dropping `TransactionAmt` entirely from the input (simulating an upstream integration bug) raised `ValueError: "columns are missing: {'TransactionAmt'}"` — again, a clear, catchable failure mode, not a silent wrong-shape output.

### 3.5 Out-of-distribution input handling

The far-out-of-distribution amount (§3.1) and the all-missing-numeric-columns case (§3.3) are this project's OOD checks: both returned a finite, in-range probability without crashing. No claim is made that either probability is *correct* — only that the system remains well-behaved.

### 3.6 Early-vs-late chronological performance (predictive robustness, from already-saved evidence — no new fit)

Reusing Stage 9's own 5-fold chronological search results for the frozen candidate (`candidate_index=4`), which are themselves ordered in time (each fold's validation window starts after the previous one's):

| Fold | Validation window (`TransactionDT`) | AP | Prevalence |
|---|---|---|---|
| 0 (earliest) | [1,624,994 – 3,007,352] | 0.47255 | 2.572% |
| 1 | [3,007,361 – 5,134,060] | 0.51617 | 3.988% |
| 2 | [5,134,603 – 7,210,402] | 0.55416 | 4.158% |
| 3 | [7,210,556 – 9,064,152] | 0.45258 | 3.474% |
| 4 (latest, still within `train`) | [9,064,164 – 11,093,269] | 0.56632 | 4.164% |

**No simple monotonic decline is visible moving forward in time within `train` itself** — fold 3 dips below fold 2 before fold 4 recovers, and all 5 folds' AP (0.453–0.566) is HIGHER than Stage 10's `final_test` AP of **0.42784**, which is chronologically after every one of these folds. This is consistent with — but does not, on its own, decompose — Stage 10's already-documented CV-to-`final_test` gap: the available evidence suggests whatever degradation exists is concentrated at the leap to the fully-out-of-sample future population, not a smooth decay visible within `train`'s own time range. This restates, rather than re-derives, Stage 10's three co-plausible explanations (selection effects, training-window/size differences, temporal population change) — no new cause is proposed here.

## 4. Demographic fairness limitations

**This dataset contains no legally or ethically protected attribute** — no age, gender, race, nationality, disability status, or any comparable field exists anywhere in the IEEE-CIS schema (confirmed at Stage 2's schema inspection and never contradicted since). This is a structural, permanent limitation of this specific dataset, not a gap this project failed to close.

Consequently:
- `ProductCD`, `card4`, and `card6` are **transaction/payment-method characteristics, not demographic proxies** — this report does not claim, imply, or test any correspondence between them and a protected class, and such a correspondence is not evaluable from this data.
- **This assessment does NOT establish, and does not claim to establish, demographic fairness.** Any future deployment with access to genuine (and lawfully collected) demographic data would need its own separate fairness audit against those actual attributes — this stage cannot substitute for that.
- The subgroup differences in §2 are reported strictly as *predictive-performance* variation across transaction types, useful for understanding where the model is weaker or stronger, and explicitly NOT as evidence of discriminatory treatment of any group of people.

## 5. Misuse and harm scenarios

| Harm scenario | Realistic? | Proposed safeguard | Implemented in this portfolio? | What a future org would still need |
|---|---|---|---|---|
| Automatic rejection/decline based solely on the score | Yes — the most obvious misuse | Human-in-the-loop review only; no auto-decline | **Scoped out since Stage 1** (the business framing names human-investigation-prioritization as the only in-scope decision) — a documented boundary, not a deployed technical block | An actual enforced policy/system control preventing auto-decline; this codebase cannot stop a future integrator from wiring the score straight into an auto-decline path |
| False accusation / unnecessary account restriction from a false positive | Yes — P@1%≈0.80-0.96 across segments means 4-20% of top-flagged transactions are still false positives | Review by a human investigator before any customer-facing action (§6); report P@K honestly rather than implying near-certainty | Reporting: yes (Stage 10, this stage). Enforcement: no | A real dispute/appeals process (§6) |
| Missed fraudulent activity (false negative) | Yes — R@1% is only ≈0.23-0.31 depending on segment | Capacity-based, not single-threshold, review; explicit non-claim of "catches all fraud" | Reporting: yes. Mitigation: no (inherent to any ranking model at a fixed review capacity) | A monitoring/feedback loop to catch missed fraud downstream (Stage 18-scope, not built) |
| Misinterpreting SHAP explanations as causal | Yes — a named Stage 11 risk | Explicit "SHAP does not identify causes of fraud" language throughout Stage 11's report/notebook | Yes, documented | Operational training for any human investigator who will actually read these explanations |
| Using the score outside this dataset's validated purpose (e.g. credit decisions, employment screening) | Yes — score-reuse creep is a common real-world failure mode | Explicit statement that this model is validated for exactly one purpose: ranking IEEE-CIS-like e-commerce transactions for fraud-investigation prioritization | Documented here and in Stage 1 | Organizational governance preventing scope creep — not a technical control this codebase can enforce |
| Privacy leakage from transaction data or exported artifacts | Low for this specific portfolio (public Kaggle research data, not live PII) but real in general | `data/raw/` excluded from git (`.gitignore`); no raw transaction rows appear in any committed report, notebook output, or figure | Yes, verified (git status checks throughout this project) | A real deployment would need encryption-at-rest, access logging, and a retention policy — none of which apply to a static research dataset |

## 6. Human oversight and recourse (proposed design — NOT an existing deployed capability)

A proposed operational process, documented here as a design recommendation only:

1. **Prioritised review queue**: the model's score ranks transactions; a fixed review capacity (the same `K_FRACTIONS` convention used since Stage 1) determines how many enter the queue — the score *initiates* review, it never *decides* an outcome by itself.
2. **Investigator evidence review**: an investigator examines the flagged transaction's actual evidence (not just the score) — this would plausibly draw on Stage 11's SHAP explanations as one input among several, with the explicit caveat (Stage 11) that SHAP shows model reasoning, not ground truth.
3. **Dispute route for affected individuals**: anyone whose transaction is held, declined, or restricted as a consequence of a flag would need a clear, accessible channel to contest that outcome and request human reconsideration — not built here, but named as a requirement.
4. **Correction mechanism**: a confirmed false positive should be correctable (e.g., releasing a wrongly-held transaction) with the correction itself recorded.
5. **Auditability**: every score-triggered review, its outcome, and any correction would need to be logged in a way that supports later audit — structurally analogous to this project's own append-only experiment registry, but for production decisions rather than experiments.

**None of items 1-5 exist as deployed capabilities in this portfolio** — this project has no API, no queue, no investigator UI, and no decision log. They are named here because Stage 1 already scoped this model's decision as "informs human investigation," which is meaningless without *some* version of this process existing downstream.

## 7. Documentation and learning

**Connections to the Notion Data Science Journal:**
- **Model evaluation**: §2's AP-to-prevalence lift is the same normalization insight from Stage 10's audit (segment AP ÷ segment's own no-skill floor), applied here to a new set of groupings on a new population — reinforcing that raw AP is never comparable across segments with different prevalence.
- **Feature engineering**: the subgroup columns (`ProductCD`, `card4`, `card6`) are raw, un-engineered columns — a reminder that fairness/robustness analysis often needs to work with the *original* business-meaningful columns, not just the transformed feature space SHAP explained in Stage 11.
- **SHAP limitations**: §5's "misinterpreting SHAP as causal" row is a direct callback to Stage 11's own central caveat — responsible-AI review is partly about making sure a prior stage's own stated limitation doesn't quietly get forgotten two stages later.
- **Statistical uncertainty**: §2's small-sample caveats (16 positives for `american express`) apply the same "don't trust a precise-looking number from too few rows" discipline as Stage 10's `MIN_SEGMENT_SAMPLES`/`MIN_SEGMENT_POSITIVES` floors.
- **Data leakage**: §3.4/3.5's schema-mismatch and OOD checks are a different kind of robustness than Stage 4's leakage work, but share the same underlying habit — don't assume a pipeline behaves correctly on an input you haven't actually tried.

**Interview framing:** be ready to explain, precisely, why "no protected attribute in this data" is a *data* limitation that no amount of modeling skill can fix (§4), and why a subgroup AP difference is not, by itself, evidence of discrimination (§2) — these are the two points most likely to be probed in a technical interview about this stage.

## 8. Scientific limitations

- No legally/ethically protected demographic attribute exists in this dataset — §4's limitation is permanent, not an oversight.
- Subgroup evidence (§2) uses an auxiliary fold-0 fit, not the Stage 10 `final_test`-evaluated model or population — a deliberately separate, train-only measurement, not a repeat or validation of Stage 10's own `ProductCD` findings.
- No per-segment bootstrap CI was computed this round (kept proportional per instruction); the smallest sufficient segments (`american express`, 16 positives) carry meaningfully more sampling noise than their point estimates alone suggest.
- Robustness checks (§3) are input-validation checks on one real row's edge cases — not an exhaustive sweep over all 434 register columns, and not a claim about genuine fraud behaviour under any of these conditions.
- The early-vs-late chronological synthesis (§3.6) restates Stage 10's existing, undecomposed explanation set — it does not identify a single proven cause for the CV-to-`final_test` gap.
- Every safeguard marked "not implemented" in §5/§6 is a real, named gap — this is a portfolio project, not a production system, and this report does not claim otherwise anywhere.

## 9. Test results and git status

**246 tests passing** (235 prior + 11 new in `tests/test_responsible_ai.py`, zero regressions). All 11 new tests passed on first run — no bug required a fix this round, stated plainly rather than forcing a "bug found" narrative. The malformed-dtype and schema-mismatch tests assert on the actually-observed `ValueError` behaviour (confirmed by running the real pipeline), not an assumed one.

`git diff --stat` for this round: new files `src/responsible_ai.py`, `tests/test_responsible_ai.py`, `scripts/run_stage12_responsible_ai.py`, this report; modified `.gitignore` (+1 line for `docs/_stage12_*_raw.json`) and `docs/experiment_registry.jsonl` (+1 record, the auxiliary fold-0 fit, `status=completed`). **Zero Stage 1-11 source files touched; zero approved preprocessing/feature-engineering/model-selection code modified; the frozen Stage 9/10 Random Forest configuration was not retrained, retuned, or reselected; `final_test` was not accessed; no new dependency was installed.**

## 10. Three interview-defense questions

1. **"How did you check for disparate performance across groups, given this dataset has no demographic data?"** — Answer: there is no protected attribute to check against, full stop; instead, subgroup performance was measured across real transaction/payment-method fields (`ProductCD`, `card4`, `card6`) using the same prevalence-normalized AP-lift methodology from the Stage 10 audit, with the explicit, repeated caveat that these are not demographic proxies and the results say nothing about discrimination against people.
2. **"What's the worst way this model's output could be misused, and what prevents that?"** — Answer: automatic decline-at-checkout with no human in the loop, which Stage 1 explicitly scoped out of this model's validated use — but that scoping is a documentation boundary, not a technical control; nothing in this codebase can stop a future integrator from wiring the raw score into an auto-decline path, which is itself an honest thing to say in an interview rather than overstating what a portfolio project's safeguards can do.
3. **"Your early-vs-late fold analysis shows no monotonic decline within `train`, but `final_test` still scores lower — what does that actually tell you?"** — Answer: it suggests whatever degradation exists is concentrated at the jump to a fully out-of-sample future population rather than a smooth drift visible within `train`'s own chronological range — but that's a pattern consistent with the existing, undecomposed explanation set from Stage 10 (selection effects, training-window differences, temporal population change), not a new proven mechanism; the honest answer is "consistent with, not proof of."
