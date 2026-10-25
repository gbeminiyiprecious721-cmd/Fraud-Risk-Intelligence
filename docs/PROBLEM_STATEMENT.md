# Problem Statement — Fraud Risk Intelligence & Decision Support System

**Status:** Revised per user review (Round 2) — see `docs/ML_WORKFLOW_STATE.md` Approval Status for whether this round is itself approved. Nothing downstream (EDA, preprocessing, modeling) has started.

**Revision note (Round 2):** three refinements applied — (1) human investigation prioritization is now the primary decision, automated declines reframed as a future extension; (2) cost assumptions made more explicitly hypothetical, with labelled sensitivity scenarios instead of a single point assumption; (3) investigation capacity K redefined around dataset-relative evaluation windows rather than anything implying real calendar/staffing data; (4) "PR-AUC" renamed to the precise **Average Precision (AP)**, matching `sklearn.metrics.average_precision_score`, with a note on how AP differs from a trapezoidal-interpolated PR-AUC.

## Evidence Classification

Every claim below is tagged so hypothetical framing never gets mistaken for a finding:

- **[VERIFIED]** — confirmed in this repo (Stage 0 structural inspection: file presence, row/column counts, schema).
- **[EXTERNAL, UNCONFIRMED]** — publicly known about the IEEE-CIS Kaggle competition this dataset comes from, but **not yet checked against our own copy** of the data. Must be confirmed in Stage 2/3 before being relied on.
- **[HYPOTHETICAL]** — an assumption invented for the simulated business framing, because this is a portfolio project with no real stakeholder. Clearly a modeling choice, not a fact, and revisable.

No model performance numbers appear anywhere in this document — none exist yet.

## 1. Business Problem

**[HYPOTHETICAL framing, used throughout this document]** The project simulates a fraud risk function at a payment processor ("the Company") handling card-not-present (e-commerce) transactions.

**Primary operational decision (this project's actual scope): ranking and prioritizing transactions for human investigation.** Given a capacity-constrained investigation queue, the model's job is to rank transactions by fraud likelihood so analysts work the highest-likelihood cases first within their fixed capacity. **Automated decline-at-checkout is explicitly out of scope for this project** and is noted only as a plausible future extension (Section 4) — it carries a materially different risk profile (an automated decline acts on a customer with no human in the loop, raising the stakes of a false positive and the bar for calibration/fairness evidence) and should not be assumed as a goal just because the same score could technically drive it.

**Decision supported:** *Given a capacity-constrained human investigation queue, which transactions should be prioritized for review, to catch the most fraud per unit of investigation effort, while bounding harm to legitimate customers?*

**Intended users / decision consumers:**
| User | Decision they make using this model |
|---|---|
| Fraud operations analyst | Which held transactions to investigate first, with what priority — a human makes the final call on every case |
| Fraud strategy / risk manager | Where to set the investigation-queue cutoff given today's capacity and risk appetite |
| Finance / chargeback team | Bears the cost when fraud is missed — a stakeholder for the false-negative cost, not a model user |
| Customer experience / support | Bears the cost when legitimate customers are investigated/delayed — a stakeholder for the false-positive cost |

**Expected value:** A model that ranks transactions by fraud likelihood should let the fraud team catch more fraud per fixed unit of investigation capacity than an un-triaged queue or simple rule-based heuristics (the implicit current baseline for most teams without a model). The differentiator this project aims to demonstrate is not "a model that scores well" but a model whose output is translated into a prioritization decision and a quantified cost trade-off — the step most portfolio projects skip.

## 2. Consequences of False Negatives and False Positives

**False Negative — fraud missed, transaction allowed:**
- **[VERIFIED]** `TransactionAmt` exists as a field in the dataset — that much is a fact about the schema, not about money lost.
- **[HYPOTHETICAL — not a confirmed loss figure]** Using `TransactionAmt` as a *proxy* for financial loss is a modeling assumption, not a measured cost. It ignores partial chargeback recovery, goods that are cancellable/not-yet-shipped, and insurance/reserve offsets a real processor might have — all of which would reduce the true loss below the raw transaction amount. Treating the two as equal would overstate the false-negative cost.
- **[EXTERNAL, UNCONFIRMED — industry mechanism, not a dataset fact]** Sustained high fraud rates can trigger card-network fraud-monitoring programs (e.g. Visa/Mastercard excessive-fraud thresholds), risking processing penalties or account restrictions — mentioned as real-world context for why this matters beyond the immediate loss, not something this dataset can confirm or quantify.

**False Positive — legitimate transaction flagged for human investigation:**
- **[HYPOTHETICAL]** Direct cost = analyst investigation time (a bounded, shared resource) spent reviewing a case that turns out to be legitimate.
- **[HYPOTHETICAL]** Harder-to-quantify cost: if investigation involves contacting the customer or delaying order fulfillment, there's a secondary friction/trust cost — smaller in scope than an outright decline (Section 1 scopes declines out), but not zero. Worth stating explicitly: a model tuned purely to minimize missed fraud, with no false-positive cost term, will over-flag and quietly consume investigation capacity on low-value cases in a way that doesn't show up in a confusion matrix unless you look for it.

These two costs are **asymmetric, not directly comparable in kind, and neither is a measured number** — a missed-fraud proxy-loss vs. an analyst-time cost. That's precisely why metric choice (Section 3) and the threshold decision cannot be "pick the highest-accuracy model" and must instead be an explicit, sensitivity-tested cost trade-off, not a single hard-coded assumption.

**Labelled sensitivity scenarios [HYPOTHETICAL — for stress-testing the threshold decision in Stage 10, not a prediction]:**

| Scenario | FN cost proxy | FP cost proxy | Why include it |
|---|---|---|---|
| A — Conservative loss | 50% of `TransactionAmt` (assumes partial recovery/insurance) | 1 analyst-review unit | Tests whether the chosen threshold is sensitive to overstating fraud loss |
| B — Full transaction value | 100% of `TransactionAmt` | 1 analyst-review unit | The simplest, most-likely-to-overstate-loss assumption from Section 2 above — included as the naive baseline to contrast against, not the recommended one |
| C — Capacity-constrained | 100% of `TransactionAmt` | 5 analyst-review units (models a team that is unusually capacity-starved, so false positives are disproportionately expensive) | Tests whether the threshold recommendation flips when investigation capacity, not fraud loss, is the binding constraint |

None of these numbers come from the dataset or from any real organization — they exist so that Stage 10's threshold recommendation can be reported as "robust across scenarios A–C" or "sensitive to assumption X," which is a defensible claim; a single unlabelled assumption would not be.

## 3. Evaluation Metrics — and Why

**Ruled out: plain accuracy.** Payment fraud is a rare-event label by the nature of the problem — this project's exact class balance is **[EXTERNAL, UNCONFIRMED]** until Stage 3 confirms it on this data, but accuracy is structurally the wrong metric at any plausible fraud rate below roughly 10-20%: a classifier that predicts "never fraud" scores high accuracy and has zero operational value. This conclusion doesn't depend on knowing the exact ratio, so it's safe to rule out now.

**Proposed primary research metric: Average Precision (AP).**
- *Precise definition, not a loose synonym:* "AP" here means exactly `sklearn.metrics.average_precision_score` — the weighted mean of precision values at each threshold, weighted by the increase in recall at that threshold. This is **not** the same computation as a trapezoidal-interpolated area under the precision-recall curve: linear interpolation between two points in PR space is not valid the way it is in ROC space (precision is not guaranteed to vary linearly between two recall levels), so a naive `np.trapz` over the PR curve can give a different, potentially optimistic number than `average_precision_score`. Any reported "AP" in this project's later stages refers specifically to the sklearn function, named precisely to avoid that ambiguity.
- *Why not ROC-AUC:* under severe class imbalance, ROC-AUC's false-positive-rate axis is divided by a huge negative-class count, so ROC-AUC can look deceptively high while precision for the actual flagged set is poor. AP is computed entirely in terms of the positive (fraud) class's precision/recall trade-off, which is what operationally matters. (This is a general property of imbalanced-classification evaluation, independent of this dataset's specific balance — [EXTERNAL, UNCONFIRMED] only in the sense that the *magnitude* of the ROC-AUC/AP gap here hasn't been measured yet.)
- *Why it's the research-stage metric:* AP summarizes ranking quality across all thresholds, which is the right comparison when choosing between candidate models (Stage 7) before any specific operating threshold is fixed.

**Proposed operational metric: precision@K / recall@K over a defined evaluation window.**
- Translates a ranked model output into the actual constraint the business faces: *"within an evaluation window containing K transactions, what fraction of the top-K-ranked are actually fraud (precision@K), and what fraction of all fraud in that window did we catch (recall@K)?"* This is the number a fraud-ops manager or a hiring manager actually reasons with — it maps directly to a capacity decision, not an abstract score.
- **[HYPOTHETICAL, and explicitly not a staffing schedule]** K is defined as a count of transactions per evaluation window drawn from this dataset's own transaction volume — e.g. "K = investigation capacity as a fraction of the held-out window's transaction count" — **not** a real calendar day, shift, or staffing number, since this dataset carries no actual roster or business-calendar information (`TransactionDT` is a relative time offset, not a timestamp tied to a real calendar — see `docs/ML_WORKFLOW_STATE.md` for that still-unconfirmed detail). The exact window size and K will be fixed in Stage 10 once the held-out evaluation set's size is known, and reported as what it is: a dataset-relative evaluation convention, not a real org's staffing plan.

**Proposed decision-layer analysis: cost-sensitive threshold selection.**
- Once a model produces a fraud-probability score, the operating threshold should be chosen by minimizing **expected total cost** = (missed-fraud count × FN cost proxy) + (false-positive count × FP cost proxy), evaluated across the labelled sensitivity scenarios A–C above — not a fixed 0.5 probability cutoff, which only happens to be cost-optimal by coincidence, and not a single hard-coded cost pair either, per the refinement above.
- This is the step that most directly demonstrates "connecting a model to money," per the quantified-business-impact capability that most distinguishes a well-paid DS role from an analyst role — with the explicit caveat that the "money" here is a labelled proxy, not a confirmed figure.

**Realistic success criteria (no invented performance numbers):**
Because no baseline has been run yet, this project cannot state "AP must exceed X" today — that number doesn't exist and inventing one would violate Stage 1's own standard. Instead, Stage 1's success criteria are:
1. A trivial baseline (majority-class predictor, or a simple rule such as "flag all transactions above amount threshold Y") is established in Stage 7 and reported alongside every subsequent model.
2. Any candidate model must be shown to beat that baseline on **both** AP and recall@K, with the margin reported honestly (large, small, or not achieved) rather than assumed in advance.
3. The final chosen threshold is justified by the cost-sensitive analysis above, reported across sensitivity scenarios A–C, not picked by default or from a single assumption.
4. The evaluation is leakage-free and reproducible per `ml-engineering-workflow`'s standards (stage 4/8 requirements) — a prerequisite for *any* of the above numbers being trustworthy.

## 4. Constraints

- **Interpretability [HYPOTHETICAL]:** fraud-ops/compliance stakeholders plausibly need a rationale per flagged transaction, not just a score — this motivates treating Stage 11 (Explainability) as load-bearing for this project, not optional polish.
- **Latency [HYPOTHETICAL]:** since the primary decision is queue prioritization for human review (Section 1), this project assumes **batch/near-real-time scoring feeding a ranked queue**, not hard real-time blocking at checkout. That assumption is what keeps the acceptable model complexity and Stage 14 API design unconstrained by sub-second latency requirements. Revisit if the future decline-at-checkout extension (below) is ever taken up.
- **Regulatory context [EXTERNAL, UNCONFIRMED — general industry context]:** fraud models that use demographic-correlated features can draw fair-lending-adjacent scrutiny in some jurisdictions. Relevant to Stage 12 (Responsible AI); not actionable yet since no features have been engineered.
- **Future extension, explicitly out of current scope [HYPOTHETICAL]:** automated decline-at-checkout, using the same or a recalibrated model, could be considered later. It is **not** a Stage 1 objective because it changes the risk profile materially — no human in the loop, a false positive directly harms a specific customer rather than consuming analyst time, and it would raise the bar on calibration, latency, and fairness evidence well above what queue-prioritization requires. Noted here so it isn't silently assumed back in at a later stage.

## 5. What This Document Does Not Do

- Does not report any dataset statistic beyond what Stage 0 already verified (row/column counts, schema, file presence).
- Does not report a fraud rate, a transaction-amount distribution, or any other value-level fact — those require EDA (Stage 3), explicitly deferred.
- Does not claim any model performance — no model exists yet.
- Does not fix the investigation-capacity number (K), the cost figures, or the latency requirement as final — they are placeholders for a real decision to be made explicitly (and revisited) when the relevant later stage needs them.
- Does not claim this dataset contains real staffing schedules, shift patterns, or calendar dates — `TransactionDT`'s real-world meaning (relative offset vs. an actual timestamp) is itself [EXTERNAL, UNCONFIRMED] until inspected in Stage 2/3; K is defined purely as a dataset-relative evaluation-window convention (Section 3), not derived from or implying any real-world schedule.
- Does not equate `TransactionAmt` with a confirmed financial loss — it is used only as one clearly labelled, sensitivity-tested proxy among several (Section 2).
