# Stage 23 — Model Explainability and Interpretability

Status: implementation verified locally; awaiting owner review. No push, merge,
deployment or Stage24 work.
Date:2026-10-31 (Europe/London).

## Purpose, authorisation and checkpoint

Help fraud operations analysts/risk managers distinguish global model reliance,
local additive model-score contributions and causal claims. This is a synthetic
portfolio decision-support feature, not a new model, fraud verdict, calibrated
probability, live-bank monitoring system or replacement for human investigation.
The owner explicitly authorised Stage23. Do not advance to Stage24.

Branch `stage23-model-explainability` was created from verified integration
`756013d724364d3d16ba1dfcfc39f85fc1679884` with a clean starting tree. Remote main
remains `52c01c919cb40cd2bedf1a41794a73e38c6cf737`. No push/merge/deploy, integration
or main modification, Render change, paid service or model fit is authorised.
Local verified commit only after all acceptance gates pass.

Frozen artifact SHA256 verified during inspection:
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
Loading and inspecting its manifest/schema is not model retraining. At that initial inspection, no SHAP,
permutation, new data scoring or expensive explanation computation had started.
No closed final-test data or prediction artifacts were accessed.

## Existing work and load-bearing findings

- Stage11 already provides SHAP/lineage/permutation utilities, reports and tests.
  Its driver FITS models; do not rerun it. Historical SHAP uses a balanced
  100/100 train sample and its own whole-train-derived feature order. Its
  permutation check uses a separate auxiliary fold-0 fit. These are historical
  experiments, not checksum-verified explanations of the deployed object.
- Preserve all old scientific reports/results. Reuse verified mapping concepts,
  not old numerical explanations as if they described the current artifact.
- The Stage13 artifact/manifest makes ordered preprocessing an identity contract.
  Exact inspection confirms392 raw inputs,378 numeric,14 categorical,813 model
  columns:378 imputed numeric values,271 missingness indicators and164 one-hot
  columns. Its preprocessor output names match the stored ordered schema exactly;
  every transformed column maps to one of the392 raw groups, with no unknowns.
- Existing `src.api.STAGE11_GLOBAL_FEATURE_IMPORTANCE` values come from the
  Stage11 driver's `global_ranking_by_original_column_top15`, defined by SUM of
  mean absolute transformed SHAP. Current UI/comment labels incorrectly call
  them permutation importance. Correct those presentation labels/provenance,
  preserving the values and synthetic generator/scoring behaviour.
- Existing Model Insights contains fixed Stage10 historical performance/service
  context. Do not alter its published scientific numbers. A separate bounded
  Explainability view is proposed because fresh artifact-specific global/local
  analysis has a different provenance and interaction model; link from existing
  Insights rather than crowding or silently replacing historical context.
- SHAP0.54.1 and slicer0.0.8 are already explicitly approved and installed locally.
  No dependency installation/upgrade is proposed. SHAP stays OFFLINE and is not
  added to `requirements-render.txt` or imported by public serving modules.

## Approaches and proposed selection

**Recommended: deployed-artifact MDI plus offline path-dependent TreeSHAP.**
MDI provides training-criterion reliance for the actual frozen ensemble; SHAP
provides an explicitly background-dependent additive account of the same model
on a fixed validation sample and fixed synthetic records. It supports local
explanations, existing installed libraries and precomputed public outputs.
Caveats: impurity/cardinality bias, correlated attribution ambiguity, approximate
implicit background assumptions and synthetic out-of-distribution inputs.

**Alternative: global MDI plus bounded raw-group permutation sensitivity.**
This can measure predictive-metric sensitivity on authorised validation labels,
but does not provide additive local contributions. Permuting correlated inputs
can create implausible rows, and redundant features can hide importance. It
would require a separate fixed scoring/sample/repetition protocol before use.
Do not automatically implement it in addition to the recommended methods.

The proposed delivered method selector includes MDI and mean absolute grouped
TreeSHAP ONLY if both validate. No permutation result, cross-method equivalence,
causal relevance or real-world fraud explanation will be invented.

## Proposed populations and access boundary

### Global explanation population

Use the existing authorised chronological validation partition:75,560 records,
TransactionDT11093842–13452406, ID fingerprint
`14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc`.
Select512 unique IDs without replacement from IDs sorted ascending, using
`numpy.random.default_rng(42)`. Selection is fixed before explanations, independent
of labels, scores, predictions or attractive plots. Read split metadata only to
select IDs; use the established ID-first physical-row reader to parse ONLY
selected validation features. Verify actual timestamps/IDs against metadata.
No outcome columns are needed/requested. All scores use the frozen preprocessor;
no fit or refit. Validation has repeatedly been reused during development and
is not a new pristine holdout. Global SHAP describes this fixed sample, not an
unbiased population guarantee; global MDI depends on fitted trees, not the sample.

### Local synthetic population

Generate the EXACT existing `_synthetic_dashboard_row` records, with unchanged
raw feature order and Python Random seed, for seeds101,202,303,404,505 and the
first48 rows per seed (240 fixed synthetic records). Generate no historical
examples for public local explanation. Verify generation/scoring against the
shared `build_dashboard_queue_payload` for n24 and n48; the first24 raw rows and
scores must agree, though seeded display statuses/risk bands are batch-relative.

Precompute240 local explanations offline for the normal48-row local cohort.
Public access is restricted to the first24 per approved seed (120 identifiers),
matching Render's current public queue. A server-owned per-app limit can restrict
public lookup to24 while local lookup permits48; no caller-controlled override.
Arbitrary seeds, unknown/malformed identifiers, historical IDs and sequence
positions outside the allowed scope return controlled errors, never inference.
No features, case notes or assignments are submitted to the explanation API.

Most synthetic numeric inputs are fixed100 and many categories are `unknown`,
which the fitted encoder may encode as all-zero. These are fictional, often
unrepresentative records. Mathematical fidelity to their model score is not
proof of useful real-world risk interpretation or an explanation on the natural
historical data manifold.

## Exact methods, mapping and output semantics

### MDI

Read `artifact.model.feature_importances_` in the exact813-column schema order.
Validate nonnegative finite values and sum1 within numerical tolerance. Group
by summing all transformed importances mapped to each original input, including
its missingness indicator and one-hot columns. Verify every transformed index
appears once, no index is omitted or reused, and grouped total equals original.
MDI measures fitted impurity reduction; it is biased by cardinality and can
spread credit across correlated predictors. It is not an accuracy guarantee.

### TreeSHAP

Use installed `shap.TreeExplainer(artifact.model,
feature_perturbation='tree_path_dependent',model_output='raw')` with exact frozen
preprocessing; no approximate SHAP or surrogate model. Infer class index from
`model.classes_==1`, verify the actual returned class-axis shape and fail closed
rather than assuming ambiguous single-output semantics. For this Random Forest,
validate that the extracted output is its uncalibrated class-1 ranking score,
not a log-odds margin or calibrated probability.

The background is the fitted trees' recorded training-path counts, not an
explicit sampled background. This does not establish the true conditional joint
distribution or solve correlation ambiguity. Do NOT repeat stronger historical
wording that path-dependent SHAP fully preserves the actual joint distribution.
Check the SHAP baseline independently against fitted tree leaf probabilities
weighted by their recorded path sample weights. No private tree structure,
thresholds or training examples are exposed publicly.

For each original input g, group SIGNED contributions FIRST:
`phi_g(x)=sum(phi_j(x) for j mapped to g)`.
Global SHAP importance is `mean(abs(phi_g(x)))` over the fixed512 sample. This is
NOT Stage11's sum-of-component-mean-absolute values; within-group cancellation
makes those different estimands. Explain the distinction instead of silently
comparing numerical ranks. Check grouped signed sums against all813 contributions.

Require `baseline + sum(all contributions) == actual frozen score` within1e-8
absolute tolerance for every global/local/probe row, before publication. Verify
against both direct fitted-model output and deployed preprocessing/scoring.
Queue scores rounded to6 decimals are reconciled with their separate rounding
bound (at most0.5e-6 plus summation tolerance); raw contributions reconcile to the
unrounded score. No contribution or baseline is labelled calibrated probability.

Local public payloads show baseline, exact model score, top10 original groups by
absolute signed contribution, and a SIGNED Other-features residual aggregating
all omitted groups. Positive and negative values respectively increase/decrease
the model score relative to this baseline. Baseline+displayed groups+Other must
reconcile within1e-8 even after serialisation. Other is not discarded or forced
positive; no arbitrary salience percentages are invented.

Historical public global data contains bounded top20 rankings, method/sample
provenance, aggregated diagnostics and limitations only. No historical row
identifiers, values, labels, individual attributions, model object or unlimited
introspection is exposed. Local payloads contain fixed synthetic identifiers
and contributions only; descriptive attributes remain visibly separate.

## Feasibility, stability and resource policy

After protocol review, run a bounded two-row compatibility/additivity probe
before the full calculation. Prior Stage11 measured about0.04s per row, but that
is not a benchmark of this artifact; record actual probe/run wall time and peak
RSS. Abort or fall back to validated global-only output if methods fail schema,
shape, output-space, additivity or resource checks. No silent tolerance relaxation,
package upgrade, new method or model retraining to make a failed probe pass.
Target complete offline run under10 minutes and4GiB RSS; monitor a subprocess
and terminate cleanly if limits are exceeded. Save completed verified evidence.
No SHAP computation on any public request or browser interaction.

Assess global ranking variability with200 deterministic resamples of16 contiguous
blocks of the512 selected rows sorted by TransactionDT/ID, advancing boundaries
past timestamp ties. Recompute grouped mean absolute SHAP from existing values;
report top10 overlap/rank spread as descriptive sample-resampling behaviour,
not a confidence guarantee, fairness result, causal validity or training stability.
If16 nonempty tie-safe blocks cannot be formed, report insufficient evidence.
Do not compute new raw-correlation clusters from closed data or infer meanings
for anonymised C/V inputs. Carry forward known correlation limitations and
explicitly state that grouping one-hot/missingness for a raw feature does not
eliminate correlation between DIFFERENT raw features.

Bounded offline synthetic-only stress probes add missing numerical values and
reference-unseen categorical values to designated fixtures, reusing the exact
preprocessor. Check safe transformation, supported output and additivity; do not
publish them as observed historical issues or evidence of rare-category fairness.
Path-dependent background is fixed by the model; sensitivity to alternative
background definitions is NOT established by these checks and will be disclosed.

## UI/API architecture and safeguards

Proposed dedicated `#/explainability` view reuses current navigation/theme/charting.
Existing Insights gains accurate historical method labels and a clear link to
new deployed-artifact explanations. Keep prior Stage10 metrics unchanged.
Provide MDI/SHAP selector, global ranking, feature interpretation/mapping panel,
method/provenance/limitations, synthetic local contribution visualisation and
baseline/Other/score reconciliation. Allow a selected synthetic transaction/case
to open its supported explanation. Distinguish deployed RF from experimental
challengers; no XGBoost promotion. No fonts/colours/unrelated layouts redesigned.

Use fixed versioned aggregate resources, read-only global and identifier-lookup
endpoints, explicit input/output allowlists and response-size bounds. Strict
identifier validation must precede lookup; no arbitrary paths or feature vectors.
Maintain existing rate limits, body caps, CORS and public prediction restrictions.
No backend case writes or case-storage/schema migration. Reset affects only
explanation controls; notes/history remain unchanged. Missing/corrupt assets or
unsupported records show clear recovery/availability messages. Validate model
checksum/ordered-schema identity to prevent explaining mismatched deployed scores.
No new Render dependencies: serving code reads validated JSON using existing
runtime libraries and does not import SHAP or the offline builder.

## Acceptance and delivery gates

- Unit/scientific tests for exact mappings/order, preprocessing, deterministic
  global/local output, baseline/grouped/serialised reconciliation, MDI totals,
  zero/negative attribution handling, insufficient stability evidence, unknown
  categories, missingness, invalid IDs/assets and aggregate privacy boundaries.
- Independent numerical checks and regeneration, with no model fit or closed
  final-test access. Every displayed explanation must match the frozen artifact.
- API tests for approved bounded identifiers, public/local scope distinction,
  read-only behaviour, output schema/size and unchanged public safeguards.
- Real Chrome at1440×900,1280×800,768×1024,390×844: selectors, charts, keyboard,
  case-to-explanation link, reset/retry, unavailable/unsupported states, responsive
  layout and preservation of populated local case data. Capture screenshots.
- Complete existing applicable regression, including all six heavy/data-contract
  tests and Stage19–22/case/security/scoring regressions. Existing heavy reference
  fits remain disposable train-only test artifacts, never the deployed model.
- Final checksum, unchanged original scientific workflow body, protected paths/
  refs, exact inventory, results/resource costs and limitations documented here
  and in a new Stage23 workflow prefix. Commit verified work locally only.

## Primary method references

[SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
documents tree-path counts, output spaces and class-wise shapes. Algorithmic
additivity and a baseline assumption do not establish causal interpretation.
[Scikit-learn correlated-feature example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)
illustrates impurity/permutation interpretation limits with redundant predictors.

No explanation result, runtime benchmark or completed test count is claimed at
this inspection/design checkpoint. This records the initial pre-computation checkpoint; approval and results follow.


## Protocol clarifications after independent design review

The independent read-only review found the design viable and identified details
to freeze before execution. The following rules settle those details; they are
part of the protocol, not implementation-dependent choices.

### Independent baseline equation

For each fitted tree t, let L_t be its leaves, w_t(l) its
`weighted_n_node_samples[l]`, and p_t(l,1) the class-1 leaf probability (normalise
the stored class-value vector by its own sum, whether stored as proportions or
counts). Compute:

`b_t = sum(w_t(l)/w_t(root) * p_t(l,1) for l in L_t)`;
`baseline = mean(b_t across all fitted trees)`.

Verify positive root weights, leaf weights summing to root weight, explicit
forest/tree class-index alignment and equality to TreeSHAP's selected baseline
within1e-10. Do not pool weights across different trees or use unweighted sample
counts. This is the fitted trees' implicit training-path expectation, not a new
estimate of future fraud prevalence or a historical sample's mean prediction.

### Resampling fixed algorithm

Sort selected512 rows by TransactionDT then TransactionID. Boundaries are
floor(i*n/16), i=1..15, advanced past timestamp ties, with0 and n as endpoints.
Require16 nonempty blocks. Use `numpy.random.default_rng(20231023)` for200
replicates; each replicate samples16 block indices uniformly with replacement
and concatenates all rows in those selected blocks. Compute the row-weighted
mean absolute GROUPED contributions over the concatenated rows, not an unweighted
mean of unequal block means. Never recompute SHAP during resampling.

Rank all392 raw groups descending by statistic, then ascending raw name to
break ties deterministically. Define top10 overlap as the intersection size of
replicate/base top10 divided by10. Publish overlap min/median/max and, for each
base top20 feature, rank min/q10/median/q90/max (linear quantiles). Rank universe
is all392, not only published features. Do not call these quantiles confidence
intervals. Zero-importance ties are name-resolved display order, not evidence of
scientifically distinct importance. If blocks cannot be formed, show explicit
insufficient stability evidence without claiming zero variability.

### Concrete probes and stopping rules

Two compatibility rows are fixed before execution: seed101/sequence0 and
seed202/sequence0, generated by the existing synthetic generator. Stress fixtures
copy seed101/sequence0, separately applying:

1. `dist1=None` (numeric missingness/imputation path).
2. `card4='__STAGE23_UNSEEN__'` (deliberately unseen category, not an observed
   historical value).
3. If eligible, replace card4 with the lexically first category having the
   lowest positive count no greater than5 in the fixed512 validation sample.
   Selection uses category counts only, never scores/SHAP. If no such category
   exists, record insufficient evidence for this probe. Rarity is relative to
   this sample, not a claim about training or population frequency.

Every probe is synthetic and remains separate from historical summaries.
Probe time/RSS and all later work count toward the same10-minute/4GiB budget.
A failed checksum, schema/order or lineage gate aborts all publication. A SHAP
compatibility/baseline/global-row additivity failure disables BOTH global and
local SHAP; only independently validated MDI may remain. If global SHAP passes
but any fixed-cohort/local/stress gate fails, disable the ENTIRE local cohort;
global SHAP may remain only if its own gates passed. Never publish a convenient
subset of passing local examples after seeing failures. Resource limits prevent
publication of incomplete results. Atomic output writes include capability flags;
unavailable methods/selectors are disabled, local absence is explained, and no
stale partial SHAP output is retained. Record the actual scientific blocker.

### Asset identity, privacy and delivery boundary

Use resource schema version1.0.0, artifact SHA256, ordered transformed-schema
fingerprint and a canonical synthetic-generator fingerprint. The latter covers
the existing function source, referenced generation constants, raw schema order,
approved seed list, Python version and row count. It excludes unrelated seeded
status/risk-band presentation. Validate a complete, unique240-identifier local
inventory against the fixed seed/sequence Cartesian product before serving.
Public per-app lookup permits only the first24 identifiers per seed; local
lookup permits48. Other locally generated seeds/batch sizes remain supported by
the existing dashboard but receive an explicit explanation-unavailable message.
Do not remap unsupported explanation IDs to a different synthetic transaction.

Store builder outputs in a SERVER-ONLY `explainability_assets/` directory outside
`ui/` and every static mount; do not add a static route for that directory.
Historical global asset contains bounded aggregates only. Full240 local asset is
never returned by an endpoint. Static-path negative tests must show that neither
that file nor private diagnostics can be fetched directly. The existing local
static UI server serves only `ui/`, so it cannot expose the server-only assets.

Validate fixed assets before serving/caching: strict required-field allowlists,
finite numerical ranges, exact identity fingerprints, permitted raw-group names,
unique IDs, complete inventories and local baseline/contribution/score checks.
JSON serialization rejects NaN/Infinity. Compare identity with the actually loaded
application artifact, not merely the asset's self-declared checksum. Maximum
on-disk global resource128KiB, local resource1MiB; individual public response16KiB.
Cache only validated immutable resources; invalid/mismatched resources fail closed
with503 and generic guidance. Lookup input is a bounded identifier only, never a
raw feature vector, path, case note or SHAP background. Public serving imports no
SHAP or offline generation driver and performs no explanation/inference work.

## Inspection checkpoint saved

Only this protocol document is uncommitted on `stage23-model-explainability`.
The model and prior scientific records remain unchanged. At that checkpoint expensive computation
and product implementation awaited protocol approval. The owner subsequently
approved it explicitly; the results below supersede that pending status.


## Implementation ruling — signed residual display

The predeclared top10-by-absolute-value selection is unchanged. Some synthetic
examples have ten positive leading groups, which would hide smaller negative
contributions within a positive Other net. The payload therefore additionally
reports Other-positive and Other-negative subtotals; their sum must equal the
same signed Other net. Charts display those two subtotals instead of duplicating
the net; tables mark them as INCLUDED components. This improves sign visibility
without selecting different examples/features, dropping any contribution or
changing the attribution method/reconciliation equation. Zero subtotals remain
zero; negative effects are never manufactured.


## Approved implementation and measured results

The owner explicitly selected “Approve MDI + TreeSHAP protocol (recommended)”.
No package was installed/upgraded. Existing SHAP0.54.1/slicer0.0.8 were used
OFFLINE only. The baseline, sample, methods and resource policy were fixed before
computation. No frozen-model fit, outcome read or closed-data access occurred.

### Provenance, mappings and mathematical validation

Fixed512 validation sample ID SHA256:
`b65d62465b93fe4bc5e969be95a177d7a6b7acfc368a0a2824600e583bac7362`.
Selection used default_rng(42) on sorted authorised validation IDs without
replacement or labels. The source reader skipped all nonselected feature rows
before parsing and checked timestamps/IDs. No sample fraud prevalence was read
or inferred. These are reused development-validation observations, not a new
independent performance evaluation.

Exact813 model columns map to392 raw groups:378 numeric value columns,
271 missingness indicators and164 one-hot columns. Both offline grouping and
runtime validation reconstruct complete, unique mapping against the fitted
ordered schema. The original preprocessing/scoring code and artifact are
unchanged. Local generation/scoring agrees with existing n24/n48 queue outputs
for all five approved seeds, including the six-decimal score-rounding boundary.

Independent weighted-leaf baseline: **0.035109660485021406**. TreeSHAP's
class-1 baseline differs by at most6.94e-18. All2 compatibility rows,512 global
rows,3 synthetic stress fixtures and240 synthetic locals passed raw, grouped and
serialised reconciliation. Largest measured raw additivity error **7.22e-16**,
well below1e-8; public serialisation is rounded to12 decimals and remains within
that tolerance. The single-feature synthetic-tree unit test independently checks
known SHAP values−0.5/+0.5 with baseline0.5. Disposable toy tests never touch or
replace the production artifact.

### Global findings — different methods, different meanings

| Rank | MDI raw group | Impurity share | Grouped TreeSHAP raw group | Mean absolute score contribution |
|---|---|---:|---|---:|
|1|C1|0.028964|C13|0.002249|
|2|V257|0.022663|card6|0.002057|
|3|C13|0.019582|C14|0.002007|
|4|V242|0.019233|C1|0.001993|
|5|V188|0.019164|C5|0.001320|

MDI sums to1 across all groups before publishing top20. Its numbers describe
fitted impurity-reduction allocation, not fraud probabilities. Grouped SHAP uses
mean absolute SIGNED group sums on the512 examples; its values are score units,
not impurity shares, causal effects or predictive permutation sensitivity.
Cross-method rank differences do not prove one method wrong or validate either
as a causal account. The new figures must not be numerically compared to the
legacy balanced200-train SHAP table as model-performance improvement/decay.

### Local findings, signed residuals and stability

All240 fixed local examples validate. Public access is limited to120 identifiers
(first24 for each seed); local48-row cohorts use the full fixed240. Unsupported
locally generated seeds/sizes receive a clear unavailable state rather than a
surrogate explanation. For seed101/sequence0, raw model score is
0.532089436879. The implicit baseline is0.035109660485; the signed Other net is
0.287499120483, consisting of Other increases0.467530031803 and Other decreases
−0.180030911320. These subtotals are INCLUDED in Other, never double-counted.
The leading ten individual raw groups remain selected by absolute attribution.
Synthetic defaults often create large departures from the baseline; they do not
represent verified customer risk or plausible counterfactual interventions.

Two hundred16-block resamples yield top10 overlap **min0.70 / median0.90 /
max1.00**. For C13 the rank range is1–3; for card6 and C14 it is1–4 and2–4.
These are descriptive selected-sample block-resampling findings, not confidence
intervals, a fairness test, background sensitivity evidence or general robustness.
The rare-category probe was eligible at five occurrences in this sample only.
Missingness, deliberately unseen categories and that sample-sparse category all
passed preprocessing and additivity; this does not establish rare-group fairness.

### Reproducibility and resource observations

The final global/local resources regenerate byte-identically:

- global-v1.json SHA256 `b6985a81087cba786d927e59967038da87804db27794429ee78f53649d84a44c`
- local-v1.json SHA256 `facb66f0d1f19d1bf3c3d60e3f9c2f69fc535d17df84555afa139fd83129b168`

The local asset changed once to add the documented signed Other subtotals; final
repeat evidence verifies BOTH final resources, not equality to the earlier
pre-subtotal schema. Earlier build evidence is retained as chronological history.
The final watched build took **39.48s**, peak **399,458,304 bytes (~381MiB)**,
within10minutes/4GiB. The initial watched run took38.82s/403,013,632 bytes.
Actual two-row probe was0.14s, global SHAP19.86s, local SHAP8.23s on this machine.
Times are local observations, not Render request latency guarantees. Resource
monitoring required native process-list permission under the Codex sandbox;
subsequent runs were fully monitored with no budget breach.

The global server asset is34,872 bytes and local asset139,477 bytes. The global
public response is about8.2KiB after excluding the392-group server-side catalogue;
individual responses are capped at16KiB. Render performs only bounded JSON
validation/cache/lookups: no SHAP imports, historical source access, transforms,
model inference or new runtime packages. No paid service or Render change.

### UI, security and source review

Dedicated Explainability preserves the existing Model Insights/Stage10 context.
The historical Stage11 values remain unchanged; only their incorrect permutation
label/provenance is corrected. New controls select MDI/SHAP, a raw feature and a
current synthetic transaction; case drawers open their exact supported score.
Charts/tables show method units, sign meanings, baseline, all displayed groups,
Other and reconciliation, with full methodology/limits. Exact tables disclose
six-decimal display precision; payload values retain12-decimal serialisation.

Resources live outside all static UI directories. No endpoint returns the full
local asset or historical per-row attributions. Server-only describes the app's
HTTP delivery boundary, not a secrecy claim about fictional synthetic source
assets. Global source data is aggregate-only. Input allowlists, actual loaded
artifact/schema/generator identity, resource sizes, numerical bounds and complete
inventories are validated before caching. No note/assignment/body data is sent
by case-to-explanation navigation; no write endpoints, store/key/schema migration,
rate-limit change or prediction-body change was added.

Independent review found one missing enforcement of the global16KiB response
bound. A regression reproduced it; the bound now applies before caching, and
re-review found no residual blockers. A direct-link batch-identity race found
by Chrome tests was fixed by waiting for verified batch identity before loading.
Corrupt/unavailable/mismatched resources and unsupported records fail clearly,
while global/case functionality remains usable. MDI-only fallback and withheld
local output are tested without retaining failed partial SHAP results.

No causal, calibrated-probability, future-performance, live-banking, background-
robustness or fairness claim is made. Anonymised meanings, correlated credit,
fixed-background assumptions, sample size/reuse and synthetic OOD limitations
remain material. Genuine analyst use would require verified semantics, realistic
eligible populations, label/case governance and appropriate human validation.


### Additional interpretation boundary

Signed summation of transformed-column SHAP is an additive reporting group;
it is not a recomputation of joint raw-group Shapley/Owen values with groups as
players. One-hot mutual exclusivity and dependencies across raw inputs remain
relevant. The explicit grouping formula, not an implied causal coalition model,
defines the published quantity.

### Final verification ledger

The first complete run passed554 tests, zero failures/skips, five inherited
warnings,332.18s. It is retained as pre-fallback-verification evidence. A final
browser regression was added to ensure MDI-only fallback never claims completed
SHAP, increasing the final suite to555 collected tests, including84 real-Chrome
cases and all six heavy/data-contract tests. Seven focused explanation Chrome
checks pass; the scientific/API suite passes23 tests. Final complete run: **555 passed, zero failures/skips, five inherited warnings,
318.99 seconds**, including84 real-Chrome cases and all six heavy/data-contract
tests. See [full regression](stage23_explainability_evidence/full-regression.log).
The existing warnings are the Starlette deprecation and sklearn logistic
penalty/convergence warnings, not failures of the deployed model.

Twelve screenshots cover both methods at all four viewports, with four extra
full-page mobile/tablet captures included in that total. Public-app checks use
local `src.render_demo:app` in a disposable profile, verify the24-record queue,
case-to-explanation link, selectors/charts/reset, unchanged populated case bytes,
no runtime errors and no desktop outer content overflow. Native keyboard and
fallback/retry/score-mismatch checks are in the browser regressions. Existing
Stage19–22/production/security tests remain in the complete suite.


## Independent numerical check and preservation

A fresh artifact load independently re-generated/scored all240 synthetic records.
Largest difference from cached scores:4.96e-13 (12-decimal serialization only).
All signed/Other subtotals reconcile; maximum serialized residual2.00e-12.
All392 MDI groups sum to1, and published top20 ranks/values match the actual
artifact. A separate root-class-proportion baseline0.0351096604850214 agrees
with the weighted-leaf and SHAP baselines. This additional audit reads no
historical features or outcomes; see
[independent checks](stage23_explainability_evidence/independent-numerical-checks.json).

Preservation verifies the frozen SHA256, entire prior workflow/science body,
unchanged fixed Stage10 panel, Stage11 source/values, Stage19/20/22 aggregates,
case-store/UI implementation and storage contracts, production preprocessing/
scoring and pinned dependencies. The production API AST differs only by new
read-only router registration; Render changes add a static JS asset and the
server-owned24-record explanation scope, preserving existing prediction and
rate-limit behaviour. No closed final-test data, new outcomes, real customer
information, paid services or Render settings were accessed/changed.

## Evidence and local delivery

- [Full555-test regression](stage23_explainability_evidence/full-regression.log)
- [Scientific/API tests](stage23_explainability_evidence/scientific-api-tests.log):23 passed
- [Focused Chrome checks](stage23_explainability_evidence/focused-browser.log):7 passed
- [Final build and reproducibility](stage23_explainability_evidence/build.json)
- [Measured resource budget](stage23_explainability_evidence/resources.json)
- [Independent review](stage23_explainability_evidence/independent-review.json)
- [Preservation](stage23_explainability_evidence/preservation.json)
- [Public browser metrics](stage23_explainability_evidence/public-browser-verification.json)
- [Desktop1280×800](stage23_explainability_evidence/explainability-shap-1280x800.png)
- [Mobile full view](stage23_explainability_evidence/explainability-shap-390x844-full.png)
- [Exact changed-file inventory](stage23_explainability_evidence/changed-files.txt)

The completion gate passes for this bounded historical/synthetic portfolio
feature. Local explanations were feasible and validated, so no scientific blocker
required their omission. No remaining blocking software defect is known.
Physical-device/assistive-technology certification and genuine operational
explanation governance are not claimed. Existing scientific limitations above
remain material. Branch `stage23-model-explainability`; local commit SHA is
reported at delivery. No push, merge, deployment, model promotion or Stage24 work.

Final inventory: **49 files**, exactly enumerated in changed-files.txt.
