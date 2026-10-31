# Stage 24 — Portfolio documentation evidence audit

Status: Stage 24 release review verified (2026-10-31); ready for PR creation. The owner now
authorises pushing this branch and opening a PR against
`render-dashboard-integration`. This supersedes the original local-only delivery
restriction recorded below. Merge, deployment, Render changes and Stage 25
remain unauthorised. The complete Phase A–I implementation brief is retained.
Audit date: 2026-10-31. Branch: `stage24-portfolio-documentation`.
Approved integration baseline: `942b6d53ae004b847a8c0eca9a9298f1fe0c3024`.

## Evidence inventory established before rewriting

| Claim or boundary | Authoritative repository evidence | Permitted interpretation |
|---|---|---|
| Human investigation prioritisation | [Problem statement](PROBLEM_STATEMENT.md) | Simulated capacity-constrained decision support; no measured bank outcome or automated decline policy |
| IEEE-CIS structure, keys and identity coverage | [Data validation](DATA_VALIDATION_REPORT.md), [EDA](EDA_REPORT.md) | Historical dataset inspection; raw restricted data is not redistributed |
| Chronology and leakage restrictions | [Leakage boundaries](LEAKAGE_BOUNDARIES_REPORT.md), [feature availability register](feature_availability_register.csv) | Relative-time partitions and explicit availability caveats; no established label-maturity guarantee |
| Train-only preprocessing | [Preprocessing report](PREPROCESSING_REPORT.md) | 392 raw inputs; fitted imputation/encoding; ordered transformed schema |
| Engineered-feature adoption | [Feature engineering report](FEATURE_ENGINEERING_REPORT.md), [register](engineered_features_register.csv) | Prototypes are not silently represented as deployed features |
| Original tuning and final evaluation | [Stage 9](STAGE9_TUNING_DESIGN_REPORT.md), [Stage 10](STAGE10_EVALUATION_PROTOCOL.md), [registry summaries](experiment_registry.jsonl) | Historical published AP 0.42784, ROC-AUC 0.86637, P@1% 0.8583, R@1% 0.2461; no re-evaluation |
| Frozen artifact identity | [Production pipeline](PRODUCTION_PIPELINE_REPORT.md), [manifest](../artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json) | Later reconstruction of the frozen configuration; historical Stage 10 metrics are not a new evaluation of this joblib |
| Responsible use | [Responsible AI assessment](RESPONSIBLE_AI_ASSESSMENT.md), [historical explainability](STAGE11_EXPLAINABILITY_REPORT.md) | No calibrated probabilities, demographic fairness certification or causal explanations |
| API, application and operational evidence | [API report](STAGE14_API_REPORT.md), [application integration](STAGE15_APPLICATION_INTEGRATION_REPORT.md), [testing](STAGE16_AUTOMATED_TESTING_REPORT.md), [deployment](STAGE17_DEPLOYMENT_REPORT.md), [monitoring](STAGE18_MONITORING_REPORT.md), [dashboard](DASHBOARD.md) | Distinguish original scientific-stage numbering from later portfolio enhancements |
| Capacity simulator | [Stage 19](STAGE19_CAPACITY_SIMULATOR.md), [aggregate](../ui/capacity-v1.json) | 201 checkpoints on reused historical validation, not on synthetic queue labels |
| Four-model comparison | [Stage 20](STAGE20_CHALLENGER_BENCHMARK.md), [aggregate](../ui/benchmark-v1.json) | Identical 75,560-row validation population with 2,617 positives; 804 checkpoints; exploratory paired block bootstrap |
| Case management and queue semantics | [Case management](STAGE21_CASE_MANAGEMENT.md), [queue corrections](STAGE21_QUEUE_FILTER_CORRECTIONS.md), `ui/case-store.js` | Synthetic browser-local cases; opened-case filters; confirmed reset; no server case writes |
| Historical monitoring | [Stage 22](STAGE22_MODEL_MONITORING.md), [aggregate](../ui/monitoring-v1.json) | Disjoint validation windows of 25,186 / 25,187 / 25,187 rows; investigation heuristics, not production accuracy estimates |
| Exact-artifact explainability | [Stage 23](STAGE23_MODEL_EXPLAINABILITY.md), `explainability_assets/` | 813 columns grouped to 392 inputs; 512 label-free global examples; 240 synthetic local explanations, 120 publicly allowlisted |
| Runtime and experimental dependencies | `requirements-render.txt`, `requirements-benchmark.txt`, `requirements.txt` | Minimal serving requirements are separate from offline challenger training and explainability computation |
| Local serving and artifact delivery | `src/render_demo.py`, `deploy/render_fetch_artifact.sh`, [Render preparation](RENDER_DEPLOYMENT_PREPARATION.md) | Single-origin synthetic adapter; private trusted artifact provisioning is required; no default retraining |

The registry review reads published metric summaries only. No closed final-test
dataset or row-level final-test predictions are opened or recomputed.

## README defects identified

- It incorrectly says no LightGBM/XGBoost/CatBoost comparison exists.
- It omits the later capacity, case-management, monitoring and explainability
  features and their boundaries.
- Its 340-test count and execution-time estimates predate the current suite.
- It presents artifact rebuilding as ordinary setup, which could replace the
  exact artifact required by the current precomputed explanation assets.
- It does not give a concise architecture and distinct evidence paths for the
  historical model evaluation, later validation research and synthetic demo.

## Prioritised audit

Verdict: substantial scientific evidence exists, but the landing page no longer
describes the current application accurately. These are reviewer-judgment scores,
not automated quality measurements.

| Dimension | Before / 5 | Evidence |
|---|---:|---|
| First impression | 3 | Problem is stated, but the current application and later features are not evident |
| Problem framing | 4 | Human-review decision and scope are explicit; financial stakes remain hypothetical |
| Technical substance | 5 | Chronological protocol, interpretation and scientific audit records |
| Code quality and structure | 5 | Shared `src/` modules, reusable pipelines, contracts and browser tests |
| Reproducibility | 3 | Pinned runtime exists, but default setup rebuilds an identity-bound artifact |
| Evidence of outcome | 4 | Historical results and demo links exist; current feature evidence is incomplete |
| Communication | 4 | Careful original evaluation, but later evidence and limitations need synthesis |

Do first: correct claims and establish population-specific results. Next: provide
a current application overview, evidence links and verified local startup.
Spare-time polish: only evidence-backed screenshots and concise navigation.
Do not spend time on decorative badges, fabricated business impact or product
redesign. Preserve historical reports rather than rewriting their chronology.

## Recorded baseline, not a fresh Stage 24 test run

The Stage 23 release review records **555 passed**, zero failures/skips, five
inherited warnings, 313.06 seconds, including 84 real-Chrome tests and all six
heavy/data-contract tests. See the Stage 23 report and its versioned evidence.
This documentation stage must label that evidence as recorded until any fresh
verification required by the remaining brief is completed.

## Preservation boundary

Expected frozen SHA256:
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.

No scientific recomputation, model modification, API change, dependency install,
Render change, push, merge or deployment is part of this audit. Original
scientific records and approved aggregates remain the sources of truth.

## Deliverables and README improvements

All 19 requested README topics are covered, with related subjects grouped:
project overview; business problem; analytics/ML/engineering objectives; dataset
attribution; preparation/leakage; original modelling; historical final-evaluation
results; same-validation challenger comparison; capacity; case management;
historical monitoring; explainability; architecture; actual stack; public demo;
local setup; testing/reproducibility; responsible AI/limitations; future work.
The old “no challenger comparison” and 340-test claims are removed. The startup
path selects the integration branch and trusted artifact, not a default retrain.

- [README](../README.md): reviewer entry point, current application and explicit
  population/result boundaries; two inline screenshots and seven workspace links.
- [Architecture](ARCHITECTURE.md): offline ML pipeline and synthetic human review
  workflow. The public deployment/inference diagram is in the
  [README architecture section](../README.md#application-architecture), making
  **three unique version-controlled Mermaid sources** in total.
- [Portfolio walkthrough](PORTFOLIO_WALKTHROUGH.md): approximately 1,058 words
  including prompts, with a six-minute navigation sequence and evidence links.
- [Portfolio case study](PORTFOLIO_CASE_STUDY.md): approximately 1,079 words on
  problem, methods, engineering, results, business interpretation and limits.
- [Workflow state](ML_WORKFLOW_STATE.md): Stage 24 update prepended; all earlier
  workflow history remains verbatim. Original reports are not rewritten.

The seven-dimensional audit above uses the GitHub portfolio advisor rubric.
The fresh verification follows the verification-before-completion skill:
completion is based on commands executed in this stage, not historical counts.

## Screenshots selected and reviewed

All selected captures were opened and visually reviewed at 1440×900. Their
paths, dimensions and SHA256 values are recorded in
[screenshot inventory](stage24_portfolio_evidence/screenshot-inventory.json).

| Workspace | Source | Decision |
|---|---|---|
| Overview | [New capture](stage24_portfolio_evidence/overview-1440x900.png) | One necessary update: older Overview images lacked the later seeded-status distinction and current navigation |
| Investigation Queue | [Stage 21 corrected queue](stage21_queue_filter_evidence/queue-case-matching-1440x900.png) | Reuse; opened case with combined filters |
| Capacity Simulator | [Stage 20 capacity](stage20_challenger_evidence/screenshots/capacity-1440x900.png) | Reuse; explicit historical validation banner |
| Benchmarking | [Stage 20 benchmark](stage20_challenger_evidence/screenshots/benchmark-1440x900.png) | Reuse; four models, uncertainty and retained RF |
| Case Management | [Stage 21 corrected workflow](stage21_queue_filter_evidence/workflow-case-matching-1440x900.png) | Reuse; one actual synthetic opened case, not seeded decisions |
| Monitoring | [Stage 22 feature drift](stage22_monitoring_evidence/monitoring-features-1440x900.png) | Reuse; historical-window and non-performance warnings |
| Explainability | [Stage 23 TreeSHAP](stage23_explainability_evidence/explainability-shap-1440x900.png) | Reuse; grouped global and signed synthetic local contributions |

The README embeds Overview and Benchmarking, with full-size links for all seven.
Older feature screenshots are labelled by their verification stage; their older
navigation is not presented as a current live snapshot. No private rows, notes,
credentials or development paths appear in these screenshots. Readme/architecture
preview images are separate documentation-rendering evidence, not new dashboard
features or invented charts.

## Numerical-claim verification

The executable [documentation verifier](stage24_portfolio_evidence/verify_documentation.py)
reads approved summaries/assets, checks counts and published rounding, and checks
the actual artifact bytes without deserializing or fitting it. Its
[results](stage24_portfolio_evidence/documentation-checks.json) cover:

| Claim | Verified source/result |
|---|---|
| Original held-out metrics | One Stage 10 registry summary: AP 0.42784, ROC-AUC 0.86637, P@1% 0.8583, R@1% 0.2461; 76,855 rows, 2,682 positives, 769 reviews, 660 positives |
| Training/schema counts | Manifest: 438,125 training rows, 392 raw inputs, 813 transformed columns |
| Validation benchmark | 75,560 rows / 2,617 positives; RF, LightGBM, XGBoost, CatBoost AP/ROC-AUC round exactly to the README table |
| Capacity | 201 checkpoints; 756 reviews and 618 positives at the 1% checkpoint; 81.75% precision / 23.61% recall |
| Comparative uncertainty | 804 model/checkpoint pairs; 1,000 paired draws, 20 chronological blocks |
| Historical monitoring | 25,186 / 25,187 / 25,187 records, with each prior window ending before the next starts |
| Explainability | 392 raw groups, 512 label-free global examples, 240 synthetic local records; source adapter constants give 24 public records × 5 seeds = 120 approved identifiers |
| Data table metadata | Published Stage 2 report: transaction 590,540 × 394; identity 144,233 × 41; EDA confirms approximately 3.5% fraud |

This is a documentation cross-check, not new scientific evaluation. The full
regression additionally executes the existing numerical/security contracts for
Stages 19–23. Historical Stage 10 and later validation metrics remain explicitly
separate in all three narrative deliverables. No causal, calibrated-probability,
measured-savings or production-superiority claim is introduced.

## Markdown, links, diagrams and public URLs

The original implementation verifier checked **145 internal Markdown targets and two inline images**, covering every internal Markdown link/anchor and relative image in
the changed Markdown files, including the workflow-state addition and its retained
history. Final counts are in the versioned JSON; there are **zero broken targets**.
Code fences are balanced and `git diff --check` passes. External evidence links
in untouched historical reports are not silently rewritten.

The [Chrome documentation verifier](stage24_portfolio_evidence/verify_browser.py)
uses the existing CDP helper, Mistune table/strikethrough rendering and a temporary
Mermaid **11.12.0** browser bundle. It adds no repository dependency. All three
Mermaid sources render to actual SVG without parse/runtime errors. All four
narrative pages render; the README images load without broken-image errors.
[Browser results](stage24_portfolio_evidence/browser-documentation-checks.json),
[README preview](stage24_portfolio_evidence/readme-preview.png) and
[architecture preview](stage24_portfolio_evidence/architecture-preview.png).

The non-publishing `POST https://api.github.com/markdown` check returns HTTP 200
for GFM, rendering the README heading, five tables, images and Mermaid code block.
[GitHub rendering evidence](stage24_portfolio_evidence/github-markdown-check.json).
This does not publish a branch or verify GitHub's eventual interactive Mermaid
version; actual SVG rendering is independently verified locally in Chrome.

[Public-link evidence](stage24_portfolio_evidence/public-link-checks.json): the
live dashboard returns HTTP 200 with its expected title after 44.35 seconds.
The canonical [Kaggle dataset page](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
is verified through official indexed content for the transaction/identity join,
partial identity coverage and relative time. The direct HTTP client receives
404, so automated direct access is **not** reported as passing. Official
[competition acknowledgement](https://www.kaggle.com/competitions/ieee-fraud-detection/overview/about-galaxy-zoo)
credits Vesta. File access requires accepting the competition rules/signing in;
no historical files are downloaded. Render's current
[Free-service documentation](https://render.com/docs/free#spinning-down-on-idle)
supports the documented cold-start limitation.

## Installation, execution and API verification

A disposable Python **3.12.15** environment was created outside the checkout and
successfully installed **only `requirements-render.txt`** and its normal
transitive requirements. The existing project environment and all dependency
manifests remain unchanged. [Installation log](stage24_portfolio_evidence/serving-install.log).

The README checksum command was executed against the actual artifact and returned
the expected SHA256. The documented `python -m uvicorn src.render_demo:app --host
127.0.0.1 --port ...` invocation was executed using that new environment (a free
port was used for verification). Chrome loaded its same-origin dashboard with
24 public synthetic records and visible charts. This does not require historical
data, challenger packages, SHAP or a second static server.

Verified GET responses from that process: `/`, `/index.html`, `/health`,
`/v1/dashboard/capacity`, `/v1/dashboard/benchmark`, `/v1/dashboard/monitoring` and
`/v1/dashboard/explainability` all return 200. Existing regression checks additionally
exercise constrained prediction, queue generation, approved local explanation
identifiers, denied requests and no case-write endpoints. Descriptions distinguish
this public `src.render_demo:app` adapter from the separate local `src.api:app`.

The verification helper initially used the wrong temporary-directory assumption,
then resolved the virtualenv executable symlink to base Python. These helper-only
issues and a temporary Chrome-profile cleanup race were corrected; the final
run exits successfully with an explicit virtualenv entry point and cleanup
after Chrome shutdown. No product or dependency correction was required. The public documentation
logs replace local checkout paths with `<repository>` before being saved.

## Fresh complete regression

Command: `.venv/bin/python -m pytest tests/ -q`.

| Result | Count |
|---|---:|
| Passed | **555** |
| Failed / errors | **0 / 0** |
| Skipped | **0** |
| Real-Chrome tests included | **84** |
| Heavy/data-contract tests included | **6** |
| Warnings | **5** |
| Duration | **314.22 seconds** |

[Full output](stage24_portfolio_evidence/full-regression.log) and
[collected marker inventory](stage24_portfolio_evidence/test-inventory.json).
Warnings are inherited: one Starlette TestClient/httpx deprecation, two
scikit-learn penalty FutureWarnings and two existing saga convergence warnings.
No runtime source changed after this run; subsequent edits affect documentation
and documentation-only verification helpers. Chrome tests cover existing capacity,
benchmark, cases, queue corrections, monitoring, explainability and production
scoring/security regressions. No hosted CI is claimed; no `.github` directory is
present at the inspected checkpoint.

## Preservation and scope

[Preservation evidence](stage24_portfolio_evidence/preservation-checks.json)
confirms the expected frozen checksum and unchanged runtime source, manifest,
scientific records, historical aggregates, case-store source and dependencies.
The original workflow body is byte-for-byte preserved below the new Stage 24
prefix. Tests fit only their existing train-only fixtures; they do not retrain,
replace or save over the frozen serving artifact. Synthetic fixtures named
`final_test` in isolation tests are not the closed historical dataset.

`main` stays at `52c01c919cb40cd2bedf1a41794a73e38c6cf737`. The local integration
branch remains at its pre-existing `d413486e962368a5ad81a445f15549da3d8075fc`;
`origin/render-dashboard-integration` remains the approved baseline
`942b6d53ae004b847a8c0eca9a9298f1fe0c3024`. Those branches are not moved or
checked out for documentation edits. No Git configuration, Render settings,
production scoring, security boundary or actual browser case data is changed.
No paid service, push, merge or deployment occurs.

## Changed-file inventory

The exact sorted inventory is [changed-files.txt](stage24_portfolio_evidence/changed-files.txt).
It contains only README/workflow documentation, this report, two portfolio
narratives, the architecture document and directly related verification assets.
No unrelated files are reverted or discarded.

## Known limitations

- Running model-backed pages from a public clone still requires the exact trusted,
  privately provisioned artifact. The raw restricted dataset is not a setup download.
- Fresh installation/startup is executed on macOS. Linux and Windows commands are
  structure-checked guidance; they are not claimed as separately executed platform runs.
- Top-level runtime versions are pinned; their transitive dependencies are not a
  complete platform-specific lockfile. The successful clean installation is logged.
- Kaggle's official URL is confirmed through indexed primary content, but direct
  automated HTTP access returned 404. Competition authentication/rules remain
  separate prerequisites; no cause for that HTTP response is asserted.
- GitHub GFM HTML and local Mermaid SVG are verified without pushing. The final
  interactive rendering on a published GitHub branch remains a release-review check.
- Existing warning messages and the original scientific/operational limitations
  remain. Historical monitoring is not live monitoring, explanation is not
  causation, synthetic cases are not an authenticated platform, and no verified
  savings, banking integration or challenger promotion is claimed.

No blocking product defect was found by the full regression or clean-runtime
smoke test. Delivery is limited to a verified local documentation commit for
owner review; Stage 25 requires separate explicit approval.


## Final release review — 2026-10-31

The owner now authorises feature-branch push and PR creation, superseding the
original local-only delivery notes above. Merge, deployment, Render changes,
model promotion/replacement and Stage 25 remain prohibited.

All original 21 files and all 19 README topics were reviewed against source and
approved summaries. An independent read-only reviewer found no blocking issue.
One wording correction specifies that the availability register excludes
`TransactionID` and `isFraud`; the broader previous phrase could imply other
included card attributes were excluded. The three diagrams match offline ML,
public FastAPI serving and browser-local synthetic human review. Walkthrough
and case-study claims remain measured and population-specific. All seven
screenshot hashes and dimensions match the reviewed assets; no new screenshot
replaces the original evidence.

The original 145 internal targets and two images passed before release edits.
The two evidence links below bring the final scope to 147 internal targets,
with zero broken targets and both images valid. All three Mermaid sources render
as SVG again in real Chrome; all four narrative pages render without missing
images. GitHub's non-publishing GFM check returns 200 with five tables, two
images and the Mermaid block. The exact final README SHA256 is recorded.

Fresh minimal serving requirements install successfully in a new disposable
Python 3.12.15 environment; `pip check` reports no broken requirements. The
single-origin startup serves the dashboard/health/approved evidence endpoints,
with 24 synthetic queue rows in real Chrome. No research dependencies are needed
for this serving path. The Kaggle direct page check still returns 404; official
primary indexed descriptions confirm the canonical dataset/attribution. No data
files are downloaded, and the HTTP access limitation is retained.

Fresh full applicable regression: **555 passed, zero failures/errors/skips,
five inherited warnings, 312.64 seconds**; includes all **84 real-Chrome tests**
and **six heavy/data-contract tests**. The original implementation's 314.22-second
run remains preserved separately. Only documentation/evidence changes follow
this fresh run; production source is unchanged.

[Release review evidence](stage24_portfolio_evidence/release-review.json) and
[fresh regression output](stage24_portfolio_evidence/release-regression.log).

Final inventory: **23 files** — the reviewed original 21 plus these two fresh
release-evidence files. The inventory and final documentation checks are updated
in place. Frozen SHA256, original scientific records, approved aggregates,
case-store code, production scoring/security and dependencies remain unchanged.
No closed final-test data/predictions are accessed. Added lines are scanned for
credentials/private development paths; no private artifact or raw dataset is
included. Existing platform, artifact-access and statistical limitations remain;
no blocking documentation or product defect was found.
