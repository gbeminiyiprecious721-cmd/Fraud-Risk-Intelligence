# Stage 25 — Final release report

**Local readiness verdict: READY WITH DOCUMENTED LIMITATIONS.**
Date: 2026-10-31 (Europe/London). Stage 25 is the final development stage.
Branch: `stage25-final-integration-release`.
Approved integration and live baseline:
`3fd88535b89d554c8955c0dcf0a4333f06ca6e92`.

The original implementation was delivered as a verified local commit,
`45edc309f7b175b5961b6a7467d7758798704e83`. The owner now authorises feature-branch
push and PR creation following independent review, superseding the original
local-only restriction. The final PR head is recorded in the handoff and GitHub.
Merge, deployment, Render changes, model promotion and any additional development
stage remain prohibited. Release-preparation edits change documentation/evidence
only; the reviewed runtime implementation remains unchanged.

## 1. Baseline and inventory

Read-only Render inspection confirms Stage 24 reached Live at the exact approved
merge: deployment `dep-dbivtes9v7es738oc7rg`, manually triggered and completed
2026-10-31 14:45:41 UTC. The service is Free, Auto-Deploy Off and previews Off.
The Stage 25 branch was created from that merge with a clean starting worktree.
`main` remains `52c01c919cb40cd2bedf1a41794a73e38c6cf737`.
[Baseline and endpoint/dependency inventory](stage25_release_evidence/baseline.json).

There are **75 intended changed files**, including 40 viewport captures and two
documentation-rendering previews. The exact sorted inventory is
[changed-files.txt](stage25_release_evidence/changed-files.txt). Only two runtime
files change: public request intake in `src/render_demo.py` and KPI presentation
in `ui/dashboard.js`. Two existing test files add six regression cases. Remaining
changes are the current verification pointers/screenshot in README and portfolio
narratives, a workflow-state prefix, this report and reproducible QA assets.
There are no dependency, model, database, authentication, backend-case-write,
paid infrastructure, CSS/font/colour or unrelated feature changes.

The public adapter exposes nine business/API paths (health, version, prediction,
queue, capacity, benchmark, monitoring and two explanation paths), explicit
HTML/CSS/JS paths and existing FastAPI documentation/schema paths: 24 route paths
in total. Serving uses `src.render_demo:app`; `src.api:app` remains a separate
local API. Artifact provisioning and the seven pinned Render requirements are
unchanged. Challenger and SHAP libraries remain offline.

## 2. Release checklist and feature integration

PASS means an executed check or observed source/deployment evidence. The release
control gates are deliberately held; they are not failed implementation checks.

| Gate | Result and evidence |
|---|---|
| Exact approved/live baseline and branch | PASS — matching remote ref, merge object, live deploy and branch base |
| Frozen model and original scientific history | PASS — checksum and protected-path comparisons; historical workflow body retained verbatim |
| Overview/KPIs, queue, sorting and shared filters | PASS — full Chrome suite; two new KPI regressions; public24 view checks |
| Capacity Simulator | PASS — 201 checkpoints, slider/numeric/reset/keyboard/pointer/chart checks |
| Model Benchmarking | PASS — four models, 804 checkpoints, controls/charts/uncertainty and retained RF |
| Case management / Workflow Board | PASS — lifecycle, reasons, fictional assignment/priority, notes/history, reset confirmation, store validation/conflicts/seed isolation |
| Historical monitoring | PASS — windows/selectors, all three dimensions, charts/reset, failure/retry and missing-data states |
| Model explainability | PASS — mapping/contribution contracts, MDI/SHAP selectors, case links, fallback and bounded public identifiers |
| Navigation, refresh and back/forward | PASS — full Chrome suite and same-origin public scoring/Back to Dashboard at all four sizes |
| Loading/error/empty/retry/fallback states | PASS — existing browser tests, including evidence failures, unavailable/corrupt storage, chart fallback and undefined metrics |
| Browser-local privacy / XSS safety | PASS — hostile note remains literal, no case-write request or note transmission; refresh/navigation/filter clear preserve cases |
| Desktop viewport and responsive access | PASS — 40 public view/viewport checks, rendered charts and no desktop outer overflow; mobile/tablet permit normal vertical/internal scrolling |
| Full applicable regression | PASS — 561 passed, zero failures/errors/skips, five inherited warnings |
| Documentation, numerical claims, Mermaid and images | PASS — final documentation JSON, GFM and actual Chrome SVG rendering; external-access limitations below |
| Minimal runtime, health, readiness script and resources | PASS — pinned environment, pip check, healthy startup, local deployment-script check and measured samples |
| Feature push / PR | AUTHORISED after independent verification; does not constitute release |
| Merge / deployment / Render changes | HELD — not authorised in PR preparation |

[Full regression](stage25_release_evidence/full-regression.log),
[test marker inventory](stage25_release_evidence/test-inventory.json),
[public-browser QA](stage25_release_evidence/public-browser-qa.json),
[scientific/security/build checks](stage25_release_evidence/release-checks.json).

## 3. Verified findings and corrections

No Critical or High defect was found in the audited scope. The following changes
are narrowly required for resource safety, cross-view consistency and statistical
accuracy; none changes model outputs or introduces a new workflow.

| Finding | Severity | Root cause and minimal correction |
|---|---|---|
| SEC-001: unlimited application buffering before size rejection | Medium | The old public middleware called `request.body()` before checking actual length. An outer ASGI receive counter now rejects the first chunk crossing the unchanged 16 KiB cap, without reading subsequent chunks; ordinary caching/replay, allowlist and rate limits remain intact. |
| UX-001: review KPI ignores opened-case status | Medium | KPI used immutable seeded `t.status`, unlike the workflow board. It now uses the existing `CaseUI.status(t)` resolver; actual case status overrides the seed once opened. Subtitle explicitly identifies case/seeded scope. |
| UX-002: empty-scope score statistics shown as measured zero | Low correctness | Empty cohorts displayed average/highest risk as 0.0000. Both now display an en dash; legitimate zero counts remain zero and nonempty formulas are unchanged. This is a statistical correction, not a cosmetic redesign. |

Tests first reproduced each defect. Stream tests failed twice before correction:
four 8 KiB chunks were consumed despite a 16 KiB cap. They now reject on the
third, with no fourth read, for missing and understated Content-Length. Approved
chunked input still scores correctly and exact-cap whitespace-padded JSON is
accepted. [Stream before-fix evidence](stage25_release_evidence/body-limit-before.log)
and [focused public checks](stage25_release_evidence/public-security-focused.log).

Real Chrome reproduced seeded Resolved → opened New remaining incorrectly at
zero in the review KPI. The new test now checks New → Resolved → In Review and
Overview navigation. A separate empty-scope test failed on fabricated 0.0000
before the dash correction. [Case-KPI failure](stage25_release_evidence/case-kpi-before.log),
[empty-score failure](stage25_release_evidence/empty-kpi-before.log),
[two passing Chrome regressions](stage25_release_evidence/kpi-focused.log).

Independent read-only reviews verified both the ASGI correction and minimal KPI
diff, including unchanged score calculations and case-storage semantics. A deep-
JSON hypothesis was disproved: bounded parsed arrays are already rejected with
403 as unapproved input, so no parser change was made.

## 4. Scientific integrity

Frozen Random Forest SHA256 remains:

```text
2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04
```

The live baseline's `/health` reports its artifact loaded, and `/version` reports
that same checksum. Local checks hash actual artifact bytes. No frozen-artifact
retraining/replacement or challenger promotion occurred. Heavy regression tests
fit their existing independent train-only fixtures without overwriting the
serving artifact. No closed final partition or row-level final-test predictions
are selected/opened; documentation uses published summaries only.

Historical Stage 10 AP **0.42784**, ROC-AUC **0.86637**, precision@1% **0.8583**
and recall@1% **0.2461** remain attributed to the original fitted-model evaluation,
not a fresh evaluation of its later reconstructed joblib. Reused validation
remains a distinct 75,560-record/2,617-positive research population. Capacity has
201 checkpoints; four-model benchmarking has 804. Monitoring's reference and two
later windows remain disjoint, chronological and sized 25,186 / 25,187 / 25,187.
Explainability retains 392 raw groups / 813 transformed columns, its 512 label-free
global sample and 240 synthetic local records, of which 120 are public.

Published values match approved registry summaries and versioned aggregates.
No approved metrics, results, aggregate bytes, model configuration or scientific
conclusions are rewritten. Scores remain uncalibrated ranks, capacity is not
verified savings, attribution is not causation, monitoring is historical and
synthetic case decisions are never presented as real investigations. Existing
validation reuse, feature-timing/label-maturity, anonymisation, fairness and
explanation-assumption limitations remain.

## 5. Security and privacy verification

A local exhaustive lookup audit allows **120 approved identifiers** and denies
**120 non-public identifiers**. During explanation requests, hooks on a separately
loaded test instance forbid prediction and preprocessing transforms: observed
calls are **zero**, and SHAP is not imported. Public explanation assets remain
precomputed/validated, with no unrestricted feature-vector inference.

Five private paths (`.env`, Git configuration, model binary, raw dataset path and
server-only local explanation asset) return 404. Eight POST/PUT/PATCH/DELETE case
write probes return 404. Unsupported prediction input returns a generic 403
without echoing its note marker or a traceback. Original route/security tests
also exercise malformed JSON, allowlist and parameter bounds, unavailable
artifacts, safe internal failures and independent rate-limit budgets.

Case creation/changes/notes remain browser-local, validated and isolated by demo
seed/batch; source storage keys/schema are unchanged. A synthetic hostile note
is displayed literally, creates no image element and executes no handler.
Chrome Network observation finds no case-write request or note transmission.
Refreshing, navigation and clearing filters preserve the same stored record.
Reset confirmation, corrupt/unsupported storage, denied storage, conflicts and
atomicity are additionally covered by the full browser suite. This is not secure,
authenticated, multi-user, cross-device or production-grade persistence.

A high-confidence token/private-key scan of tracked source/config/docs found no
matches, no tracked `.env` and no tracked model binary. This is a scoped pattern
scan, not proof that every possible encoded secret is absent. Final added-line
and inventory review excludes private artifacts, raw datasets and sensitive
local filesystem paths. [Scan evidence](stage25_release_evidence/secret-pattern-scan.json).

Structured/application logs do not include the rejected synthetic request's note
marker. [Readiness/logging probe](stage25_release_evidence/deployment-health-check.json).
Standard access metadata is not claimed to be absent. No authentication,
database, security subscription or backend case write capability is added.

## 6. Regression results and warnings

Command: `.venv/bin/python -m pytest tests/ -q` after all runtime corrections.

| Result | Final count |
|---|---:|
| Passed | **561** |
| Failed / errors / skipped | **0 / 0 / 0** |
| Real-Chrome tests included | **86** |
| Heavy/data-contract tests included | **6** |
| Warnings | **5** |
| Duration | **299.81 seconds** |

The earlier 559-pass/294.05-second run is retained as interim evidence before the
two KPI tests; it is not substituted for the final 561-pass result. The initial
23-pass and later 59-pass focused API results remain labelled by their scope.
Subsequent changes affect documentation/verification helpers, not runtime code.

Warning review: one Starlette TestClient/httpx deprecation affects test tooling;
two scikit-learn penalty FutureWarnings concern an existing logistic-regression
test API; two saga ConvergenceWarnings arise in the existing bounded solver
fixture. They are inherited maintenance/scientific caveats, not failures of the
frozen Random Forest or production runtime. No dependency upgrade or alteration
to historical modelling decisions is used to silence them. The separate local
audit records its one repeated TestClient deprecation explicitly. No hosted CI
or zero-warning claim is made.

## 7. Chrome and responsive evidence

All ten SPA views were exercised against the **public adapter** at **1440×900,
1280×800, 768×1024 and 390×844**, producing **40 viewport captures**. Charts have
positive rendered dimensions. Desktop body and outer view-container dimensions
fit the viewport without unnecessary outer scrolling. Mobile/tablet vertical
scrolling and intentional internal table/Kanban scrolling remain supported;
a viewport capture is not a claim that all mobile content fits above the fold.

Four additional same-origin workflows verify capacity and benchmark controls
(including keyboard/pointer interactions), monitoring dimensions/reset,
explanation methods, fictional case edits, literal hostile notes, persistence,
filter clear, actual public single-transaction scoring and Back to Dashboard.
There are zero runtime errors in that run. The full 86-test Chrome suite covers
history, sorting/combined filters, reset confirmation, error/retry/fallback,
keyboard access and storage failure/isolation/conflict edge cases.

[Public-browser results](stage25_release_evidence/public-browser-qa.json) and
[screenshot hashes](stage25_release_evidence/screenshot-checksums.json).
Representative views: [desktop Overview](stage25_release_evidence/overview-1440x900.png),
[1280 Overview](stage25_release_evidence/overview-1280x800.png),
[tablet Workflow](stage25_release_evidence/workflow-768x1024.png),
[mobile Overview](stage25_release_evidence/overview-390x844.png).
All traffic/profiles are local and disposable; owner browser storage is untouched.

The new harness initially cleared storage but reused document state, triggering
the intended optimistic-edit protection. Distinct query navigation forces a new
document after the clear, and the actual allowlisted transaction supplies the
storage key. No product conflict safeguard was weakened to make the harness pass.

## 8. Performance, startup and Render compatibility

| Measured observation | Result and scope |
|---|---|
| Local healthy startup | **1.37253 seconds** from process launch to healthy artifact-ready response |
| Artifact startup event | **0.2856 seconds** reported by application startup; excludes imports/process launch |
| Sampled local server peak RSS | **220,561,408 bytes ≈ 210.3 MiB**, sampled every 10 ms across public QA; excludes Chrome |
| Five local public24 queue samples | **0.051728–0.097595 seconds**; illustrative request samples, not latency percentiles or a load test |
| Observed live baseline memory | **186,126,340–201,064,450 bytes ≈ 177.5–191.8 MiB**, 23 samples from 14:46–15:08 UTC |
| Reported live memory limit | **536,870,900 bytes ≈ 512 MiB** |

Local measurements are on macOS in the disposable minimal serving environment,
not promises about Render Linux CPU/startup/peak memory. Live samples concern
Stage 24's deployed baseline, not a Stage 25 deployment or startup/load peak.
[Render metric evidence](stage25_release_evidence/render-baseline-metrics.json)
and public-browser performance fields retain exact values.

All seven runtime pins match the manifest requirements; `pip check` finds no
broken requirements. SHAP, XGBoost, LightGBM and CatBoost are absent from the
minimal environment, yet all public workspaces function. No runtime SHAP,
challenger training, paid infrastructure or unnecessary dependency is introduced.
The four existing shell deployment scripts pass `bash -n`; the existing
`deploy/health_check.sh` succeeds against a temporary healthy public process and
checks JSON readiness, not HTTP 200 alone. No private artifact fetch is rerun,
model binary overwritten or external deployment executed.

Live health/version initially return 200 with the expected artifact. One later
health request reaches its configured 45-second read timeout; a subsequent audit
returns healthy 200. This is consistent with the documented Free-instance
cold-start/availability limitation, but its exact cause is not independently
diagnosed. Both observations are retained, not hidden behind the recovery.
[Network observations](stage25_release_evidence/network-observations.json).

## 9. Documentation and reproducibility

README retains all 19 required topics and correct population attribution; its
verification counts, Overview capture and current report pointers now reflect
Stage 25. The architecture diagrams, walkthrough, case study and Stage 19–24
scientific/engineering evidence were reviewed. Historical records remain intact.
The current three Mermaid sources render as real SVG in Chrome, and the four
narrative documents render with working images. GFM rendering is checked without
publishing a branch. Final verification checks **252 internal Markdown targets and two inline images**
across 13 documents, with zero broken targets. Actual counts and errors are in
[documentation checks](stage25_release_evidence/documentation-checks.json).

The official IEEE-CIS Kaggle URL and Vesta attribution are correct according to
primary indexed competition content. Direct automated access still returns 404;
this is an explicitly unsuccessful HTTP availability check, not an invented pass
or a reason to link an unrelated dataset. No historical data files are downloaded.
The exact trusted artifact remains privately provisioned. Setup/startup is
verified on macOS; Windows/Linux instructions are not claimed as separate local
platform executions. Top-level pins are not a full transitive/platform lockfile.

Reproduce with the existing authorised environment and trusted artifact:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/verify_stage25_release.py --docs-only
.venv/bin/python scripts/verify_stage25_release.py --serving-python <minimal-env>/bin/python
.venv/bin/python scripts/verify_stage25_browser.py --serving-python <minimal-env>/bin/python
```

Provision `<minimal-env>` with Python 3.12.15 and only `requirements-render.txt`;
research/test requirements provide the verification tools. Never load an
untrusted joblib, rebuild the frozen artifact as a setup shortcut or reopen the
closed final-test boundary. The verifiers do not require historical datasets;
the six existing heavy/data-contract tests use their approved train-only inputs.

## 10. Blocker assessment, limitations and rollback

**No unresolved Critical/High or release-blocking defect was found.** The three
verified correctness/resource defects are fixed and regression-tested locally.
Readiness is **READY WITH DOCUMENTED LIMITATIONS**, not an assertion of a banking
production platform, comprehensive penetration-test certification or immunity
to denial of service.

Remaining limits include private artifact access; reused validation and unknown
label maturity; uncalibrated scores/incomplete fairness evidence; non-causal
explanations; historical monitoring; synthetic/browser-local cases; transient
Free-host availability; unverified exact proxy client attribution and in-process
rate budgets; upstream allocation of a crossing ASGI chunk; CDN/fallback and
platform/transitive dependency assumptions; existing warnings; direct Kaggle
HTTP failure; and no separate Windows/Linux setup run. GitHub's final interactive
Mermaid renderer cannot be checked on an unpublished branch; local SVG and GFM
checks do not claim that final hosted interaction was exercised.

Rollback reference is the approved current live commit:
`3fd88535b89d554c8955c0dcf0a4333f06ca6e92`.
Rolling back to it would also restore the documented pre-fix intake/KPI behaviour;
it is a known baseline, not a claim that the corrected candidate is already live.

Future deployment verification plan, **only after separate owner authorisation**:

1. Independently review the exact local commit and inventory; then authorise any
   feature push, PR and normal controlled merge, retaining `main` untouched.
2. Keep Render Free and Auto-Deploy Off. If deployment is approved, manually
   deploy only the reviewed integration commit; do not change configuration.
3. Confirm the new deploy SHA and artifact checksum; health must report both
   `status=ok` and `artifact_loaded=true`. Allow for documented cold starts.
4. Use a small approved synthetic smoke check for public24 queue, fixed-fixture
   scoring, explanation allow/deny and read-only evidence. Confirm existing local
   case storage remains usable, then inspect resource/error metadata without
   logging or submitting real customer data.
5. If acceptance fails, use the known rollback deployment after the owner's
   release decision. No automatic deployment/rollback or new development stage
   is triggered by this report.

All delivery work remains on the Stage 25 feature branch. Original local-only
implementation evidence is preserved. Current authorisation permits a feature
push and PR after independent verification; it does not permit merge, deployment,
Render changes, challenger promotion or an additional development stage.


## Independent final review and PR preparation — 2026-10-31

The complete 75-file committed diff, inventory and report were independently
reviewed from approved base 3fd88535 to implementation 45edc309. No blocking
correctness/security finding or unnecessary file was found. The three correction
regressions re-execute in the fresh complete suite: streamed bodies stop at the
cap-crossing chunk; approved exact-cap and chunked bodies still score; actual
opened-case status overrides seeded labels through resolution/reopening; empty
cohorts display undefined score statistics rather than zero.

Fresh full regression: **561 passed, zero failures/errors/skips, five inherited
warnings, 299.44 seconds**. It includes **86 real-Chrome tests and all six heavy/
data-contract tests**. The original 299.81-second implementation run remains
preserved; a clearly separated independent-review run is appended to the same
full-regression log. No warning is silenced and no runtime source is changed.

Public same-origin QA independently repeats **40 viewport checks and four case/
scoring workflows** at all four required sizes, with no runtime errors or
unnecessary desktop outer overflow. Fresh measured native startup is 1.262086s
and sampled server peak RSS is 222085120 bytes (about 211.8 MiB), excluding Chrome.
These support minimal-runtime compatibility, not a Render load/startup guarantee.
The original screenshots and measurements are retained rather than overwritten.

Exhaustive fresh local access checks allow 120 approved synthetic IDs and deny 120
non-public IDs; inference/transform hooks record zero calls and SHAP is not
imported. Five private asset paths and eight backend case-write method probes
remain denied. Literal-note/XSS and browser persistence checks pass again.
Content types, no-store cache headers and CORS are independently checked: an
unapproved origin receives no allow-origin header, its preflight is rejected,
and the approved local origin matches without credentials. This is focused
header evidence, not a claim of comprehensive CSP/frame-policy hardening.

All 252 documentation targets and two inline images pass again; the three
Mermaid sources are rerendered in real Chrome. The existing seven minimal runtime
pins pass pip check with SHAP/challengers absent. Frozen checksum, scientific
records, historical aggregates, case storage, scoring contracts and protected
branches remain unchanged. No closed final partition/predictions are accessed.

Fresh review, browser, security and header results are appended as
`independent_pr_review` in the existing release-checks JSON. This preserves the
exact 75-file inventory. Existing limitations and rollback reference 3fd88535
remain valid. Readiness remains **READY WITH DOCUMENTED LIMITATIONS** for an
explicitly authorised controlled merge after PR review; the project is not
claimed fully released. Merge, Render deployment and live post-deployment checks
require separate authorisation and verification.
