# Stage 15 — dashboard UX recovery (2026-10-30)

## Recovered checkpoint

Branch: `dashboard-viewport-refinement`; HEAD: `9e9741a` (PR #3 merge).
The owner identifies this as the last confirmed live Render commit; live
Render state was not independently queried or changed. The local
`render-dashboard-integration` ref remains at `d413486`; it is stale relative
to the owner's checkpoint. No refs were moved.

Five files were already modified on arrival: `ui/dashboard.css`,
`ui/dashboard.html`, `ui/dashboard.js`, `ui/index.html`, and
`tests/test_dashboard_routing_browser.py` (361 insertions, 115 deletions).
The original diff was saved to `/private/tmp/fraud-stage15-recovered.patch`.
No existing changes were reset, discarded, or replaced wholesale.

The original ML Stage 15 (Application Integration) is separately approved
and closed. This report concerns the later **Live Dashboard UX, Navigation
and Viewport Refinement** stage named in the owner's recovery request.
`HANDOFF.md` contains obsolete early-project notes; `DASHBOARD.md` records
through dashboard Stage 12, with a Stage 13 deployment addendum. No separate
complete Stage 15 acceptance specification was found in those documents.
Scope below therefore follows the owner's named requirements and the
recovered implementation; unrecorded prior conversation cannot be recovered
from Git.

## Requirement status

| Requirement | Recovered state | Recovery outcome |
|---|---|---|
| Back to Dashboard | Implemented on single-score page | Browser return to Overview checked |
| Desktop viewport layouts | Flex content regions, internal queue/board scrolling, four primary Analytics charts plus fifth supplementary chart | Short-desktop queue collapse reproduced at 1024×600; minimum content height added so outer scroll preserves access |
| Navigation | Existing six-view hash router preserved | Existing routing/history/shared-selection tests retained; disclosure badge navigation checked |
| Filters | Shared filters, global scope count, highlighted Reset all filters | Empty-scope and reset transitions tested; stale product-chart message fixed |
| Badges | Synthetic disclosure retained, uncalibrated badge links to Responsible AI, service badge has status role | Disclosure navigation checked; existing service behavior retained |
| Responsive design | Mobile document scrolling and desktop internal scrolling retained | Width/visible-heading checks across all six views at 1440×900, 1280×800, 1024×600, 390×844; usable desktop table/chart assertions |
| Verification and handoff | One existing browser selector updated, no Stage 15 completion record | Six new parametrized browser cases added; this report records recovery |

## Changes made during recovery

Only two product corrections were needed: reset the product-chart message
when its scope transitions from Low-only to zero matching rows; prevent
`.view-fill` collapsing below a usable 320px on desktop. The latter permits
the existing outer scroll container to handle insufficient height.
All other recovered UI implementation is retained.

Browser checks use the repository's existing pytest/CDP harness and installed
Chrome, against real local API and static servers. These are automated DOM
interaction/layout checks, not physical-device or screen-reader certification.
The original 12 browser tests passed before recovery edits. Both newly found
defects were reproduced before correction.

The recovery pass did not train, tune or evaluate models, change raw data,
install dependencies, change API/security code, push, merge or deploy. The
initial UI regression pass excluded heavy and full-data-contract tests;
the owner subsequently authorized those tests for the final review below. Owner review/approval remains pending; no next
stage is authorized by this report.

Frozen model SHA-256:
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.

## Initial recovery verification

`.venv/bin/python -m pytest tests/ -m 'not heavy and not data_contract_full' -q`
completed with **399 passed, 6 deselected, 5 warnings** (38.87s), including
all **18 real-Chrome browser cases**, with no skips. Warnings concern existing
Starlette/httpx deprecation and the existing small SAGA test's sklearn
parameter/convergence warnings. `git diff --check` passed. Frozen artifact
checksum was recomputed after verification and is unchanged.

## Owner-requested final review — 2026-10-30

### Additional defects reproduced and corrected

Final screenshot review and scroll measurements found three further sizing
issues missed by the initial width/heading checks:

- Overview's compact chart/table height rules lost to later base CSS;
  seven KPI cards also wrapped at 1280px. The overview preview now takes
  remaining flex space, with its own scrolling table. Desktop KPI cards
  stay in seven columns from 1200px, while mobile keeps its existing grid.
- Workflow's intermediate `.kanban-scroll-wrap` had no bounded flex size,
  letting the board grow to 1461px instead of scrolling within its columns.
  That wrapper is now bounded; column headings remain visible while card
  lists scroll independently.
- Queue's 320px minimum content region caused a small outer overflow at
  1280×800. Its minimum is now 240px, preserving useful table height while
  fitting that viewport. Smaller desktop heights still allow outer scroll
  rather than hiding content.

Desktop footer and content padding were reduced to give the data region
more room. These changes preserve the existing components, routing and
state. New browser assertions check actual outer-scroll dimensions,
Kanban column bounds, a real chart pointer click, and scoring through the
single-transaction page followed by Back to Dashboard.

### Screenshot evidence

All images are real installed-Chrome viewport captures at device scale 1,
with a scored synthetic Sequence 1 batch from the actual local API.
Viewport dimensions stayed fixed while each page rendered; these are not
stitched full-page captures. The fifth Analytics chart remains reachable
by intentional **internal** Analytics scrolling below its four primary
charts. Tables and Kanban columns intentionally scroll within their panels.

| View | 1440×900 | 1280×800 |
|---|---|---|
| Overview | [Screenshot](stage15_ux_verification/1440x900-overview.png) | [Screenshot](stage15_ux_verification/1280x800-overview.png) |
| Investigation Queue | [Screenshot](stage15_ux_verification/1440x900-queue.png) | [Screenshot](stage15_ux_verification/1280x800-queue.png) |
| Analytics | [Screenshot](stage15_ux_verification/1440x900-analytics.png) | [Screenshot](stage15_ux_verification/1280x800-analytics.png) |
| Workflow | [Screenshot](stage15_ux_verification/1440x900-workflow.png) | [Screenshot](stage15_ux_verification/1280x800-workflow.png) |
| Model Insights | [Screenshot](stage15_ux_verification/1440x900-insights.png) | [Screenshot](stage15_ux_verification/1280x800-insights.png) |
| Responsible Use | [Screenshot](stage15_ux_verification/1440x900-responsible-ai.png) | [Screenshot](stage15_ux_verification/1280x800-responsible-ai.png) |

Machine-readable scroll measurements:
[viewport-metrics.json](stage15_ux_verification/viewport-metrics.json).

### Design review and limitations

Review follows the premium dark design recorded in `docs/DASHBOARD.md`
(Stage 3 and the Stage 8–10 visual audits): dark layered panels, persistent
left navigation, blue active state, colored KPI borders/risk pills,
consistent panel spacing, restrained typography and clearly separated
historical metrics and synthetic live scores. The original inspiration
mockup is not available as an authoritative current pixel specification;
this review uses the documented design and prior approved screenshots.
Owner visual approval remains pending.

Automated desktop checks are Chrome-based, at default zoom, with synthetic
demo data. They do not certify Safari/Firefox, physical-device input,
screen-reader output, or every zoom/viewport combination. Charts retain
the existing external Chart.js dependency. Kanban remains deliberately
read-only: selecting cards opens details; dragging or persisting status
changes is not an implemented feature. No new live Render verification
or deployment occurred.

### Final viewport and interaction findings

All 12 final screenshots were visually inspected. No clipped headings,
overlapping panels, missing charts or unintended outer scrollbars remain
at the requested sizes. The document/body dimensions equal the requested
viewport exactly; `.view-container.scrollHeight === clientHeight` for
every view (721px at 1440×900; 621px at 1280×800). Overview's preview table
has 220px / 103px of scrolling height respectively, showing about five /
two transaction rows initially; all eight preview rows remain reachable.
Queue has 371px / 220px for its scrolling table. Workflow columns stay
within the viewport, with independent internal card-list scrolling.

In addition to pytest's real-Chrome checks, a separate real-browser pass
verified ProductCD, card-network and risk-band filtering/reset, amount
filtering, sortable-header state, an actual pointer click on a ProductCD
bar, all five demo sequences and reload. All 16 recorded checks passed,
with no JavaScript exceptions observed. Evidence:
[interaction-checks.json](stage15_ux_verification/interaction-checks.json).
The regression suite also exercises a real pointer click on a risk-band
slice, opening/closing a Kanban detail panel, real `/v1/predict` scoring,
and returning from the single-score page to Overview.

No remaining defect was observed in the verified scope. The browser,
accessibility, external-chart dependency and live-Render limitations above
remain; this is ready for owner review, not a recorded owner approval.

### Separately requested heavy/data-contract result

`.venv/bin/python -m pytest tests/ -m 'heavy or data_contract_full' -vv`
returned **6 passed, 399 deselected, 1 warning in 138.15s** at the time of
that separate run. This comprises five full-**train** data-contract tests
and one full-training equivalence regression. The latter intentionally fits
two temporary in-memory models under the existing approved configuration;
it does not save/replace the frozen artifact, tune parameters, evaluate
`final_test`, or alter approved scientific results. The additional three
browser acceptance cases were added after that separate run.
[Separate test log](stage15_ux_verification/heavy-data-contract-tests.txt).

### Complete regression result and preservation checks

`.venv/bin/python -m pytest tests/ -q` returned **408 passed, 5 warnings
in 194.69s (3:14)**, exit code 0, with **zero exclusions, zero skips,
zero failures**. This includes **21 real-Chrome cases** and all six
heavy/data-contract tests. The five warnings are the existing Starlette/
httpx deprecation and two occurrences each of sklearn's penalty deprecation
and small-SAGA-test convergence warning. No dependency changes were made.
[Full regression log](stage15_ux_verification/full-regression-tests.txt).

The final browser interaction test completed a real single-transaction
scoring request successfully, displayed a risk score in [0,1], kept error
and unavailable panels hidden, then used Back to Dashboard and verified
Overview with a loaded queue. Existing history, shared selection,
filters, empty-state/reset and mobile-width regression tests all passed.

Frozen artifact SHA-256 was recomputed after the complete test run and
remains `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
Manifest SHA-256 is
`37c8985a6871abfade9b12469bcceef6099b741f46b420812b7eec5f91f21ed7`.
No `src/`, artifact manifest or approved scientific report changes appear
in the Git diff. `git diff --check` passed. Screenshot integrity hashes
are saved in [screenshot-checksums.json](stage15_ux_verification/screenshot-checksums.json).

HEAD remains `9e9741af555f4324907a0541319586407b10f4f3`.
Exact `git status --short --branch` for owner approval:

```text
## dashboard-viewport-refinement
 M docs/ML_WORKFLOW_STATE.md
 M tests/test_dashboard_routing_browser.py
 M ui/dashboard.css
 M ui/dashboard.html
 M ui/dashboard.js
 M ui/index.html
?? docs/STAGE15_UX_RECOVERY.md
?? docs/stage15_ux_verification/
```

All work remains uncommitted. No push, merge or deployment occurred.
Stage 15 owner approval remains pending; no subsequent stage was begun.

## Stage 16 — authorized release preparation (2026-10-30)

The owner approved Stage 15 for release preparation and explicitly
instructed committing the verified changes, pushing
`dashboard-viewport-refinement`, and opening a PR targeting
`render-dashboard-integration`. The pending-approval and uncommitted-status
statements above are historical snapshots from the Stage 15 handoff.
Merge, deployment and Render setting changes remain outside authorization.

### Diff and scientific preservation review

Reviewed all modified UI files, the browser-test diff, this report, the
workflow-state addition, all verification JSON/test logs and the existing
12 screenshot evidence files. No unrelated modification was found; nothing
was excluded, reverted or discarded. All 12 screenshot integrity hashes
still match the reviewed images.

The `docs/ML_WORKFLOW_STATE.md` change is confined to the dashboard recovery
note above `## Project Identity`. The entire original scientific workflow
record from `## Project Identity` to EOF is byte-for-byte identical to
`9e9741a`, preserving all approved Stages 0–19 decisions and results.
The new note records Stage 15 approval and the narrowly authorized Stage 16
release-preparation actions; it does not reopen or renumber ML stages.
No `src/`, model artifact/manifest, dependencies, deployment configuration,
raw data, or scientific report is included in the change set.

Fetched the actual GitHub target branch before release preparation:
`render-dashboard-integration` is at
`9e9741af555f4324907a0541319586407b10f4f3`, the feature branch's parent
checkpoint. The stale local integration ref was not moved. The PR should
therefore propose only the Stage 15 UI, browser tests and evidence, plus
this directly related release-preparation documentation.

### Deployment boundary

No deployment is performed by this stage. PR review/approval, an explicit
merge decision and separately authorized deployment remain outstanding.
Render auto-deploy/settings were not inspected or changed during this
stage; a future release operator must verify their state before merging
or deploying. No GitHub workflow/deployment configuration is introduced
by this PR. Existing cross-browser/accessibility verification limitations
remain documented above.

### Fresh pre-commit verification

Stage 16 repeated `.venv/bin/python -m pytest tests/ -q` on the reviewed
release tree: **408 passed, 5 warnings in 181.31s (3:01)**, exit code 0,
with no skips or exclusions. This includes all 21 real-Chrome cases and
all six heavy/data-contract tests. The warnings are the same existing
Starlette/httpx and sklearn deprecation/convergence warnings.
[Fresh pre-commit test log](stage15_ux_verification/stage16-precommit-tests.txt).
`git diff --check` passed. The frozen model SHA-256 was recomputed and
remains `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
All product changes are the verified Stage 15 implementation; Stage 16
only updates directly related approval/release-preparation documentation
and records the fresh test evidence.
