# Stage 21 — Interactive Investigation Case Management

Status: implementation and verification complete locally; ready for owner review. No push, merge or deployment.
Base: `7ffe91f995bd4bea7296e04ef408d9c3600869a7`.
Branch: `stage21-case-management`.

## Purpose and architecture (recorded before implementation)

A synthetic fraud analyst can open, assign, prioritise, document, resolve and
reopen investigation cases inside the existing queue/Workflow Board and shared
transaction drawer. Model scores and risk ranks are immutable context, not
analyst decisions. No real identity, fraud verdict, automatic block or monetary
claim is introduced. No database, authentication, backend write API or package
addition is required. Render Free serves static JavaScript only.

Chosen design: a pure validated case-store module plus a drawer/controller
module, integrated by small hooks in existing dashboard routing/rendering.
A separate app would duplicate working navigation; a server store violates the
approved browser-local/no-database constraint. These implementation choices
are the implementer's calls under the owner's explicit instruction to document
architecture, then implement and verify without another approval cycle.

Version 1 records contain stable `CASE-DEMO-<seed>-<sequence>` IDs, the exact
allowlisted transaction ID, seed/batch-size scope, immutable synthetic score
and batch-relative band, status, fictional analyst, priority, notes, resolution
reason, timestamps and activity. Only successfully fetched synthetic batches
with fixed seeds 101/202/303/404/505 and 24 (public) or 48 (local) records supply
eligible IDs. Storage keys include seed and batch size: risk bands depend on
batch composition and are never transferred between scopes.

One case per transaction, including resolved cases: opening an existing case
shows it; reopening requires a reason. New cases start New. Non-resolved lanes
can change freely; resolving requires a reason, and a resolved case can only
reopen to In Review with a reason. Assignment supports Unassigned and three
explicitly fictional analyst identities. Priority is independent of risk band.

Existing unopened transaction cards retain their deterministic seeded lanes,
clearly marked unopened; actual case overlays show analyst-controlled state.
This preserves existing queue/board exploration and deterministic empty local
state while making workload KPIs count only opened cases. No seeded status is
claimed to be a real investigation event. No drag/drop: native buttons/selects
provide the complete keyboard workflow.

History is replay-validated, append-only through supported controls, and
bounded (64 events, 16 notes, 500 characters per note, 300 per reason, 48 cases
per scope, 393,216 UTF-16 code units per stored scope). Once a limit is reached, changes are
rejected with an explicit message; history is never silently truncated.
JSON restores require exact schema versions, enum/ID/type/length checks,
chronological timestamps and event-to-current-state consistency. Corrupt or
unsupported entries are not executed or silently trusted; explicit reset
permits recovery. Storage denial/quota failure keeps changes in memory with
a visible warning, never claims persistence and never sends case text to API.
Reset is scoped to the current batch and requires a native confirmation.
Multiple-tab edits are rejected on conflicting stored revisions; reload or
explicit reset is required. Local history is not a secure server audit log;
users/extensions can modify or delete it outside the application.

All user text is rendered via textContent. The UI warns against entering real
customer/personal/sensitive information; fictional assignments are closed enums.
No automatic PII detection or secure storage claim is made. Drawer focus is
trapped while open, Escape closes and restores focus to its opener. Existing
shared product/card/band/search controls remain; additional case status,
analyst and priority filters apply coherently to Queue and Workflow Board.

## Implementation and verification plan

1. Add pure store and executable schema/transition/storage/seed tests (red,
   then green). Validate replay and bounds rather than trusting stored fields.
2. Add case controls, notes/history, workload KPIs and filters using the existing
   drawer/board. Preserve score calculations and ranking, and test actual user
   create/assign/resolve/reopen/refresh/reset interactions in real Chrome.
3. Test unavailable/corrupt/version/quota/conflicting storage and hostile plain
   text; capture 1440×900, 1280×800, 768×1024 and 390×844 evidence.
4. Run complete regression including six heavy/data-contract tests, preserve
   frozen checksum and all previous scientific records verbatim, record counts,
   limitations and exact files, and commit verified work locally only.

No push, merge, deployment, Render settings changes, model training/replacement
or closed final-test access is authorised. Stage 22 is not authorised.


## Review and corrective verification

Independent read-only review found one P2 issue: a multi-field Apply operation
could partially save at the 64-event history boundary. A real-Chrome regression
first reproduced changed local storage despite a rejected submission. Updates
now build and replay-validate all requested events against a cloned case and
persist once; failure leaves case state, history, local storage and the board
unchanged. Form drafts are retained on error so reasons can be corrected.
No additional concrete security, data-boundary or persistence finding was
reported. Keyboard focus returns to a replacement card if re-rendering removed
the original opener. Tests enforce that note HTML is literal text and no case
mutation calls fetch. Existing historical/scientific controls remain separate.


## Implemented operation and controls

Inspect a synthetic Queue row or Workflow card and choose **Open investigation
case**. The same stable case is shown on subsequent inspections; there is at
most one case per eligible transaction. Assign a fictional analyst, select
priority/status and Apply. Resolve with a reason, or reopen a resolved case to
In Review with a reason. Resolved cases cannot receive notes, assignment or
priority changes until reopened. Reopening clears the current resolution field
but preserves the previous reason in activity history. Notes use a dedicated
Add timestamped note button; validation errors preserve unsaved form drafts.

Queue/Workflow controls filter by actual opened-case status, analyst, priority
and immutable risk band; Not opened selects staged transactions without cases.
Global search accepts transaction IDs or their stable CASE-DEMO equivalents.
All active filters are reflected in the shared scope count and can be reset.
Opened/active/unassigned-active/resolved KPIs are scoped to the filtered rows;
seeded unopened lane counts are deliberately not claimed as opened-case KPIs.
Normal controls provide no delete/edit-history function. Confirmed reset clears
only the current seed/batch-size scope, with old records in other scopes retained.

Persistence reads and saves strictly validate immutable synthetic context and
replay every activity event to reconstruct the current state. Externally altered
state/history inconsistencies, unknown schema versions and invalid IDs/enums
are rejected. Every multi-field edit is one atomic state/storage operation.
Quota/denial warns of memory-only state; stale multi-tab saves fail explicitly.
A page refresh is required to reload another tab's current version. No automatic
cross-tab merging or silent corrupt-data overwrite is claimed. A deliberate
reset is the recovery path for invalid stored data.

## Evidence and verification methodology

- 21 pure-store contract scenarios run inside the installed real Chrome using
  in-memory storage adapters; no Node runtime, package or test framework was added.
- Ten lifecycle/storage/filter browser scenarios plus the activity-limit atomic
  regression exercise the actual UI. Four lifecycle variants cover 1440×900,
  1280×800, 768×1024 and 390×844. Native confirmation acceptance/cancellation,
  keyboard Tab containment/Escape, full refresh, source-seed changes, exact text,
  no-fetch case changes and safe storage-failure recovery are checked.
- The existing 31 dashboard/browser cases remain in the complete suite, including
  Stage 19 capacity, Stage 20 benchmark, charts, navigation and original scoring.
- A public API test checks both static assets and 404 responses for POST requests
  to proposed case-write URLs. No backend case route exists.
- Sixteen screenshots in `stage21_case_evidence/` show Queue, Workflow, Case Details
  and local history at all four viewports. `viewport-metrics.json` confirms zero
  unnecessary outer content overflow on both desktop sizes and no body horizontal
  overflow. Kanban columns and the drawer have internal scroll areas; narrow
  viewports intentionally retain document/table/board scrolling.
- Frozen checksum, all prior workflow text, historical JSON aggregates, production
  preprocessing/scoring/API security, requirements and protected branch refs are
  checked separately in `preservation.json`.

Focused checks: **32 Stage 21 Chrome cases passed**, zero skips, after the
atomic-update correction. Full-suite totals are recorded below when complete.

## Remaining limitations and trade-offs

This is not secure, shared, authenticated or authoritative case management.
Local records can be altered/deleted with developer tools, extensions, browser
settings, origin changes or cleared storage. Persistence is per browser profile,
device, origin, seed and batch size; there is no cross-device sync or backup.
Private browsing, denied storage and quota errors can prevent persistence.
Memory-only warnings mean refresh may lose unsaved local state, and a failed
reset cannot guarantee the browser's old persistent entry was removed.

Free-text warnings and bounds do not detect every possible personal/sensitive
string: users must enter fictional demonstration text only. No PII scanning,
authenticated analyst identity, encryption or tamper-proof audit claim is made.
History records local client timestamps (nondecreasing within each case), not
a trusted server clock. All identity names are explicitly fictional enum values.

Cases preserve the immutable model rank/score/band from the eligible current
synthetic batch; local analyst status and priority are not model predictions,
fraud labels, learning targets, automatic declines or monetary impact estimates.
No scientific result is recomputed. Local/public batches differ in size and
batch-relative bands, hence separate storage scopes even for matching IDs.

No drag/drop is added; keyboard-accessible native controls provide the complete
workflow. Focus is trapped within the shared drawer and restored on close.
Native confirm requires an explicit reset decision. Screen-reader/physical-device
accessibility certification is not claimed; all required viewport tests use
real Chrome with emulated dimensions, not four physical devices. The drawer's
context, notes and full history intentionally require internal scrolling.

Record/event/note/size limits intentionally stop further edits instead of deleting
old activity. Clear/reset is a destructive local demo action requiring confirmation;
there is no archive/export. Concurrent tabs use optimistic conflict detection,
not a distributed transaction or atomic cross-tab compare-and-swap: reload after
another tab's edits. Existing Chart.js CDN availability remains an inherited
external dependency. No database, authentication, backend case writes, paid
infrastructure or deployment change is introduced.


Additional local smoke checks use the actual public API's capped 24-record
response and confirm that its exact synthetic population/bands are accepted
by the case schema. Native Enter-key input opens a case through the real UI.
Evidence: `stage21_case_evidence/public-and-keyboard-smoke.json`. No production
service or Render setting is read or changed for these local smoke checks.


A separate native Enter-key check after the initial full 485-test pass exposed
focus loss when the now-hidden Open case button was removed from the tab order.
A native-input regression reproduced the empty activeElement ID. Successful
keyboard creation now focuses the analyst selector, and Tab trapping recovers
focus if it ever leaves the open drawer. The final suite includes this added
regression; the initial 485-test log is retained as historical pre-fix evidence,
not the final completion result. Refreshing returns to the dashboard's default
sequence; selecting another sequence restores that scope's persisted cases.


## Changed-file inventory

The exact tracked delivery inventory is recorded in
[changed-files.txt](stage21_case_evidence/changed-files.txt), including itself.
Core changes are `ui/case-store.js` (pure state/storage validator),
`ui/case-ui.js` (safe DOM controls), small integrations in the existing
`ui/dashboard.js`, `ui/dashboard.html` and `ui/dashboard.css`, two fixed static
asset routes in `src/render_demo.py`, browser/store/public-API regressions,
`scripts/verify_case_browser.py`, this report/workflow entry and local evidence.
No model artifact, dataset, backend case-write API, dependency or deployment
configuration file changes are included.


## Local reproduction

Use the existing local project environment; no new installation is required:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_case_browser.py
```

The screenshot script launches only local test servers and real headless Chrome,
uses fictional synthetic cases, and stops its processes afterwards. It requires
free local ports 8000/8080, as does the established browser regression fixture.
Run browser captures and browser test modules sequentially. Browser contract
checks require the installed Chrome already used by the project; no Node,
Selenium, Playwright, database, authentication package or paid service is added.


## Final verification and delivery

Final complete regression: **486 passed, 0 failed, 0 skipped**, five existing
warnings, **253.66s**. This includes all **six heavy/data-contract tests** and
**63 real-Chrome cases**. Stage 21 adds 32 browser tests (21 pure-store scenarios
and 11 UI/lifecycle/error/accessibility regressions) plus one public static-
asset/no-case-write API test to the prior 453-test suite. No failing test was
excluded. Final evidence: [full-regression.log](stage21_case_evidence/full-regression.log).
Warnings are the existing Starlette/httpx deprecation, sklearn penalty
deprecation and convergence warnings; no dependency was upgraded to hide them.
The heavyweight test trains disposable independent reference models using
approved train rows; it never overwrites or retrains the frozen artifact.

Focused Stage 21 checks: **32 passed**, zero skips. Screenshots: **16**, four
views at each required viewport, with desktop outer content overflow **0px**.
The 485-test pre-focus run is explicitly historical, superseded by the final
486-test run above. Both independently identified defects have reproducing
red-to-green regressions and are covered by the final complete suite.

- [Focused Chrome results](stage21_case_evidence/focused-browser.log)
- [Viewport measurements](stage21_case_evidence/viewport-metrics.json)
- [1280×800 Workflow](stage21_case_evidence/workflow-1280x800.png)
- [1280×800 Case Details](stage21_case_evidence/case-detail-1280x800.png)
- [Mobile notes/history](stage21_case_evidence/case-history-390x844.png)
- [Independent review and fixes](stage21_case_evidence/independent-review.json)
- [Public-batch and native keyboard smoke](stage21_case_evidence/public-and-keyboard-smoke.json)
- [Preservation checks](stage21_case_evidence/preservation.json)
- [Exact 37-file inventory](stage21_case_evidence/changed-files.txt)

Frozen Random Forest SHA256 before/after remains
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
All prior workflow records are byte-identical after the new Stage 21 entry.
Production scoring/preprocessing, public safeguards, Stage 19/20 aggregates,
model manifest, dependencies and protected main/integration refs are unchanged.
Shared server changes are limited to two fixed JavaScript file responses.
Python compilation, `pip check` and `git diff --check` pass. No closed final-test
data/predictions were accessed. No Render configuration or deployed service
was changed. The frozen model was not retrained/replaced, and no challenger
was promoted.

There is no known unresolved functional or local implementation blocker.
Browser-local, non-authoritative storage, possible quota/denial loss, bounded
history, optimistic cross-tab conflicts and unverified assistive-technology/
physical-device behaviour remain explicit limits above. Verified changes are
committed locally only on `stage21-case-management`. No push, merge, deployment
or Stage 22 work is authorised by this delivery.
