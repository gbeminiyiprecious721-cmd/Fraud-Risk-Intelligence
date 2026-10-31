# Stage 21 — Queue filter and UX corrections

Date: 2026-10-31 (Europe/London). Status: correction complete; release review verified for the owner-authorised branch push and PR. The root-cause record below was written before implementation.
Base: `d419da2dfa11ff0dda85f99a094e114f4e733e3e`.
Branch: `stage21-queue-filter-corrections`.

## Actual reproduced cause

A real, clean Chrome profile against the LOCAL public Render entrypoint returns
24 deterministic synthetic records (seed 101) and no opened investigation cases.
Status New/In Review/Resolved, fictional analysts and all three priorities return
zero rows by design: those predicates inspect the case store, not seeded display
statuses. The generic “No synthetic transactions match” message and unqualified
Status/Analyst/Priority labels conceal this distinction. Seeded Resolved is a
fabricated display lane, never evidence of an analyst resolution.

Unassigned is inconsistent: its predicate substitutes Unassigned when no case
exists, so it returns all 24 unopened transactions. This differs from the owner's
reported zero, but is reproducible in a clean state; combinations of other active
case filters can subsequently reduce it to zero. Risk Band is duplicated in two
selects which synchronise the same value. Clear case filters unexpectedly clears
this transaction-wide band as well. Reproduction evidence:
`stage21_queue_filter_evidence/reproduction-before.json`.

## Chosen correction and scope

Keep opened-case filtering instead of mixing seeded display lanes into actual
case statuses. This preserves the existing case-store contract and never invents
analyst decisions or priorities. Explicitly label case-only scope. Unassigned
means an OPENED case whose fictional analyst is Unassigned (including a resolved
case unless another status filter excludes it); unopened records match only the
explicit Not opened option and unfiltered transaction views.

Empty states distinguish no opened cases, zero matching cases and transaction
criteria that hide otherwise eligible cases, always retaining the actual batch
population count and guidance. Transaction filters (risk, product, card, amount,
search) combine with case filters using AND and do not mutate data. Queue and
Workflow use exactly the same filtered rows. The shared risk selector is the only
Risk Band control; its stable filter-band ID remains available to chart actions.
Clear case filters clears only status/analyst/priority. Clear all filters clears
all view predicates. Reset Demo Cases remains a separate, confirmed destructive
current-scope action, explicitly naming notes, assignments and history; no filter
clear action calls the store's reset method.

Implementation is targeted to existing toolbar, messages and predicates. No case
schema migration, new backend routes, new dependencies, cosmetic redesign,
model/scientific changes, data access, deployment, push or merge is planned.
All existing local cases, immutable scoring, API safeguards and historical
aggregates must be preserved. No Stage 22 work is authorised.

The initial brief ended mid-sentence in Phase C. The owner subsequently supplied
the remaining Phases C–F requirements during this task. Completed work was
preserved; the correction was not restarted. The continuation explicitly
authorises a verified local commit and prohibits push, merge and deployment.


Unopened row badges expose Not opened as their display state, retaining the
seeded lane only as explicitly labelled contextual text. They use the existing
neutral badge style instead of the green/purple actual-case status styling.
Opened case badges retain their actual validated status. This is a semantic
distinction using existing design tokens, not a dashboard redesign.

The existing Clear all filters action is now shared and reachable from Workflow
as well as Queue. It resets risk/product/card/amount/search plus case predicates,
while Clear case filters leaves every transaction predicate intact. Both retain
byte-identical stored notes, assignments and event history.


## Correction details and verified behaviour

- Case status New/In Review/Escalated/Resolved matches only actual opened cases.
  Not opened explicitly selects records with no case. Selecting Not opened with
  analyst/priority is an impossible AND combination and receives an explanation,
  not a silent filter reset or fabricated assignment.
- Unassigned means an opened case without a fictional analyst, not an unopened
  transaction. Default Normal priority belongs only to a case that was opened;
  no priority is generated for unopened rows. Fictional assignments and updated
  priorities match the exact affected case in Queue and Workflow.
- Empty messages distinguish no opened cases, existing cases not matching the
  case selection, transaction filters hiding eligible cases, and all records
  already opened. They retain the actual batch population (24 public  / 48 local).
  The shared scope count and active-filter badge count all AND predicates once.
- A single Risk Band selector, with the original filter-band ID, is reachable
  in the shared toolbar. Historical capacity/benchmark views hide inapplicable
  synthetic controls. Existing chart clicks still drive the same selector.
- Clear case filters clears only status/analyst/priority. Clear all filters
  clears all case predicates, risk, product, card, amount and transaction/case-ID
  search. Neither action updates the case store or case timestamps/history.
- Reset Demo Cases is separate: the confirmation explicitly says locally saved
  notes, assignments, priorities and case history will be discarded for the
  current seed/batch scope, and other demo batches remain unchanged.

Four targeted regressions cover clean no-case state, matching opened cases,
all case/transaction filters combined (eight active predicates), scope counts,
Queue/Workflow agreement, batch switches and reset/clear distinctions. Initial
regressions reproduced the defects before implementation. Seeded row badge
semantics also have a reproducing regression. One pre-existing Stage 21 test
now addresses the single shared risk control rather than its removed duplicate;
its case-filter assertion remains in place with the clarified empty message.
All previous 63 browser cases passed alongside the first three new corrections
(66/66); the final complete run additionally includes the batch/reset test.

## Public demo and responsive evidence

A separate real-Chrome run used the local public Render entrypoint, preserving
its exact 24-record cap and fixed seed101 rather than assuming the local API's
48-row population. All four required dimensions passed:1440×900, 1280×800,
768×1024 and 390×844. Sixteen screenshots show the default Queue, clear no-case
empty state, a matching assigned/priority-filtered case and corresponding
Workflow Board. Desktop outer content overflow is zero and body horizontal
width remains within the viewport. Mobile retains intentional scrolling.

Byte-for-byte native browser storage comparisons confirm that filter clears
preserve saved notes, assignments and chronological history. Native reset
cancellation preserves them; explicit acceptance removes the current scope.
No synthetic model scores, backend protections or storage keys were changed.
Evidence: [public-browser-verification.json](stage21_queue_filter_evidence/public-browser-verification.json)
and [public-browser.log](stage21_queue_filter_evidence/public-browser.log).

The independent read-only source review found no concrete defect within the
supplied requirements. It confirmed the scope/predicate/reset/count consistency
and no backend, store/schema, score or privacy change. Its review is recorded
in [independent-review.json](stage21_queue_filter_evidence/independent-review.json).
The source review preceded the supplied continuation; all Phases C–F acceptance
checks are covered by the final verification below.


## Continuation requirements and keyboard checks

The owner's supplied continuation confirms transaction/case distinctions,
recoverable empty states, no redundancy, consistent terminology, accurate counts,
keyboard-accessible dropdowns, responsive layouts and no unrelated styling.
It requires the full regression including all six heavy/data-contract checks,
all prior stages/security/science preserved, and local commit only.

Four added native-keyboard browser variants cover 1440×900, 1280×800,768×1024
and390×844. They focus Status, Analyst and Priority, choose options with native
select type-ahead, commit with Enter, verify empty-state/count changes, and use
Enter on Clear case filters. They also choose the shared Risk Band using native
keys, verify nonempty matching rows and exactly one active band predicate, then
activate Clear all filters with Enter. No programmatic value/change dispatch
substitutes for those native dropdown selections. On this installed macOS
headless Chrome, ArrowDown dispatch alone did not commit native-popup choices;
the automation uses the browser's working native type-ahead interaction instead.
No product-specific keyboard interception was needed or added.

The completed first full run passed 490 tests, zero failures/skips, five existing
warnings, 256.74 seconds. Its log is retained as pre-keyboard-extension evidence. The
expanded final suite additionally includes all four native-keyboard variants.
No new model training or evaluation is part of this correction; the inherited
heavy regression builds disposable independent train-only references, never
retraining or overwriting the frozen artifact.


## Before / after and action semantics

| Selection / action | Before | Corrected behaviour |
|---|---|---|
| New/In Review/Resolved on a clean batch | Zero rows with generic transaction-empty text | Zero OPENED cases, actual transaction count retained, explicit scope and recovery guidance |
| Unassigned on a clean batch | All unopened transactions incorrectly matched | No opened cases match; Unassigned is a case assignment state only |
| Fictional analyst or priority with no cases | Zero rows without explanation | Explicit no-opened-cases explanation; no fabricated assignments/priorities |
| Case field after opening/updating | Matches actual store state | Same validated store semantics, now consistently labelled and tested |
| Seeded Resolved display label | Styled like a resolved case | Neutral Not opened badge; seeded text is context, not an analyst decision |
| Risk Band | Two synchronised dropdowns | One shared selector, used by Queue, Workflow and existing chart actions |
| Clear case filters | Also cleared risk | Clears only case status/analyst/priority; all transaction criteria/data retained |
| Clear all filters | Queue-only action | Shared action clears every view predicate; saved case bytes unchanged |
| Reset Demo Cases | Generic reset confirmation | Explicit destructive local-scope confirmation names notes, assignments, priorities/history |

Empty messages name the corresponding shared Clear case filters/Clear all filters
controls, which remain available in Queue and Workflow. Clearing is not deletion.
All filters combine using AND; a zero intersection never removes source data.
The filter badge counts risk once, not twice. Transaction/case search, product,
card, amount and demo-batch changes retain existing synthetic scoring contracts.

Reproduce tests with `.venv/bin/python -m pytest -q` and targeted checks with
`.venv/bin/python -m pytest tests/test_dashboard_routing_browser.py -k queue_correction -q`.
Public-browser evidence was captured against local `src.render_demo:app` on
port 8000 and the existing static dashboard server on 8080, not by altering or
writing to the live Render service. The Chrome profile was clean and dedicated
to synthetic verification; no existing owner browser-local cases were touched.

## Preserved boundaries and outstanding limitations

The case-store implementation/schema, storage keys and persisted records are
unchanged. No backend file, dependency, model artifact or historical aggregate
changes are included. All prior workflow/scientific text is retained verbatim
below the new correction note. Frozen SHA256 remains
`2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
No closed final-test data/predictions were opened. No paid infrastructure,
Render setting change, push, merge or deployment was performed.

Existing browser-local limits remain: records are not authenticated, encrypted,
shared or a secure server audit log; users/extensions can alter/delete them.
Storage denial/quota can cause explicitly warned memory-only persistence.
Case filters persist across synthetic view and batch changes by design, so a
new batch without matching cases can show zero; the guidance and counts make
that state explicit. Model risk bands are batch-relative, not investigator
priority or a fraud verdict. Tests emulate viewport sizes in real Chrome; no
physical-device or assistive-technology certification is claimed.


## Final verification and local delivery

Final complete regression: **494 passed, 0 failed, 0 skipped**, five existing
warnings, **263.81 seconds**. Includes all six heavy/data-contract tests and
**71 real-Chrome cases**. The baseline was 486 tests; the correction adds four
filter/data/reset regressions and four native-keyboard viewport variants.
No failing or heavy test was excluded. The prior 490-test run is retained as
pre-keyboard-extension evidence, superseded by this complete 494-test run.

- [Full regression](stage21_queue_filter_evidence/full-regression.log)
- [Focused filter checks](stage21_queue_filter_evidence/focused-browser.log): 4 passed
- [Native keyboard checks](stage21_queue_filter_evidence/native-keyboard.log): 4 passed
- [Before reproduction](stage21_queue_filter_evidence/reproduction-before.json)
- [Public24-record verification](stage21_queue_filter_evidence/public-browser-verification.json)
- [1280×800 Queue](stage21_queue_filter_evidence/queue-default-1280x800.png)
- [1280×800 explained empty state](stage21_queue_filter_evidence/queue-case-empty-1280x800.png)
- [390×844 matched case](stage21_queue_filter_evidence/queue-case-matching-390x844.png)
- [Preservation](stage21_queue_filter_evidence/preservation.json)
- [Changed-file inventory (35 after release review)](stage21_queue_filter_evidence/changed-files.txt)

Sixteen screenshots verify public 24-record Queue and Workflow at all four
required sizes. Desktop outer content overflow is 0px; no body horizontal
overflow or browser runtime error was observed. Existing Stage 19/20 charts,
scoring, API security and Stage 21 case lifecycle/refresh/schema/storage/error
regressions all remain in the complete run. Native keyboard dropdown selection,
Enter activation and non-destructive recovery pass across the required sizes.
The public reset cancellation/acceptance checks confirm the destructive action
is distinct from clearing filters and remains explicitly confirmed.

All prior workflow/scientific records are byte-identical beneath the new
correction note. Frozen SHA256 is unchanged before/after; backend code,
case-store/schema/keys, dependencies and historical JSON files are unchanged.
Main and integration refs are unchanged. `git diff --check` and `pip check`
pass. No final-test access, paid infrastructure, model change, push, merge,
deployment or Render configuration change occurred.

All supplied Phases A–F requirements are implemented and verified. No known
unresolved correction defect or local implementation blocker remains. Existing
browser-local security/persistence and physical-device/accessibility limits
above remain. Verified changes are committed locally on
`stage21-queue-filter-corrections`; release and Stage 22 are not authorised.

## Release preparation review (2026-10-31)

The owner subsequently authorised pushing the correction branch and opening a
PR against `render-dashboard-integration`, superseding the local-only delivery
restriction above. Merge, deployment and Render changes remain prohibited.
Reviewed implementation commit: `aadd4c01719dd0f4928109ab0b2be3d992812a2a`.
The complete original 33-file inventory matched exactly. Independent source
review found no concrete defects; no runtime correction was necessary.

Fresh complete regression: **494 passed, 0 failed, 0 skipped**, five existing
warnings, **266.02 seconds**. All **71 real-Chrome cases** and **six heavy/data-contract
tests** ran. See [release regression](stage21_queue_filter_evidence/release-regression.log).
The warnings remain the existing Starlette deprecation and sklearn logistic
penalty/convergence warnings; they do not concern the frozen production model.

An additional clean-profile local public-app check passed at 1440×900, 1280×800,
768×1024 and 390×844. It independently verified refresh persistence, seed202
isolation and restoration of seed101, matching Queue/Workflow intersections,
byte-identical case storage after both filter clears, and native reset cancel
and accept. Desktop content overflow was zero; no runtime errors were seen.
The existing 16 committed screenshots remain applicable. See
[release review evidence](stage21_queue_filter_evidence/release-review.json).

Frozen model SHA256 was independently rechecked and unchanged. Backend routes,
production scoring/security, case-store schema, dependencies, historical
aggregates and all earlier scientific workflow text remain unchanged. No closed
final-test data was accessed. Remote `main` remains
`52c01c919cb40cd2bedf1a41794a73e38c6cf737`; target integration remains
`d419da2dfa11ff0dda85f99a094e114f4e733e3e` before the PR.

Release inventory is **35 files**: the approved original33 plus two fresh
release evidence files. The report, workflow note and inventory were updated
in place. No product-code changes were required during release review.
The limitations above remain; there are no identified blocking defects.
