# Model Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Validated frozen-artifact global MDI/TreeSHAP and bounded synthetic local contributions.
**Architecture:** Offline SHAP only; server-only versioned assets validated against loaded artifact/generator. GET endpoints expose aggregate globals and bounded synthetic lookups. Dedicated existing-router view plus case link.
**Tech Stack:** Existing SHAP0.54.1, NumPy/pandas, FastAPI, Chart.js, pytest/CDP Chrome. No installs.
**Spec:** docs/STAGE23_MODEL_EXPLAINABILITY.md (owner approved MDI+TreeSHAP protocol).

## Global Constraints
- Only stage23-model-explainability from756013d; local verified commit only.
- No frozen-artifact fit/change, final-test/prediction access, push/merge/deploy or Render changes.
- SHA2562fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04.
- Exact813-order preprocessing,512 label-free validation sample;240 fixed synthetic locals/public120.
- Baseline/group/serialized additivity1e-8; implicit path baseline cross-check1e-10.
- No new dependencies; no public SHAP/inference or case writes.

## Review Focus
- Mapping cancellation: signed sum before mean absolute, not legacy sum meanabs.
- Multi-output/class indexing: explicit classes_[1] and SHAP tensor checks; no assumed output space.
- Asset privacy/identity: no static directory, mismatched assets fail closed.
- Local request races/outdated scores: selected ID and displayed score must match.
- Unsupported local seeds/sizes: explicit unavailable without disturbing global view/case store.

### Task1 — numeric methods and bounded offline build
Files src/deployed_explainability.py, scripts/build_explainability_assets.py, scripts/run_explainability_build.py, tests/test_deployed_explainability.py, explainability_assets/*.json.
- [x] Write failing mapping/baseline/group/reconciliation/stability contracts, then implement and pass.
- [x] Run bounded two-row probe, fixed512 validation and240 synthetic cohorts/stress fixtures, with10m/4GiB watchdog.
- [x] Verify deterministic regeneration, all local/public queue scores, grouped math and resampling; save aggregate evidence only.

### Task2 — validated read-only API and browser view
Files src/explanation_assets.py, src/explanation_routes.py, minimal src/api.py/src/render_demo.py registration; ui/explainability.js and scoped dashboard edits; tests/test_explanation_routes.py.
- [x] Failing tests for static/privacy bounds, IDs, identity/schema corruption and read-only methods; implement and pass.
- [x] Method/feature/local selectors, signed chart+Other/baseline/score, case link, retry/reset, precise historical labels and limitations. Preserve closed results.

### Task3 — full verification and local delivery
Files browser test additions, scripts/verify_explainability_browser.py, docs/stage23_explainability_evidence/*, report/workflow.
- [x] Real Chrome4sizes including keyboard, case storage, empty states, scope/security and prior views; screenshots.
- [x] Independent source/science/security review, full regression including six heavy tests, checksum/protected paths/inventory.
- [x] Document findings/costs/limits and commit verified changes locally; do not advance Stage24.

## Execution rulings
- Native implementation continues under the owner's explicit implementation/protocol approval; no repeated approval gate.
- Generator identity uses Python major/minor (Random algorithm family), with full patch version retained in build evidence; avoids patch-only deployment mismatch. Exact deterministic cohort/score checks remain mandatory.
- Ruling: retain top10 absolute groups and split the existing Other net into included positive/negative subtotals for display; identical mathematical decomposition, no post-hoc feature/example selection.

- Verification: raw SHAP max error7.22e-16; final assets byte-identical; fresh-artifact240 scores/392 MDI groups independently match. Watched build39.48s/~381MiB.
- Source review response-size gap fixed with failure-first regression; re-review clear. Direct-link identity race and MDI-only disclosure covered by Chrome tests.
- Final suite555 passed/0 failed/0 skipped, five existing warnings,318.99s.84 Chrome and six heavy tests included. Protected files/refs/science preserved. Local commit is the final delivery action.
