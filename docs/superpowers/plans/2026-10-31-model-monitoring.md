# Model Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Historical, label-free drift intelligence using three disjoint validation windows.
**Architecture:** Offline bounded inference produces a validated immutable aggregate. A fixed GET route serves it; the dashboard renders charts/details with native selectors and monitoring-only reset.
**Tech Stack:** Existing NumPy/pandas/SciPy, FastAPI, Chart.js, pytest, real Chrome.
**Spec:** docs/STAGE22_MODEL_MONITORING.md

## Global Constraints
- Only stage22-model-monitoring; local commit only, no push/merge/deploy.
- Frozen SHA2562fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04 unchanged.
- No closed final-test/outcome reads, production scoring change, case writes or new dependencies.
- Owner selected earliest validation third; all scores out-of-sample, reused development evidence.

## Review Focus
- Constant/all-missing features: explicit bins or insufficient evidence, no fabricated zero drift.
- Reference-unseen categories: common vocabulary and explicit counts, not automatic schema invalidity.
- Invalid aggregate: fail closed with unavailable response, no partial misleading chart.
- Historical view switches: no case storage or synthetic filter mutation.
- Narrow viewport: readable controls and chart tables without horizontal body overflow.

### Task1 — numerical contract and offline pipeline
Files: src/drift_intelligence.py, scripts/build_monitoring_aggregate.py, tests/test_drift_intelligence.py, ui/monitoring-v1.json.
Interfaces: split_windows(metadata), numeric_comparison(reference,comparison), categorical_comparison(reference,comparison), validate_aggregate(payload), build_payload(frame,scores).
- [x] Write tests for window ties/separation, known PSI/KS/JS, missing/constant/unknown/insufficient cases; observe failure.
- [x] Implement pure calculations, strict schemas and authorised ID-first reader with no outcome column.
- [x] Verify known fixtures independently, generate twice and compare canonical aggregate checksums.

### Task2 — read-only service and monitoring UI
Files: src/monitoring_routes.py, minimal router/static registration in src/api.py and src/render_demo.py; ui/monitoring.js, dashboard.html/js/css; tests/test_monitoring_routes.py.
- [x] Add endpoint tests for aggregate-only shape, missing/corrupt evidence and unsupported writes; observe failures.
- [x] Serve schema-validated fixed aggregate. Add native window/feature/mode/reset/retry controls, ranking/detail/score/quality/trend charts and exact values.
- [x] Preserve existing security, API routes and case store; no runtime inference.

### Task3 — browser, regression and evidence
Files: tests/test_dashboard_routing_browser.py (add monitoring tests), docs/stage22_monitoring_evidence/*, docs/STAGE22_MODEL_MONITORING.md, docs/ML_WORKFLOW_STATE.md.
- [x] Test all four sizes, selector/reset/keyboard, charts, navigation and case storage preservation, API failure recovery.
- [x] Run full suite including six heavy/data-contract tests; capture screenshots and actual counts.
- [x] Independent source review, preservation checks, inventory, local verified commit. No next-stage work.


## Execution ledger
- Owner chose earliest validation third; original training-reference proposal superseded before calculation.
- Numerical and API tests first failed on absent modules; implemented contracts then passed. Review mutation tests first exposed four validator gaps, then passed after fixes.
- Independent review found incomplete score/bin invariants and undefined chart shares; resolved and re-reviewed cleanly.
- Browser inspection found compressed desktop charts/overflow; targeted contained layout fixes, horizontal ranking and explicit plot-area tests pass.
- Final complete regression525 passed/0 failed/0 skipped, five inherited warnings,292.42s; all77 Chrome and six heavy tests included.
- No stage advancement, external writes, deployment or new dependencies. Local verified commit is the final delivery action.
