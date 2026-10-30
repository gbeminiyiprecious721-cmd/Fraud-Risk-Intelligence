# Investigation Capacity Simulator Implementation Plan

**Goal:** Extend the existing dashboard with historical validation top-K workload analysis.
**Architecture:** Use the trusted train-only frozen artifact for validation inference once, independently verify all checkpoint counts, publish only a compact versioned aggregate, and render a dedicated hash-routed view using existing FastAPI/Chart.js patterns.
**Tech stack:** Existing Python/numpy/pandas/sklearn/FastAPI, vanilla JS and Chart.js; no new dependency.
**Spec:** Owner's Stage 19 Investigation Capacity Simulator brief supplied 2026-10-30.

## Evidence and design decisions

- Published Stage 10 final-test summary metrics remain historical, separate and unchanged; no final-test predictions or derived artifacts will be opened.
- The trusted Stage 13 artifact manifest declares train-only fitting (438125 rows), checksum 2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04. The established validation population is eligible for inference without retraining. It was consulted in development, so estimates are selection-biased development backtesting, not independent generalization.
- Only one legitimate population is used: no misleading population selector.
- Deterministic ranking is descending score then ascending TransactionID. Capacity checkpoints span 0–100% in 0.5% increments, with integer K using ceiling; numeric input snaps to a computed checkpoint. No inferred/interpolated performance metrics.
- Random review means uniform sampling without replacement; show expected counts, not a simulated or guaranteed outcome. Marginal yield refers only to adjacent computed blocks.
- At K=0 precision is undefined; recall is undefined when there are no positives. Counts still remain defined.
- Synthetic dashboard filters/batches do not affect this historical population; simulator labels must make that separation visible.

## Tasks and checks

- [x] Write failing unit tests for ranking, counts, boundaries, invalid inputs, tied scores, monotonic capture, random expectation and independent calculations.
- [x] Implement pure metric/aggregate functions in `src/capacity_simulator.py`; add a reproducible validation-only builder in `scripts/build_capacity_aggregate.py`. Features/labels must be parsed only for validation rows; first scan only IDs to locate physical rows. Verify train fingerprint against manifest, disjointness, schema and checksum. Do not save row-level predictions.
- [x] Independently compare every checkpoint to sklearn confusion/precision/recall calculations and record provenance in `ui/capacity-v1.json`.
- [x] Add fixed read-only aggregate endpoint to both local/public app without altering prediction/security handlers. Add only the required frontend JS asset route.
- [x] Integrate dedicated `#/capacity` view and `ui/capacity.js`: synchronized slider/numeric input, KPIs, two charts, random baseline, marginal yield, Reset, assumptions and explicit error states.
- [x] Add real-browser tests for inputs/charts/reset/routes, desktop/tablet/mobile sizes, original six views and original scoring. Capture screenshots and run the entire applicable suite including heavy/data-contract tests.
- [x] Preserve scientific workflow tail; add separate Stage 19 enhancement state/report, verification evidence, limitations. Verify checksum and branch boundaries; commit locally only.

## Review focus

No final-test access, no model fit, no row-level public output, no threshold/calibration claim, no monetary estimates, denominator correctness, stable ties, no hidden connection to synthetic filters, chart availability/error fallback, keyboard-accessible controls and natural mobile scrolling. Full legacy regression tests may fit temporary models for their existing train-only equivalence checks; Stage 19 generation performs inference only and never replaces the frozen artifact.
