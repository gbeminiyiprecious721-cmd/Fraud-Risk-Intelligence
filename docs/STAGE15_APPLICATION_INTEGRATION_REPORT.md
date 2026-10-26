# Stage 15 — Local Application Integration

**Status: implementation complete, not yet marked done.** A minimal static HTML/CSS/vanilla-JS page that makes a real HTTP call to the existing Stage 14 API, for a human-oriented demonstration — not a dashboard, not a commercial product, not deployed anywhere. `final_test` was not accessed; the frozen model, the Stage 13 artifact, and Stage 13's/14's implementations were not modified beyond one narrowly-scoped CORS addition.

## 1. Architecture

| Piece | What it is |
|---|---|
| `ui/index.html` | One static file — no framework, no build step, no new dependency. Inline CSS + vanilla JS. |
| `src/api.py` | The existing Stage 14 API, with one addition: `CORSMiddleware`, scoped to two explicit local origins. |

The UI never talks to the model, the preprocessing pipeline, or the artifact directly — it only calls `POST http://127.0.0.1:8000/v1/predict` over real HTTP, exactly as any other API client would.

## 2. The narrowly scoped CORS change

```python
ALLOWED_UI_ORIGINS = ["http://127.0.0.1:8080", "http://localhost:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_UI_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

- **No wildcard origin.** `127.0.0.1` and `localhost` are listed as two *separate* origins deliberately — browsers treat them as different origins even though they resolve to the same host.
- **`allow_credentials=False`** — there are no cookies or auth tokens to protect in this demonstration; nothing requires sending credentials cross-origin.
- **Only `GET`/`POST`** and only the `Content-Type` header — the UI needs nothing else.
- Both local servers are documented to bind to `127.0.0.1` only (§6) — never `0.0.0.0`, never a public interface.

## 3. Real local API integration evidence (not simulated)

Both real servers were started for real, and the UI's actual embedded 392-field synthetic payload was extracted directly from `ui/index.html` and sent to the running API with `curl`, using the exact `Origin` header a browser would send — this is what the page's own `fetch()` call does, verified at the HTTP level:

| Check | Result |
|---|---|
| `GET http://127.0.0.1:8080/index.html` | `200`, `content-type: text/html` |
| `GET http://127.0.0.1:8000/health` | `200` |
| `POST /v1/predict` with the real embedded payload, `Origin: http://127.0.0.1:8080` | `200`, `access-control-allow-origin: http://127.0.0.1:8080`, `fraud_probability=0.5201984944879683` (displayed as `0.52 / 1.00` via `toFixed(2)`) |
| Same request, `TransactionAmt` removed | `422`, CORS header still present (so the browser WOULD expose this error to the page), `{"detail": "Missing required raw column(s): ['TransactionAmt']", "error_category": "SchemaValidationError"}` |
| API process killed mid-session, request retried | `curl` exit code 7 ("Failed to connect") — exactly the failure `fetch()` would reject on, landing in the page's `catch` block → `showUnavailable()`, never a zero score |
| API restarted, same request retried | `200`, **identical** `fraud_probability` (0.5201984944879683) — confirms clean recovery, no state corruption from the stop/restart |
| `OPTIONS /v1/predict` preflight, `Origin: http://evil.example.com` | `400`, no `access-control-allow-origin` header — matches the automated test exactly |
| Both demonstration servers stopped afterward | Confirmed via `ps aux` — no orphaned `uvicorn` or `http.server` process remained |

## 4. UI states — loading / success / error / unavailable

Implemented in `ui/index.html`'s `hideAllOutcomePanels()`/`setLoading()`/`showResult()`/`showError()`/`showUnavailable()`:

- **Loading** begins by calling `hideAllOutcomePanels()` *before* the `fetch()` call starts — this is what guarantees a new request can never leave a stale, previous score visible while a new one is in flight or has failed.
- **Success** shows the score (2 decimal places), the artifact/manifest version, and a collapsible "technical details" section with the full raw JSON response — never shown by default, so the main view stays simple.
- **Error** (4xx) shows the server's own `detail` message — safe to display, since the client already knows what it sent (Stage 14's own design decision, reused here).
- **Unavailable** (network failure or a 5-second `AbortController` timeout) shows a fixed, non-alarming message with retry instructions — **never** a zero score, confirmed in §3 above with a real killed server, not just reasoned about.

## 5. Calibration-safe score presentation

- The score is rounded to 2 decimal places (`toFixed(2)`) — not the raw 16-significant-figure float the API actually returns (that remains visible only in the optional "technical details" section).
- The caveat **"This is a relative risk ranking score — not a verified, calibrated probability of real-world fraud"** is shown immediately next to every displayed score, and restated again in a dedicated, always-visible "Important" section on the page — not only on success.
- A separate, equally prominent line states the score must never trigger an automatic fraud accusation or payment rejection, tying the UI directly to Stage 1's own scoped decision (human investigation prioritisation only).
- No threshold, no "flagged"/"cleared" label, and no case-outcome claim is generated by the real integration — the only fabricated-looking content on the page (a 3-row "review queue") is inside a dashed-border box explicitly labeled **"ILLUSTRATIVE MOCKUP — fabricated example, not real model output"**, visually and textually separated from the real API result above it.

## 6. CORS tests and configuration

6 new tests in `tests/test_api.py` (30 total in that file, 297 project-wide):

| Test | Verifies |
|---|---|
| `test_cors_allowed_origin_receives_the_allow_origin_header` | An allowed origin gets `access-control-allow-origin` on a real `POST` |
| `test_cors_disallowed_origin_does_not_receive_the_allow_origin_header` | A disallowed origin's request still succeeds server-side (CORS is a browser-enforced control, not a server refusal) but gets no CORS header — the browser is what blocks the page from reading it |
| `test_cors_real_preflight_for_allowed_origin` | A genuine `OPTIONS` preflight (with `Access-Control-Request-Method`/`-Headers`) for an allowed origin returns `200` with the right headers |
| `test_cors_real_preflight_for_disallowed_origin_is_not_approved` | The same preflight for a disallowed origin returns `400` with no CORS header — verified directly, not assumed |
| `test_cors_does_not_allow_credentials` | `allow_credentials=False` is read directly from the app's own middleware stack |
| `test_cors_allowed_origins_are_127_0_0_1_and_localhost_only` | No wildcard origin exists anywhere in the configuration |

All 6 passed on first run; the real running server's behavior (§3) matched the `TestClient`-based automated tests exactly, both for the allowed and the disallowed case.

## 7. Investigator-oriented framing

The page explicitly states the intended decision consumer, per this project's own Stage 1 business framing, is a **human fraud investigator prioritising reviews under a limited capacity** — never an automated decision-maker. No case management, customer accounts, batch scoring, live SHAP, or investigation system was built — the one small illustrative "review queue" mockup included is clearly, visually, and textually marked as fabricated and separate from the real model output above it (§5).

## 8. Accessibility — what was and was not verified

**Honestly disclosed limitation: no GUI browser or screenshot tool was available in this session** (the environment's browser-automation plugin was not connected). The following could **not** be independently verified by actually rendering the page, tabbing through it, or capturing a screenshot — and this report does not claim otherwise:

- Actual visual layout/appearance on a real desktop browser window.
- Actual keyboard tab order and focus behavior as experienced by a user.
- Any screen-reader behavior.

**What was verified, precisely, by direct inspection of the real file (`ui/index.html`) and computation, not assumption:**
- The scoring control is a native `<button type="button">` element with no `tabindex` manipulation anywhere in the file — native HTML buttons are keyboard-focusable and activate on Enter/Space by default in every major browser, by the HTML specification itself, not by any custom script.
- No CSS rule anywhere in the file sets `outline: none` or otherwise removes a focus indicator — confirmed by a direct text search of the file; the only `outline` rule present *adds* a 3px visible focus ring.
- The status region (`#status`) uses `role="status"` and `aria-live="polite"`, so a screen reader would be told about loading/completion without needing focus to move there — a documented ARIA pattern, not independently confirmed with an actual screen reader.
- Text contrast ratios were computed directly from the chosen colors (WCAG's relative-luminance formula, by hand, not asserted): body text `#e8e8e8` on background `#0f1115` ≈ **15.4:1**; muted text `#a9adb8` on the same background ≈ **8.4:1**; the button's dark text `#0f1115` on its accent background `#5aa9ff` ≈ **7.7:1** — all three comfortably clear WCAG AA's 4.5:1 minimum for normal text, and the first two clear AAA's 7:1 as well.

This is a real, bounded set of code-level accessibility properties — not a substitute for an actual assistive-technology or rendered-browser check, which this environment could not perform this round.

## 9. Quality assurance

- **Artifact checksum** (`artifacts/frozen_rf_fraud_risk_v1.joblib`) independently recomputed and confirmed **identical** (`2fb0cbc4...`) both before and after this round's work.
- **Full project suite**: **297 passed, 0 failed, 0 skipped** (291 prior + 6 new CORS tests, zero regressions), including the heavyweight Stage 13 full-training regression test.
- **`git diff --stat`**: `src/api.py` (+21 lines, the CORS addition only), `tests/test_api.py` (+70 lines, the 6 new CORS tests), new `ui/` directory (`index.html`). **Zero Stage 1-13 source files touched**; Stage 14's existing endpoints, response shape, logging, and error handling are byte-for-byte unchanged aside from the one approved CORS addition.

## 10. Known limitations

- **No GUI/browser verification was possible this session** (§8) — the visual layout, actual keyboard interaction, and screen-reader behavior are *designed for* but not independently confirmed.
- The UI has no build step and no bundler — this is deliberate (no new dependency authorised), not an oversight, but it also means no linting/type-checking exists for the inline JavaScript.
- `ALLOWED_UI_ORIGINS` hardcodes port `8080` for the UI's static server — starting the demo on a different port requires editing `src/api.py`, a manual step documented in §6 of `ui/index.html` itself.
- This remains everything Stage 14 already disclosed: `fraud_probability` is uncalibrated; no decision threshold exists; no authentication, rate limiting, TLS, or monitoring exists; the artifact is a verified reconstruction, not Stage 10's literal historical object; the negative-`TransactionAmt` policy remains unresolved. None of these were revisited or weakened this round.

## Why application integration matters even without a commercial product

A model that only a script can call is still, in a real sense, unfinished — not because the HTTP wrapper (Stage 14) was insufficient, but because "can a human actually look at this and make a decision with it" is a different, harder question than "does the API return 200." This stage is where every carried-forward caveat (uncalibrated score, no threshold, human-review-only decision scope) either survives contact with an actual screen a person would read, or gets quietly lost in translation. Writing the calibration caveat into the UI's markup directly, next to the number it describes, is the concrete, portfolio-visible proof that this project's own discipline about not overclaiming was followed all the way to the last visible surface — not just in a Markdown report a hiring manager might never scroll to.

## Three interview-defense questions

1. **"What does the end user see if the model is wrong or unavailable?"** — Never a silently-wrong or silently-blank result: a 4xx shows the real server-provided reason (§3); a genuinely unreachable API shows an explicit "service unavailable, retry" message (§4), verified for real by killing the API process mid-session and confirming the exact failure mode a browser's `fetch()` would hit.
2. **"How did you decide what to show vs. hide from the consumer?"** — The raw 16-digit probability, the full JSON response, and the request id are all real but demoted to an optional "technical details" section; the one thing that is never optional or collapsible is the calibration caveat, because that's the single fact most likely to be silently lost if this page were handed to someone without this project's full context.
3. **"Why didn't you just build this with React/a dashboard framework?"** — Because the stage explicitly authorised zero new dependencies and no build system, and a 392-field demonstration page genuinely doesn't need one — using a framework here would have been solving a problem this project doesn't have, at the cost of an approval round-trip for no real benefit.

## 11. Real browser verification (2026-10-26, addendum)

The user accepted the API integration and automated tests, and asked specifically for real browser rendering/interaction evidence, since none had been available in the implementation round above. **This time, real evidence was obtained**, using only the already-installed Google Chrome (version **155.0.8059.85**) in its built-in headless screenshot mode — no new tool, package, or browser binary was installed.

### Browser capability check
`/Applications/Google Chrome.app` and `/Applications/Safari.app` are both already installed on this machine. Chrome's own command-line headless mode (`--headless=new --screenshot=<path>`) renders a real page with the real engine and writes a real PNG — this is a built-in Chrome capability, not a new dependency. No paid service, no Playwright/Selenium install, no browser binary download was used.

### Method
Both real servers were started (API on `127.0.0.1:8000`, static UI on `127.0.0.1:8080`, exactly as documented). For interaction (clicking the real button, not just loading the page), a small **temporary, same-origin verification harness** (`ui/_verify_harness.html`) was created, used, and then **deleted** before this report was finalized — it is not part of the committed UI and does not appear in the final `git diff`. It loaded the real `index.html` in a same-origin `<iframe>` (so the real page's real `fetch()` calls work under the real CORS policy) and dispatched a real DOM `.click()`/`.focus()` on the real button — equivalent to what a user's mouse click or keyboard activation produces, run through the actual rendering engine, not simulated via HTTP alone.

### Results — all against the real, running local servers

| # | Screenshot | What it shows | Verified |
|---|---|---|---|
| 1 | `docs/stage15_screenshots/01_ready_state.png` | Initial page load, no interaction yet | Real rendering: heading, badge, button, all sections, correct dark theme and layout |
| 2 | `docs/stage15_screenshots/02_success_after_click.png` | After a real click, API running | Real score **0.52 / 1.00** displayed, uncalibrated-probability caveat visible, artifact version `1.0.0`/manifest `1.1.0` shown, console panel confirms **no errors** |
| 3 | `docs/stage15_screenshots/03_error_invalid_payload.png` | After a real click with `TransactionAmt` removed from the real payload | Real `422` surfaced as "The API rejected this request. Missing required raw column(s): ['TransactionAmt']" — no stale/zero score, no console errors |
| 4 | `docs/stage15_screenshots/04_unavailable_api_down.png` | API process genuinely killed (`kill`, confirmed via `ps`) before this click | Real "Scoring service unavailable... try again" message — again, no stale or zero score, no console errors |
| 5 | `docs/stage15_screenshots/05_retry_after_restore.png` | API genuinely restarted, same click repeated | Real score **0.52 / 1.00** again — identical to #2, confirming clean recovery with no state corruption |
| 6 | `docs/stage15_screenshots/06_keyboard_activation.png` | `.focus()` + a dispatched `Enter` `keydown`, then `.click()` | Same successful result as #2 |
| 7 | `docs/stage15_screenshots/07_focus_outline.png` | `.focus()` only, no click | The button's real, rendered **white 3px focus ring** is visible — the `:focus-visible` CSS rule works as intended in a real browser, not just in source |

**One precise, honest caveat on #6**: synthetic `KeyboardEvent`s dispatched via JavaScript do not reliably trigger a browser's *default* action for a focused `<button>` (a known browser-engine limitation, not specific to this page) — the harness's `.click()` call after the dispatched `Enter` keydown models the outcome a real physical key-press produces, rather than proving the synthetic event alone triggered it. What **is** directly, separately proven is more fundamental and not affected by that caveat: the button is a native, unmodified `<button>` element (confirmed by source inspection, §8 of the original report), it is genuinely focusable (screenshot #7, a real rendered focus ring with no synthetic click involved), and native `<button>` elements activate on a real Enter/Space key press by HTML specification, independent of any JavaScript on the page.

### Console errors
No console error, unhandled promise rejection, or `console.error` call was observed in **any** of the 7 real runs — captured via real `error`/`unhandledrejection` event listeners attached to the real iframe's real `window`, not inferred.

### What remains unverified
Actual mouse-driven clicking (as opposed to a DOM `.click()` call), real physical keyboard hardware events, and screen-reader software behavior were still not exercised — those require a human at a real input device or screen-reader software, which this environment cannot simulate even with Chrome available. This is named explicitly, not glossed over.

### Recommendation on Stage 15 closure
With this addendum, every item in the user's original verification checklist now has **real evidence**, not reasoning-only: desktop rendering, button interaction, an actual score displayed after a real API response, the persistent uncalibrated-score warning, loading-adjacent success/error states, no stale score after a failed request, a successful retry after restoring the API, basic keyboard focusability (with the one precise caveat above), and zero observed console errors. The only remaining gap — literal human-hand/screen-reader interaction — is a materially smaller, clearly-scoped residual limitation rather than the prior round's much larger "no browser evidence at all" gap. **Recommendation: this is sufficient real evidence to close Stage 15**, with the remaining human/assistive-technology gap recorded as a standing limitation rather than blocking closure — but the decision is the user's, as always.

**Stage 15 is not marked complete by this round.** Stage 16 has not begun, nothing was deployed, `final_test` was not accessed, and the verification harness used to produce this evidence has been deleted and does not appear in the committed UI.
