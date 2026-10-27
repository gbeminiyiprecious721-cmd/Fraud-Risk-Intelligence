"""Stage 8: real-browser tests for the dashboard SPA's hash-based client-
side router and its shared global state (current demo batch, selected
transaction, filters) -- the two things no server-side pytest test can
verify, since both live entirely in ui/dashboard.js.

These drive a real, headless Chrome instance via the Chrome DevTools
Protocol, using only already-pinned project dependencies (`requests`,
`websocket-client` -- see requirements.txt). No new test framework
(Selenium/Playwright/Jest) was introduced, matching this project's
existing "prefer existing dependencies" convention and its established
pattern (Stages 3-8) of verifying frontend behavior against a real
running server rather than mocking the DOM.

Marked `@pytest.mark.browser` (registered in pytest.ini, same convention
as the existing `heavy`/`data_contract_full` markers) and self-skip via
`pytest.skip` if Chrome or a free port can't be found -- these have a
real external dependency (an installed Chrome browser) that the project's
other tests don't, so they must never hard-fail a run on a machine
without Chrome.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests
import websocket

ROOT_DIR = Path(__file__).resolve().parent.parent
UI_DIR = ROOT_DIR / "ui"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]


def _find_chrome() -> str | None:
    import os

    env_path = os.environ.get("CHROME_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=1).status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


class CDPSession:
    """Minimal Chrome DevTools Protocol client: one tab, synchronous
    request/response, with console messages collected as they arrive."""

    def __init__(self, cdp_port: int, start_url: str):
        self._base = f"http://127.0.0.1:{cdp_port}"
        # Deliberately NOT `requests.put(f"{base}/json/new?{start_url}")`
        # when start_url has a #fragment: HTTP clients strip URL fragments
        # before sending the request line (a fragment is a client-side-only
        # concept), so a hash would silently never reach Chrome's control
        # endpoint. Create the tab at about:blank, then Page.navigate to
        # the real URL as a JSON body value instead, which has no such
        # stripping problem.
        tab = requests.put(f"{self._base}/json/new?about:blank").json()
        self.tab_id = tab["id"]
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=15)
        self._id = 0
        self.console_messages: list[tuple] = []
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Page.navigate", {"url": start_url})

    def send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        this_id = self._id
        self.ws.send(json.dumps({"id": this_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == this_id:
                return msg
            if msg.get("method") == "Runtime.consoleAPICalled":
                p = msg["params"]
                args = [a.get("value", a.get("description", "")) for a in p.get("args", [])]
                self.console_messages.append((p.get("type"), args))
            if msg.get("method") == "Runtime.exceptionThrown":
                self.console_messages.append(("exception", msg["params"]["exceptionDetails"].get("text")))

    def evaluate(self, expr: str):
        """Send `expr` to the browser tab via CDP's Runtime.evaluate and
        return its JS-side result. Safe here specifically: `expr` is
        always a literal authored in this file (optionally f-string-
        interpolated with a fixed, hardcoded view name from this file's
        own VIEWS list below -- never user input or data from an
        untrusted source), and the tab belongs to a headless Chrome
        process this module's own `browser_env` fixture launched and will
        tear down -- this is the standard CDP automation pattern every
        browser-testing tool (Selenium/Playwright/Puppeteer) uses
        internally, not a generic code-execution sink."""
        r = self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r["result"]["result"].get("value")

    def evaluate_json(self, expr: str):
        return json.loads(self.evaluate(expr))

    def wait_for_queue_data(self, timeout: float = 10.0) -> None:
        """Poll for the async /v1/dashboard/queue fetch to actually have
        populated the table, instead of a fixed sleep -- a fixed delay
        was observed to flake under a cold first request (no server
        warm-up yet in a freshly started test module)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length > 0"):
                return
            time.sleep(0.2)
        raise AssertionError("Queue data never loaded within timeout -- dashboard.js may have failed to fetch /v1/dashboard/queue.")

    def click(self, selector: str) -> None:
        self.send("Runtime.evaluate", {"expression": f"document.querySelector({selector!r}).click();"})

    def navigate(self, url: str, settle: float = 2.0) -> None:
        self.send("Page.navigate", {"url": url})
        time.sleep(settle)

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            requests.get(f"{self._base}/json/close/{self.tab_id}", timeout=5)


@pytest.fixture(scope="module")
def browser_env():
    chrome_path = _find_chrome()
    if not chrome_path:
        pytest.skip("No Chrome binary found (set CHROME_PATH to override) -- browser tests require a real Chrome install.")

    # Fixed, not random, ports: ui/dashboard.js's own local-dev origin
    # detection (IS_LOCAL_DEV_SPLIT_ORIGIN) hardcodes "static server on
    # 8080, API on 8000" -- the project's own documented local-dev
    # convention (see docs/DASHBOARD.md). A random static port was tried
    # first and confirmed to silently break every data-dependent test:
    # the browser's own fetch went to the wrong origin, so the dashboard
    # genuinely never loaded any data -- not a bug in the dashboard, a bug
    # in not matching the one origin pair it actually recognizes.
    api_port = 8000
    static_port = 8080
    cdp_port = _free_port()
    for port, label in [(api_port, "API"), (static_port, "static")]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                pytest.skip(f"Port {port} ({label}) is already in use by another process -- free it and re-run.")

    api_proc = subprocess.Popen(
        [".venv/bin/uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    static_proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(static_port), "--directory", str(UI_DIR)],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for(f"http://127.0.0.1:{api_port}/health", timeout=15):
        api_proc.terminate()
        static_proc.terminate()
        pytest.skip("Local API server did not become ready in time.")
    if not _wait_for(f"http://127.0.0.1:{static_port}/dashboard.html", timeout=10):
        api_proc.terminate()
        static_proc.terminate()
        pytest.skip("Local static server did not become ready in time.")

    import tempfile

    profile_dir = tempfile.mkdtemp(prefix="chrome-browser-test-")
    chrome_proc = subprocess.Popen(
        [
            chrome_path,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={profile_dir}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for(f"http://127.0.0.1:{cdp_port}/json/version", timeout=10):
        chrome_proc.terminate()
        api_proc.terminate()
        static_proc.terminate()
        pytest.skip("Headless Chrome did not expose its DevTools Protocol port in time.")

    env = {"cdp_port": cdp_port, "static_origin": f"http://127.0.0.1:{static_port}"}
    try:
        yield env
    finally:
        for proc in (chrome_proc, api_proc, static_proc):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def page(browser_env):
    session = CDPSession(browser_env["cdp_port"], f"{browser_env['static_origin']}/dashboard.html")
    session.wait_for_queue_data()
    try:
        yield session
    finally:
        session.close()


pytestmark = pytest.mark.browser


def test_default_route_is_overview_with_no_hash(page):
    state = page.evaluate_json(
        "JSON.stringify({hash: location.hash, activeView: document.querySelector('.view.active').dataset.view})"
    )
    assert state["hash"] == ""
    assert state["activeView"] == "overview"


@pytest.mark.parametrize("view", ["queue", "analytics", "workflow", "insights", "responsible-ai", "overview"])
def test_clicking_each_nav_link_routes_correctly(page, view):
    page.click(f'.sidenav-list a[data-view="{view}"]')
    time.sleep(0.6)
    state = page.evaluate_json(
        """
        JSON.stringify({
          hash: location.hash,
          activeView: document.querySelector('.view.active').dataset.view,
          ariaCurrentView: (document.querySelector('.sidenav-list a[aria-current="page"]') || {}).dataset
            ? document.querySelector('.sidenav-list a[aria-current="page"]').dataset.view
            : null
        })
        """
    )
    assert state["hash"] == f"#/{view}"
    assert state["activeView"] == view
    assert state["ariaCurrentView"] == view


def test_direct_hash_navigation_resolves_correct_view_on_fresh_load(browser_env):
    session = CDPSession(browser_env["cdp_port"], f"{browser_env['static_origin']}/dashboard.html#/analytics")
    session.wait_for_queue_data()
    time.sleep(0.3)  # let Chart.js finish creating the just-rendered canvases
    try:
        state = session.evaluate_json(
            "JSON.stringify({activeView: document.querySelector('.view.active').dataset.view, chartCount: document.querySelectorAll('.analytics-grid canvas').length})"
        )
        assert state["activeView"] == "analytics"
        assert state["chartCount"] == 5
    finally:
        session.close()


def test_back_and_forward_restore_previous_views(page):
    for view in ["queue", "analytics", "workflow"]:
        page.click(f'.sidenav-list a[data-view="{view}"]')
        time.sleep(0.4)

    page.send("Runtime.evaluate", {"expression": "history.back();"})
    time.sleep(0.4)
    after_one_back = page.evaluate_json("JSON.stringify({view: document.querySelector('.view.active').dataset.view})")
    assert after_one_back["view"] == "analytics"

    page.send("Runtime.evaluate", {"expression": "history.back();"})
    time.sleep(0.4)
    after_two_back = page.evaluate_json("JSON.stringify({view: document.querySelector('.view.active').dataset.view})")
    assert after_two_back["view"] == "queue"

    page.send("Runtime.evaluate", {"expression": "history.forward();"})
    time.sleep(0.4)
    after_forward = page.evaluate_json("JSON.stringify({view: document.querySelector('.view.active').dataset.view})")
    assert after_forward["view"] == "analytics"


def test_selected_transaction_survives_navigating_to_another_view(page):
    """The core shared-state contract: openDrawer() sets one global
    state.selectedId, and navigating between views must never clear it."""
    page.click('.sidenav-list a[data-view="queue"]')
    time.sleep(0.6)
    clicked_id = page.evaluate(
        """
        (function() {
          var row = document.querySelector('#queue-tbody tr[data-id]');
          var id = row.dataset.id;
          row.click();
          return id;
        })()
        """
    )
    assert clicked_id
    drawer_after_click = page.evaluate_json(
        "JSON.stringify({open: document.getElementById('detail-drawer').classList.contains('open'), id: document.getElementById('drawer-id').textContent})"
    )
    assert drawer_after_click["open"] is True
    assert drawer_after_click["id"] == clicked_id

    page.click('.sidenav-list a[data-view="workflow"]')
    time.sleep(0.6)
    drawer_after_nav = page.evaluate_json(
        "JSON.stringify({open: document.getElementById('detail-drawer').classList.contains('open'), id: document.getElementById('drawer-id').textContent})"
    )
    assert drawer_after_nav["open"] is True
    assert drawer_after_nav["id"] == clicked_id


def test_global_filter_propagates_to_overview_and_workflow(page):
    page.click('.sidenav-list a[data-view="queue"]')
    time.sleep(0.6)
    queue_state = page.evaluate_json(
        """
        (function() {
          var sel = document.getElementById('filter-product');
          sel.value = sel.options[1].value;
          sel.dispatchEvent(new Event('change', {bubbles:true}));
          return JSON.stringify({product: sel.value, rows: document.querySelectorAll('#queue-tbody tr[data-id]').length});
        })()
        """
    )
    assert queue_state["rows"] > 0

    page.click('.sidenav-list a[data-view="workflow"]')
    time.sleep(0.6)
    kanban_total = page.evaluate("document.querySelectorAll('.kanban-card').length")
    assert kanban_total == queue_state["rows"]

    page.click('.sidenav-list a[data-view="queue"]')
    time.sleep(0.4)
    page.click("#reset-filters-btn")
    time.sleep(0.3)


def test_no_console_errors_across_full_navigation_sweep(browser_env):
    session = CDPSession(browser_env["cdp_port"], f"{browser_env['static_origin']}/dashboard.html")
    session.wait_for_queue_data()
    try:
        for view in ["queue", "analytics", "workflow", "insights", "responsible-ai", "overview"]:
            session.click(f'.sidenav-list a[data-view="{view}"]')
            time.sleep(0.5)
        errors = [m for m in session.console_messages if m[0] == "exception"]
        assert not errors, f"Console exceptions during navigation: {errors}"
    finally:
        session.close()
