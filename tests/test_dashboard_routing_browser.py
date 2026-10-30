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
            # Stage 15 split Analytics into a 2x2 "primary" grid (4 charts,
            # .analytics-grid-primary) plus one standalone 5th chart
            # (.analytics-grid-wide-standalone) so the primary 4 can fill the
            # view's available height without the 5th stealing space from
            # them (see docs/DASHBOARD.md) -- both live inside the active
            # Analytics view, so scoping the count to the view (not the old
            # single .analytics-grid container, which no longer wraps all 5)
            # is what still proves "5 charts, fresh load" that this test checks.
            "JSON.stringify({activeView: document.querySelector('.view.active').dataset.view, chartCount: document.querySelectorAll('.view[data-view=\"analytics\"] canvas').length})"
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


def test_product_chart_distinguishes_no_high_risk_from_no_matches(page):
    page.click('.sidenav-list a[data-view="analytics"]')
    page.evaluate("document.getElementById('filter-band').value = 'Low'; document.getElementById('filter-band').dispatchEvent(new Event('change'))")
    assert page.evaluate("document.getElementById('chart-by-product-empty').textContent") == "No High or Critical transactions in the current scope."
    page.evaluate("document.getElementById('filter-search').value = 'no-such-transaction'; document.getElementById('filter-search').dispatchEvent(new Event('input'))")
    time.sleep(0.5)
    assert page.evaluate("document.getElementById('chart-by-product-empty').textContent") == "No synthetic transactions match the current filters."
    assert "Showing 0 of" in page.evaluate("document.getElementById('global-scope-status').textContent")
    page.click('#reset-filters-btn')
    assert page.evaluate("document.getElementById('chart-by-product-empty').hidden")
    assert "Showing all" in page.evaluate("document.getElementById('global-scope-status').textContent")


@pytest.mark.parametrize('width,height', [(1440, 900), (1280, 800), (1024, 600), (390, 844)])
def test_refined_views_fit_width_and_keep_content_accessible(page, width, height):
    page.send('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False})
    for view in ['overview', 'queue', 'analytics', 'workflow', 'insights', 'responsible-ai']:
        page.click(f'.sidenav-list a[data-view="{view}"]')
        time.sleep(0.2)
        metrics = page.evaluate_json("JSON.stringify({width: innerWidth, body: document.body.scrollWidth, heading: document.querySelector('.view.active h2').getBoundingClientRect().height})")
        assert metrics['body'] <= metrics['width'], (view, metrics)
        assert metrics['heading'] > 0
    if width > 900:
        page.click('.sidenav-list a[data-view="queue"]')
        time.sleep(0.2)
        assert page.evaluate("document.querySelector('.table-scroll-full').clientHeight > 0")
        page.click('.sidenav-list a[data-view="analytics"]')
        time.sleep(0.2)
        assert page.evaluate("Array.from(document.querySelectorAll('.analytics-grid-primary canvas')).every(c => c.getBoundingClientRect().height > 0)")


def test_single_score_back_link_and_disclosure_badge(page, browser_env):
    page.navigate(f"{browser_env['static_origin']}/index.html")
    page.click('.back-link')
    page.wait_for_queue_data()
    assert page.evaluate('location.hash') == '#/overview'
    page.click('.badge-uncalibrated')
    time.sleep(0.2)
    assert page.evaluate("document.querySelector('.view.active').dataset.view") == 'responsible-ai'


@pytest.mark.parametrize('width,height', [(1440, 900), (1280, 800)])
def test_desktop_views_use_local_scroll_regions(page, width, height):
    page.send('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False})
    for view in ['overview', 'queue', 'analytics', 'workflow', 'insights', 'responsible-ai']:
        page.click(f'.sidenav-list a[data-view="{view}"]')
        time.sleep(0.3)
        assert page.evaluate("document.body.scrollHeight <= innerHeight"), view
        assert page.evaluate("document.querySelector('.view-container').scrollHeight <= document.querySelector('.view-container').clientHeight + 1"), view
    page.click('.sidenav-list a[data-view="workflow"]')
    time.sleep(0.3)
    assert page.evaluate("Array.from(document.querySelectorAll('.kanban-column')).every(e => e.getBoundingClientRect().bottom <= document.querySelector('.view-container').getBoundingClientRect().bottom)")
    assert page.evaluate("Array.from(document.querySelectorAll('.kanban-cards')).some(e => e.scrollHeight > e.clientHeight)")


def test_kanban_chart_click_and_real_single_transaction_scoring(page, browser_env):
    page.send('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 900, 'deviceScaleFactor': 1, 'mobile': False})
    page.click('.sidenav-list a[data-view="workflow"]')
    time.sleep(0.3)
    selected = page.evaluate("document.querySelector('.kanban-card').dataset.id")
    page.click('.kanban-card')
    assert page.evaluate("document.getElementById('drawer-id').textContent") == selected
    assert page.evaluate("document.getElementById('detail-drawer').classList.contains('open')")
    page.click('#drawer-close')
    assert not page.evaluate("document.getElementById('detail-drawer').classList.contains('open')")
    page.click('.sidenav-list a[data-view="analytics"]')
    time.sleep(0.4)
    assert page.evaluate("Array.from(document.querySelectorAll('.view[data-view=\"analytics\"] canvas')).every(c => Chart.getChart(c))")
    # Send genuine browser pointer events to the first nonzero doughnut arc.
    target = page.evaluate_json("""JSON.stringify((()=>{const c=Chart.getChart(document.getElementById('chart-band-breakdown')); const i=c.data.datasets[0].data.findIndex(n=>n>0); const e=c.getDatasetMeta(0).data[i]; const a=(e.startAngle+e.endAngle)/2; const r=(e.innerRadius+e.outerRadius)/2; const b=c.canvas.getBoundingClientRect(); return {x:b.x+e.x+Math.cos(a)*r,y:b.y+e.y+Math.sin(a)*r,band:c.data.labels[i]};})())""")
    for event in ['mouseMoved', 'mousePressed', 'mouseReleased']:
        page.send('Input.dispatchMouseEvent', {'type': event, 'x': target['x'], 'y': target['y'], 'button': 'left', 'clickCount': 1})
    time.sleep(0.4)
    assert page.evaluate("document.getElementById('filter-band').value") == target['band']
    page.click('.sidenav-list a[data-view="queue"]')
    time.sleep(0.3)
    page.click('#reset-filters-btn')
    assert page.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length") == 48
    page.navigate(f"{browser_env['static_origin']}/index.html")
    page.click('#score-btn')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and page.evaluate("document.getElementById('result').hidden"):
        time.sleep(0.2)
    assert not page.evaluate("document.getElementById('result').hidden")
    assert 0 <= float(page.evaluate("document.getElementById('score-value').textContent").split(' / ')[0]) <= 1
    assert page.evaluate("document.getElementById('error').hidden && document.getElementById('unavailable').hidden")
    page.click('.back-link')
    page.wait_for_queue_data()
    assert page.evaluate("document.querySelector('.view.active').dataset.view") == 'overview'


@pytest.mark.parametrize('width,height', [(1440, 900), (1280, 800), (768, 1024), (390, 844)])
def test_capacity_workspace_controls_charts_and_layout(page, width, height):
    page.send('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False})
    page.click('.sidenav-list a[data-view="capacity"]')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and page.evaluate("document.getElementById('capacity-workspace').hidden"):
        time.sleep(.1)
    assert not page.evaluate("document.getElementById('capacity-workspace').hidden")
    assert page.evaluate("document.querySelector('.view.active').dataset.view") == 'capacity'
    assert page.evaluate("document.querySelector('.topbar .badge-synthetic').textContent") == 'Historical validation / portfolio'
    assert page.evaluate("document.getElementById('capacity-count').value") == '756'
    page.evaluate("document.getElementById('capacity-slider').value=0;document.getElementById('capacity-slider').dispatchEvent(new Event('input'))")
    assert page.evaluate("document.getElementById('capacity-count').value") == '0'
    assert 'Not defined' in page.evaluate("document.getElementById('capacity-kpis').textContent")
    page.evaluate("document.getElementById('capacity-count').value=75560;document.getElementById('capacity-count').dispatchEvent(new Event('change'))")
    assert page.evaluate("document.getElementById('capacity-slider').value") == '200'
    assert '100.00%' in page.evaluate("document.getElementById('capacity-kpis').textContent")
    page.evaluate("document.getElementById('capacity-count').value=999999;document.getElementById('capacity-count').dispatchEvent(new Event('change'))")
    assert page.evaluate("document.getElementById('capacity-count').value") == '75560'
    page.click('#capacity-reset')
    assert page.evaluate("document.getElementById('capacity-count').value") == '756'
    assert page.evaluate("document.body.scrollWidth <= innerWidth")
    assert page.evaluate("Chart.getChart(document.getElementById('capacity-recall-chart')).data.datasets.length") == 3
    if width > 900:
        assert page.evaluate("document.querySelector('.view-container').scrollHeight <= document.querySelector('.view-container').clientHeight+1")
        assert page.evaluate("['capacity-recall-chart','capacity-precision-chart'].every(id => {const c=Chart.getChart(document.getElementById(id));return c.chartArea.bottom-c.chartArea.top >= 70;})")
        # Genuine pointer input to the measured curve at the 50% checkpoint.
        target = page.evaluate_json("""JSON.stringify((()=>{const c=Chart.getChart(document.getElementById('capacity-recall-chart'));const p=c.getDatasetMeta(0).data[100];const b=c.canvas.getBoundingClientRect();return {x:b.x+p.x,y:b.y+p.y}})())""")
        for event in ['mouseMoved','mousePressed','mouseReleased']:
            page.send('Input.dispatchMouseEvent', {'type':event,'x':target['x'],'y':target['y'],'button':'left','clickCount':1})
        time.sleep(.2)
        assert page.evaluate("document.getElementById('capacity-slider').value") == '100'
    page.click('.capacity-decision summary')
    assert page.evaluate("document.querySelector('.capacity-decision').open")
    assert page.evaluate("document.getElementById('capacity-counts').getBoundingClientRect().height > 0")
    page.click('.capacity-decision summary')
    page.click('.sidenav-list a[data-view="overview"]')
    time.sleep(.2)
    page.click('.sidenav-list a[data-view="capacity"]')
    time.sleep(.2)
    assert page.evaluate("document.querySelector('.view.active').dataset.view") == 'capacity'
    assert not [x for x in page.console_messages if x[0] in {'exception', 'error'}]


def test_capacity_evidence_failure_is_explicit_and_retry_recovers(browser_env):
    page = CDPSession(browser_env['cdp_port'], 'about:blank')
    try:
        page.send('Network.enable')
        page.send('Network.setBlockedURLs', {'urls':['*v1/dashboard/capacity*']})
        page.navigate(browser_env['static_origin'] + '/dashboard.html#/capacity')
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and page.evaluate("document.getElementById('capacity-retry').hidden"):
            time.sleep(.1)
        assert page.evaluate("document.getElementById('capacity-workspace').hidden")
        assert not page.evaluate("document.getElementById('capacity-retry').hidden")
        assert 'No simulator results' in page.evaluate("document.getElementById('capacity-status').textContent")
        page.send('Network.setBlockedURLs', {'urls':[]})
        page.click('#capacity-retry')
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and page.evaluate("document.getElementById('capacity-workspace').hidden"):
            time.sleep(.1)
        assert not page.evaluate("document.getElementById('capacity-workspace').hidden")
        page.evaluate("document.getElementById('capacity-slider').focus()")
        page.send('Input.dispatchKeyEvent', {'type':'keyDown','key':'ArrowRight','code':'ArrowRight','windowsVirtualKeyCode':39})
        page.send('Input.dispatchKeyEvent', {'type':'keyUp','key':'ArrowRight','code':'ArrowRight','windowsVirtualKeyCode':39})
        assert page.evaluate("document.getElementById('capacity-slider').value") == '3'
        assert not [x for x in page.console_messages if x[0]=='exception']
    finally:
        page.close()


@pytest.mark.parametrize('width,height', [(1440,900),(1280,800),(768,1024),(390,844)])
def test_benchmark_controls_charts_population_and_responsive_layout(page,width,height):
    page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
    page.click('.sidenav-list a[data-view="benchmark"]')
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and page.evaluate("document.getElementById('benchmark-workspace').hidden"):time.sleep(.1)
    assert not page.evaluate("document.getElementById('benchmark-workspace').hidden")
    assert page.evaluate("document.querySelector('.view.active').dataset.view")=='benchmark'
    assert page.evaluate("document.querySelector('.topbar .badge-synthetic').textContent")=='Historical validation / portfolio'
    assert page.evaluate("document.querySelectorAll('#benchmark-table tr').length")==4
    assert page.evaluate("document.getElementById('benchmark-count').value")=='756'
    page.evaluate("document.getElementById('benchmark-slider').value=0;document.getElementById('benchmark-slider').dispatchEvent(new Event('input'))")
    assert 'Undefined' in page.evaluate("document.getElementById('benchmark-table').textContent")
    page.evaluate("document.getElementById('benchmark-count').value=999999;document.getElementById('benchmark-count').dispatchEvent(new Event('change'))")
    assert page.evaluate("document.getElementById('benchmark-count').value")=='75560'
    assert page.evaluate("document.getElementById('benchmark-slider').value")=='200'
    page.evaluate("document.getElementById('benchmark-metric').value='roc_auc';document.getElementById('benchmark-metric').dispatchEvent(new Event('change'))")
    assert page.evaluate("Chart.getChart(document.getElementById('benchmark-metric-chart')).data.datasets[0].label")=='ROC-AUC'
    page.click('#benchmark-reset')
    assert page.evaluate("document.getElementById('benchmark-count').value")=='756'
    assert page.evaluate("document.getElementById('benchmark-metric').value")=='average_precision'
    page.evaluate("document.getElementById('benchmark-slider').focus()")
    for event in ['keyDown','keyUp']:page.send('Input.dispatchKeyEvent',{'type':event,'key':'ArrowRight','code':'ArrowRight','windowsVirtualKeyCode':39})
    assert page.evaluate("document.getElementById('benchmark-slider').value")=='3'
    assert page.evaluate("document.body.scrollWidth<=innerWidth")
    if width>900:
        assert page.evaluate("document.querySelector('.view-container').scrollHeight<=document.querySelector('.view-container').clientHeight+1")
        target=page.evaluate_json("""JSON.stringify((()=>{const c=Chart.getChart(document.getElementById('benchmark-capacity-chart'));const p=c.getDatasetMeta(0).data[100];const b=c.canvas.getBoundingClientRect();return {x:b.x+p.x,y:b.y+p.y}})())""")
        for event in ['mouseMoved','mousePressed','mouseReleased']:page.send('Input.dispatchMouseEvent',{'type':event,'x':target['x'],'y':target['y'],'button':'left','clickCount':1})
        time.sleep(.2)
        assert page.evaluate("document.getElementById('benchmark-slider').value")=='100'
    page.click('.benchmark-findings summary')
    assert page.evaluate("document.querySelector('.benchmark-findings').open")
    assert 'Paired AP differences' in page.evaluate("document.getElementById('benchmark-limits').textContent")
    page.click('.benchmark-findings summary')
    page.click('.sidenav-list a[data-view="capacity"]')
    page.click('.sidenav-list a[data-view="benchmark"]')
    assert not [item for item in page.console_messages if item[0] in {'exception','error'}]


def test_benchmark_unavailable_evidence_and_retry(browser_env):
    session=CDPSession(browser_env['cdp_port'],'about:blank')
    try:
        session.send('Network.enable')
        session.send('Network.setBlockedURLs',{'urls':['*v1/dashboard/benchmark*']})
        session.navigate(browser_env['static_origin']+'/dashboard.html#/benchmark')
        assert session.evaluate("document.getElementById('benchmark-workspace').hidden")
        assert not session.evaluate("document.getElementById('benchmark-retry').hidden")
        assert 'No comparison' in session.evaluate("document.getElementById('benchmark-status').textContent")
        session.send('Network.setBlockedURLs',{'urls':[]})
        session.click('#benchmark-retry')
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and session.evaluate("document.getElementById('benchmark-workspace').hidden"):time.sleep(.1)
        assert not session.evaluate("document.getElementById('benchmark-workspace').hidden")
    finally:session.close()


@pytest.mark.parametrize('check', ['schema','duplicate','resolveReason','reopenReason','assignment','transitions','noteBounds','appendOnly','seedIsolation','persistence','corrupt','unsupported','quota','unavailable','tamperedHistory','crossBatch','conflict','reset','hostileText','eventLimit','invalidBatch'])
def test_stage21_case_store_contract_in_chrome(page,check):
    page.evaluate((ROOT_DIR/'tests/stage21_store_checks.js').read_text())
    result=page.evaluate(f"(()=>{{try{{stage21StoreChecks.{check}();return 'pass'}}catch(e){{return e.message}}}})()")
    assert result=='pass',result


def _stage21_ready(page,browser_env):
    page.evaluate('localStorage.clear()')
    page.navigate(browser_env['static_origin']+'/dashboard.html?stage21-reset=1#/workflow')
    page.wait_for_queue_data()


def _stage21_select(page,id,value,event='change'):
    page.evaluate(f"document.getElementById({id!r}).value={value!r};document.getElementById({id!r}).dispatchEvent(new Event({event!r}))")


@pytest.mark.parametrize('width,height',[(1440,900),(1280,800),(768,1024),(390,844)])
def test_stage21_full_case_lifecycle_persistence_filters_and_keyboard(page,browser_env,width,height):
    _stage21_ready(page,browser_env)
    page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
    page.evaluate("window.caseRequests=[];window.originalFetch=window.fetch;window.fetch=(...args)=>{caseRequests.push(args);return originalFetch(...args)}")
    page.click('.kanban-card')
    transaction=page.evaluate("document.getElementById('drawer-id').textContent")
    score=page.evaluate("document.getElementById('drawer-score').textContent")
    page.click('#case-open')
    assert page.evaluate("!document.getElementById('case-detail').hidden"), page.evaluate("document.getElementById('case-message').textContent")
    _stage21_select(page,'case-analyst','Fictional Analyst B');_stage21_select(page,'case-priority','Urgent');_stage21_select(page,'case-status','In Review')
    page.click('#case-save')
    _stage21_select(page,'case-note','<img src=x onerror=alert(1)> Fictional demo only')
    page.click('#case-add-note')
    assert page.evaluate("document.querySelectorAll('#case-notes img').length")==0
    assert '<img' in page.evaluate("document.getElementById('case-notes').textContent")
    _stage21_select(page,'case-status','Resolved');page.click('#case-save')
    assert 'reason' in page.evaluate("document.getElementById('case-message').textContent")
    _stage21_select(page,'case-reason','Synthetic review completed');page.click('#case-save')
    assert 'resolved' in page.evaluate("document.getElementById('case-history').textContent")
    _stage21_select(page,'case-status','In Review');_stage21_select(page,'case-reason','Fictional second look');page.click('#case-save')
    assert 'reopened' in page.evaluate("document.getElementById('case-history').textContent")
    assert page.evaluate("document.getElementById('drawer-score').textContent")==score
    assert page.evaluate('caseRequests.length')==0
    page.evaluate("document.getElementById('drawer-close').focus()")
    page.send('Input.dispatchKeyEvent',{'type':'keyDown','key':'Tab','code':'Tab','windowsVirtualKeyCode':9,'modifiers':8})
    assert page.evaluate("document.getElementById('detail-drawer').contains(document.activeElement)")
    page.send('Input.dispatchKeyEvent',{'type':'keyDown','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
    assert not page.evaluate("document.getElementById('detail-drawer').classList.contains('open')")
    _stage21_select(page,'case-analyst-filter','Fictional Analyst B')
    assert page.evaluate("document.querySelectorAll('.kanban-card').length")==1
    assert 'Showing 1 of' in page.evaluate("document.getElementById('global-scope-status').textContent")
    page.click('.sidenav-list a[data-view="queue"]')
    assert page.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length")==1
    page.click('#case-clear-filters')
    page.navigate(browser_env['static_origin']+'/dashboard.html#/workflow');page.wait_for_queue_data()
    page.click(f'.kanban-card[data-id="{transaction}"]')
    assert page.evaluate("document.getElementById('case-analyst').value")=='Fictional Analyst B'
    assert 'Fictional second look' in page.evaluate("document.getElementById('case-history').textContent")
    page.click('#drawer-close')
    _stage21_select(page,'batch-select','202');time.sleep(.8)
    assert 'Opened cases: 0' in page.evaluate("document.getElementById('case-workload').textContent")
    _stage21_select(page,'batch-select','101');time.sleep(.8)
    assert 'Opened cases: 1' in page.evaluate("document.getElementById('case-workload').textContent")
    assert page.evaluate('document.body.scrollWidth<=innerWidth')
    if width>900:assert page.evaluate("document.querySelector('.view-container').scrollHeight<=document.querySelector('.view-container').clientHeight+1")
    page.evaluate('localStorage.clear()')


def test_stage21_reset_confirmation_and_storage_error_ui(page,browser_env):
    _stage21_ready(page,browser_env);page.click('.kanban-card');page.click('#case-open');page.click('#drawer-close')
    for accept in [False,True]:
        page.evaluate("setTimeout(()=>document.getElementById('case-reset').click(),0)");time.sleep(.1)
        page.send('Page.handleJavaScriptDialog',{'accept':accept})
        assert ('Opened cases: 0' if accept else 'Opened cases: 1') in page.evaluate("document.getElementById('case-workload').textContent")
    page.evaluate("Storage.prototype.setItem=function(){throw new DOMException('quota','QuotaExceededError')}")
    page.click('.kanban-card');page.click('#case-open')
    assert 'memory' in page.evaluate("document.getElementById('case-message').textContent")
    assert not page.evaluate("document.getElementById('case-detail').hidden")
    page.click('#drawer-close');page.evaluate('localStorage.clear()')


def test_stage21_search_all_case_filters_and_empty_state(page,browser_env):
    _stage21_ready(page,browser_env);page.click('.kanban-card');page.click('#case-open')
    case_id=page.evaluate("document.getElementById('case-identity').textContent.split(' · ')[0]")
    band=page.evaluate("document.getElementById('drawer-band-pill').textContent")
    _stage21_select(page,'case-priority','High');page.click('#case-save');page.click('#drawer-close')
    _stage21_select(page,'filter-search',case_id,'input')
    _stage21_select(page,'case-status-filter','New');_stage21_select(page,'case-priority-filter','High');_stage21_select(page,'case-risk-filter',band)
    assert page.evaluate("document.querySelectorAll('.kanban-card').length")==1
    _stage21_select(page,'case-status-filter','Resolved')
    assert page.evaluate("document.querySelectorAll('.kanban-card').length")==0
    assert 'No matching transactions' in page.evaluate("document.getElementById('kanban-board').textContent")
    page.click('#reset-filters-btn')
    assert page.evaluate("document.querySelectorAll('.kanban-card').length")==48
    page.evaluate('localStorage.clear()')


@pytest.mark.parametrize('stored',['{broken','{"schemaVersion":999}'])
def test_stage21_corrupt_and_unsupported_storage_ui_recovers_after_explicit_reset(page,browser_env,stored):
    page.evaluate(f"localStorage.clear();localStorage.setItem('fri.demo.cases.v1.101.48',{stored!r})")
    page.navigate(browser_env['static_origin']+'/dashboard.html?invalid-store=1#/workflow');page.wait_for_queue_data()
    page.click('.kanban-card');page.click('#case-open')
    assert page.evaluate("document.getElementById('case-detail').hidden")
    assert 'Reset Demo Cases' in page.evaluate("document.getElementById('case-message').textContent")
    page.click('#drawer-close');page.evaluate("setTimeout(()=>document.getElementById('case-reset').click(),0)");time.sleep(.1);page.send('Page.handleJavaScriptDialog',{'accept':True})
    page.click('.kanban-card');page.click('#case-open')
    assert not page.evaluate("document.getElementById('case-detail').hidden")
    page.evaluate('localStorage.clear()')


def test_stage21_unavailable_storage_remains_usable_and_honest(page,browser_env):
    page.send('Page.addScriptToEvaluateOnNewDocument',{'source':"Storage.prototype.getItem=function(){throw Error('storage denied')}"})
    page.navigate(browser_env['static_origin']+'/dashboard.html?storage-denied=1#/workflow');page.wait_for_queue_data()
    page.click('.kanban-card');page.click('#case-open')
    assert not page.evaluate("document.getElementById('case-detail').hidden")
    assert 'memory' in page.evaluate("document.getElementById('case-message').textContent")


def test_stage21_rejected_multifield_edit_is_atomic_at_activity_limit(page,browser_env):
    _stage21_ready(page,browser_env);page.click('.kanban-card');page.click('#case-open');page.click('#drawer-close')
    page.evaluate("""(()=>{const key='fri.demo.cases.v1.101.48',d=JSON.parse(localStorage.getItem(key)),c=d.cases[0];for(let i=0;i<62;i++){const value=i%2?'Normal':'High';c.history.push({type:'priority',at:c.updatedAt,before:c.priority,after:value,text:''});c.priority=value;d.revision++;}localStorage.setItem(key,JSON.stringify(d))})()""")
    page.navigate(browser_env['static_origin']+'/dashboard.html?activity-limit=1#/workflow');page.wait_for_queue_data()
    before=page.evaluate("localStorage.getItem('fri.demo.cases.v1.101.48')")
    board=page.evaluate("document.getElementById('kanban-board').textContent")
    page.click('.kanban-card');_stage21_select(page,'case-analyst','Fictional Analyst B');_stage21_select(page,'case-status','In Review');page.click('#case-save')
    assert 'limit' in page.evaluate("document.getElementById('case-message').textContent")
    assert page.evaluate("localStorage.getItem('fri.demo.cases.v1.101.48')")==before
    assert page.evaluate("document.getElementById('case-history').children.length")==63
    assert 'New' in page.evaluate("document.getElementById('case-identity').textContent")
    page.click('#drawer-close')
    assert page.evaluate("document.getElementById('kanban-board').textContent")==board
    page.evaluate('localStorage.clear()')


def test_stage21_native_keyboard_creation_keeps_focus_in_case_controls(page,browser_env):
    _stage21_ready(page,browser_env);page.click('.kanban-card');time.sleep(.2)
    page.evaluate("document.getElementById('case-open').focus()")
    page.send('Input.dispatchKeyEvent',{'type':'keyDown','key':'Enter','code':'Enter','windowsVirtualKeyCode':13,'text':'\r','unmodifiedText':'\r'})
    page.send('Input.dispatchKeyEvent',{'type':'keyUp','key':'Enter','code':'Enter','windowsVirtualKeyCode':13})
    assert not page.evaluate("document.getElementById('case-detail').hidden")
    assert page.evaluate('document.activeElement.id')=='case-analyst'
    page.send('Input.dispatchKeyEvent',{'type':'keyDown','key':'Tab','code':'Tab','windowsVirtualKeyCode':9})
    assert page.evaluate("document.getElementById('detail-drawer').contains(document.activeElement)")
    page.evaluate('localStorage.clear()')
