"""Render documentation and smoke-test the README's constrained startup path.

Uses existing research/test dependencies and real Chrome, plus a temporary
Mermaid 11.12.0 browser bundle. Adds no serving/research dependencies. Run after
provisioning the disposable serving environment documented in the Stage 24 report:
    .venv/bin/python docs/stage24_portfolio_evidence/verify_browser.py --serving-python <temporary-env>/bin/python
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import html
import importlib.util
import json
import subprocess
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mistune
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/stage24_portfolio_evidence"
spec = importlib.util.spec_from_file_location("existing_browser_helpers", ROOT / "tests/test_dashboard_routing_browser.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
STYLE = "body{font:16px/1.55 Arial,sans-serif;color:#1f2328;margin:32px auto;padding:0 28px;max-width:1000px}img{max-width:100%}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:7px;border:1px solid #d0d7de}pre{background:#f6f8fa;padding:16px;overflow:auto}code{font-size:13px}.mermaid{margin:20px 0}svg{max-width:100%;height:auto}a{color:#0969da}h1,h2{border-bottom:1px solid #d0d7de}"


class Renderer(mistune.HTMLRenderer):
    def block_code(self, code, info=None):
        if info == "mermaid":
            return '<div class="mermaid">' + html.escape(code) + '</div>'
        return super().block_code(code, info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-python", type=Path, required=True)
    args = parser.parse_args()
    bundle_url = "https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.min.js"
    response = requests.get(bundle_url, timeout=60)
    response.raise_for_status()
    bundle = response.content
    md = mistune.create_markdown(renderer=Renderer(), plugins=["table", "strikethrough"])
    texts = {p: (ROOT / p).read_text() for p in ["README.md", "docs/ARCHITECTURE.md", "docs/PORTFOLIO_WALKTHROUGH.md", "docs/PORTFOLIO_CASE_STUDY.md"]}
    script = '<script src="/__mermaid.js"></script><script>mermaid.initialize({startOnLoad:false,securityLevel:"strict",theme:"default"});window.renderDone=false;mermaid.run().then(()=>window.renderDone=true).catch(e=>window.renderError=String(e));</script>'
    pages = {f"/__preview-{i}.html": ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Stage 24 documentation preview</title><style>' + STYLE + '</style><body>' + md(text) + script + '</body></html>').encode() for i, text in enumerate(texts.values())}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in pages or self.path == "/__mermaid.js":
                body = bundle if self.path == "/__mermaid.js" else pages[self.path]
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript" if self.path == "/__mermaid.js" else "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cdp_port, api_port = helpers._free_port(), helpers._free_port()
    chrome = api = session = None
    results = {"mermaid_version": "11.12.0", "mermaid_bundle_sha256": hashlib.sha256(bundle).hexdigest(), "preview": [], "serving": {}}
    profile_context = tempfile.TemporaryDirectory(prefix="stage24-chrome-")
    try:
        profile = profile_context.name
        chrome = subprocess.Popen([helpers._find_chrome(), "--headless", "--disable-gpu", "--no-sandbox", f"--remote-debugging-port={cdp_port}", f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert helpers._wait_for(f"http://127.0.0.1:{cdp_port}/json/version")
        session = helpers.CDPSession(cdp_port, "about:blank")
        session.send("Emulation.setDeviceMetricsOverride", {"width":1440,"height":900,"deviceScaleFactor":1,"mobile":False})
        for i, name in enumerate(texts):
            session.navigate(f"http://127.0.0.1:{server.server_port}/__preview-{i}.html")
            for _ in range(80):
                if session.evaluate("window.renderDone || window.renderError"): break
                time.sleep(.1)
            state = session.evaluate_json("JSON.stringify({done:window.renderDone,error:window.renderError||null,svg:document.querySelectorAll('.mermaid svg').length,broken:Array.from(document.images).filter(i=>!i.complete||!i.naturalWidth).map(i=>i.getAttribute('src'))})")
            assert state["done"] and not state["error"], (name, state)
            assert not state["broken"], (name, state)
            expected = 1 if i == 0 else 2 if i == 1 else 0
            assert state["svg"] == expected, state
            results["preview"].append({"file": name, **state})
            if i < 2:
                screenshot = session.send("Page.captureScreenshot", {"format":"png","captureBeyondViewport":False})["result"]["data"]
                (OUT / ("readme-preview.png" if i == 0 else "architecture-preview.png")).write_bytes(base64.b64decode(screenshot))
        # Preserve the virtualenv entry point: resolving this symlink would
        # invoke the base interpreter without the freshly installed packages.
        serving_python = args.serving_python.absolute()
        assert serving_python.exists(), "Create the disposable serving environment first"
        api = subprocess.Popen([str(serving_python), "-m", "uvicorn", "src.render_demo:app", "--host", "127.0.0.1", "--port", str(api_port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        origin = f"http://127.0.0.1:{api_port}"
        # A new virtualenv can spend time importing/initialising compiled
        # scientific packages before Uvicorn reaches application startup.
        assert helpers._wait_for(origin + "/health", timeout=60)
        for route in ["/", "/index.html", "/health", "/v1/dashboard/capacity", "/v1/dashboard/benchmark", "/v1/dashboard/monitoring", "/v1/dashboard/explainability"]:
            r = requests.get(origin + route, timeout=10)
            assert r.status_code == 200, (route, r.status_code)
            results["serving"][route] = r.status_code
        session.navigate(origin)
        session.wait_for_queue_data()
        time.sleep(1)
        queue_rows = session.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length")
        assert queue_rows == 24
        screenshot = session.send("Page.captureScreenshot", {"format":"png","captureBeyondViewport":False})["result"]["data"]
        (OUT / "overview-1440x900.png").write_bytes(base64.b64decode(screenshot))
        results["serving"]["synthetic_queue_rows"] = queue_rows
        results["serving"]["runtime"] = "Disposable Python 3.12.15 environment; existing requirements-render.txt only"
        results["rendering_limit"] = "Real Mermaid SVG in local GFM-style Chrome preview; GitHub Markdown API is checked separately. GitHub's final interactive Mermaid renderer is not checked through a published branch."
    finally:
        if session:
            session.close()
        for proc in (api, chrome):
            if proc:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
        server.shutdown()
        profile_context.cleanup()
    (OUT / "browser-documentation-checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
