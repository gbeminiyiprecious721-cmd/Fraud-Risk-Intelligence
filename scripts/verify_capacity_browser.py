"""Capture Stage 19 real-Chrome screenshots/measurements using existing CDP helpers."""
import base64
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('dashboard_browser_tests', ROOT / 'tests/test_dashboard_routing_browser.py')
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)
OUT = ROOT / 'docs/stage19_capacity_evidence'


def main():
    OUT.mkdir(exist_ok=True)
    fixture = browser.browser_env.__wrapped__()
    env = next(fixture)
    rows = []
    try:
        for width, height in [(1440, 900), (1280, 800), (768, 1024), (390, 844)]:
            page = browser.CDPSession(env['cdp_port'], 'about:blank')
            try:
                page.send('Emulation.setDeviceMetricsOverride', {'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                page.navigate(env['static_origin'] + '/dashboard.html#/capacity')
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and page.evaluate("document.getElementById('capacity-workspace').hidden"):
                    time.sleep(.1)
                assert not page.evaluate("document.getElementById('capacity-workspace').hidden")
                time.sleep(.4)
                metrics = page.evaluate_json("""JSON.stringify(Object.fromEntries(['body','.view-container','.capacity-view','#capacity-workspace','.capacity-controls','.capacity-kpis','.capacity-charts','.capacity-decision'].map(s=>{const e=document.querySelector(s),b=e.getBoundingClientRect();return [s,{height:e.clientHeight,scrollHeight:e.scrollHeight,width:e.clientWidth,scrollWidth:e.scrollWidth,top:b.top,bottom:b.bottom}]})))""")
                rows.append({'width':width,'height':height,'metrics':metrics,'browser_exceptions':[m for m in page.console_messages if m[0] in {'exception','error'}], 'plot_heights':page.evaluate_json("JSON.stringify(['capacity-recall-chart','capacity-precision-chart'].map(id=>{const c=Chart.getChart(document.getElementById(id));return c.chartArea.bottom-c.chartArea.top}))")})
                shot = page.send('Page.captureScreenshot', {'format':'png','captureBeyondViewport':False})['result']['data']
                (OUT / f'capacity-{width}x{height}.png').write_bytes(base64.b64decode(shot))
            finally:
                page.close()
        (OUT / 'viewport-metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows, indent=2))
    finally:
        try: next(fixture)
        except StopIteration: pass


if __name__ == '__main__':
    main()
