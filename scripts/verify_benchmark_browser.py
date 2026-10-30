"""Capture Stage 20 dashboard evidence with real Chrome and existing CDP helpers."""
import base64
import importlib.util
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dashboard_browser_tests',ROOT/'tests/test_dashboard_routing_browser.py')
browser=importlib.util.module_from_spec(spec);spec.loader.exec_module(browser)
OUT=ROOT/'docs/stage20_challenger_evidence/screenshots'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fixture=browser.browser_env.__wrapped__();env=next(fixture);evidence=[]
    try:
        for width,height in [(1440,900),(1280,800),(768,1024),(390,844)]:
            views=['overview','queue','analytics','workflow','insights','responsible-ai','capacity','benchmark'] if width>900 else ['benchmark']
            page=browser.CDPSession(env['cdp_port'],'about:blank')
            try:
                page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                page.navigate(env['static_origin']+'/dashboard.html')
                page.wait_for_queue_data()
                for view in views:
                    page.click(f'.sidenav-list a[data-view="{view}"]')
                    if view in ['capacity','benchmark']:
                        deadline=time.monotonic()+10
                        while time.monotonic()<deadline and page.evaluate(f"document.getElementById('{view}-workspace').hidden"):time.sleep(.1)
                        assert not page.evaluate(f"document.getElementById('{view}-workspace').hidden")
                    time.sleep(.3)
                    metrics=page.evaluate_json("""JSON.stringify((()=>{const c=document.querySelector('.view-container');return {bodyWidth:document.body.scrollWidth,viewportWidth:innerWidth,bodyHeight:document.body.scrollHeight,viewportHeight:innerHeight,contentHeight:c.clientHeight,contentScrollHeight:c.scrollHeight,activeView:document.querySelector('.view.active').dataset.view}})())""")
                    assert metrics['bodyWidth']<=width
                    if width>900:assert metrics['contentScrollHeight']<=metrics['contentHeight']+1,metrics
                    exceptions=[m for m in page.console_messages if m[0] in {'exception','error'}]
                    assert not exceptions,exceptions
                    evidence.append({'width':width,'height':height,'view':view,'metrics':metrics,'browser_exceptions':exceptions})
                    image=page.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['result']['data']
                    (OUT/f'{view}-{width}x{height}.png').write_bytes(base64.b64decode(image))
            finally:page.close()
    finally:
        try:next(fixture)
        except StopIteration:pass
        (OUT.parent/'viewport-metrics.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'{len(evidence)} screenshots; no desktop outer content overflow or browser errors')

if __name__=='__main__':main()
