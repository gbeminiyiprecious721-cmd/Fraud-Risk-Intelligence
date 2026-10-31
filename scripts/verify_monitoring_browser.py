"""Real-Chrome public-entrypoint Stage22 screenshots using existing CDP fixtures.

Local processes and a disposable profile only; never touches Render or owner cases.
"""
import base64
import importlib.util
import inspect
import json
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('monitoring_browser_tests',ROOT/'tests/test_dashboard_routing_browser.py')
browser=importlib.util.module_from_spec(spec);spec.loader.exec_module(browser)
OUT=ROOT/'docs/stage22_monitoring_evidence'


def main():
    # Reuse the fixture's exact Chrome lifecycle/CORS ports; substitute only
    # the public app module for independent public-safeguard verification.
    source=inspect.getsource(browser.browser_env.__wrapped__).replace('@pytest.fixture(scope="module")\n','').replace('def browser_env():','def public_browser_env():').replace('src.api:app','src.render_demo:app')
    namespace=dict(browser.__dict__);exec(source,namespace)
    fixture=namespace['public_browser_env']();env=next(fixture);evidence=[]
    OUT.mkdir(exist_ok=True)
    try:
        for width,height in [(1440,900),(1280,800),(768,1024),(390,844)]:
            page=browser.CDPSession(env['cdp_port'],'about:blank')
            try:
                page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                page.navigate(env['static_origin']+'/dashboard.html#/monitoring');page.wait_for_queue_data();browser._monitoring_wait(page)
                before=page.evaluate('JSON.stringify(localStorage)')
                for mode in ['features','scores','quality']:
                    browser._stage21_select(page,'monitoring-mode',mode)
                    browser._stage21_select(page,'monitoring-window','V3')
                    browser._stage21_select(page,'monitoring-feature','dist1')
                    time.sleep(.3)
                    metrics=page.evaluate_json("JSON.stringify((()=>{const c=document.querySelector('.view-container');return {bodyWidth:document.body.scrollWidth,width:innerWidth,height:innerHeight,contentHeight:c.clientHeight,contentScrollHeight:c.scrollHeight,mainChart:Chart.getChart('monitoring-main-chart').height,detailChart:Chart.getChart('monitoring-detail-chart').height}})())")
                    assert metrics['bodyWidth']<=width
                    if width>900:assert metrics['contentScrollHeight']<=metrics['contentHeight']+1,metrics
                    assert metrics['mainChart']>0 and metrics['detailChart']>0
                    assert not [m for m in page.console_messages if m[0] in ['exception','error']]
                    image=page.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['result']['data']
                    name=f'monitoring-{mode}-{width}x{height}.png';(OUT/name).write_bytes(base64.b64decode(image))
                    evidence.append({'image':name,'mode':mode,'metrics':metrics})
                    if width<=900:
                        bounds=page.send('Page.getLayoutMetrics')['result']['cssContentSize']
                        full=page.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':width,'height':bounds['height'],'scale':1}})['result']['data']
                        (OUT/f'monitoring-{mode}-{width}x{height}-full.png').write_bytes(base64.b64decode(full))

                page.click('#monitoring-reset')
                assert page.evaluate('JSON.stringify(localStorage)')==before
                page.click('.sidenav-list a[data-view="queue"]')
                assert page.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length")==24
            finally:page.close()
    finally:
        try:next(fixture)
        except StopIteration:pass
    (OUT/'public-browser-verification.json').write_text(json.dumps({'entrypoint':'local src.render_demo:app','synthetic_queue_rows':24,'case_storage_unchanged':True,'runtime_errors':0,'screenshots':evidence},indent=2)+'\n')
    print(f'{len(evidence)} screenshots; public24 queue, all dimensions, charts, reset and four viewport checks pass.')


if __name__=='__main__':main()
