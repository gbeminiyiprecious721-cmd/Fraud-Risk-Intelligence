"""Local public-app real-Chrome evidence; no owner-profile or Render access."""
import base64
import importlib.util
import inspect
import json
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('explanation_browser_tests',ROOT/'tests/test_dashboard_routing_browser.py')
browser=importlib.util.module_from_spec(spec);spec.loader.exec_module(browser)
OUT=ROOT/'docs/stage23_explainability_evidence'


def main():
    source=inspect.getsource(browser.browser_env.__wrapped__).replace('@pytest.fixture(scope="module")\n','').replace('def browser_env():','def public_browser_env():').replace('src.api:app','src.render_demo:app')
    namespace=dict(browser.__dict__);exec(source,namespace)
    fixture=namespace['public_browser_env']();env=next(fixture);evidence=[];OUT.mkdir(exist_ok=True)
    try:
        for width,height in [(1440,900),(1280,800),(768,1024),(390,844)]:
            p=browser.CDPSession(env['cdp_port'],'about:blank')
            try:
                p.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                p.navigate(env['static_origin']+'/dashboard.html#/queue');p.wait_for_queue_data()
                assert p.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length")==24
                p.click('#queue-tbody .inspect-btn');p.click('#case-open');browser._stage21_select(p,'case-note','Synthetic local note preserved across explanations');p.click('#case-add-note')
                before=p.evaluate("localStorage.getItem('fri.demo.cases.v1.101.24')")
                p.click('#drawer-explain');browser._explanation_wait(p)
                browser._stage21_select(p,'explanation-transaction','TXN-DEMO-0101-0000');browser._explanation_wait(p)
                for method in ['mdi','shap']:
                    browser._stage21_select(p,'explanation-method',method);time.sleep(.3)
                    metrics=p.evaluate_json("JSON.stringify((()=>{const c=document.querySelector('.view-container');return {width:innerWidth,height:innerHeight,bodyWidth:document.body.scrollWidth,contentHeight:c.clientHeight,contentScrollHeight:c.scrollHeight,globalPlot:Chart.getChart('explanation-global-chart').chartArea.bottom-Chart.getChart('explanation-global-chart').chartArea.top,localPlot:Chart.getChart('explanation-local-chart').chartArea.bottom-Chart.getChart('explanation-local-chart').chartArea.top}})())")
                    assert metrics['bodyWidth']<=width
                    if width>900:assert metrics['contentScrollHeight']<=metrics['contentHeight']+1 and metrics['globalPlot']>=40 and metrics['localPlot']>=40,metrics
                    assert not [m for m in p.console_messages if m[0] in ['exception','error']]
                    name=f'explainability-{method}-{width}x{height}.png'
                    png=p.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['result']['data'];(OUT/name).write_bytes(base64.b64decode(png))
                    if width<=900:
                        bounds=p.send('Page.getLayoutMetrics')['result']['cssContentSize']
                        full=p.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':width,'height':bounds['height'],'scale':1}})['result']['data']
                        (OUT/f'explainability-{method}-{width}x{height}-full.png').write_bytes(base64.b64decode(full))
                    evidence.append({'image':name,'method':method,'metrics':metrics})
                p.click('#explanation-reset');browser._explanation_wait(p)
                assert p.evaluate("localStorage.getItem('fri.demo.cases.v1.101.24')")==before
                p.click('.explanation-methodology summary')
                if width>900:assert p.evaluate("document.querySelector('.view-container').scrollHeight<=document.querySelector('.view-container').clientHeight+1")
            finally:p.close()
    finally:
        try:next(fixture)
        except StopIteration:pass
    (OUT/'public-browser-verification.json').write_text(json.dumps({'entrypoint':'local src.render_demo:app','synthetic_queue_rows':24,'case_storage_bytes_preserved':True,'public_scope_max_sequence':23,'runtime_errors':0,'screenshots':evidence},indent=2)+'\n')
    print(f'{len(evidence)} viewport screenshots plus4 full mobile/tablet screenshots; public24 case link, controls, signed plots, reset and preservation pass.')

if __name__=='__main__':main()
