"""Stage 21 real-Chrome UI evidence, with synthetic browser-local cases only."""
import base64
import json
import time
from pathlib import Path
from verify_benchmark_browser import browser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/stage21_case_evidence'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fixture=browser.browser_env.__wrapped__();env=next(fixture);evidence=[]
    try:
        for width,height in [(1440,900),(1280,800),(768,1024),(390,844)]:
            page=browser.CDPSession(env['cdp_port'],'about:blank')
            try:
                page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                page.navigate(env['static_origin']+'/dashboard.html#/workflow');page.wait_for_queue_data()
                page.evaluate('localStorage.clear()')
                page.navigate(env['static_origin']+f'/dashboard.html?capture={width}#/workflow');page.wait_for_queue_data()
                page.click('.kanban-card');transaction=page.evaluate("document.getElementById('drawer-id').textContent");page.click('#case-open')
                for id,value in [('case-analyst','Fictional Analyst A'),('case-priority','High'),('case-status','In Review')]:page.evaluate(f'document.getElementById({id!r}).value={value!r}')
                page.click('#case-save');page.evaluate("document.getElementById('case-note').value='Fictional demo: requested a second review of synthetic transaction context.'");page.click('#case-add-note');page.click('#drawer-close')
                for view in ['workflow','queue','case-detail','case-history']:
                    if view=='case-detail':page.click(f'#queue-tbody .inspect-btn[data-id="{transaction}"]')
                    elif view=='case-history':page.evaluate("document.getElementById('case-notes').scrollIntoView({block:'start'})")
                    else:page.click(f'.sidenav-list a[data-view="{view}"]')
                    time.sleep(.3)
                    metrics=page.evaluate_json("""JSON.stringify((()=>{const c=document.querySelector('.view-container');return {bodyWidth:document.body.scrollWidth,viewportWidth:innerWidth,contentHeight:c.clientHeight,contentScrollHeight:c.scrollHeight,drawerOpen:document.getElementById('detail-drawer').classList.contains('open'),errors:[]}})())""")
                    assert metrics['bodyWidth']<=width
                    if width>900 and view not in ['case-detail','case-history']:assert metrics['contentScrollHeight']<=metrics['contentHeight']+1,metrics
                    errors=[x for x in page.console_messages if x[0] in {'exception','error'}];assert not errors,errors
                    evidence.append({'view':view,'width':width,'height':height,'metrics':metrics})
                    shot=page.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['result']['data']
                    (OUT/f'{view}-{width}x{height}.png').write_bytes(base64.b64decode(shot))
            finally:page.close()
    finally:
        try:next(fixture)
        except StopIteration:pass
        (OUT/'viewport-metrics.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'{len(evidence)} real-Chrome screenshots; no desktop outer overflow or browser errors')

if __name__=='__main__':main()
