"""Stage 25 real-Chrome public-app and resource evidence, local processes only.

Run from the repository root with the existing research/test environment:
    .venv/bin/python scripts/verify_stage25_browser.py --serving-python <minimal-env>/bin/python
Uses the existing CDP helper and a disposable Chrome profile. No Render traffic,
owner browser profile, historical rows, model fitting or SHAP computation.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/stage25_release_evidence'
spec = importlib.util.spec_from_file_location('stage25_browser_helpers', ROOT / 'tests/test_dashboard_routing_browser.py')
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-python', type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    api_port, cdp_port = browser._free_port(), browser._free_port()
    origin = f'http://127.0.0.1:{api_port}'
    profile = tempfile.TemporaryDirectory(prefix='fri-stage25-chrome-')
    log = tempfile.TemporaryFile(mode='w+')
    server = chrome = None
    stop = threading.Event()
    rss = []
    samples = []
    result = {'entrypoint':'src.render_demo:app', 'historical_data_access':False, 'viewports':[], 'case_checks':[]}

    def sample_memory():
        proc = psutil.Process(server.pid)
        while not stop.is_set():
            try: rss.append(proc.memory_info().rss)
            except psutil.NoSuchProcess: break
            stop.wait(.01)

    def wait(expression, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if page.evaluate(expression): return
            time.sleep(.1)
        raise AssertionError(expression)

    def navigate(view):
        page.click(f'.sidenav-list a[data-view="{view}"]')
        wait(f"document.querySelector('.view.active').dataset.view === {json.dumps(view)}")
        if view == 'capacity': wait("!document.getElementById('capacity-workspace').hidden")
        if view == 'benchmark': wait("!document.getElementById('benchmark-workspace').hidden")
        if view == 'monitoring': browser._monitoring_wait(page)
        if view == 'explainability': browser._explanation_wait(page)
        time.sleep(.2)

    try:
        started = time.monotonic()
        server = subprocess.Popen([str(args.serving_python.absolute()), '-m', 'uvicorn', 'src.render_demo:app', '--host', '127.0.0.1', '--port', str(api_port)], cwd=ROOT, stdout=log, stderr=log)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        deadline = started + 60
        while time.monotonic() < deadline:
            try:
                health = requests.get(origin+'/health', timeout=1)
                if health.status_code == 200 and health.json().get('artifact_loaded'): break
            except requests.RequestException: pass
            time.sleep(.1)
        else: raise AssertionError('Minimal serving environment did not become healthy')
        ready_seconds = time.monotonic() - started
        version = requests.get(origin+'/version',timeout=5).json()
        assert version['artifact_sha256'] == '2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
        # Measured local request samples; not a production latency/load guarantee.
        for seed in [101,202,303,404,505]:
            t = time.monotonic()
            response = requests.get(origin+f'/v1/dashboard/queue?seed={seed}',timeout=10)
            assert response.status_code == 200 and len(response.json()['transactions']) == 24
            samples.append({'route':'/v1/dashboard/queue','seed':seed,'seconds':round(time.monotonic()-t,6)})
        chrome = subprocess.Popen([browser._find_chrome(),'--headless','--disable-gpu','--no-sandbox',f'--remote-debugging-port={cdp_port}',f'--user-data-dir={profile.name}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        assert browser._wait_for(f'http://127.0.0.1:{cdp_port}/json/version')
        for width,height in [(1440,900),(1280,800),(768,1024),(390,844)]:
            page = browser.CDPSession(cdp_port,'about:blank')
            try:
                page.send('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                # A different query forces a new document after clearing storage;
                # hash-only navigation would retain the old optimistic-edit baseline.
                page.navigate(origin+f'/?qa={width}#/overview');page.wait_for_queue_data()
                page.evaluate('localStorage.clear()')
                page.navigate(origin+'/#/overview');page.wait_for_queue_data()
                assert page.evaluate("document.querySelectorAll('#queue-tbody tr[data-id]').length") == 24
                views = page.evaluate_json("JSON.stringify(Array.from(document.querySelectorAll('.sidenav-list a[data-view]')).map(a=>a.dataset.view))")
                for view in views:
                    navigate(view)
                    metric = page.evaluate_json("JSON.stringify((()=>{const c=document.querySelector('.view-container');return {view:document.querySelector('.view.active').dataset.view,width:innerWidth,height:innerHeight,bodyWidth:document.body.scrollWidth,bodyHeight:document.body.scrollHeight,containerHeight:c.clientHeight,containerScrollHeight:c.scrollHeight,charts:Array.from(document.querySelectorAll('.view.active canvas')).map(c=>({id:c.id,rendered:!!Chart.getChart(c),width:c.width,height:c.height}))}})())")
                    assert metric['bodyWidth'] <= width,metric
                    if width>900:
                        assert metric['bodyHeight'] <= height and metric['containerScrollHeight']<=metric['containerHeight']+1,metric
                    assert all(c['rendered'] and c['width']>0 and c['height']>0 for c in metric['charts']),metric
                    name=f'{view}-{width}x{height}.png'
                    (OUT/name).write_bytes(base64.b64decode(page.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['result']['data']))
                    result['viewports'].append({'image':name,**metric})
                # Existing meaningful controls checks work unchanged on fixed public24 data.
                browser.test_capacity_workspace_controls_charts_and_layout(page,width,height)
                browser.test_benchmark_controls_charts_population_and_responsive_layout(page,width,height)
                navigate('monitoring')
                for mode in ['scores','quality','features']:
                    browser._stage21_select(page,'monitoring-mode',mode)
                    assert page.evaluate("Chart.getChart('monitoring-detail-chart').width")>0
                browser._stage21_select(page,'monitoring-window','V3')
                page.click('#monitoring-reset')
                assert page.evaluate("document.getElementById('monitoring-window').value")=='V2'
                navigate('explainability')
                for method in ['mdi','shap']:
                    browser._stage21_select(page,'explanation-method',method)
                    browser._explanation_wait(page)
                    assert page.evaluate("Chart.getChart('explanation-global-chart').width")>0
                navigate('queue')
                page.send('Network.enable')
                observed=[];receive=page.ws.recv
                def capture(*a,**kw):
                    raw=receive(*a,**kw);event=json.loads(raw)
                    if event.get('method')=='Network.requestWillBeSent':observed.append(event['params']['request'])
                    return raw
                page.ws.recv=capture
                transaction=page.evaluate("document.querySelector('#queue-tbody .inspect-btn').dataset.id")
                page.click('#queue-tbody .inspect-btn');page.click('#case-open')
                note='Stage25 synthetic <img src=x onerror=window.__stage25Xss=1> note'
                for control,value in [('case-analyst','Fictional Analyst A'),('case-priority','High'),('case-status','In Review')]:browser._stage21_select(page,control,value)
                page.click('#case-save')
                browser._stage21_select(page,'case-note',note);page.click('#case-add-note')
                assert page.evaluate("!window.__stage25Xss && !document.querySelector('#case-notes img')")
                assert note in page.evaluate("document.getElementById('case-notes').textContent")
                seed=int(transaction.split('-')[2])
                key=f'fri.demo.cases.v1.{seed}.24'
                stored=page.evaluate(f'localStorage.getItem({json.dumps(key)})')
                assert stored, page.evaluate("JSON.stringify({keys:Object.keys(localStorage),message:document.getElementById('case-message').textContent})")
                page.click('#drawer-close');navigate('workflow');navigate('queue')
                page.click('#case-clear-filters');page.click('#reset-filters-btn')
                assert page.evaluate(f'localStorage.getItem({json.dumps(key)})')==stored
                page.navigate(origin+'/#/queue');page.wait_for_queue_data()
                assert page.evaluate(f'localStorage.getItem({json.dumps(key)})')==stored
                page.click(f'.inspect-btn[data-id="{transaction}"]')
                assert note in page.evaluate("document.getElementById('case-notes').textContent")
                assert not any(note in r.get('postData','') for r in observed)
                assert not any(r['method'] in ['POST','PUT','PATCH','DELETE'] for r in observed)
                page.ws.recv=receive
                page.click('#drawer-close')
                # Real browser scoring and Back to Dashboard on same-origin public app.
                page.navigate(origin+'/index.html');page.click('#score-btn')
                wait("!document.getElementById('result').hidden")
                assert page.evaluate("document.getElementById('error').hidden && document.getElementById('unavailable').hidden")
                score=float(page.evaluate("document.getElementById('score-value').textContent").split(' / ')[0]);assert 0<=score<=1
                page.click('.back-link');page.wait_for_queue_data()
                assert page.evaluate("document.querySelector('.view.active').dataset.view")=='overview'
                assert page.evaluate(f'localStorage.getItem({json.dumps(key)})')==stored
                assert not [e for e in page.console_messages if e[0] in ['error','exception']],page.console_messages
                result['case_checks'].append({'width':width,'height':height,'assignment_priority_status':True,'literal_note_xss_safe':True,'note_not_transmitted':True,'no_case_write_requests':True,'refresh_and_navigation_persistence':True,'filter_clear_preserves_cases':True,'single_scoring_and_back':True})
            finally: page.close()
        stop.set();sampler.join(timeout=2)
        log.flush();log.seek(0)
        startup=[json.loads(line) for line in log.read().splitlines() if line.startswith('{') and '"event": "startup"' in line]
        result['performance']={'environment':'macOS, disposable minimal serving venv; not Render runtime or load benchmark','startup_health_ready_seconds':round(ready_seconds,6),'artifact_startup_event':startup[0] if startup else None,'rss_sample_interval_seconds':.01,'process_peak_rss_bytes':max(rss),'public_queue_request_samples':samples,'includes_chrome_memory':False,'frozen_sha256':version['artifact_sha256']}
        result['runtime_errors']=0
        (OUT/'public-browser-qa.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'view_checks':len(result['viewports']),'viewport_workflow_checks':len(result['case_checks']),'runtime_errors':0,'performance':result['performance']},indent=2))
    finally:
        stop.set()
        for proc in [chrome,server]:
            if proc:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        profile.cleanup();log.close()


if __name__=='__main__':main()
