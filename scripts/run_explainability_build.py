"""Resource watchdog for approved offline SHAP build (all work, including probes)."""
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/stage23_explainability_evidence';OUT.mkdir(exist_ok=True)

def main():
    started=time.monotonic();peak=0
    with (OUT/'build.json').open('w') as stdout,(OUT/'build.stderr').open('w') as stderr:
        p=subprocess.Popen([sys.executable,str(ROOT/'scripts/build_explainability_assets.py')],cwd=ROOT,stdout=stdout,stderr=stderr)
        try:
            process=psutil.Process(p.pid)
            while p.poll() is None:
                try:peak=max(peak,process.memory_info().rss+sum(c.memory_info().rss for c in process.children(recursive=True)))
                except psutil.NoSuchProcess:pass
                if time.monotonic()-started>600 or peak>4*1024**3:
                    p.terminate();p.wait(timeout=10);raise RuntimeError('Approved10minute/4GiB explanation build budget exceeded')
                time.sleep(.2)
        finally:
            if p.poll() is None:p.kill();p.wait()
        result={'exit_code':p.returncode,'wall_seconds':time.monotonic()-started,'rss_peak_bytes':peak,'limit_seconds':600,'limit_bytes':4*1024**3}
        (OUT/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
        if p.returncode:raise SystemExit(p.returncode)
    print(json.dumps(result))

if __name__=='__main__':main()
