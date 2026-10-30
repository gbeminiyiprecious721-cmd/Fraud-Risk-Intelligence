"""Fixed read-only Stage 20 aggregate; no training imports or record access."""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()
_AGGREGATE = Path(__file__).resolve().parents[1] / 'ui/benchmark-v1.json'


@router.get('/v1/dashboard/benchmark')
def historical_benchmark():
    if not _AGGREGATE.is_file():
        return JSONResponse(status_code=503, content={'detail':'Benchmark evidence unavailable.'})
    return FileResponse(_AGGREGATE, media_type='application/json', headers={'Cache-Control':'no-store'})
