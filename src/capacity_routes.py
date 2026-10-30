"""Fixed, read-only historical aggregate; no raw records, inference or user paths."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()
_AGGREGATE = Path(__file__).resolve().parent.parent / 'ui/capacity-v1.json'


@router.get('/v1/dashboard/capacity')
def historical_capacity():
    if not _AGGREGATE.is_file():
        return JSONResponse(status_code=503, content={'detail': 'Historical capacity aggregate unavailable.'})
    return FileResponse(_AGGREGATE, media_type='application/json', headers={'Cache-Control': 'no-store'})
