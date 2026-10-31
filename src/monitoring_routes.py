"""Fixed, read-only, schema-validated aggregate. No raw data or model imports."""
import json
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.drift_intelligence import validate_aggregate

router=APIRouter()
AGGREGATE=Path(__file__).resolve().parents[1]/'ui/monitoring-v1.json'

@router.get('/v1/dashboard/monitoring')
def historical_monitoring():
    try:
        if AGGREGATE.stat().st_size>262144:raise ValueError('Oversized evidence')
        payload=json.loads(AGGREGATE.read_text())
        validate_aggregate(payload)
    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError):
        return JSONResponse(status_code=503,content={'detail':'Historical monitoring evidence unavailable.'})
    return JSONResponse(payload,headers={'Cache-Control':'no-store'})
