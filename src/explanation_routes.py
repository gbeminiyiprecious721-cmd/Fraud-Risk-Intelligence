"""Fixed read-only explanation assets; no SHAP, offline data or model inference."""
import json
from pathlib import Path
from fastapi import APIRouter,Request,Query
from fastapi.responses import JSONResponse
from src.explanation_assets import FROZEN,SEEDS,validate_assets,generator_fingerprint
from src.production_pipeline import compute_ordered_schema_fingerprint
router=APIRouter()
ROOT=Path(__file__).resolve().parents[1]
GLOBAL=ROOT/'explainability_assets/global-v1.json'
LOCAL=ROOT/'explainability_assets/local-v1.json'


def resources(request):
    artifact=getattr(request.app.state,'artifact',None);manifest=getattr(request.app.state,'manifest',None)
    if artifact is None or not manifest or manifest.get('artifact_sha256')!=FROZEN:raise ValueError('Artifact unavailable/mismatched')
    identity=(str(GLOBAL),str(LOCAL),manifest['artifact_sha256'],compute_ordered_schema_fingerprint(artifact.schema.transformed_feature_names),generator_fingerprint(artifact.schema.raw_columns))
    cache=getattr(request.app.state,'validated_explanations',None)
    if cache and cache[0]==identity:return cache[1:]
    if GLOBAL.stat().st_size>131072 or LOCAL.stat().st_size>1048576:raise ValueError('Oversized asset')
    g,l=validate_assets(json.loads(GLOBAL.read_text()),json.loads(LOCAL.read_text()),artifact)
    public=g.model_dump(exclude={'groups'});records={r.transaction_id:r.model_dump() for r in l.records}
    if len(json.dumps(public,separators=(',',':'),allow_nan=False).encode())>16384:raise ValueError('Oversized global output')
    request.app.state.validated_explanations=(identity,public,records)
    return public,records


def unavailable():return JSONResponse(status_code=503,content={'detail':'Validated explanation resources unavailable.'})


@router.get('/v1/dashboard/explainability')
def global_explanation(request:Request):
    try:public,_=resources(request)
    except (OSError,ValueError,TypeError,KeyError,IndexError):return unavailable()
    return JSONResponse(public,headers={'Cache-Control':'no-store'})


@router.get('/v1/dashboard/explanation')
def local_explanation(request:Request,transaction_id:str=Query(max_length=24,pattern=r'^TXN-DEMO-\d{4}-\d{4}$')):
    parts=transaction_id.split('-');seed=int(parts[2]);sequence=int(parts[3])
    limit=getattr(request.app.state,'explanation_max_sequence',48)
    if seed not in SEEDS or not 0<=sequence<limit:return JSONResponse(status_code=404,content={'detail':'Explanation unavailable for this synthetic identifier.'})
    try:public,records=resources(request)
    except (OSError,ValueError,TypeError,KeyError,IndexError):return unavailable()
    if not public['capabilities']['local_shap']:return unavailable()
    record=records.get(transaction_id)
    if record is None:return JSONResponse(status_code=404,content={'detail':'Explanation unavailable for this synthetic identifier.'})
    return JSONResponse(record,headers={'Cache-Control':'no-store'})
