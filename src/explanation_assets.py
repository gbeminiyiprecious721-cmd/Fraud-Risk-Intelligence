"""Strict fixed-resource contracts. No SHAP import, historical access or inference."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import platform
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

FROZEN='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
SEEDS=(101,202,303,404,505)
LIMITATIONS=[
 'Explanations describe the deployed frozen Random Forest, not experimental challengers or fraud causation.',
 'Scores and contributions use the uncalibrated class-1 model-score scale; they are not calibrated fraud probabilities.',
 'TreeSHAP uses recorded training-path weights as an implicit background, not a verified causal or true conditional distribution.',
 'Correlated raw inputs can share attribution ambiguously. Grouping encodings does not remove between-feature dependence.',
 'Global SHAP describes512 fixed validation examples; validation was repeatedly reused during development.',
 'MDI measures fitted impurity reduction and is biased by cardinality/correlation; it is not predictive permutation sensitivity.',
 'Synthetic records contain fixed numeric defaults and unknown categories and may lie outside the natural historical manifold.',
 'Resampling summaries describe this sample only, not confidence, fairness, causal validity or alternative-background robustness.',
 'Anonymised C/V features have no verified business meaning. Contributions support human investigation, never establish a fraud verdict.',
]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def generator_fingerprint(raw_columns):
    from src import api
    return digest({'source':inspect.getsource(api._synthetic_dashboard_row),'raw_columns':list(raw_columns),
                   'constants':{k:getattr(api,k) for k in ['_DASHBOARD_VARY_NUMERIC','_DASHBOARD_PRODUCT_CODES','_DASHBOARD_CARD4','_DASHBOARD_CARD6','_DASHBOARD_EMAIL_DOMAINS']},
                   'seeds':SEEDS,'n':48,'python':'.'.join(platform.python_version_tuple()[:2])})


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)

class Components(Strict):
    numeric:int=Field(ge=0,le=1)
    missingness_indicator:int=Field(ge=0,le=1)
    categorical_onehot:int=Field(ge=0,le=813)

class Importance(Strict):
    name:str
    value:float=Field(ge=0,le=813)
    components:Components

class Population(Strict):
    partition:Literal['validation']
    n:Literal[512]
    eligible:Literal[75560]
    id_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    labels_used:Literal[False]

class Spread(Strict):
    name:str
    quantiles:list[float]=Field(min_length=5,max_length=5)

class Stability(Strict):
    status:Literal['available','insufficient']
    reason:str|None=None
    seed:int|None=None
    replicates:int|None=None
    blocks:int|None=None
    row_weighted:bool|None=None
    overlap_min:float|None=None
    overlap_median:float|None=None
    overlap_max:float|None=None
    rank_spread:list[Spread]|None=None

class Capabilities(Strict):
    mdi:Literal[True]
    global_shap:bool
    local_shap:bool

class GlobalAsset(Strict):
    schema_version:Literal['1.0.0']
    model_sha256:Literal[FROZEN]
    ordered_schema_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    generator_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    groups:dict[str,Components]
    methods:dict[Literal['mdi','shap'],list[Importance]]
    population:Population
    baseline:float|None=Field(default=None,ge=0,le=1)
    mdi_total:float
    capabilities:Capabilities
    stability:Stability
    limitations:list[str]
    unavailable_reason:str|None=None

class Contribution(Strict):
    name:str
    value:float=Field(ge=-813,le=813)

class LocalRecord(Strict):
    transaction_id:str
    baseline:float=Field(ge=0,le=1)
    score:float=Field(ge=0,le=1)
    contributions:list[Contribution]=Field(min_length=10,max_length=10)
    other:float=Field(ge=-813,le=813)
    other_positive:float=Field(ge=0,le=813)
    other_negative:float=Field(ge=-813,le=0)

class LocalAsset(Strict):
    schema_version:Literal['1.0.0']
    model_sha256:Literal[FROZEN]
    ordered_schema_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    generator_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    records:list[LocalRecord]=Field(max_length=240)


def validate_assets(global_data,local_data,artifact):
    """Check identity against the loaded artifact, not self-declared resource IDs."""
    from src.production_pipeline import compute_ordered_schema_fingerprint
    g=GlobalAsset.model_validate(global_data);l=LocalAsset.model_validate(local_data)
    order=compute_ordered_schema_fingerprint(artifact.schema.transformed_feature_names)
    if tuple(artifact.preprocessor.get_feature_names_out())!=artifact.schema.transformed_feature_names:raise ValueError('Schema mismatch')
    gen=generator_fingerprint(artifact.schema.raw_columns)
    if any(x.ordered_schema_sha256!=order or x.generator_sha256!=gen for x in [g,l]):raise ValueError('Identity mismatch')
    if set(g.groups)!=set(artifact.schema.raw_columns) or sum(sum(c.model_dump().values()) for c in g.groups.values())!=813:raise ValueError('Group mismatch')
    expected={name:{'numeric':0,'missingness_indicator':0,'categorical_onehot':0} for name in artifact.schema.raw_columns}
    cats=[name for name in expected if name not in artifact.schema.numeric_columns]
    for name in artifact.schema.transformed_feature_names:
        if name.startswith('numeric_plain__'):raw=name[len('numeric_plain__'):];kind='numeric'
        elif name.startswith('numeric_indicator__missingindicator_'):raw=name[len('numeric_indicator__missingindicator_'):];kind='missingness_indicator'
        elif name.startswith('numeric_indicator__'):raw=name[len('numeric_indicator__'):];kind='numeric'
        elif name.startswith('categorical__'):
            suffix=name[len('categorical__'):];matches=[c for c in cats if suffix.startswith(c+'_')]
            if not matches:raise ValueError('Unknown encoded group')
            raw=max(matches,key=len);kind='categorical_onehot'
        else:raise ValueError('Unknown transform')
        expected[raw][kind]+=1
    if any(g.groups[name].model_dump()!=components for name,components in expected.items()):raise ValueError('Incorrect component lineage')
    if abs(g.mdi_total-1)>1e-9 or g.limitations!=LIMITATIONS:raise ValueError('Invalid method metadata')
    expected_methods={'mdi','shap'} if g.capabilities.global_shap else {'mdi'}
    if set(g.methods)!=expected_methods or g.capabilities.local_shap and not g.capabilities.global_shap:raise ValueError('Invalid capabilities')
    for method,rows in g.methods.items():
        if len(rows)!=20 or len({r.name for r in rows})!=20:raise ValueError('Invalid ranking')
        if [(r.name,r.value) for r in rows]!=sorted([(r.name,r.value) for r in rows],key=lambda v:(-v[1],v[0])):raise ValueError('Unsorted ranking')
        for row in rows:
            if row.name not in g.groups or row.components!=g.groups[row.name]:raise ValueError('Invalid mapping')
        if method=='mdi' and sum(r.value for r in rows)>1+1e-9:raise ValueError('Invalid MDI')
    if g.capabilities.global_shap and g.baseline is None:raise ValueError('Missing baseline')
    s=g.stability
    if s.status=='available':
        if not g.capabilities.global_shap or (s.seed,s.replicates,s.blocks,s.row_weighted)!=(20231023,200,16,True):raise ValueError('Invalid resampling')
        overlap=[s.overlap_min,s.overlap_median,s.overlap_max]
        if any(v is None or not 0<=v<=1 for v in overlap) or overlap!=sorted(overlap):raise ValueError('Invalid stability bounds')
        if not s.rank_spread or [v.name for v in s.rank_spread]!=[v.name for v in g.methods['shap']]:raise ValueError('Invalid stability features')
        for row in s.rank_spread:
            if row.quantiles!=sorted(row.quantiles) or any(not 1<=v<=392 for v in row.quantiles):raise ValueError('Invalid rank spread')
    elif not s.reason:raise ValueError('Missing insufficiency reason')
    records={r.transaction_id:r for r in l.records}
    expected={f'TXN-DEMO-{seed:04d}-{i:04d}' for seed in SEEDS for i in range(48)} if g.capabilities.local_shap else set()
    if set(records)!=expected or len(records)!=len(l.records):raise ValueError('Invalid local inventory')
    for r in records.values():
        if abs(r.baseline-g.baseline)>1e-10 or len({c.name for c in r.contributions})!=10 or any(c.name not in g.groups for c in r.contributions):raise ValueError('Invalid local groups')
        if abs(r.other_positive+r.other_negative-r.other)>1e-8:raise ValueError('Other subtotals mismatch')
        if abs(r.baseline+sum(c.value for c in r.contributions)+r.other-r.score)>1e-8:raise ValueError('Local reconciliation failed')
        if len(json.dumps(r.model_dump()).encode())>16384:raise ValueError('Oversized local output')
    return g,l
