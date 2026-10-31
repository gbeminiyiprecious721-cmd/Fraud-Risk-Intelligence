"""Historical top-K aggregates. Pure calculations; no model fitting or dataset access."""
from __future__ import annotations

from numbers import Integral

import numpy as np


def build_capacity_points(labels, scores, transaction_ids, capacities) -> list[dict]:
    """Rank by descending score, ascending ID; return exact checkpoint counts.

    Undefined precision (K=0) and recall (no positives) use None, never zero.
    Random review is a uniform without-replacement expectation, not a guarantee.
    """
    y, score, ids = np.asarray(labels), np.asarray(scores), np.asarray(transaction_ids)
    if any(a.ndim != 1 for a in (y, score, ids)) or not len(y) or len(y) != len(score) or len(y) != len(ids):
        raise ValueError('Nonempty, aligned one-dimensional arrays required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Binary labels and finite scores within [0,1] required')
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique integer transaction IDs required')
    ks = list(capacities)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0 or k > len(y) for k in ks):
        raise ValueError('Capacity must be an integer between zero and population size')
    if ks != sorted(set(ks)) or not ks:
        raise ValueError('Capacities must be nonempty, unique and increasing')
    cumulative = np.r_[0, np.cumsum(y[np.lexsort((ids, -score))], dtype=np.int64)]
    n, positives = len(y), int(y.sum())
    points = []
    for k in ks:
        k = int(k)
        tp = int(cumulative[k])
        fp, fn = k - tp, positives - tp
        points.append(dict(k=k, capacity_fraction=k / n, tp=tp, fp=fp, fn=fn,
                           tn=n - positives - fp, precision=tp / k if k else None,
                           recall=tp / positives if positives else None,
                           random_tp_expected=k * positives / n,
                           random_precision_expected=positives / n if k else None,
                           random_recall_expected=k / n if positives else None))
    return points
