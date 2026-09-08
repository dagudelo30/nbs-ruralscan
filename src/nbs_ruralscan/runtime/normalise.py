"""Normalisation methods for T2/T5 rows -- distinct from `fuzzy.py` (T4 suitability, which
encodes an ecological/agronomic optimum via trapezoidal-family membership functions).
Risk/priority variables (T2, T5) instead ask "how severe is this, relative to what's observed
across the AOI (or a fixed standard)" -- hence percentile clipping / min-max / log / linear
decay rather than a biophysical response curve.

Implements the four `normalisation_method` values actually present in T2_climate_risk.csv:
    percentile_clip   -- clip to [p_low, p_high] percentile of the AOI's own distribution,
                          then min-max scale the clipped range to [0,1]
    min_max            -- plain (x - min) / (max - min) over the AOI
    log_transform       -- log1p(x) then min-max (for heavy-tailed variables like population
                            density, where a few very-high cells shouldn't dominate the scale)
    linear_decay        -- 1 at `suitable_max`, ramping to 0 at `unsuitable_min` (a monotonic
                            decay, e.g. adaptive capacity falling off with distance to a road)

`directionality` (T2 field: `positive_risk` | `negative_risk`) is applied on top: `negative_risk`
means a HIGH raw value REDUCES risk (e.g. adaptive capacity), so the normalised layer is
inverted (1 - x) before entering the risk composite -- every layer that reaches `compose_risk`
in `climate_risk.py` is on the same "higher = more risk" polarity.
"""

from __future__ import annotations

import numpy as np


def percentile_clip(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def min_max(x: np.ndarray) -> np.ndarray:
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return np.zeros_like(x)
    lo, hi = valid.min(), valid.max()
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def log_transform(x: np.ndarray) -> np.ndarray:
    return min_max(np.log1p(np.clip(x, a_min=0, a_max=None)))


def linear_decay(
    x: np.ndarray, suitable_max: float, unsuitable_min: float
) -> np.ndarray:
    if unsuitable_min == suitable_max:
        return np.where(x <= suitable_max, 1.0, 0.0)
    frac = (x - suitable_max) / (unsuitable_min - suitable_max)
    return np.clip(1.0 - frac, 0.0, 1.0)


_DISPATCH = {
    "percentile_clip": lambda x, p: percentile_clip(
        x, p.get("p_low", 2), p.get("p_high", 98)
    ),
    "min_max": lambda x, p: min_max(x),
    "log_transform": lambda x, p: log_transform(x),
    "linear_decay": lambda x, p: linear_decay(
        x, p["suitable_max"], p["unsuitable_min"]
    ),
}


def normalise(
    x: np.ndarray, method: str, params: dict, directionality: str = "positive_risk"
) -> np.ndarray:
    """Dispatch + apply directionality. Raises KeyError for an unimplemented method (loud, not
    a silent pass-through -- an un-normalised 0-100+ raster mixed additively with 0-1 layers
    would silently dominate any composite)."""
    if method not in _DISPATCH:
        raise KeyError(
            f"no normalisation implemented for method={method!r}; known: {sorted(_DISPATCH)}"
        )
    out = _DISPATCH[method](x, params)
    if directionality == "negative_risk":
        out = 1.0 - out
    return out
