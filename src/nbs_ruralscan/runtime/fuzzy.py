"""Fuzzy standardisation — M1 spec Sec 6.2: turn a raw raster into a 0-1 suitability layer per
`T4.relationship_type` + `T4.relationship_params`.

Implements the relationship types actually present in the agroforestry recipe (verified against
`schema/recipes/agroforestry/T4_suitability_mappings.json`):

    trapezoidal        -- abs_min, opt_low, opt_high, abs_max (rise / plateau / fall)
    linear_increasing   -- abs_min, opt_low (0 below abs_min, ramps to 1 at opt_low, 1 above)
    linear_decreasing   -- same 4 params as trapezoidal; "decreasing" describes the semantics
                            (higher raw value = less suitable), the shape math is identical
    ranked_classes      -- either a numeric ordinal ramp (abs_min/opt_low/opt_high/abs_max on
                            the class code itself) or a categorical `class_map` (raw class ->
                            suitability, e.g. ESA WorldCover codes -> agroforestry suitability)
    threshold           -- binary exclusion/inclusion. Numeric `threshold` -> direct compare.
                            String `threshold` (a category label, e.g. an IUCN protection class)
                            means the source layer must already encode the category numerically
                            and a per-dataset code map is needed -- flagged, not guessed.

Pure numpy: operates on `xarray.DataArray.values` or plain ndarrays; callers keep the xarray
wrapper (coords/attrs) on the caller side.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def trapezoidal(
    x: np.ndarray, abs_min: float, opt_low: float, opt_high: float, abs_max: float
) -> np.ndarray:
    """0 below abs_min, ramps to 1 at opt_low, plateau at 1 until opt_high, ramps to 0 at abs_max.

    Defensive against malformed recipe rows (seen in the real data: `tree_canopy_cover` has
    `abs_max < opt_high` by one unit, likely a data-entry slip) -- clamps rather than producing
    NaN/negative-width ramps, and logs a warning so it surfaces for literature-team review
    instead of silently mis-scoring pixels.
    """
    abs_min, opt_low, opt_high, abs_max = (
        float(v) for v in (abs_min, opt_low, opt_high, abs_max)
    )
    if abs_max < opt_high:
        logger.warning(
            "trapezoidal params malformed (abs_max=%s < opt_high=%s) -- clamping abs_max to "
            "opt_high, i.e. a hard cliff instead of a ramp on the high side. Flag this T4 row "
            "for review.",
            abs_max,
            opt_high,
        )
        abs_max = opt_high
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    rising = (x >= abs_min) & (x < opt_low) & (opt_low > abs_min)
    if opt_low > abs_min:
        out[rising] = (x[rising] - abs_min) / (opt_low - abs_min)
    plateau = (x >= opt_low) & (x <= opt_high)
    out[plateau] = 1.0
    falling = (x > opt_high) & (x <= abs_max) & (abs_max > opt_high)
    if abs_max > opt_high:
        out[falling] = 1.0 - (x[falling] - opt_high) / (abs_max - opt_high)
    return np.clip(out, 0.0, 1.0)


def linear_increasing(x: np.ndarray, abs_min: float, opt_low: float) -> np.ndarray:
    """0 below abs_min, ramps linearly to 1 at opt_low, 1 above opt_low."""
    abs_min, opt_low = float(abs_min), float(opt_low)
    x = np.asarray(x, dtype=float)
    if opt_low == abs_min:
        return np.where(x >= opt_low, 1.0, 0.0)
    out = (x - abs_min) / (opt_low - abs_min)
    return np.clip(out, 0.0, 1.0)


def linear_decreasing(
    x: np.ndarray, abs_min: float, opt_low: float, opt_high: float, abs_max: float
) -> np.ndarray:
    """Same shape as `trapezoidal` -- kept as a distinct name because T4 encodes the
    "higher raw value is worse" semantics as a label, not a different formula; the four
    breakpoints already describe a peak-then-decline shape (see `distance_to_road`:
    abs_min=0, opt_low=0, opt_high=3, abs_max=20 -> fully suitable within 3 km, tapering to
    unsuitable by 20 km)."""
    return trapezoidal(x, abs_min, opt_low, opt_high, abs_max)


def ranked_classes(x: np.ndarray, params: dict) -> np.ndarray:
    """Either a categorical lookup (`class_map`: {raw_code: suitability}) or a numeric ordinal
    ramp on the class code itself (same params as `trapezoidal`)."""
    if "class_map" in params:
        class_map = {str(k): float(v) for k, v in params["class_map"].items()}
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape, dtype=float)
        unmapped = np.ones(x.shape, dtype=bool)
        for code, suit in class_map.items():
            # Numeric comparison, not string comparison: real rasters (e.g. GEE pulls) come
            # back as float64, where 10.0 stringifies to "10.0" -- never equal to the class_map
            # key "10". Confirmed against a real run: this bug alone made 100% of land_cover
            # pixels fall through to "unmapped" even though the class_map's codes (10, 20,
            # 40...) exactly match ESA WorldCover's real scheme. np.isclose sidesteps float
            # representation entirely.
            match = np.isclose(x, float(code))
            out[match] = suit
            unmapped &= ~match
        if unmapped.any():
            logger.warning(
                "ranked_classes: %d pixel(s) had a raw class code not in class_map "
                "(defaulted to 0 -- unsuitable). Check the dataset's value scheme against "
                "the recipe's class_map.",
                int(unmapped.sum()),
            )
        return out
    return trapezoidal(
        x, params["abs_min"], params["opt_low"], params["opt_high"], params["abs_max"]
    )


def threshold(x: np.ndarray, params: dict) -> np.ndarray:
    """Binary suitability. Numeric `threshold` -> direct compare. String `threshold` (a
    category label like an IUCN protection class) means the raw layer must already be a
    dataset-specific numeric code -- without that per-dataset mapping this degrades to
    treating the input as already-binary (0/1) and only applies the polarity flip, with a
    loud warning. Build the real code map (e.g. WDPA IUCN category -> int) before trusting
    this for a protected-area exclusion in production."""
    above_is_suitable = bool(params.get("above_is_suitable", True))
    thr = params.get("threshold")
    x = np.asarray(x, dtype=float)
    if isinstance(thr, (int, float)):
        above = x >= float(thr)
    else:
        logger.warning(
            "threshold param %r is a category label, not a number -- no per-dataset code "
            "map implemented yet. Treating input as already-binary (0/1) and applying only "
            "the above_is_suitable polarity. Verify against the real dataset's value scheme "
            "before trusting this exclusion.",
            thr,
        )
        above = x >= 0.5
    return np.where(above == above_is_suitable, 1.0, 0.0)


def _trapezoidal_partial_params(x: np.ndarray, params: dict) -> np.ndarray:
    """Some T4 rows are labelled `trapezoidal` but only carry `abs_min`/`opt_low` (e.g.
    `aridity_index` in the real agroforestry recipe) -- an open-ended "more is always better
    past this point" shape, which is `linear_increasing` in substance. Falls back to that
    rather than raising, but logs it as a recipe-content gap (missing opt_high/abs_max), not a
    silent assumption.
    """
    logger.warning(
        "trapezoidal row missing opt_high/abs_max (params=%s) -- treating as an open-ended "
        "ramp (linear_increasing shape). Flag this T4 row so the recipe author adds the "
        "upper breakpoints, or confirms open-ended is intentional.",
        params,
    )
    return linear_increasing(x, params["abs_min"], params["opt_low"])


_DISPATCH = {
    "trapezoidal": lambda x, p: (
        trapezoidal(x, p["abs_min"], p["opt_low"], p["opt_high"], p["abs_max"])
        if "opt_high" in p and "abs_max" in p
        else _trapezoidal_partial_params(x, p)
    ),
    "linear_increasing": lambda x, p: linear_increasing(x, p["abs_min"], p["opt_low"]),
    "linear_decreasing": lambda x, p: linear_decreasing(
        x, p["abs_min"], p["opt_low"], p["opt_high"], p["abs_max"]
    ),
    "ranked_classes": lambda x, p: ranked_classes(x, p),
    "threshold": lambda x, p: threshold(x, p),
}


def standardise(
    x: np.ndarray, relationship_type: str, relationship_params: dict
) -> np.ndarray:
    """Dispatch to the right membership function per `T4.relationship_type`. Raises KeyError
    (not a silent pass-through) for a relationship_type this module doesn't implement yet --
    an unstandardised 0-1 mismatch would silently corrupt the weighted overlay downstream."""
    if relationship_type not in _DISPATCH:
        raise KeyError(
            f"no fuzzy standardisation implemented for relationship_type={relationship_type!r} "
            f"-- known types: {sorted(_DISPATCH)}"
        )
    return _DISPATCH[relationship_type](x, relationship_params)
