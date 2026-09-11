"""M1 -- Suitability -> Opportunity Space. Orchestrates spec Sec 6.1-6.8, reusing the shipped
`mcda.py` math core (CRITIC/AHP/reconcile/weighted_overlay/sensitivity_perturb/quartile_classify)
and the new `fuzzy.py` standardisation. Nothing here is NbS-specific -- every rule comes from
T4 rows the caller passes in.

Pipeline (mirrors M1_suitability.md Sec 6, function names match its Sec 13 suggestions):

    assemble_variables   -> {variable: DataArray}         (6.1, calls data_loaders per row)
    standardise_stack     -> {variable: ndarray 0-1}        (6.2, fuzzy.standardise per row)
    reduce_correlated     -> (kept_vars, cluster_log)        (6.3, |r|>0.7 pairwise clustering)
    derive_weights         -> weights ndarray + weight_log   (6.4, CRITIC + T4.weight_default AHP)
    weighted_overlay       -> suitability ndarray (0-1)      (6.6, mcda.weighted_overlay)
    sensitivity            -> SensitivityResult              (6.7, mcda.sensitivity_perturb)
    classify_and_fingerprint -> (classes, legend, fingerprint dict)  (6.8, mcda.quartile_classify)

Structural exclusion masking (6.5) is intentionally thin here: the draft-0 agroforestry recipe
doesn't yet carry a dedicated exclusion-layer list (per-recipe Sec 7.3 in the spec), so this
module treats any `threshold` relationship_type row as a hard mask (0 -> excluded) rather than
folding it into the weighted sum -- a defensible default, flagged so it's revisited once a
recipe explicitly separates "structural constraint" (exclusion) from "quality gradient"
(weighted criterion) per row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from nbs_ruralscan.data_loaders import load_variable
from nbs_ruralscan.runtime.fuzzy import standardise
from nbs_ruralscan.runtime.mcda import (
    SensitivityResult,
    ahp_matrix_from_weights,
    ahp_weights,
    critic_weights,
    quartile_classify,
    reconcile_weights,
    sensitivity_perturb,
    weighted_overlay,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolutionAuditRow:
    variable: str
    dataset_id: str
    native_resolution_m: float | None
    analysis_resolution_m: int
    is_synthetic: bool
    recommended_source: str | None


# Known raw-value transforms required BEFORE fuzzy standardisation, per each dataset's own
# T1.preprocessing_notes. Only applied to real (non-synthetic) pulls -- the synthetic fallback
# already draws values in the range T4's relationship_params expect (see _plausible_range),
# so it needs no correction; only the real source needs converting to match.
#
# accessibility_travel_time (nelson_accessibility_2015): T1 says "Travel time minutes; invert
# if framing as accessibility (1 / time)". T4's relationship_type=linear_increasing with
# opt_low=1 confirms the recipe was written expecting an already-inverted 0-1-ish accessibility
# score, not raw minutes -- confirmed as a real mismatch (raw minutes would make nearly every
# pixel saturate at suitability=1.0, since almost anywhere is >=1 minute from a city). Uses
# 1/(1+x) rather than a bare 1/x to avoid a divide-by-zero exactly in-city (x=0), while still
# giving accessibility=1.0 there and decaying smoothly toward 0 as travel time grows -- and it
# lands neatly inside T4's own [0, opt_low=1] ramp.
def _elevation_to_slope_degrees(elevation_da) -> np.ndarray:
    """Real terrain slope (degrees) from a raw elevation grid -- not a per-pixel lookup, uses
    each cell's difference against its neighbours (standard central-difference gradient
    method), so this only lives here, not in `_RAW_VALUE_TRANSFORMS`.

    Pixel spacing is read from the DataArray's own x/y coordinates (already the exact
    analysis grid after `_align_to_grid`) and converted from degrees to metres -- longitude
    degrees shrink with latitude (cos(lat)), so this is computed per-row, not as one constant
    factor for the whole AOI.
    """
    elevation = elevation_da.values.astype(float)
    lats = elevation_da.y.values
    lons = elevation_da.x.values

    dx_deg = float(np.mean(np.diff(lons))) if len(lons) > 1 else 1.0
    dy_deg = float(np.mean(np.diff(lats))) if len(lats) > 1 else 1.0
    metres_per_deg_lat = 111_320.0
    dy_m = abs(dy_deg) * metres_per_deg_lat
    dx_m_per_row = abs(dx_deg) * metres_per_deg_lat * np.cos(np.radians(lats))
    dx_m_per_row = np.clip(dx_m_per_row, 1.0, None)  # guard against a pole-adjacent AOI

    dz_dy, dz_dx = np.gradient(elevation, axis=(0, 1))
    dz_dy = dz_dy / dy_m
    dz_dx = dz_dx / dx_m_per_row[:, None]

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)


_RAW_VALUE_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "accessibility_travel_time": lambda x: 1.0 / (1.0 + x),
    # SoilGrids stores pH x 10 (ISRIC's own documented convention: "Soil pH x 10 in H2O...
    # to convert to pH values divide by 10"). Confirmed against a real run: the raw fill
    # value (61.22) only makes sense as pH once divided by 10 (-> 6.1, squarely inside T4's
    # expected 5.0-8.5 range) -- without this, every soil_ph pixel would blow straight past
    # T4's abs_max=8.5 and saturate at suitability=0 everywhere.
    "soil_ph": lambda x: x / 10.0,
    # SoilGrids stores SOC in dg/kg (ISRIC docs.isric.org: conversion factor x10 -> g/kg).
    # 10 g/kg = 1%, so raw/100 gives percent organic carbon directly. T4's own thresholds
    # (abs_min=0.5, opt_low=0.8) were left unchanged -- they match a real published range
    # (a Kenya agroforestry study: SOC rising 0.8% -> 1.4% under adoption), so the fix here
    # is the missing unit conversion, not the recipe's thresholds.
    "soil_organic_carbon": lambda x: x / 100.0,
    # CGIAR-CSI Global Aridity Index v3.1 (figshare, confirmed 2025-07 release): "Aridity
    # Index values... have been multiplied by a factor of 10,000 to derive and distribute the
    # data as integers". raw/10000 recovers the real P/PET ratio.
    "aridity_index": lambda x: x / 10_000.0,
}


def _plausible_range(relationship_params) -> tuple[float, float]:
    """Derive a raw-value range to draw synthetic placeholder data from, straight from the
    T4 row's own thresholds -- e.g. annual_precipitation's abs_min/abs_max (770-1500mm), not
    an uninformed [0,1]. Falls back to (0,1) for categorical rows (`class_map`) where "raw
    value" isn't numeric, and widens single-sided rows (linear_increasing only has
    abs_min/opt_low) by 3x past opt_low so the synthetic data doesn't cluster at the boundary.
    """
    import json

    if isinstance(relationship_params, str):
        relationship_params = json.loads(relationship_params)
    if "class_map" in relationship_params:
        return (
            0.0,
            1.0,
        )  # categorical -- caller's fuzzy fn does the string lookup itself
    lo = relationship_params.get("abs_min")
    hi = relationship_params.get("abs_max")
    if lo is None:
        return (0.0, 1.0)
    if hi is None:
        opt_low = relationship_params.get("opt_low", lo)
        hi = lo + 3 * max(opt_low - lo, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))


def _target_grid(bbox: tuple[float, float, float, float], resolution_m: int):
    """The single reference grid every variable gets forced onto before stacking -- same
    formula `data_loaders._load_gee_asset` uses to build its GEE request, computed once here
    so every layer (real or synthetic, whatever its native resolution/alignment) lands on
    identical pixel edges. Without this, GEE's returned grid for one dataset can differ by a
    pixel from another's (or from the synthetic fallback's), and `np.stack` in `derive_weights`
    fails with "all input arrays must have the same shape" -- confirmed against a real run.
    """
    import math

    from affine import Affine

    minx, miny, maxx, maxy = bbox
    resolution_deg = resolution_m / 111_320
    width = max(1, math.ceil((maxx - minx) / resolution_deg))
    height = max(1, math.ceil((maxy - miny) / resolution_deg))
    transform = Affine(resolution_deg, 0, minx, 0, -resolution_deg, maxy)
    return transform, (height, width)


def _align_to_grid(
    da, transform, shape, variable: str, categorical: bool, dst_crs: str = "EPSG:4326"
):
    """Reproject/resample one variable's raster onto the shared target grid. Handles xee's
    dimension naming (`lon`/`lat` rather than rioxarray's default `x`/`y`) and missing CRS
    metadata defensively -- neither is guaranteed identical between a GEE pull and the
    synthetic fallback.

    Reprojection can leave a few border pixels as NaN when the source grid's extent doesn't
    exactly match the target's (confirmed against a real run -- rioxarray fills
    outside-of-source-footprint pixels with nodata, defaulting to NaN for float data). A few
    NaN pixels are enough to make CRITIC's correlation matrix (mcda.py) return NaN weights
    entirely, so any NaN introduced here is filled rather than left to propagate.

    `categorical` (True for `ranked_classes`/`threshold` T4 rows, e.g. land_cover) matters for
    *how* it's filled: the layer MEAN is meaningless for class codes (confirmed against a real
    run -- land_cover's ESA WorldCover codes are 10/20/.../100, and filling gaps with their
    mean, ~58.8, isn't a valid code, so every gap-filled pixel then failed `ranked_classes`'s
    class_map lookup even after that lookup's own float-comparison bug was fixed). Categorical
    layers are filled with the MODE (most frequent valid class) instead; continuous layers keep
    the mean.
    """
    if "lon" in da.dims and "lat" in da.dims:
        da = da.rename({"lon": "x", "lat": "y"})
    if da.rio.crs is None:
        da = da.rio.write_crs(dst_crs)
    is_synthetic = bool(da.attrs.get("is_synthetic", False))
    da = da.rio.reproject(dst_crs, shape=shape, transform=transform)
    n_nan = int(da.isnull().sum())
    fill_mask = np.isnan(da.values)  # True donde no habia dato real, antes de rellenar
    if n_nan > 0:
        valid = da.values[~np.isnan(da.values)]
        if valid.size == 0:
            fill_value = 0.0
        elif categorical:
            values, counts = np.unique(valid, return_counts=True)
            fill_value = float(values[np.argmax(counts)])
        else:
            fill_value = float(valid.mean())
        logger.warning(
            "%r: reprojecting onto the shared grid left %d/%d pixel(s) as NaN (source "
            "extent didn't exactly cover the target grid) -- filled with the layer's %s "
            "(%.4g) so downstream weighting stays finite.",
            variable,
            n_nan,
            da.size,
            "mode" if categorical else "mean",
            fill_value,
        )
        da = da.fillna(fill_value)
    da.attrs["fill_mask"] = (
        fill_mask  # True = relleno artificial, no dato real -- para graficar
    )
    da.attrs["is_synthetic"] = is_synthetic
    da.attrs["variable"] = variable
    return da


def load_and_align_one_variable(
    var: str,
    row: pd.Series,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    dataset_id_override: str | None = None,
    country_iso3: str | None = None,
) -> tuple[object | None, ResolutionAuditRow | None]:
    """Pull + align ONE variable, standalone -- the exact same logic `assemble_variables` runs
    per T4 row, extracted so it can be called on its own (one variable, one plot, one sanity
    check) before ever entering the full M1 stack. This is the single source of truth: the
    audit notebook and `assemble_variables` both call this, so nothing can drift between "what
    I checked by eye" and "what the pipeline actually runs".

    Returns (DataArray, audit_row) on success, (None, None) if the dataset_id isn't in T1 or
    alignment fails -- logged either way, never a silent None.
    """
    transform, shape = _target_grid(bbox, resolution_m)
    dataset_id = dataset_id_override or row["dataset_id"]
    matches = t1[t1["dataset_id"] == dataset_id]
    if matches.empty:
        logger.warning("%r: dataset_id=%r not found in T1 -- skipping", var, dataset_id)
        return None, None

    dataset_row = matches.iloc[0].to_dict()
    value_range = _plausible_range(row["relationship_params"])
    da = load_variable(
        var,
        dataset_row,
        bbox,
        resolution_m=resolution_m,
        synthetic_value_range=value_range,
        country_iso3=country_iso3,
    )
    try:
        categorical = row["relationship_type"] in ("ranked_classes", "threshold")
        da = _align_to_grid(da, transform, shape, variable=var, categorical=categorical)
    except Exception as exc:  # noqa: BLE001 -- alignment/reprojection can fail in many GDAL/rioxarray-specific ways; any failure should skip this one variable, not crash the whole M1 run
        logger.warning(
            "%r: could not align to the shared analysis grid (%s: %s) -- skipping this "
            "variable for this run rather than crashing the whole M1 stack.",
            var,
            type(exc).__name__,
            exc,
        )
        return None, None

    # Apply any required raw-value transform BEFORE fuzzy standardisation, per T1's own
    # preprocessing_notes -- only for real pulls; the synthetic fallback already matches
    # T4's expected scale (see _plausible_range), so transforming it too would double-apply.
    if var in _RAW_VALUE_TRANSFORMS and not da.attrs.get("is_synthetic", False):
        transformed_values = _RAW_VALUE_TRANSFORMS[var](da.values)
        da = da.copy(data=transformed_values)
        da.attrs["is_synthetic"] = False
        da.attrs["variable"] = var
        logger.info("%r: applied raw-value transform per T1.preprocessing_notes", var)

    # "slope" is a special case, not a simple elementwise scale factor: T1's srtm_dem_* source
    # is raw ELEVATION (metres), not slope. Confirmed as a real bug against a live run -- T4's
    # thresholds expect slope in DEGREES with abs_max=24, so feeding raw elevation (0-1768m
    # across Haiti) straight through collapsed suitability to 0 at nearly every pixel in the
    # country, silently. Needs neighbouring-pixel differencing (a real terrain-slope
    # calculation), not a per-pixel lambda, so it can't live in _RAW_VALUE_TRANSFORMS -- that
    # dict only ever sees da.values in isolation, not the surrounding grid or pixel spacing.
    if var == "slope" and not da.attrs.get("is_synthetic", False):
        da = da.copy(data=_elevation_to_slope_degrees(da))
        da.attrs["is_synthetic"] = False
        da.attrs["variable"] = var
        logger.info("%r: converted raw elevation (m) to terrain slope (degrees)", var)

    audit_row = ResolutionAuditRow(
        variable=var,
        dataset_id=dataset_id,
        native_resolution_m=dataset_row.get("spatial_resolution_m"),
        analysis_resolution_m=resolution_m,
        is_synthetic=bool(da.attrs.get("is_synthetic", False)),
        recommended_source=da.attrs.get("recommended_source"),
    )
    return da, audit_row


def assemble_variables(
    t4: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    dataset_ids: dict[str, str] | None = None,
    country_iso3: str | None = None,
) -> tuple[dict[str, object], list[ResolutionAuditRow]]:
    """6.1 -- pull every T4 variable's raster for the AOI, one at a time, via
    `load_and_align_one_variable` (the single source of truth also used by the per-variable
    audit notebook). `dataset_ids` optionally overrides T4's own `dataset_id` per variable
    (e.g. with a BIND-resolved id). `country_iso3` is required for any T1 row with
    access_type=reference_table (country-level statistics) -- without it, those variables
    always fall back to synthetic regardless of what's in schema/reference/country_statistics.csv.

    Skips any row whose own T4.justification marks it an M2b Stream-B operational lever
    ("Filter/flag, never summed") -- confirmed against 8 real T4 rows carrying that exact
    marker (accessibility_travel_time, electrification_index, tenure_security,
    conflict_fragility_index, extension_governance, finance_credit_access, market_value_chain,
    labour_availability). These are meant for the not-yet-built M2b (project/investment risk)
    module, not M1's weighted suitability overlay -- summing them into M1 would silently
    contradict what the recipe's own justification field already says. Not dropped from T4,
    not un-loadable (the per-variable audit notebook still calls `load_and_align_one_variable`
    directly per row, unaffected by this M1-level filter), just kept out of the sum here.
    """
    layers = {}
    audit: list[ResolutionAuditRow] = []
    for _, row in t4.iterrows():
        var = row["variable"]
        if "M2b Stream-B" in str(row.get("justification", "")):
            logger.info(
                "%r: T4 marks this an M2b Stream-B lever ('Filter/flag, never summed') -- "
                "excluded from M1's weighted overlay pending M2b's own implementation.",
                var,
            )
            continue
        override = (dataset_ids or {}).get(var)
        da, audit_row = load_and_align_one_variable(
            var,
            row,
            t1,
            bbox,
            resolution_m,
            dataset_id_override=override,
            country_iso3=country_iso3,
        )
        if da is None:
            continue
        layers[var] = da
        audit.append(audit_row)
    return layers, audit


def standardise_stack(
    layers: dict[str, object], t4: pd.DataFrame
) -> dict[str, np.ndarray]:
    """6.2 -- fuzzy-standardise every layer per its T4 relationship_type/params."""
    out = {}
    for var, da in layers.items():
        row = t4[t4["variable"] == var].iloc[0]
        params = row["relationship_params"]
        if isinstance(params, str):
            import json

            params = json.loads(params)
        out[var] = standardise(da.values, row["relationship_type"], params)
    return out


def reduce_correlated(
    standardised: dict[str, np.ndarray],
    threshold: float = 0.7,
    exclude_from_correlation: set[str] | None = None,
) -> tuple[list[str], dict]:
    """6.3 -- pairwise Kendall's tau across standardised bands; cluster |tau| > threshold;
    one representative per cluster, chosen by highest variance (Ani's principle 1 default --
    T4.is_cluster_default expert override not yet wired in, since draft-0 doesn't populate it).
    Returns (kept_variables, cluster_log) where cluster_log maps every variable to its
    representative.

    Kendall's tau, not Pearson -- confirmed as the better fit against a live run: T4's fuzzy
    functions routinely saturate large fractions of the AOI at exactly 1.0 (trapezoidal
    plateaus, linear_increasing ceilings), producing heavy ties that Pearson's linear
    assumption doesn't handle well and can inflate (two variables saturated for unrelated
    reasons can still read as "correlated" in Pearson terms). Kendall is rank-based, built for
    exactly this tied/non-linear situation, and doesn't assume a linear relationship the way
    Pearson does.

    `exclude_from_correlation` (categorical/threshold-type T4 rows, e.g. land_cover,
    protected_area_status) never enter the correlation graph at all -- confirmed as a real
    risk, not just a style concern: the representative-by-variance rule means a
    low-cardinality categorical/binary variable could outscore a genuinely continuous variable
    it happens to correlate with, silently dropping that continuous variable from the
    analysis. Each excluded variable is returned as its own singleton cluster instead, exactly
    as if nothing correlated with it.
    """
    exclude_from_correlation = exclude_from_correlation or set()
    variables = [v for v in standardised if v not in exclude_from_correlation]
    flat = {v: standardised[v].ravel().astype(float) for v in variables}

    # union-find over the |tau| > threshold graph
    parent = {v: v for v in variables}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in combinations(variables, 2):
        xa, xb = flat[a], flat[b]
        if np.std(xa) == 0 or np.std(xb) == 0:
            continue  # constant layer (e.g. synthetic edge case) -- correlation undefined
        tau, _p_value = kendalltau(xa, xb)
        if not np.isnan(tau) and abs(tau) > threshold:
            union(a, b)

    clusters: dict[str, list[str]] = {}
    for v in variables:
        clusters.setdefault(find(v), []).append(v)

    cluster_log = {}
    kept = []
    for members in clusters.values():
        representative = max(members, key=lambda v: np.var(flat[v]))
        kept.append(representative)
        for m in members:
            cluster_log[m] = representative

    for v in exclude_from_correlation:
        if v in standardised:
            kept.append(v)
            cluster_log[v] = v

    return kept, cluster_log


def derive_weights(
    kept_variables: list[str],
    standardised: dict[str, np.ndarray],
    t4: pd.DataFrame,
    alpha: float = 0.4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """6.4 -- CRITIC (objective, computed from the actual standardised rasters) reconciled with
    T4.weight_default (subjective/literature-derived) via alpha=0.4 (framework default: 60%
    objective, 40% expert). No AHP pairwise matrix exists in the draft-0 recipe (only a scalar
    `weight_default` per row), so the subjective input is built by round-tripping weight_default
    through `ahp_matrix_from_weights` -> `ahp_weights` (a perfectly-consistent matrix by
    construction) rather than skipping AHP silently -- keeps the same reconcile_weights contract
    the rest of the framework (M4) expects.
    """
    stack = np.stack([standardised[v] for v in kept_variables], axis=-1)
    n_obs = stack.reshape(-1, stack.shape[-1])

    # Defensive guard (kept here, not in the shipped mcda.py, since it's a caller-side data
    # concern): a zero-variance column -- e.g. a hazard that never occurs anywhere in a small
    # AOI, or a degenerate synthetic layer -- makes CRITIC's correlation matrix divide by zero
    # (NaN weight). Add a tiny deterministic jitter to any such column so CRITIC stays finite;
    # its resulting weight will be small anyway since CRITIC rewards variance.
    zero_var_cols = np.where(n_obs.std(axis=0) == 0)[0]
    if len(zero_var_cols) > 0:
        logger.warning(
            "%d variable(s) have zero variance across the AOI (%s) -- adding negligible "
            "jitter so CRITIC weighting stays finite instead of producing NaN.",
            len(zero_var_cols),
            [kept_variables[i] for i in zero_var_cols],
        )
        rng = np.random.default_rng(0)
        for c in zero_var_cols:
            n_obs[:, c] += rng.normal(0, 1e-9, size=n_obs.shape[0])

    objective = critic_weights(n_obs)

    subjective_raw = (
        t4.set_index("variable")
        .loc[kept_variables, "weight_default"]
        .to_numpy(dtype=float)
    )
    subjective_raw = subjective_raw / subjective_raw.sum()
    ahp_matrix = ahp_matrix_from_weights(subjective_raw)
    subjective = ahp_weights(ahp_matrix)

    final = reconcile_weights(ahp=subjective, critic=objective, alpha=alpha)

    weight_log = pd.DataFrame(
        {
            "variable": kept_variables,
            "weight_default_t4": subjective_raw,
            "ahp_weight": subjective,
            "critic_weight": objective,
            "final_weight_alpha_0.4": final,
        }
    )
    return final, weight_log


def apply_structural_exclusions(
    standardised: dict[str, np.ndarray], t4: pd.DataFrame
) -> np.ndarray | None:
    """6.5 -- any `threshold`-type row (a hard constraint, e.g. protected-area status) becomes
    a boolean exclusion mask (1 = keep, 0 = excluded) rather than entering the weighted sum as
    a graded criterion. Returns None if no threshold-type rows exist for this recipe."""
    threshold_vars = t4[t4["relationship_type"] == "threshold"]["variable"].tolist()
    threshold_vars = [v for v in threshold_vars if v in standardised]
    if not threshold_vars:
        return None
    mask = np.ones_like(next(iter(standardised.values())), dtype=bool)
    for v in threshold_vars:
        mask &= standardised[v] > 0.5
    return mask


def classify_and_fingerprint(
    suitability: np.ndarray,
) -> tuple[np.ndarray, dict[int, str], dict]:
    """6.8 -- classify + a minimal fingerprint (full T5-driven fingerprint with population/
    farm/production breakdowns is M3's job once characterisation.py exists; this is the M1-only
    slice: area by class)."""
    classes, legend = quartile_classify(suitability)
    valid = ~np.isnan(suitability)
    fingerprint = {
        "total_cells": int(valid.sum()),
        "cells_by_class": {legend[c]: int((classes == c).sum()) for c in legend},
        "opportunity_space_share": float(
            ((classes == 3) | (classes == 4)).sum() / max(valid.sum(), 1)
        ),
    }
    return classes, legend, fingerprint


def run_m1(
    t4: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    dataset_ids: dict[str, str] | None = None,
    correlation_threshold: float = 0.7,
    sensitivity_runs: int = 50,
    country_iso3: str | None = None,
) -> dict:
    """End-to-end M1 for one NbS + AOI. Returns every intermediate the notebook wants to show
    (per the "explain every step" requirement) plus the final classified surface.
    `country_iso3` (e.g. "HTI") is required for reference_table-backed variables to resolve --
    without it every one of those falls back to synthetic no matter what's in
    schema/reference/country_statistics.csv.
    """
    layers, resolution_audit = assemble_variables(
        t4, t1, bbox, resolution_m, dataset_ids, country_iso3=country_iso3
    )
    standardised = standardise_stack(layers, t4)
    # Categorical/threshold T4 rows never compete for correlation grouping against continuous
    # ones -- see reduce_correlated's own docstring for why (Pearson assumes continuous data,
    # and the variance-based representative rule could let a categorical variable silently
    # absorb and drop a real continuous one it happens to correlate with).
    categorical_vars = set(
        t4[t4["relationship_type"].isin(["ranked_classes", "threshold"])]["variable"]
    )
    kept_variables, cluster_log = reduce_correlated(
        standardised, correlation_threshold, exclude_from_correlation=categorical_vars
    )

    # 6.5 says threshold-type variables become a hard mask, never a weighted-sum criterion --
    # confirmed as a real bug against a live run: derive_weights/weighted_overlay were given
    # ALL of kept_variables, threshold rows included, so e.g. protected_area_status (T4
    # weight_default=0.526, the single largest weight in the recipe) was being counted BOTH as
    # a normal weighted term AND as the final hard mask -- double-counted, not excluded. Since
    # a threshold variable's standardised value is 1.0 everywhere it isn't excluded, weighting
    # it into the sum adds a flat, non-discriminating bonus across most of the AOI instead of
    # letting genuinely continuous variables (slope, tree_canopy_cover) carry their full share.
    threshold_vars = set(t4[t4["relationship_type"] == "threshold"]["variable"])
    summed_variables = [v for v in kept_variables if v not in threshold_vars]

    weights, weight_log = derive_weights(summed_variables, standardised, t4)
    exclusion_mask = apply_structural_exclusions(standardised, t4)

    # Water isn't a criterion to average against the rest -- it's a hard exclusion, same
    # mechanism as protected_area_status. Confirmed as a real gap against a live run: the
    # weighted overlay was mixing land_cover's correct water penalty (code 80 -> suitability
    # 0.0) with every OTHER variable's NaN-fill-with-mean strategy treating ocean pixels as
    # "typical Haiti land" -- giving the sea a diluted, misleadingly non-zero score instead of
    # being excluded outright. Uses land_cover's raw (pre-standardisation) value, since
    # standardised["land_cover"] is already fuzzy-transformed and the class code is gone by
    # then.
    if "land_cover" in layers:
        water_mask = layers["land_cover"].values != 80
        exclusion_mask = (
            water_mask if exclusion_mask is None else (exclusion_mask & water_mask)
        )

    stack = np.stack([standardised[v] for v in summed_variables], axis=-1)
    suitability = weighted_overlay(stack, weights)
    if exclusion_mask is not None:
        suitability = np.where(exclusion_mask, suitability, 0.0)

    sensitivity: SensitivityResult = sensitivity_perturb(
        stack, weights, n=sensitivity_runs, scale=0.10
    )
    classes, legend, fingerprint = classify_and_fingerprint(suitability)

    return {
        "layers": layers,
        "resolution_audit": resolution_audit,
        "standardised": standardised,
        "kept_variables": summed_variables,
        "excluded_as_hard_mask": sorted(threshold_vars & set(kept_variables)),
        "cluster_log": cluster_log,
        "weights": weights,
        "weight_log": weight_log,
        "exclusion_mask": exclusion_mask,
        "suitability": suitability,
        "sensitivity": sensitivity,
        "classes": classes,
        "legend": legend,
        "fingerprint": fingerprint,
    }
