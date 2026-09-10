"""Tests for schema_loader — run against the real schema/ tree (agroforestry recipe + Haiti)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nbs_ruralscan.runtime.schema_loader import (
    aoi_contexts,
    load_recipe,
    load_recipe_table,
    load_root_table,
    recipe_readiness,
)

SCHEMA_ROOT = Path(__file__).resolve().parent.parent / "schema"


def test_load_root_table_t7_has_haiti():
    t7 = load_root_table(SCHEMA_ROOT, "T7")
    hti = t7[(t7["context_type"] == "admin_country") & (t7["context_id"] == "HTI")]
    assert len(hti) == 1
    assert hti.iloc[0]["context_name"] == "Haiti"


def test_load_root_table_rejects_recipe_table():
    with pytest.raises(KeyError):
        load_root_table(SCHEMA_ROOT, "T4")


def test_load_recipe_table_missing_nbs_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="forest_restoration"):
        load_recipe_table(SCHEMA_ROOT, "forest_restoration", "T4")


def test_load_recipe_agroforestry_loads_all_eight_tables():
    recipe = load_recipe(SCHEMA_ROOT, "agroforestry")
    assert recipe.nbs_id == "agroforestry"
    assert len(recipe.t0) == 1
    assert len(recipe.t4) > 0
    assert len(recipe.t7) > 0


def test_aoi_contexts_resolves_haiti():
    t7 = load_root_table(SCHEMA_ROOT, "T7")
    assert aoi_contexts(t7, "HTI") == {"HTI"}


def test_aoi_contexts_unknown_country_raises():
    t7 = load_root_table(SCHEMA_ROOT, "T7")
    with pytest.raises(KeyError):
        aoi_contexts(t7, "ZZZ")


def test_recipe_readiness_agroforestry_matches_known_evidence_state():
    recipe = load_recipe(SCHEMA_ROOT, "agroforestry")
    readiness = recipe_readiness(recipe)
    assert readiness["nbs_id"] == "agroforestry"
    assert readiness["n_suitability_variables"] == 19
    # Confirmed against T4 n_sources column: 13 rows have n_sources > 0 (slope through
    # tree_canopy_cover + tenure/finance), 6 Stream-B rows are still at 0.
    assert readiness["n_with_literature_evidence"] == 13
    assert readiness["n_placeholder"] == 6


def test_accessibility_travel_time_raw_value_transform():
    """T1.preprocessing_notes says the raw Oxford/MAP band is minutes, to be inverted before
    it reaches T4's linear_increasing(opt_low=1) standardisation. Confirms the transform lands
    exactly where T4 expects it: 0 minutes (in-city) -> 1.0, growing travel time -> decays
    toward 0, never negative or >1."""
    from nbs_ruralscan.runtime.suitability import _RAW_VALUE_TRANSFORMS

    transform = _RAW_VALUE_TRANSFORMS["accessibility_travel_time"]
    minutes = np.array([0, 1, 9, 59, 599])
    accessibility = transform(minutes)
    assert accessibility[0] == 1.0  # exactly in a city
    assert (accessibility[1:] < accessibility[:-1]).all()  # strictly decreasing
    assert (accessibility > 0).all() and (accessibility <= 1.0).all()


def test_soil_ph_raw_value_transform():
    """SoilGrids stores pH x 10 (ISRIC docs). A raw fill value of 61.22 should become 6.1,
    inside T4's expected 5.0-8.5 range."""
    from nbs_ruralscan.runtime.suitability import _RAW_VALUE_TRANSFORMS

    transform = _RAW_VALUE_TRANSFORMS["soil_ph"]
    assert transform(np.array([61.22]))[0] == pytest.approx(6.122)


def test_soil_organic_carbon_raw_value_transform():
    """SoilGrids stores SOC in dg/kg (ISRIC docs: x10 -> g/kg; 10 g/kg = 1%). raw/100 should
    land real Haiti-range values (262-1266 dg/kg observed) inside a plausible 2-13% organic
    carbon range, not the raw 3-4 digit dg/kg scale."""
    from nbs_ruralscan.runtime.suitability import _RAW_VALUE_TRANSFORMS

    transform = _RAW_VALUE_TRANSFORMS["soil_organic_carbon"]
    percent = transform(np.array([262, 1266]))
    assert percent[0] == pytest.approx(2.62)
    assert percent[1] == pytest.approx(12.66)


def test_elevation_to_slope_degrees_known_gradient():
    """A synthetic exact-tilted-plane DEM (rises 100m every 1000m east-west, flat north-south)
    has a known, constant slope everywhere: arctan(100/1000) = 5.71 degrees. Confirmed against
    a real run: feeding raw SRTM elevation (metres, up to 1768m across Haiti) straight into
    T4's slope thresholds (abs_max=24) collapsed suitability to 0 almost everywhere -- this is
    the fix, computing actual terrain slope in degrees before standardisation."""
    import numpy as np
    import xarray as xr

    from nbs_ruralscan.runtime.suitability import _elevation_to_slope_degrees

    resolution_deg = 1000 / 111_320
    lons = np.arange(0, 20 * resolution_deg, resolution_deg)
    lats = np.arange(0, 15 * resolution_deg, resolution_deg)
    elevation = np.zeros((len(lats), len(lons)))
    for j in range(len(lons)):
        elevation[:, j] = j * 100

    da = xr.DataArray(elevation, coords={"y": lats, "x": lons}, dims=("y", "x"))
    slope = _elevation_to_slope_degrees(da)
    assert slope[5:10, 5:15] == pytest.approx(5.71, abs=0.05)


def test_elevation_to_slope_degrees_flat_terrain_is_zero():
    import numpy as np
    import xarray as xr

    from nbs_ruralscan.runtime.suitability import _elevation_to_slope_degrees

    da = xr.DataArray(
        np.full((10, 10), 250.0),
        coords={"y": np.arange(10) * 0.01, "x": np.arange(10) * 0.01},
        dims=("y", "x"),
    )
    slope = _elevation_to_slope_degrees(da)
    assert slope == pytest.approx(0.0, abs=1e-6)


def test_reduce_correlated_never_merges_excluded_categorical_variable():
    """Confirmed as a real risk, not just theoretical: without exclude_from_correlation, a
    categorical/binary variable can numerically out-correlate and out-variance a genuinely
    continuous variable it happens to correlate with, becoming the cluster's "representative"
    and silently dropping the continuous variable from the whole analysis. This constructs
    that exact scenario (a binary variable built as a perfect >0.5 threshold of a continuous
    one) and confirms exclude_from_correlation keeps them both."""
    import numpy as np

    from nbs_ruralscan.runtime.suitability import reduce_correlated

    rng = np.random.default_rng(0)
    continuous = rng.uniform(0, 1, (10, 10))
    categorical = (continuous > 0.5).astype(float)  # perfectly correlated, on purpose
    standardised = {
        "variable_continua": continuous,
        "variable_categorica": categorical,
        "otra_continua": rng.uniform(0, 1, (10, 10)),
    }

    # Without the fix: the continuous variable gets absorbed into the categorical one.
    kept_before, _ = reduce_correlated(standardised, threshold=0.7)
    assert "variable_continua" not in kept_before

    # With the fix: both survive as separate kept variables.
    kept_after, log_after = reduce_correlated(
        standardised, threshold=0.7, exclude_from_correlation={"variable_categorica"}
    )
    assert "variable_continua" in kept_after
    assert "variable_categorica" in kept_after
    assert log_after["variable_categorica"] == "variable_categorica"
