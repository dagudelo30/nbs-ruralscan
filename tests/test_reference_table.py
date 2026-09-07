"""Tests for the reference_table access_type pattern (country-level statistics)."""

from __future__ import annotations

from pathlib import Path

from nbs_ruralscan.data_loaders import load_variable

REPO_ROOT = Path(__file__).resolve().parent.parent
HAITI_BBOX = (-74.5, 18.0, -71.6, 20.1)


def test_reference_table_missing_row_falls_back_to_synthetic():
    """Haiti has no filled-in value yet (template ships with TODO rows) -- must degrade to
    synthetic, never silently invent a governance/finance score."""
    dataset_row = {"access_type": "reference_table", "dataset_id": "wb_wgi_governance"}
    da = load_variable(
        "extension_governance", dataset_row, HAITI_BBOX, resolution_m=5000, country_iso3="HTI"
    )
    assert da.attrs["is_synthetic"] is True


def test_reference_table_real_value_broadcasts_uniformly(tmp_path):
    """A filled-in row becomes a real, spatially-uniform layer -- same value at every pixel,
    since a country statistic has no within-country spatial variation."""
    table = tmp_path / "country_statistics.csv"
    table.write_text(
        "variable,country_iso3,value,year,source_dataset_id,note\n"
        "extension_governance,SLE,-0.85,2023,wb_wgi_governance,test fixture\n"
    )
    from nbs_ruralscan.data_loaders import _load_reference_table

    da, value = _load_reference_table(
        "extension_governance", "SLE", HAITI_BBOX, 5000 / 111_320, table_path=str(table)
    )
    assert value == -0.85
    assert da.attrs["is_synthetic"] is False
    # Floating-point note: identical values can still yield a non-exact-zero std due to
    # summation order in numpy's variance formula -- use a tolerance, not strict equality.
    assert float(da.values.std()) < 1e-9
    assert float(da.values.flat[0]) == -0.85


def test_reference_table_unknown_country_returns_none(tmp_path):
    table = tmp_path / "country_statistics.csv"
    table.write_text("variable,country_iso3,value,year,source_dataset_id,note\n")
    from nbs_ruralscan.data_loaders import _load_reference_table

    da, value = _load_reference_table(
        "extension_governance", "ZZZ", HAITI_BBOX, 5000 / 111_320, table_path=str(table)
    )
    assert da is None
    assert value is None
