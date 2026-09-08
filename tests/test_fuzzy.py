"""Tests for fuzzy standardisation (M1 spec Sec 6.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nbs_ruralscan.runtime.fuzzy import (
    linear_increasing,
    ranked_classes,
    standardise,
    threshold,
    trapezoidal,
)

SCHEMA_ROOT = Path(__file__).resolve().parent.parent / "schema"


def test_trapezoidal_shape():
    x = np.array([-1, 0, 2.5, 5, 8, 24, 30])
    out = trapezoidal(x, abs_min=0, opt_low=0, opt_high=5, abs_max=24)
    # below abs_min -> 0; in [opt_low, opt_high] -> 1; above abs_max -> 0
    assert out[0] == 0.0
    assert out[3] == 1.0  # x=5, opt_low
    assert out[6] == 0.0  # x=30, past abs_max


def test_trapezoidal_malformed_clamps_not_crashes():
    # real recipe bug: abs_max < opt_high (tree_canopy_cover / mean_annual_temperature)
    out = trapezoidal(
        np.array([10, 22, 30]), abs_min=5, opt_low=20, opt_high=25.5, abs_max=21.3
    )
    assert not np.isnan(out).any()
    assert (out >= 0).all() and (out <= 1).all()


def test_linear_increasing_saturates_above_opt_low():
    out = linear_increasing(np.array([-5, 0, 0.5, 1, 5]), abs_min=0, opt_low=1)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert out[2] == pytest.approx(0.5)


def test_ranked_classes_class_map_lookup():
    x = np.array([10, 40, 999])  # 999 = unmapped code
    out = ranked_classes(x, {"class_map": {"10": 1.0, "40": 0.5}})
    assert out[0] == 1.0
    assert out[1] == 0.5
    assert out[2] == 0.0  # unmapped defaults to unsuitable


def test_threshold_numeric_compare():
    out = threshold(np.array([0, 1, 2]), {"threshold": 1, "above_is_suitable": False})
    # above_is_suitable=False means values >= threshold are UNsuitable
    assert list(out) == [1.0, 0.0, 0.0]


def test_standardise_unknown_type_raises():
    with pytest.raises(KeyError):
        standardise(np.array([1.0]), "not_a_real_type", {})


def test_all_agroforestry_t4_rows_standardise_without_nan():
    """Every relationship_type + relationship_params combination actually present in the
    agroforestry recipe must produce finite output in [0, 1] — this is the real content the
    pipeline will standardise, not a synthetic edge case."""
    t4 = json.loads(
        (
            SCHEMA_ROOT / "recipes" / "agroforestry" / "T4_suitability_mappings.json"
        ).read_text()
    )
    test_x = np.linspace(-5, 50, 20)
    for row in t4:
        out = standardise(test_x, row["relationship_type"], row["relationship_params"])
        assert not np.isnan(out).any(), row["variable"]
        assert out.min() >= 0.0 and out.max() <= 1.0, row["variable"]
