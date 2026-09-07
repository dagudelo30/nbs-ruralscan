from __future__ import annotations

from pathlib import Path

import pandas as pd

from nbs_ruralscan.runtime.climate_risk import relevant_hazards, run_m2
from nbs_ruralscan.runtime.schema_loader import load_recipe

SCHEMA_ROOT = Path(__file__).resolve().parent.parent / "schema"


def test_relevant_hazards_agroforestry():
    recipe = load_recipe(SCHEMA_ROOT, "agroforestry")
    hazards = relevant_hazards(recipe.t3)
    assert "drought" in hazards
    assert "flood" in hazards


def test_run_m2_end_to_end_haiti():
    recipe = load_recipe(SCHEMA_ROOT, "agroforestry")
    bbox = (-74.5, 18.0, -71.6, 20.1)
    result = run_m2(recipe.t2, recipe.t3, recipe.t1, bbox, resolution_m=10000)
    assert result["composite_risk"].min() >= 0.0
    assert result["composite_risk"].max() <= 1.0 + 1e-9
    assert len(result["hazard_layers"]) > 0
