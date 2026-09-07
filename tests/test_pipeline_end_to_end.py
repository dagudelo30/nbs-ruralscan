"""End-to-end smoke test: the full M1-M6 chain for one real recipe (agroforestry) + AOI (Haiti),
against the real schema on disk. Synthetic data throughout (no GEE credentials in test
environments) -- this test's job is to catch wiring breakage between modules, not to validate
scientific output."""

from __future__ import annotations

from pathlib import Path

from nbs_ruralscan.runtime.characterisation import run_m3
from nbs_ruralscan.runtime.climate_risk import run_m2
from nbs_ruralscan.runtime.handoff import handoff_card
from nbs_ruralscan.runtime.hotspots import run_m4
from nbs_ruralscan.runtime.schema_loader import load_recipe
from nbs_ruralscan.runtime.scorecard import run_m5
from nbs_ruralscan.runtime.suitability import run_m1

SCHEMA_ROOT = Path(__file__).resolve().parent.parent / "schema"
HAITI_BBOX = (-74.5, 18.0, -71.6, 20.1)


def test_full_chain_agroforestry_haiti():
    recipe = load_recipe(SCHEMA_ROOT, "agroforestry")

    m1 = run_m1(recipe.t4, recipe.t1, HAITI_BBOX, resolution_m=10000)
    opp_mask = m1["classes"] >= 3

    m2 = run_m2(recipe.t2, recipe.t3, recipe.t1, HAITI_BBOX, resolution_m=10000)
    assert m2["composite_risk"] is not None

    m3 = run_m3(recipe.t5, recipe.t1, HAITI_BBOX, resolution_m=10000, opp_mask=opp_mask)
    assert len(m3["standardised_priorities"]) > 0

    ttl_weights = {v: "M" for v in m3["standardised_priorities"]}
    m4 = run_m4(m3["standardised_priorities"], ttl_weights, opp_mask, m1["suitability"])
    assert m4["hotspot"].shape == m1["suitability"].shape

    m5 = run_m5(recipe.t6, recipe.t5, recipe.t2, m3["problem_distribution"])
    assert len(m5["scorecard"]) == len(recipe.t6)

    card = handoff_card("agroforestry", m4["ranked_units"], m5["economic_profile_scoping_grade"])
    assert card["current_stage"] == "Scoping (this tool)"
