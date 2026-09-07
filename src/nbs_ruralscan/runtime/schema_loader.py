"""Load T0-T7 schema tables for a given NbS recipe and AOI - the runtime side of the schema
contract described in `schema/spec.md`.

Cross-NbS tables (T1 datasets, T2 climate-risk variables, T5 opportunity-space variables, T7
geographic contexts) live at the schema root and are shared across every recipe. Per-NbS tables
(T0 registry, T3 hazard x farming, T4 suitability mappings, T6 scorecard) live under
`schema/recipes/<nbs_id>/`. Nothing here hardcodes an NbS or a country: adding one is a new
recipe folder / a new T7 row, never a code change (schema.html, "How they connect").

Stdlib + pandas only. No GEE, no xarray - this module only ever touches CSV/JSON config, never
a pixel (the config-plane-vs-data-plane split: T0-T7 stores *how to find and reason about* a
dataset, not the dataset itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ROOT_TABLES: dict[str, str] = {
    "T1": "T1_data_registry",
    "T2": "T2_climate_risk",
    "T5": "T5_opportunity_space",
    "T7": "T7_geographic_context",
}
_RECIPE_TABLES: dict[str, str] = {
    "T0": "T0_nbs_registry",
    "T3": "T3_nbs_hazard_farming",
    "T4": "T4_suitability_mappings",
    "T6": "T6_nbs_scorecard",
}


def load_root_table(schema_root: str | Path, table: str) -> pd.DataFrame:
    """Load one cross-NbS table (T1 / T2 / T5 / T7) from the schema root."""
    if table not in _ROOT_TABLES:
        raise KeyError(f"{table!r} is not a root table; expected one of {list(_ROOT_TABLES)}")
    path = Path(schema_root) / f"{_ROOT_TABLES[table]}.csv"
    return pd.read_csv(path)


def load_recipe_table(schema_root: str | Path, nbs_id: str, table: str) -> pd.DataFrame:
    """Load one per-NbS table (T0 / T3 / T4 / T6) from `schema/recipes/<nbs_id>/`.

    Raises FileNotFoundError with a clear message if the recipe hasn't been authored yet
    (e.g. forest_restoration only has T0 today) rather than a bare pandas error.
    """
    if table not in _RECIPE_TABLES:
        raise KeyError(f"{table!r} is not a recipe table; expected one of {list(_RECIPE_TABLES)}")
    path = Path(schema_root) / "recipes" / nbs_id / f"{_RECIPE_TABLES[table]}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no {table} table for recipe {nbs_id!r} at {path} "
            "- recipe not yet authored for this NbS?"
        )
    return pd.read_csv(path)


@dataclass
class Recipe:
    """All eight tables resolved for one NbS - what M0 hands to M1-M6."""

    nbs_id: str
    t0: pd.DataFrame
    t1: pd.DataFrame
    t2: pd.DataFrame
    t3: pd.DataFrame
    t4: pd.DataFrame
    t5: pd.DataFrame
    t6: pd.DataFrame
    t7: pd.DataFrame


def load_recipe(schema_root: str | Path, nbs_id: str) -> Recipe:
    """Load every table for `nbs_id`. Root tables carry every NbS's rows (they're the shared
    framework layer, see DEDUP_NOTES.md); callers filter by `variable`/`dataset_id` as needed
    rather than expecting root tables to be pre-filtered to one NbS.
    """
    return Recipe(
        nbs_id=nbs_id,
        t0=load_recipe_table(schema_root, nbs_id, "T0"),
        t1=load_root_table(schema_root, "T1"),
        t2=load_root_table(schema_root, "T2"),
        t3=load_recipe_table(schema_root, nbs_id, "T3"),
        t4=load_recipe_table(schema_root, nbs_id, "T4"),
        t5=load_root_table(schema_root, "T5"),
        t6=load_recipe_table(schema_root, nbs_id, "T6"),
        t7=load_root_table(schema_root, "T7"),
    )


def aoi_contexts(t7: pd.DataFrame, admin_country: str) -> set[str]:
    """The T7 context_ids that apply to a single-country AOI.

    Today this returns just the `admin_country` context id itself. AEZ / farming_system /
    hydrobasin overlap requires an actual geometry intersection against the AOI polygon, which
    is a spatial join done in M1's variable-assembly step (xarray/geopandas), not a config
    lookup - this function only resolves the *config* side (M0 spec Sec 6.1/6.3), same split as
    binding.resolve_binding's `aoi_contexts` parameter, which this feeds.
    """
    match = t7[(t7["context_type"] == "admin_country") & (t7["context_id"] == admin_country)]
    if match.empty:
        raise KeyError(
            f"{admin_country!r} not found in T7 as admin_country - "
            "add a row before running M0 for this AOI"
        )
    return {admin_country}


def recipe_readiness(recipe: Recipe) -> dict:
    """M0 spec Sec 6.2/6.3 - variable counts + literature-evidence readiness for the Setup tab
    / notebook sanity check. Not the full data-readiness check (Sec 6.3, which needs T1
    access-route info); this is the recipe-content half of it.

    `n_sources` (not `evidence_ids`, which is largely unpopulated in draft-0/1 recipes) is the
    live signal for "this row is backed by literature, not a placeholder" - confirmed against
    the agroforestry recipe, where `evidence_ids` is empty across the board but `n_sources`
    ranges 0-16 per row.
    """
    t4 = recipe.t4
    if "n_sources" in t4.columns:
        n_sources = pd.to_numeric(t4["n_sources"], errors="coerce").fillna(0)
        n_with_evidence = int((n_sources > 0).sum())
    else:
        n_with_evidence = 0
    n_priority = (
        int((recipe.t5["mcda_role"] == "priority").sum())
        if "mcda_role" in recipe.t5.columns
        else None
    )
    return {
        "nbs_id": recipe.nbs_id,
        "n_suitability_variables": len(t4),
        "n_with_literature_evidence": n_with_evidence,
        "n_placeholder": len(t4) - n_with_evidence,
        "n_priority_variables_root": n_priority,
    }
