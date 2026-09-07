"""M2 -- Rural Climate Risk (Mode A: Hazard x Exposure). Spec: M2_climate_risk.md.

Per the spec's own "recommended v0 scope": Mode A only; Mode B's vulnerability composition
(needs sensitivity + adaptive_capacity layers combined -- a Brayden methodology call, per the
module's open questions) is deliberately deferred behind a NotImplementedError rather than
guessed at.

Answers: "where are rural populations and production systems exposed to climate hazards this
NbS can plausibly mitigate?" -- kept separate from M1 suitability (where the NbS could work
biophysically) per the spec's explicit design.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from nbs_ruralscan.data_loaders import load_variable
from nbs_ruralscan.runtime.normalise import normalise

logger = logging.getLogger(__name__)


def relevant_hazards(t3: pd.DataFrame) -> list[str]:
    """Sec 6.1 -- hazards this NbS mitigates for rural livelihoods (risk_role in
    {livelihood_mitigation, both}). A hazard with no T3 row for this NbS is NOT included --
    recipe authors must populate T3 explicitly, per the spec."""
    mitigating = t3[t3["risk_role"].isin(["livelihood_mitigation", "both"])]
    return sorted(mitigating["hazard_type"].dropna().unique().tolist())


def _load_and_normalise_t2_rows(
    rows: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
) -> dict[str, np.ndarray]:
    out = {}
    for _, row in rows.iterrows():
        matches = t1[t1["dataset_id"] == row["dataset_id"]]
        if matches.empty:
            logger.warning("T2 variable %r: dataset_id=%r not in T1 -- skipping", row["variable"], row["dataset_id"])
            continue
        dataset_row = matches.iloc[0].to_dict()
        da = load_variable(row["variable"], dataset_row, bbox, resolution_m=resolution_m)
        params = row["normalisation_params"]
        if isinstance(params, str):
            params = json.loads(params) if params else {}
        out[row["variable"]] = normalise(
            da.values, row["normalisation_method"], params or {}, row.get("directionality", "positive_risk")
        )
    return out


def assemble_hazards(
    t2: pd.DataFrame,
    t1: pd.DataFrame,
    hazards: list[str],
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    scenario_type: str = "baseline",
) -> dict[str, np.ndarray]:
    """Sec 6.2 -- one normalised (0-1) layer per hazard, for the requested scenario."""
    rows = t2[
        (t2["risk_component"] == "hazard")
        & (t2["hazard_type"].isin(hazards))
        & (t2["scenario_type"] == scenario_type)
    ]
    if rows.empty:
        logger.warning(
            "no T2 hazard rows matched hazards=%s scenario_type=%r -- risk composite will be empty",
            hazards,
            scenario_type,
        )
    return _load_and_normalise_t2_rows(rows, t1, bbox, resolution_m)


def assemble_exposure(
    t2: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
) -> dict[str, np.ndarray]:
    """Sec 6.3 -- exposure layers (rural population, cropland, etc). Baseline only -- exposure
    isn't projected forward; the future risk surface uses baseline exposure x future hazard."""
    rows = t2[t2["risk_component"] == "exposure"]
    return _load_and_normalise_t2_rows(rows, t1, bbox, resolution_m)


def compose_risk(
    hazard_layers: dict[str, np.ndarray],
    exposure_layers: dict[str, np.ndarray],
    t2: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Sec 6.5, Mode A: `Risk_i = H_i x E_mean` per hazard, weighted sum across hazards using
    T2.weight_default, normalised to 0-1. Exposure is combined as an unweighted mean across
    its own layers first (population + cropland + ... all contribute equally to "how much is
    at stake here") -- a simplifying default; recipe-specific exposure weighting is future
    work, not guessed at here.
    """
    if not hazard_layers:
        raise ValueError("no hazard layers to compose -- check relevant_hazards()/assemble_hazards()")

    exposure = (
        np.mean(np.stack(list(exposure_layers.values()), axis=-1), axis=-1)
        if exposure_layers
        else np.ones_like(next(iter(hazard_layers.values())))
    )

    hazard_names = list(hazard_layers.keys())
    weights = t2.set_index("variable").loc[hazard_names, "weight_default"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    per_hazard_risk = {name: hazard_layers[name] * exposure for name in hazard_names}
    stack = np.stack([per_hazard_risk[n] for n in hazard_names], axis=-1)
    composite = (stack * weights).sum(axis=-1)
    composite = composite / max(composite.max(), 1e-9)  # renormalise to 0-1

    meta = {
        "mode": "A",
        "formula": "Risk = sum_i(w_i * Hazard_i * mean(Exposure))",
        "hazards_used": hazard_names,
        "hazard_weights": dict(zip(hazard_names, weights.tolist())),
        "exposure_layers_used": list(exposure_layers.keys()),
    }
    return composite, meta


def compose_vulnerability(*args, **kwargs):
    """Mode B -- deliberately not implemented (spec's own recommended v0 scope). Needs a
    Brayden methodology call on the composition function (weighted geometric mean vs additive
    vs PCA-derived) before it can be built without guessing."""
    raise NotImplementedError(
        "Mode B (Hazard x Exposure x Vulnerability) requires a methodology decision on how "
        "sensitivity + adaptive_capacity combine into vulnerability -- see M2_climate_risk.md "
        "Sec 'Open questions for Session C', item 2. Not guessed at here; use Mode A."
    )


def run_m2(
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    scenario_type: str = "baseline",
) -> dict:
    """End-to-end M2, Mode A, for one NbS + AOI + scenario."""
    hazards = relevant_hazards(t3)
    hazard_layers = assemble_hazards(t2, t1, hazards, bbox, resolution_m, scenario_type)
    exposure_layers = assemble_exposure(t2, t1, bbox, resolution_m)
    composite, meta = compose_risk(hazard_layers, exposure_layers, t2)
    return {
        "hazards": hazards,
        "hazard_layers": hazard_layers,
        "exposure_layers": exposure_layers,
        "composite_risk": composite,
        "meta": meta,
    }
