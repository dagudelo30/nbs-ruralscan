"""M3 -- Opportunity Space Characterisation. Spec: M3_characterisation.md (ratified v1.0).

Within M1's opportunity-space mask, describes who lives there and what development problems
(poverty, biodiversity, climate exposure) are most present -- the "fingerprint" M4 weights into
hotspots and the TTL reads directly. M3 *describes*; M4 (next module) *prioritises*.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nbs_ruralscan.data_loaders import load_variable
from nbs_ruralscan.runtime.normalise import min_max

logger = logging.getLogger(__name__)


def assemble_priorities(
    t5: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    opp_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sec 6.1/6.3 -- load + min-max standardise every T5 row where `mcda_role == 'priority'`,
    clipped to the opportunity-space mask (cells outside the mask -> NaN, so downstream stats
    only ever see in-mask pixels). Rows with no `dataset_id` yet (real gap in the current
    recipe: soil_erosion_risk, production_gap) are skipped with a warning rather than guessed.
    """
    priorities = t5[(t5["mcda_role"] == "priority") & t5["dataset_id"].notna()]
    skipped = t5[(t5["mcda_role"] == "priority") & t5["dataset_id"].isna()]["variable"].tolist()
    if skipped:
        logger.warning(
            "%d priority variable(s) have no dataset_id in T5 yet -- skipped: %s. These need "
            "a dataset assigned before M3/M4 can use them.",
            len(skipped),
            skipped,
        )

    out = {}
    for _, row in priorities.iterrows():
        matches = t1[t1["dataset_id"] == row["dataset_id"]]
        if matches.empty:
            logger.warning("%r: dataset_id=%r not in T1 -- skipping", row["variable"], row["dataset_id"])
            continue
        dataset_row = matches.iloc[0].to_dict()
        da = load_variable(row["variable"], dataset_row, bbox, resolution_m=resolution_m)
        standardised = min_max(da.values)
        if row.get("directionality_of_concern") == "lower_is_more_concern":
            standardised = 1.0 - standardised
        if opp_mask is not None:
            standardised = np.where(opp_mask, standardised, np.nan)
        out[row["variable"]] = standardised
    return out


def extract_fingerprint(
    t5: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    opp_mask: np.ndarray,
) -> dict:
    """Sec 6.4 -- descriptor rows (`mcda_role == 'descriptor'`) give context, never weighted
    into MCDA: rural population, accessibility, farm size, production value within the mask."""
    descriptors = t5[(t5["mcda_role"] == "descriptor") & t5["dataset_id"].notna()]
    fingerprint = {
        "total_cells": int(np.sum(opp_mask)),
        "opportunity_space_share": float(np.mean(opp_mask)),
    }
    for _, row in descriptors.iterrows():
        matches = t1[t1["dataset_id"] == row["dataset_id"]]
        if matches.empty:
            continue
        dataset_row = matches.iloc[0].to_dict()
        da = load_variable(row["variable"], dataset_row, bbox, resolution_m=resolution_m)
        in_mask = da.values[opp_mask.astype(bool)] if opp_mask.shape == da.values.shape else da.values
        fingerprint[f"{row['variable']}_mean_in_opp_space"] = float(np.nanmean(in_mask))
    return fingerprint


def problem_distribution(standardised: dict[str, np.ndarray]) -> pd.DataFrame:
    """Sec 6.5 -- share of opportunity-space land at each severity level, per priority
    variable, for the "what this NbS can address" bars M5 pairs against."""
    bins = [0, 0.25, 0.5, 0.75, 1.0001]
    labels = ["Low", "Moderate", "High", "Very High"]
    rows = []
    for var, arr in standardised.items():
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            continue
        classes = pd.cut(pd.Series(valid), bins=bins, labels=labels, right=False)
        counts = classes.value_counts(normalize=True).reindex(labels).fillna(0.0)
        rows.append({"variable": var, **counts.to_dict()})
    return pd.DataFrame(rows)


def run_m3(
    t5: pd.DataFrame,
    t1: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    opp_mask: np.ndarray,
) -> dict:
    """End-to-end M3 for one AOI, given M1's opportunity-space mask."""
    standardised = assemble_priorities(t5, t1, bbox, resolution_m, opp_mask)
    fingerprint = extract_fingerprint(t5, t1, bbox, resolution_m, opp_mask)
    distribution = problem_distribution(standardised)
    return {
        "standardised_priorities": standardised,
        "fingerprint": fingerprint,
        "problem_distribution": distribution,
    }
