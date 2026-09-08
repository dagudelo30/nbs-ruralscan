"""M5 -- NbS Scorecard and Response. Spec: M5_scorecard.md (ratified v1.0).

Explicitly QUALITATIVE per the spec: renders Likert-style effects (T6) plus an indicative
economic archetype. NOT ecosystem-service valuation, NOT a cost-benefit analysis -- those are
downstream, signposted (not computed) in M6. Mostly table assembly; no heavy compute, matching
the spec's own characterisation of this module.
"""

from __future__ import annotations

import pandas as pd


def load_scorecard(
    t6: pd.DataFrame, t5_all: pd.DataFrame, t2: pd.DataFrame
) -> pd.DataFrame:
    """Sec 6.1 -- T6.variable_id references one of three things, confirmed against the real
    agroforestry recipe (not assumed): a T5 variable (any mcda_role -- both priority rows like
    `rural_poverty` AND descriptor rows like `accessibility_travel_time` appear), a T2 hazard
    (`hazard_type`, e.g. `drought`), or a standalone economic-only row with no spatial variable
    at all (`establishment_cost`). Only truly unrecognised ids raise -- that would be a real
    recipe error, not a modelling choice this module should paper over.
    """
    valid_ids = (
        set(t5_all["variable"])
        | set(t2["hazard_type"].dropna())
        | {
            "establishment_cost"  # economic-only rows have no T2/T5 counterpart by design
        }
    )
    orphans = t6[~t6["variable_id"].isin(valid_ids)]
    if not orphans.empty:
        raise ValueError(
            "T6 rows reference variable_id(s) not found in T5, T2, or the known "
            "economic-only set: " + str(orphans["variable_id"].tolist())
        )
    cols = [
        "variable_id",
        "effect_direction",
        "effect_confidence",
        "effect_mechanism",
        "conditionality",
    ]
    return t6[cols]


def pair_with_problems(
    scorecard: pd.DataFrame, problem_distribution: pd.DataFrame
) -> pd.DataFrame:
    """Sec 6.2 -- join response strength (T6) to how present the problem is in the opportunity
    space (M3's problem_distribution). This is the "what this NbS can address, where the
    problem actually is" read."""
    merged = scorecard.merge(
        problem_distribution, left_on="variable_id", right_on="variable", how="left"
    )
    return merged.drop(columns=["variable"], errors="ignore")


def economic_profile(t6: pd.DataFrame) -> dict:
    """Sec 6.3 -- establishment cost band / financing archetype, read as-is from T6. Renders
    whatever the recipe populated; never computes a new number -- explicitly NOT a CBA."""
    econ_rows = t6[t6["economic_indicator_type"].notna()]
    return {
        row["economic_indicator_type"]: row["economic_value_range"]
        for _, row in econ_rows.iterrows()
    }


def flag_cautions(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Sec 6.5 -- surface negative/caution Likert entries explicitly, not buried."""
    negative = {"moderate_negative", "strong_negative"}
    return scorecard[scorecard["effect_direction"].isin(negative)]


def run_m5(
    t6: pd.DataFrame,
    t5_all: pd.DataFrame,
    t2: pd.DataFrame,
    problem_distribution: pd.DataFrame,
) -> dict:
    """End-to-end M5 for one NbS, given M3's problem_distribution."""
    scorecard = load_scorecard(t6, t5_all, t2)
    paired = pair_with_problems(scorecard, problem_distribution)
    econ = economic_profile(t6)
    cautions = flag_cautions(scorecard)
    return {
        "scorecard": scorecard,
        "paired_with_problems": paired,
        "economic_profile_scoping_grade": econ,
        "cautions": cautions,
    }
