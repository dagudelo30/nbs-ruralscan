"""M4 -- Priority Hotspots (MCDA). Spec: M4_hotspots.md (ratified v1.0).

Within the opportunity space, combines the TTL's weighted development priorities (M3's
standardised layers) into a single hotspot score. Explicitly reuses `mcda.weighted_overlay` +
`mcda.quartile_classify` -- per the spec's own instruction ("Reuse, don't duplicate") -- rather
than re-implementing the MCDA math a second time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbs_ruralscan.runtime.mcda import quartile_classify, weighted_overlay


def map_conceptual_weights(
    settings: dict[str, str], scale: tuple[float, float, float, float] = (0.0, 1.0, 2.0, 3.0)
) -> dict[str, float]:
    """Sec 6.3 -- map TTL settings {priority_id: '-'|'L'|'M'|'H'} to normalised numeric
    weights. '-' priorities are dropped before normalising (a TTL choosing "not a priority"
    should get zero weight, not a share of the total)."""
    level_map = {"-": scale[0], "L": scale[1], "M": scale[2], "H": scale[3]}
    raw = {k: level_map[v] for k, v in settings.items() if level_map[v] > 0}
    total = sum(raw.values())
    if total == 0:
        raise ValueError("at least one priority must be weighted above '-'")
    return {k: v / total for k, v in raw.items()}


def build_priority_stack(
    standardised_priorities: dict[str, np.ndarray], weights: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    """Sec 6.1 -- assemble the (H, W, K) stack in a fixed variable order matching `weights`."""
    ordered_vars = [v for v in weights if v in standardised_priorities]
    missing = [v for v in weights if v not in standardised_priorities]
    if missing:
        raise KeyError(f"weighted priorities missing from M3 output: {missing}")
    stack = np.stack([np.nan_to_num(standardised_priorities[v], nan=0.0) for v in ordered_vars], axis=-1)
    return stack, ordered_vars


def hotspot_overlay(stack: np.ndarray, weights: dict[str, float], ordered_vars: list[str], opp_mask: np.ndarray) -> np.ndarray:
    """Sec 6.5 -- thin wrapper over `mcda.weighted_overlay`, re-applying the opportunity-space
    mask (NaN outside) so the hotspot score is only ever defined where M1 says the NbS could
    work at all."""
    w = np.array([weights[v] for v in ordered_vars])
    score = weighted_overlay(stack, w)
    return np.where(opp_mask, score, np.nan)


def apply_project_risk_scope(
    hotspot: np.ndarray, project_risk: np.ndarray | None, threshold: float = 0.75, mode: str = "flag"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sec 6.6 -- M2b filter/scope, never summed into the score (spec's explicit rule). Returns
    (hotspot, high_risk_flag_mask). `project_risk` is None when M2b hasn't been run (it's
    optional per the spec) -- returns the hotspot unchanged and no flag."""
    if project_risk is None:
        return hotspot, None
    high_risk = project_risk >= threshold
    if mode == "exclude":
        return np.where(high_risk, np.nan, hotspot), high_risk
    return hotspot, high_risk  # "flag" mode: UI highlights, score untouched


def bivariate(suitability: np.ndarray, hotspot: np.ndarray, bins: int = 5) -> np.ndarray:
    """Sec 6.7 -- 5x5 suitability x priority classification, combined into a single 1-25 code
    (row-major: `(suit_class - 1) * bins + priority_class`)."""
    suit_classes, _ = quartile_classify(suitability) if bins == 4 else _n_class(suitability, bins)
    hot_classes, _ = quartile_classify(hotspot) if bins == 4 else _n_class(hotspot, bins)
    return (suit_classes.astype(int) - 1) * bins + hot_classes.astype(int)


def _n_class(x: np.ndarray, n: int) -> tuple[np.ndarray, dict]:
    valid = ~np.isnan(x)
    edges = np.quantile(x[valid], np.linspace(0, 1, n + 1)[1:-1]) if valid.any() else []
    classes = np.digitize(x, edges) + 1
    classes = np.where(valid, classes, 0)
    return classes, {i: f"class_{i}" for i in range(1, n + 1)}


def rank_units(
    hotspot: np.ndarray,
    stack: np.ndarray,
    ordered_vars: list[str],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Sec 6.8, simplified for a single-AOI run without a real admin-unit vector yet (that
    needs the GADM boundary pulled via T7 -- a later addition once real geometry is wired in).
    Returns AOI-level summary stats + the top-weighted-contribution variables, which is the
    content `rank_units` would otherwise aggregate per admin unit."""
    w = np.array([weights[v] for v in ordered_vars])
    contributions = np.nanmean(stack, axis=(0, 1)) * w
    order = np.argsort(contributions)[::-1]
    top_drivers = [(ordered_vars[i], float(contributions[i])) for i in order[:3]]
    return pd.DataFrame(
        {
            "mean_hotspot_score": [float(np.nanmean(hotspot))],
            "share_very_high": [float(np.nanmean(hotspot > 0.75))],
            "top_drivers": [top_drivers],
        }
    )


def run_m4(
    standardised_priorities: dict[str, np.ndarray],
    ttl_weights: dict[str, str],
    opp_mask: np.ndarray,
    suitability: np.ndarray,
    project_risk: np.ndarray | None = None,
) -> dict:
    """End-to-end M4 given M3's output + M1's suitability/mask + TTL-supplied conceptual
    weights (e.g. {'rural_poverty': 'H', 'drought_hazard': 'M', ...})."""
    weights = map_conceptual_weights(ttl_weights)
    stack, ordered_vars = build_priority_stack(standardised_priorities, weights)
    hotspot = hotspot_overlay(stack, weights, ordered_vars, opp_mask)
    hotspot, risk_flag = apply_project_risk_scope(hotspot, project_risk)
    biv = bivariate(suitability, hotspot)
    ranked = rank_units(hotspot, stack, ordered_vars, weights)
    return {
        "weights": weights,
        "hotspot": hotspot,
        "project_risk_flag": risk_flag,
        "bivariate": biv,
        "ranked_units": ranked,
    }
