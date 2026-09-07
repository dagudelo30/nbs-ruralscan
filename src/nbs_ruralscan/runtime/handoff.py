"""M6 -- Implementation Hand-off. Spec: M6_handoff.md.

Content, not computation: the narrative bridge from scoping to feasibility. Points to the
valuation toolbox from Stocktake Table 6 (InVEST, RUSLE, IPCC Tier 1-3 carbon accounting, CBA,
MCDA) as NEXT steps -- never computes them here. This module's entire job is assembling what
already exists (M4's ranked units, M5's economic archetype, T0's context) into one hand-off
card; it is the project's explicit scope boundary.
"""

from __future__ import annotations

STAGE_PATHWAY = [
    "Scoping (this tool)",
    "Pre-feasibility",
    "Feasibility & design",
    "Implementation",
]

FEASIBILITY_METHODS_TABLE_6 = {
    "nature_based": ["InVEST", "RUSLE (soil erosion)", "IPCC Tier 1-3 carbon accounting"],
    "integrated": ["Cost-benefit analysis (CBA)", "Multi-criteria decision analysis (MCDA)"],
}


def handoff_card(
    nbs_id: str,
    ranked_units: "object",
    economic_profile: dict,
    iplc_overlap: bool = False,
    context_sensitive_vars: list[str] | None = None,
) -> dict:
    """Assemble the per-NbS hand-off card. No new numbers are computed -- everything here is
    read from M4/M5 output or the recipe's own T0 content."""
    card = {
        "nbs_id": nbs_id,
        "stage_pathway": STAGE_PATHWAY,
        "current_stage": "Scoping (this tool)",
        "economic_snapshot": {
            **economic_profile,
            "caveat": "Planning-level, scoping-grade estimate. Full CBA belongs at the "
            "feasibility stage -- see feasibility_methods below.",
        },
        "recommended_next_steps": [
            "Validate priority units identified by M4 in the field.",
            "Stakeholder and tenure mapping (especially where iplc_lands overlap is flagged).",
            "Full cost-benefit and financing scan.",
            "Safeguards / ESG screening.",
        ],
        "feasibility_methods": FEASIBILITY_METHODS_TABLE_6,
        "proportionality_note": "Simple methods (benefit transfer, proxy indicators) suffice "
        "for most sites; reserve detailed spatial modelling for high-priority hotspots only.",
        "fpic_required": bool(iplc_overlap),
    }
    if context_sensitive_vars:
        card["country_endorsed_source_prompts"] = [
            f"Confirm a country-endorsed source for {v!r} before this run's figures are used "
            "in a feasibility decision (scoping used a global default)."
            for v in context_sensitive_vars
        ]
    return card
