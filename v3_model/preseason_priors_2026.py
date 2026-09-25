"""
preseason_priors_2026 - loader for the frozen 2026 Andy/Drew preseason
offense/defense decomposition (handoff Section 7).

This is the authoritative 2026 Week-1 prior source per the handoff -
NOT `rating.build_previous_season_priors`, which is a leakage-free
historical-research proxy only (previous-season EPA), never intended
as the live 2026 anchor.

WHAT THIS PRODUCES, AND WHY IT ISN'T WIRED INTO rating.py YET
----------------------------------------------------------------
The output here (`offense_strength`, `defense_strength`) is in
POINTS - it's a decomposition of "Centered Week 1 Prior," itself an
aggregate power rating in points. `rating.build_pregame_state` and
its `previous_season_priors` argument operate in EPA-like latent
units instead. The handoff is explicit that the points<->EPA
conversion contract for 2026 has NOT been frozen and warns:
"Do not independently invent one." So this module deliberately stops
at producing the frozen points-scale decomposition and does not
attempt to convert it into rating.py's EPA scale or wire it into
build_pregame_state's previous_season_priors slot - doing either
would be inventing the missing conversion, which is exactly what's
been flagged as off-limits until your own project settles it.

THE FROZEN DECOMPOSITION (reproduced exactly, verified below)
----------------------------------------------------------------
    overall_prior = "Centered Week 1 Prior" (Power Ratings sheet) -
        NOT "Final Week 1 Rating" (Core V2 Prior sheet), which bakes
        in temporary Week-1 QB/personnel/injury adjustments and must
        not be treated as persistent team strength (handoff Section 8).
    offense_unit_composite = mean(QB/OC Consensus, Skill Consensus,
        OL Consensus)               [Offense sheet]
    defense_unit_composite = mean(Pass Rush Consensus,
        Coverage Consensus, Run Defense Consensus)   [Defense sheet]
    Missing unit cells filled with that column's league mean, for
        composition only, before averaging.
    offense_unit_z / defense_unit_z = each composite standardized
        across the 32 teams using POPULATION SD (ddof=0).
    composition_delta = offense_unit_z - defense_unit_z, used at
        scale 1.0 (no additional multiplier).
    offense_strength = 0.5 * (overall_prior + composition_delta)
    defense_strength = 0.5 * (overall_prior - composition_delta)

    This guarantees offense_strength + defense_strength ==
    overall_prior exactly, by construction (verified to float
    precision in load_2026_preseason_priors and in the test suite).
"""

from __future__ import annotations

import pandas as pd


def load_2026_preseason_priors(xlsx_path: str) -> pd.DataFrame:
    """
    Load and decompose the Andy/Drew 2026 Deep Dive preseason workbook.

    Expects sheets "Power Ratings", "Offense", "Defense" with headers
    on the second row (row index 1), matching
    2026_Deep_Dive_Priors_Core_V2.xlsx's actual layout.

    Returns one row per team (32 total) with columns:
        team, team_name, overall_prior,
        offense_unit_composite, defense_unit_composite,
        offense_unit_z, defense_unit_z, composition_delta,
        offense_strength, defense_strength
    All in POINTS - see module docstring before using these alongside
    anything in rating.py.
    """
    power = pd.read_excel(xlsx_path, sheet_name="Power Ratings", header=1)
    # The sheet has a trailing footnote row that lands in the Team
    # column as a long sentence, not a 2-3 letter team code - drop it.
    power = power[power["Team"].astype(str).str.len() <= 3].copy()
    power = power[["Team", "Team Name", "Centered Week 1 Prior"]].rename(
        columns={"Team": "team", "Team Name": "team_name", "Centered Week 1 Prior": "overall_prior"}
    )

    off = pd.read_excel(xlsx_path, sheet_name="Offense", header=1)[
        ["Team", "QB/OC Consensus", "Skill Consensus", "OL Consensus"]
    ].rename(columns={"Team": "team"})
    deff = pd.read_excel(xlsx_path, sheet_name="Defense", header=1)[
        ["Team", "Pass Rush Consensus", "Coverage Consensus", "Run Defense Consensus"]
    ].rename(columns={"Team": "team"})

    # Missing unit cells -> league mean, for composition only.
    for col in ["QB/OC Consensus", "Skill Consensus", "OL Consensus"]:
        off[col] = off[col].fillna(off[col].mean())
    for col in ["Pass Rush Consensus", "Coverage Consensus", "Run Defense Consensus"]:
        deff[col] = deff[col].fillna(deff[col].mean())

    off["offense_unit_composite"] = off[
        ["QB/OC Consensus", "Skill Consensus", "OL Consensus"]
    ].mean(axis=1)
    deff["defense_unit_composite"] = deff[
        ["Pass Rush Consensus", "Coverage Consensus", "Run Defense Consensus"]
    ].mean(axis=1)

    merged = power.merge(off[["team", "offense_unit_composite"]], on="team").merge(
        deff[["team", "defense_unit_composite"]], on="team"
    )
    if len(merged) != 32:
        raise ValueError(
            f"Expected exactly 32 teams after merging Power Ratings/Offense/"
            f"Defense sheets, got {len(merged)}. Check the workbook layout "
            f"hasn't changed."
        )

    off_mean = merged["offense_unit_composite"].mean()
    off_std = merged["offense_unit_composite"].std(ddof=0)
    def_mean = merged["defense_unit_composite"].mean()
    def_std = merged["defense_unit_composite"].std(ddof=0)

    merged["offense_unit_z"] = (merged["offense_unit_composite"] - off_mean) / off_std
    merged["defense_unit_z"] = (merged["defense_unit_composite"] - def_mean) / def_std
    merged["composition_delta"] = merged["offense_unit_z"] - merged["defense_unit_z"]
    merged["offense_strength"] = 0.5 * (merged["overall_prior"] + merged["composition_delta"])
    merged["defense_strength"] = 0.5 * (merged["overall_prior"] - merged["composition_delta"])

    # Defense-in-depth: this identity must hold by construction. If it
    # doesn't, something in the decomposition above is wrong.
    reconstructed = merged["offense_strength"] + merged["defense_strength"]
    max_diff = (reconstructed - merged["overall_prior"]).abs().max()
    assert max_diff < 1e-9, (
        f"offense_strength + defense_strength should equal overall_prior "
        f"exactly; max observed difference was {max_diff}."
    )

    return merged[
        [
            "team", "team_name", "overall_prior",
            "offense_unit_composite", "defense_unit_composite",
            "offense_unit_z", "defense_unit_z", "composition_delta",
            "offense_strength", "defense_strength",
        ]
    ].reset_index(drop=True)
