import pandas as pd
import pytest

from scripts.build_early_injury_watchlist import build_watchlist


def test_prior_week_is_unverified_and_current_official_supersedes():
    schedule = pd.DataFrame([dict(game_id="2026_04_A_B", season=2026, week=4,
                                  away_team="A", home_team="B")])
    prior = pd.DataFrame([dict(team="A", player_name="One", gsis_id="1", position_group="DB",
                               game_status="Out", source_season=2026, source_week=3)])
    current = pd.DataFrame(columns=["team", "player_name", "game_status", "source_season", "source_week"])
    early = build_watchlist(prior, current, schedule, 2026, 4)
    assert early.iloc[0].verification == "PRIOR_WEEK_ONLY_UNVERIFIED"
    assert pd.isna(early.iloc[0].model_adjustment_points)
    current = pd.DataFrame([dict(team="A", player_name="One", game_status="Questionable",
                                 source_season=2026, source_week=4),
                            dict(team="B", player_name="New", game_status="Out",
                                 source_season=2026, source_week=4)])
    updated = build_watchlist(prior, current, schedule, 2026, 4)
    assert len(updated) == 2
    assert set(updated.verification) == {"CURRENT_WEEK_OFFICIAL"}
    assert updated.loc[updated.player_name.eq("One"), "current_game_status"].iloc[0] == "Questionable"
    assert updated.model_adjustment_points.isna().all()
    with pytest.raises(ValueError, match="different season/week"):
        build_watchlist(prior, current.assign(source_week=3), schedule, 2026, 4)
