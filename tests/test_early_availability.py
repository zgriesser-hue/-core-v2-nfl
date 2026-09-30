from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.early_availability import apply_early_availability, load_early_availability


def test_week4_daniels_is_provisional_and_not_double_priced(tmp_path):
    ledger = pd.DataFrame([dict(
        season=2026, week=4, team="WAS", player="Jayden Daniels", position_group="QB",
        assessment="PROVISIONAL_QB", source_url="https://www.commanders.com/source",
        source_published_at_utc="2026-09-25T18:46:00Z",
        observed_at_utc="2026-09-28T20:55:00Z", reviewed_by="research",
        reason="Team says week-to-week; Mariota scenario", adjustment_points=-1.5,
        out_adjustment_points=-1.5,
    )])
    path = tmp_path / "availability.csv"
    ledger.to_csv(path, index=False)
    asof = datetime(2026, 9, 28, 21, 10, tzinfo=timezone.utc)
    loaded = load_early_availability(path, 2026, 4, asof)
    weekly = pd.DataFrame(columns=["season", "week", "team", "player", "category",
                                   "adjustment_points", "out_adjustment_points", "exclude_from_auto"])
    combined = apply_early_availability(weekly, loaded)
    assert combined.iloc[0].category == "QB_PROVISIONAL"
    assert combined.iloc[0].adjustment_points == -1.5
    assert combined.iloc[0].out_adjustment_points == -1.5
    with pytest.raises(ValueError, match="already priced"):
        apply_early_availability(combined, loaded)
    with pytest.raises(ValueError, match="future-dated"):
        load_early_availability(path, 2026, 4, datetime(2026, 9, 27, tzinfo=timezone.utc))


def test_available_player_has_no_adjustment(tmp_path):
    ledger = pd.DataFrame([dict(
        season=2026, week=4, team="WAS", player="Jayden Daniels", position_group="QB",
        assessment="AVAILABLE", source_url="https://www.commanders.com/source",
        source_published_at_utc="2026-10-01T12:00:00Z",
        observed_at_utc="2026-10-01T12:01:00Z", reviewed_by="research",
        reason="Starter confirmed", adjustment_points=0, out_adjustment_points=0,
    )])
    path = tmp_path / "availability.csv"
    ledger.to_csv(path, index=False)
    loaded = load_early_availability(path, 2026, 4, datetime(2026, 10, 1, 13, tzinfo=timezone.utc))
    weekly = pd.DataFrame(columns=["team", "player", "category", "adjustment_points"])
    assert apply_early_availability(weekly, loaded).empty


def test_confirmed_defensive_out_is_valued_only_in_v3(tmp_path):
    row = dict(season=2026, week=4, team="B", player="Defender B",
               position_group="DB", assessment="CONFIRMED_OUT",
               source_url="https://example.com/team-report",
               source_published_at_utc="2026-09-28T18:00:00Z",
               observed_at_utc="2026-09-28T18:01:00Z",
               reviewed_by="research", reason="Team confirmed out",
               adjustment_points=0, out_adjustment_points=0)
    path = tmp_path / "availability.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    ledger = load_early_availability(
        path, 2026, 4, datetime(2026, 9, 28, 19, tzinfo=timezone.utc)
    )
    weekly = pd.DataFrame(columns=["team", "player", "category", "adjustment_points"])
    assert apply_early_availability(weekly, ledger).empty
    row["adjustment_points"] = -1.8
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="derived from frozen V3 PVAR"):
        load_early_availability(path, 2026, 4,
                                datetime(2026, 9, 28, 19, tzinfo=timezone.utc))
