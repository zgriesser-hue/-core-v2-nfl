import pandas as pd

from scripts.v3_performance import append_releases, grade_releases


def test_release_is_immutable_and_late_runs_do_not_rewrite():
    board = pd.DataFrame([dict(season=2026, week=4, game_key="2026_W4_A_H", away_team="A",
                               home_team="H", fair_home_spread_v3=5.0,
                               market_home_spread=-3.0, personnel_status="reviewed")])
    schedule = pd.DataFrame([dict(season=2026, week=4, away_team="A", home_team="H",
                                  gameday="2026-10-01", gametime="20:20", game_type="REG")])
    t = pd.Timestamp("2026-10-01T15:00:00Z")
    market = pd.DataFrame([dict(game_key="2026_W4_A_H", market_home_spread=-3.0,
                                market_snapshot_timestamp_et="2026-10-01T14:30:00Z")])
    one = append_releases(board, schedule, pd.DataFrame(), t, market)
    assert len(one) == 1 and one.model_side.iloc[0] == "H"
    assert len(append_releases(board, schedule, one, t + pd.Timedelta(minutes=1), market)) == 1
    changed = board.assign(fair_home_spread_v3=4.0)
    two = append_releases(changed, schedule, one, t + pd.Timedelta(minutes=2), market)
    assert len(two) == 2 and two.fair_home_margin_v3.tolist() == [5.0, 4.0]
    assert len(append_releases(changed, schedule, two, pd.Timestamp("2026-10-02T01:00:00Z"), market)) == 2
    no_quote = append_releases(board, schedule, pd.DataFrame(), t)
    assert pd.isna(no_quote.release_market_home_spread.iloc[0])


def test_ats_and_close_use_first_release_and_correct_spread_sign():
    releases = pd.DataFrame([
        dict(season=2026, week=4, game_key="g", away_team="A", home_team="H",
             release_timestamp_utc="2026-10-01T15:00:00Z", kickoff_utc="2026-10-02T00:20:00Z",
             fair_home_margin_v3=5.0, release_market_home_spread=-3.0,
             market_snapshot_timestamp_utc="2026-10-01T14:30:00Z",
             model_side="H", personnel_status="reviewed"),
        dict(season=2026, week=4, game_key="g", away_team="A", home_team="H",
             release_timestamp_utc="2026-10-01T17:00:00Z", kickoff_utc="2026-10-02T00:20:00Z",
             fair_home_margin_v3=4.0, release_market_home_spread=-4.0,
             market_snapshot_timestamp_utc="2026-10-01T16:30:00Z",
             model_side="H", personnel_status="reviewed")])
    grades = pd.DataFrame([dict(game_key="g", away_score=17, home_score=21)])
    closes = pd.DataFrame([dict(game_key="g", closing_home_spread=-6.0,
                                snapshot_timestamp_utc="2026-10-02T00:00:00Z",
                                kickoff_utc="2026-10-02T00:20:00Z")])
    result = grade_releases(releases, grades, closes).iloc[0]
    assert result.absolute_error == 1.0
    assert result.ats_result == "Win" and result.ats_margin == 1.0
    assert result.projection_vs_close_points == -1.0
    assert result.market_clv_points == 3.0
    invalid = grade_releases(releases, grades, closes.assign(snapshot_timestamp_utc="2026-10-02T01:00:00Z"))
    assert pd.isna(invalid.projection_vs_close_points.iloc[0])
    away = releases.iloc[:1].assign(fair_home_margin_v3=1.0, model_side="A")
    away_result = grade_releases(away, grades, closes).iloc[0]
    assert away_result.ats_result == "Loss" and away_result.ats_margin == -1.0
    assert away_result.projection_vs_close_points == 5.0
    assert away_result.market_clv_points == -3.0
