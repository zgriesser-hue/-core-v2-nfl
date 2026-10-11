import pandas as pd

from qb_research import build_qb_games, pregame_switch_signals, evaluate_loso


def test_scramble_and_pass_are_attributed_to_same_qb():
    pbp = pd.DataFrame([
        dict(season=2025, week=1, season_type="REG", game_id="a", posteam="BAL", defteam="ATL", passer_player_id="L", passer_player_name="Jackson", rusher_player_id=None, rusher_player_name=None, pass_attempt=1, sack=0, qb_scramble=0, qb_epa=.2, epa=.2),
        dict(season=2025, week=1, season_type="REG", game_id="a", posteam="BAL", defteam="ATL", passer_player_id=None, passer_player_name=None, rusher_player_id="L", rusher_player_name="Jackson", pass_attempt=0, sack=0, qb_scramble=1, qb_epa=None, epa=.5),
    ])
    games = build_qb_games(pbp)
    assert len(games) == 1
    assert games.iloc[0].dropbacks == 2
    assert abs(games.iloc[0].epa_sum - .7) < 1e-10


def test_current_game_performance_does_not_enter_its_pregame_signal():
    rows = []
    for week in range(1, 4):
        for team, opponent in (("BAL", "ATL"), ("ATL", "BAL")):
            qb = "H" if team == "BAL" and week == 3 else ("L" if team == "BAL" else "A")
            rows.append(dict(season=2025, week=week, game_id=f"2025_{week:02d}_ATL_BAL", team=team, opponent=opponent, qb_id=qb, qb_name=qb, dropbacks=30, epa_sum=3.))
    games = pd.DataFrame(rows)
    control = pd.DataFrame([dict(season=2025, week=3, game_id="2025_03_ATL_BAL", home_team="BAL", away_team="ATL", home_margin=0, pred_v30=0)])
    first = pregame_switch_signals(games, control)
    games.loc[games.week.eq(3), "epa_sum"] = -1000
    second = pregame_switch_signals(games, control)
    assert first.iloc[0].eligible
    assert first.iloc[0].switch_game
    assert first.iloc[0].qb_delta_35 == second.iloc[0].qb_delta_35


def test_loso_coefficient_excludes_held_out_season():
    panel = pd.DataFrame([
        dict(season=season, eligible=True, switch_game=True, qb_delta_35=2., pred_v30=0., home_margin=margin)
        for season, margin in ((2022, 1.), (2023, 1.), (2024, 100.)) for _ in range(20)
    ])
    folds = evaluate_loso(panel)
    held = folds.loc[(folds.season.eq(2024)) & folds.segment.eq("switch")].iloc[0]
    assert held.coefficient < 1.
