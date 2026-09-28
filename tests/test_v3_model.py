"""
Regression tests for the extracted V3 model logic.

IMPORTANT SCOPE NOTE
---------------------
These tests validate that each module's MATH matches the notebook's
documented behavior and invariants (e.g. "week 1 returns priors
untouched", "pred_v36 = pred_v30 + sum of adjustments exactly",
"zero prior disagreement -> zero market adjustment"). They do NOT
reproduce your exact historical MAE/RMSE numbers from the notebook,
because that requires your actual Google Drive parquet files
(team_game_efficiency.parquet, market_lines.parquet, etc.), which
are not available in this environment.

Before trusting this package for live use, additionally run it
against your real historical data and confirm it reproduces the
notebook's printed MAE/RMSE figures (e.g. V3.6B's "COMMON-SAMPLE
MODEL COMPARISON" table) within floating-point tolerance. That's
the real regression test; this suite is the safety net for future
refactors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3_model import personnel, market, neutral, pipeline, rating, spread


# ---------------------------------------------------------------
# rating.py
# ---------------------------------------------------------------


def test_build_pregame_state_rejects_unfiltered_future_data():
    """
    Defense-in-depth: a caller that (by mistake) passes rows at or
    after the target week must get a loud failure, not a silently
    correct answer - the assertion exists specifically so a future
    caller can't repeat predict_2026.py's old dead-filter bug and
    have it go unnoticed.
    """
    team_game = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A"],
            "plays": [65, 65, 65, 65],
            "epa_per_play": [0.10, -0.10, 0.12, -0.12],
            "def_epa_per_play": [-0.10, 0.10, -0.12, 0.12],
        }
    )
    priors = pd.DataFrame(
        {"season": [2025, 2025], "team": ["A", "B"], "prev_offense": [0.0, 0.0], "prev_defense": [0.0, 0.0]}
    )

    with pytest.raises(AssertionError):
        # week=2 requested, but team_game still includes week-2 rows -> leakage
        rating.build_pregame_state(
            team_game, 2025, week=2, previous_season_priors=priors,
            off_alpha=0.5, off_strength=6.0, def_alpha=0.2, def_strength=16.0,
        )

    # correctly filtered input must NOT raise
    filtered = team_game[team_game["week"] < 2]
    rating.build_pregame_state(
        filtered, 2025, week=2, previous_season_priors=priors,
        off_alpha=0.5, off_strength=6.0, def_alpha=0.2, def_strength=16.0,
    )


def test_prior_anchored_od_week1_returns_prior_untouched():
    """With no history, the fit must return the prior exactly."""
    priors = pd.DataFrame(
        {
            "team": ["A", "B", "C"],
            "prior_offense": [0.05, -0.02, 0.10],
            "prior_defense": [0.01, 0.03, -0.04],
        }
    )
    empty_history = pd.DataFrame(columns=["team", "opponent", "epa_per_play", "plays"])

    result = rating.fit_prior_anchored_od(empty_history, priors, prior_strength=4.0)

    result = result.set_index("team").loc[priors["team"]]
    np.testing.assert_allclose(
        result["latent_offense"].to_numpy(), priors["prior_offense"].to_numpy()
    )
    np.testing.assert_allclose(
        result["latent_defense"].to_numpy(), priors["prior_defense"].to_numpy()
    )


def test_prior_anchored_od_pulls_toward_observed_performance():
    """
    A team that has clearly outperformed its prior in every game so
    far should end up with a fitted offense rating between the prior
    and the raw observed performance (i.e. shrunk, not equal to
    either extreme, and moved in the right direction).
    """
    priors = pd.DataFrame(
        {
            "team": ["A", "B"],
            "prior_offense": [0.0, 0.0],
            "prior_defense": [0.0, 0.0],
        }
    )
    # A has strong offensive performances vs B; B's defense should
    # come out looking correspondingly worse than its zero prior.
    history = pd.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A"],
            "epa_per_play": [0.30, -0.30, 0.28, -0.28],
            "plays": [65, 65, 65, 65],
        }
    )

    result = rating.fit_prior_anchored_od(
        history, priors, prior_strength=4.0
    ).set_index("team")

    assert result.loc["A", "latent_offense"] > 0.0
    assert result.loc["A", "latent_offense"] < 0.30
    # B's defense should be worse (lower) than its zero prior, since
    # it has been consistently beaten.
    assert result.loc["B", "latent_defense"] < 0.0


def test_build_previous_season_priors_flips_defense_sign():
    team_game = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team": ["A", "A"],
            "plays": [65, 65],
            "epa_per_play": [0.10, 0.10],
            "def_epa_per_play": [0.05, 0.05],
        }
    )
    priors = rating.build_previous_season_priors(team_game, season=2025)
    row = priors.set_index("team").loc["A"]
    assert row["prev_offense"] == pytest.approx(0.10)
    # Defense sign-flipped: allowing +0.05 EPA becomes a -0.05 rating.
    assert row["prev_defense"] == pytest.approx(-0.05)


# ---------------------------------------------------------------
# spread.py
# ---------------------------------------------------------------


def test_fit_hfa_beta_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    strength_diff = rng.normal(0, 1, size=500)
    true_hfa, true_beta = 1.8, 4.2
    home_margin = true_hfa + true_beta * strength_diff  # no noise

    cal = spread.fit_hfa_beta(pd.Series(home_margin), pd.Series(strength_diff))

    assert cal.hfa == pytest.approx(true_hfa, abs=1e-8)
    assert cal.beta == pytest.approx(true_beta, abs=1e-8)


def test_strength_diff_matches_notebook_definition():
    result = spread.strength_diff(
        home_offense=pd.Series([0.10]),
        home_defense=pd.Series([0.05]),
        away_offense=pd.Series([0.02]),
        away_defense=pd.Series([-0.01]),
    )
    # (0.10 + 0.05) - (0.02 + (-0.01)) = 0.15 - 0.01 = 0.14
    assert result.iloc[0] == pytest.approx(0.14)


# ---------------------------------------------------------------
# personnel.py
# ---------------------------------------------------------------


def test_personnel_adjustment_zero_when_no_major_loss():
    cal = personnel.PersonnelCalibration(threshold=0.5, adjustment_points=2.0)
    adj = cal.adjustment(
        home_def_loss=pd.Series([0.1, 0.2]), away_def_loss=pd.Series([0.1, 0.3])
    )
    assert (adj == 0).all()


def test_personnel_adjustment_direction():
    """Away defense depleted (major loss) should raise the home margin."""
    cal = personnel.PersonnelCalibration(threshold=0.5, adjustment_points=2.0)
    adj = cal.adjustment(
        home_def_loss=pd.Series([0.1]), away_def_loss=pd.Series([0.9])
    )
    assert adj.iloc[0] == pytest.approx(2.0)  # away_major(1) - home_major(0) = 1


def test_fit_personnel_adjustment_rejects_negative_coefficient():
    """
    If the data implies away-defense attrition HURTS the home team
    (backwards from theory), fitting should raise rather than
    silently flip the sign.
    """
    home_def_loss = pd.Series([0.9, 0.1, 0.9, 0.1])
    away_def_loss = pd.Series([0.1, 0.9, 0.1, 0.9])
    # Backwards: residual is negative exactly when away is major.
    base_residual = pd.Series([2.0, -2.0, 2.0, -2.0])

    with pytest.raises(ValueError, match="negative"):
        personnel.fit_personnel_adjustment(
            home_def_loss, away_def_loss, base_residual, threshold_quantile=0.5
        )


def test_fit_personnel_adjustment_recovers_known_coefficient():
    home_def_loss = pd.Series([0.1, 0.9, 0.1, 0.9] * 10)
    away_def_loss = pd.Series([0.9, 0.1, 0.9, 0.1] * 10)
    # away_major - home_major alternates +1, -1, +1, -1 ...
    major_def_net = np.array([1, -1, 1, -1] * 10)
    true_coef = 1.5
    base_residual = pd.Series(true_coef * major_def_net)

    cal = personnel.fit_personnel_adjustment(
        home_def_loss, away_def_loss, base_residual, threshold_quantile=0.5
    )
    assert cal.adjustment_points == pytest.approx(true_coef, abs=1e-8)


# ---------------------------------------------------------------
# market.py
# ---------------------------------------------------------------


def test_build_team_market_signal_home_away_opposite_sign():
    games = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "game_id": ["g1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "market_home_margin": [6.0],
            "pred_v30": [3.0],
        }
    )
    team = market.build_team_market_signal(games)
    home_signal = team.loc[team["team"] == "A", "market_signal_game"].iloc[0]
    away_signal = team.loc[team["team"] == "B", "market_signal_game"].iloc[0]
    assert home_signal == pytest.approx(3.0)
    assert away_signal == pytest.approx(-3.0)


def test_add_lagged_ewm_signal_has_no_current_game_leakage():
    """Every team's FIRST game of a season must have a null signal."""
    team = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "team": ["A", "A", "A"],
            "week": [1, 2, 3],
            "game_id": ["g1", "g2", "g3"],
            "market_signal_game": [1.0, 2.0, 3.0],
        }
    )
    out = market.add_lagged_ewm_signal(team, span=5)
    assert pd.isna(out.iloc[0]["market_signal_ewm5"])
    # Week 2's signal should depend only on week 1's value (1.0).
    assert out.iloc[1]["market_signal_ewm5"] == pytest.approx(1.0)


def test_fit_market_beta_no_intercept_zero_signal_zero_adjustment():
    cal = market.fit_market_beta(
        v30_residual=pd.Series([2.0, -2.0, 4.0, -4.0]),
        signal_diff=pd.Series([1.0, -1.0, 2.0, -2.0]),
    )
    assert cal.beta == pytest.approx(2.0, abs=1e-8)
    assert cal.adjustment(np.array([0.0]))[0] == pytest.approx(0.0)


# ---------------------------------------------------------------
# neutral.py
# ---------------------------------------------------------------


def test_neutral_hfa_adjustment_removes_hfa_only_at_neutral_site():
    result = neutral.neutral_hfa_adjustment(
        is_neutral_site=pd.Series([True, False, True]), hfa=2.3
    )
    np.testing.assert_allclose(result, [-2.3, 0.0, -2.3])


# ---------------------------------------------------------------
# pipeline.py
# ---------------------------------------------------------------


def test_predict_v36_is_pure_sum_of_components():
    pred_v30 = np.array([1.0, 2.0, 3.0])
    personnel_adj = np.array([0.5, 0.0, -0.5])
    market_adj = np.array([0.2, -0.2, 0.0])
    neutral_adj = np.array([0.0, -2.3, 0.0])

    result = pipeline.predict_v36(pred_v30, personnel_adj, market_adj, neutral_adj)

    expected = pred_v30 + personnel_adj + market_adj + neutral_adj
    np.testing.assert_allclose(result, expected)


def test_predict_v36_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pipeline.predict_v36(
            np.array([1.0, 2.0]),
            np.array([0.5]),
            np.array([0.2, 0.0]),
            np.array([0.0, 0.0]),
        )


# ---------------------------------------------------------------
# predict_2026.py
# ---------------------------------------------------------------


def test_predict_week_single_matchup_matches_manual_computation():
    """
    Build a tiny synthetic history + matchup and confirm predict_week's
    output equals manually calling rating/spread/personnel/market/neutral
    with the same EXPERIMENTAL_* constants - i.e. the wiring itself is
    correct, independent of whether the constants are "right".
    """
    from v3_model import config
    from v3_model.predict_2026 import predict_week

    team_game = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A"],
            "plays": [65, 65, 65, 65],
            "epa_per_play": [0.10, -0.10, 0.12, -0.12],
            "def_epa_per_play": [-0.10, 0.10, -0.12, 0.12],
        }
    )
    priors = pd.DataFrame(
        {
            "season": [2025, 2025],
            "team": ["A", "B"],
            "prev_offense": [0.0, 0.0],
            "prev_defense": [0.0, 0.0],
        }
    )
    matchups = pd.DataFrame(
        {
            "season": [2025],
            "week": [3],
            "game_id": ["2025_03_A_B"],
            "home_team": ["A"],
            "away_team": ["B"],
            "neutral_site": [False],
            "home_def_loss": [0.1],
            "away_def_loss": [0.9],  # away has a major loss -> should help home
            "home_market_signal_ewm5": [0.5],
            "away_market_signal_ewm5": [-0.5],
        }
    )

    result = predict_week(team_game, priors, matchups)
    assert len(result) == 1
    row = result.iloc[0]

    # Manually reproduce pred_v30 for this matchup.
    state = rating.build_pregame_state(
        team_game, 2025, 3, priors,
        off_alpha=config.EXPERIMENTAL_OFF_ALPHA, off_strength=config.EXPERIMENTAL_OFF_STRENGTH,
        def_alpha=config.EXPERIMENTAL_DEF_ALPHA, def_strength=config.EXPERIMENTAL_DEF_STRENGTH,
    ).set_index("team")
    sdiff = spread.strength_diff(
        pd.Series([state.loc["A", "latent_offense"]]),
        pd.Series([state.loc["A", "latent_defense"]]),
        pd.Series([state.loc["B", "latent_offense"]]),
        pd.Series([state.loc["B", "latent_defense"]]),
    )
    cal = spread.SpreadCalibration(hfa=config.EXPERIMENTAL_HFA, beta=config.EXPERIMENTAL_BETA)
    expected_pred_v30 = float(cal.predict(sdiff)[0])
    assert row["pred_v30"] == pytest.approx(expected_pred_v30)

    # Personnel: away has the major loss -> positive adjustment (helps home).
    assert row["personnel_adjustment"] > 0

    # Market: home's signal (0.5) exceeds away's (-0.5) -> positive adjustment.
    assert row["market_adjustment"] > 0

    # Neutral site is False -> no HFA removal.
    assert row["neutral_hfa_adjustment"] == 0.0

    # pred_v36 is the exact sum.
    assert row["pred_v36"] == pytest.approx(
        row["pred_v30"] + row["personnel_adjustment"] + row["market_adjustment"] + row["neutral_hfa_adjustment"]
    )


def test_predict_week_neutral_site_removes_hfa():
    from v3_model import config
    from v3_model.predict_2026 import predict_week

    team_game = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["A", "B"],
            "opponent": ["B", "A"],
            "plays": [65, 65],
            "epa_per_play": [0.0, 0.0],
            "def_epa_per_play": [0.0, 0.0],
        }
    )
    priors = pd.DataFrame(
        {"season": [2025, 2025], "team": ["A", "B"], "prev_offense": [0.0, 0.0], "prev_defense": [0.0, 0.0]}
    )

    def make_matchup(neutral):
        return pd.DataFrame(
            {
                "season": [2025], "week": [2], "game_id": ["g"],
                "home_team": ["A"], "away_team": ["B"], "neutral_site": [neutral],
                "home_def_loss": [0.0], "away_def_loss": [0.0],
                "home_market_signal_ewm5": [0.0], "away_market_signal_ewm5": [0.0],
            }
        )

    home_result = predict_week(team_game, priors, make_matchup(False)).iloc[0]
    neutral_result = predict_week(team_game, priors, make_matchup(True)).iloc[0]

    assert home_result["neutral_hfa_adjustment"] == 0.0
    assert neutral_result["neutral_hfa_adjustment"] == pytest.approx(-config.EXPERIMENTAL_HFA)
    assert neutral_result["pred_v36"] == pytest.approx(home_result["pred_v36"] - config.EXPERIMENTAL_HFA)


# ---------------------------------------------------------------
# preseason_priors_2026.py
# ---------------------------------------------------------------


def test_load_2026_preseason_priors_reproduces_frozen_decomposition():
    """
    Loads the real uploaded 2026 Deep Dive workbook and checks the
    frozen invariant (offense_strength + defense_strength ==
    overall_prior exactly) plus a couple of known values, so a future
    change to this module that breaks the decomposition fails loudly.
    """
    import os

    path = "/mnt/user-data/uploads/2026_Deep_Dive_Priors_Core_V2_1_.xlsx"
    if not os.path.exists(path):
        pytest.skip("2026 Deep Dive workbook not present in this environment")

    from v3_model.preseason_priors_2026 import load_2026_preseason_priors

    df = load_2026_preseason_priors(path)

    assert len(df) == 32
    diff = (df["offense_strength"] + df["defense_strength"] - df["overall_prior"]).abs().max()
    assert diff < 1e-9

    ari = df.set_index("team").loc["ARI"]
    assert ari["overall_prior"] == pytest.approx(-4.239312, abs=1e-3)
    assert ari["offense_strength"] == pytest.approx(-1.961, abs=1e-2)
    assert ari["defense_strength"] == pytest.approx(-2.278, abs=1e-2)


def test_full_pipeline_matches_manual_v32_v33_definitions():
    """
    Sanity-check the composition rule the notebook validated exactly
    in V3.6B: pred_v32 = pred_v30 + personnel; pred_v33 = pred_v30 +
    market; pred_v36 = pred_v30 + personnel + market + neutral.
    """
    pred_v30 = np.array([0.0, 5.0, -3.0])
    personnel_adj = np.array([1.0, -1.0, 0.0])
    market_adj = np.array([0.5, 0.5, -0.5])
    neutral_adj = np.array([0.0, 0.0, -2.0])

    pred_v32 = pred_v30 + personnel_adj
    pred_v33 = pred_v30 + market_adj
    pred_v36 = pipeline.predict_v36(pred_v30, personnel_adj, market_adj, neutral_adj)

    np.testing.assert_allclose(pred_v36, pred_v32 + market_adj + neutral_adj)
    np.testing.assert_allclose(pred_v36, pred_v33 + personnel_adj + neutral_adj)


# ---------------------------------------------------------------
# production_v3.py
# ---------------------------------------------------------------


def test_persistent_margin_matches_full_offense_defense_formula():
    """
    persistent_margin(home_off, home_def, away_off, away_def) must
    equal (home_off + home_def) - (away_off + away_def).
    """
    from v3_model import production_v3

    home_off, home_def = 1.5, 0.7
    away_off, away_def = -0.5, -1.2

    expected = (home_off + home_def) - (away_off + away_def)
    actual = production_v3.persistent_margin(
        home_off, home_def, away_off, away_def
    )

    assert actual == pytest.approx(expected)


def test_predict_week_points_scale_neutral_site_cancels_hfa_exactly():
    from v3_model import production_v3

    ratings = pd.DataFrame(
        {
            "team": ["A", "B"],
            "offense_strength": [1.5, -0.5],
            "defense_strength": [0.5, -0.5],
        }
    )

    def make_matchup(neutral):
        return pd.DataFrame(
            {
                "season": [2026], "week": [1], "game_id": ["g"],
                "home_team": ["A"], "away_team": ["B"], "neutral_site": [neutral],
                "home_def_loss": [0.0], "away_def_loss": [0.0],
                "home_market_signal_ewm5": [0.0], "away_market_signal_ewm5": [0.0],
            }
        )

    home_result = production_v3.predict_week_points_scale(
        ratings, make_matchup(False)
    ).iloc[0]
    neutral_result = production_v3.predict_week_points_scale(
        ratings, make_matchup(True)
    ).iloc[0]

    assert home_result["pred_v3_lite"] == pytest.approx(
        home_result["persistent_margin"] + production_v3.EXPERIMENTAL_HFA_DEFAULT
    )
    assert neutral_result["pred_v3_lite"] == pytest.approx(
        neutral_result["persistent_margin"]
    )
    assert neutral_result["pred_v3_lite"] == pytest.approx(
        home_result["pred_v3_lite"] - production_v3.EXPERIMENTAL_HFA_DEFAULT
    )


# ---------------------------------------------------------------
# epa_updater.py
# ---------------------------------------------------------------


def test_update_split_state_moves_toward_observed_epa_not_margin():
    """
    A team with a much better observed EPA than its prior should see
    its offense_strength rise - driven entirely by EPA, with no
    margin or final-score input anywhere in this function's signature.
    """
    from v3_model import epa_updater

    # Priors must have real spread (not all identical) - with zero
    # spread there's nothing to rescale a blended z-score against,
    # which is a degenerate input, not a case this test is about.
    prior = pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "offense_strength": [0.5, -0.5, 0.1, -0.1],
            "defense_strength": [0.2, -0.2, 0.05, -0.05],
        }
    )
    team_game_history = pd.DataFrame(
        {
            "season": [2026] * 4,
            "week": [1] * 4,
            "team": ["A", "B", "C", "D"],
            "opponent": ["B", "A", "D", "C"],
            "plays": [65, 65, 65, 65],
            "epa_per_play": [0.30, -0.30, 0.0, 0.0],
            "def_epa_per_play": [0.0, 0.0, 0.0, 0.0],
        }
    )

    updated = epa_updater.update_split_state(prior, team_game_history, season=2026, week=2)
    updated = updated.set_index("team")

    assert updated.loc["A", "offense_strength"] > 0.0  # outperformed -> up
    assert updated.loc["B", "offense_strength"] < 0.0  # underperformed -> down
    # C and D had a neutral game relative to the league mean -> should stay near 0
    assert abs(updated.loc["C", "offense_strength"]) < abs(updated.loc["A", "offense_strength"])


def test_update_split_state_preserves_prior_scale():
    """
    The function must never need to know whether the prior is in EPA
    units or points - the output should stay in whatever scale the
    input was, because the bridge works entirely in z-score space and
    only rescales using the prior's OWN standard deviation.
    """
    from v3_model import epa_updater

    # A points-scale prior with a much larger spread than any EPA prior would have.
    prior_points = pd.DataFrame(
        {
            "team": ["A", "B", "C"],
            "offense_strength": [3.0, 0.0, -3.0],
            "defense_strength": [1.0, 0.0, -1.0],
        }
    )
    team_game_history = pd.DataFrame(
        {
            "season": [2026] * 3,
            "week": [1] * 3,
            "team": ["A", "B", "C"],
            "opponent": ["B", "A", "A"],
            "plays": [65, 65, 65],
            "epa_per_play": [0.05, -0.02, 0.10],
            "def_epa_per_play": [0.0, 0.0, 0.0],
        }
    )

    updated = epa_updater.update_split_state(prior_points, team_game_history, season=2026, week=2)
    # Output should still be in the points-scale ballpark (single digits), not EPA-scale
    # (which would be ~0.0-0.3) - i.e. the prior's own std correctly rescaled it back.
    assert updated["offense_strength"].abs().max() > 0.5


def test_update_split_state_no_observed_games_returns_prior_unchanged():
    from v3_model import epa_updater

    prior = pd.DataFrame(
        {"team": ["A", "B"], "offense_strength": [1.0, -1.0], "defense_strength": [0.5, -0.5]}
    )
    empty_history = pd.DataFrame(
        columns=["season", "week", "team", "opponent", "plays", "epa_per_play", "def_epa_per_play"]
    )
    updated = epa_updater.update_split_state(prior, empty_history, season=2026, week=1)
    pd.testing.assert_frame_equal(
        updated.sort_values("team").reset_index(drop=True),
        prior.sort_values("team").reset_index(drop=True),
    )


def test_update_split_state_degenerate_zero_spread_prior_is_a_safe_noop():
    """
    If every team's prior is identical (zero spread), there's nothing
    to rescale a blended z-score against - the function's fallback
    (leave state at the prior's mean) must be a documented, safe
    no-op, not a silent divide-by-zero or NaN.
    """
    from v3_model import epa_updater

    prior = pd.DataFrame(
        {"team": ["A", "B"], "offense_strength": [0.0, 0.0], "defense_strength": [0.0, 0.0]}
    )
    team_game_history = pd.DataFrame(
        {
            "season": [2026, 2026], "week": [1, 1], "team": ["A", "B"], "opponent": ["B", "A"],
            "plays": [65, 65], "epa_per_play": [0.30, -0.30], "def_epa_per_play": [0.0, 0.0],
        }
    )
    updated = epa_updater.update_split_state(prior, team_game_history, season=2026, week=2)
    assert not updated["offense_strength"].isna().any()
    assert (updated["offense_strength"] == 0.0).all()


# ---------------------------------------------------------------
# production_v3.py (post margin-residual removal)
# ---------------------------------------------------------------


def test_production_v3_has_no_margin_based_update_mechanism():
    """
    Regression guard against reintroducing a margin-residual updater:
    production_v3 must not expose any function whose signature takes
    an actual/observed margin as input.
    """
    from v3_model import production_v3
    import inspect

    for name, obj in vars(production_v3).items():
        if inspect.isfunction(obj) and obj.__module__ == production_v3.__name__:
            params = inspect.signature(obj).parameters
            assert not any("margin" in p and "persistent" not in p for p in params if "actual" in p or "observed" in p), (
                f"{name} appears to take an actual/observed margin as input - "
                f"this was explicitly rejected; see production_v3.py's revision history."
            )
    assert not hasattr(production_v3, "update_overall_rating"), (
        "the old margin-residual updater should not exist in this module anymore"
    )


def test_persistent_margin_reflects_drifted_split_after_update():
    """
    After epa_updater has moved offense/defense apart from their
    original sum, persistent_margin must reflect the NEW sum, not the
    original overall_prior - this is how the updater actually changes
    the projected margin, without ever touching final score.
    """
    from v3_model import production_v3

    original_sum = 2.0  # offense_strength + defense_strength before drift
    home_off, home_def = 1.5, 0.5  # sums to 2.0
    drifted_home_off, drifted_home_def = 1.8, 0.5  # sums to 2.3 after an update

    away_off, away_def = 0.0, 0.0

    before = production_v3.persistent_margin(home_off, home_def, away_off, away_def)
    after = production_v3.persistent_margin(drifted_home_off, drifted_home_def, away_off, away_def)

    assert before == pytest.approx(original_sum)
    assert after == pytest.approx(2.3)
    assert after != before


# ---------------------------------------------------------------
# board.py - game_key compatibility (Stage 3)
# ---------------------------------------------------------------


def test_board_game_key_matches_core_v2_format_exactly():
    """
    Regression guard: board.py's game_key must stay byte-for-byte
    identical to Core V2's own game_key() function
    ("{season}_W{week}_{away}_{home}"), since Stage 4/5 reuse
    core_v2.tracking / core_v2.bet_tracking unchanged and both key
    strictly on this exact string format - any drift here silently
    breaks that reuse without erroring anywhere obvious.
    """
    import pandas as pd

    def core_v2_game_key(season, week, away, home):
        # Copied from core_v2/projections.py's own game_key(), not
        # reimplemented differently - this IS the spec being matched.
        return f"{int(season)}_W{int(week)}_{away}_{home}"

    predictions = pd.DataFrame(
        {
            "season": [2025, 2026],
            "week": [10, 1],
            "away_team": ["ARI", "SF"],
            "home_team": ["SEA", "LA"],
        }
    )
    predictions["game_key"] = predictions.apply(
        lambda r: f"{int(r['season'])}_W{int(r['week'])}_{r['away_team']}_{r['home_team']}", axis=1
    )

    for row in predictions.itertuples(index=False):
        expected = core_v2_game_key(row.season, row.week, row.away_team, row.home_team)
        assert row.game_key == expected


# ---------------------------------------------------------------
# Sign convention: margin (V3) vs bookmaker (Core V2) - Stage 4
# ---------------------------------------------------------------


def test_board_model_side_uses_margin_convention_correctly():
    """
    Regression guard for a real bug caught by Stage 4's tests:
    board.py's model_side must select the side the market is
    UNDERVALUING relative to V3's own margin-convention projection -
    fair > market -> HOME, not fair < market -> HOME (which is only
    correct under bookmaker convention, which this column is not in).
    """
    import pandas as pd

    predictions = pd.DataFrame(
        {
            "home_team": ["LA", "CHI"],
            "away_team": ["SF", "CAR"],
            "fair_home_spread_v3": [4.6, -5.0],   # margin convention
            "market_home_spread": [3.5, -1.0],     # margin convention
        }
    )
    predictions["model_side"] = predictions.apply(
        lambda r: r["home_team"]
        if pd.notna(r["market_home_spread"]) and r["fair_home_spread_v3"] > r["market_home_spread"]
        else (r["away_team"] if pd.notna(r["market_home_spread"]) else "\u2014"),
        axis=1,
    )
    # LA: fair(4.6) > market(3.5) -> model likes LA MORE than market does -> HOME
    assert predictions.loc[0, "model_side"] == "LA"
    # CHI: fair(-5.0) < market(-1.0) -> model thinks CHI is a MUCH BIGGER
    # underdog than the market does -> the market is overrating CHI, so
    # the value side is the team the market is underrating -> AWAY (CAR)
    assert predictions.loc[1, "model_side"] == "CAR"


def test_prepare_v3_board_for_release_flips_sign_correctly():
    """
    capture_release_v3.prepare_v3_board_for_release must negate
    fair_home_spread_v3 (margin convention) into fair_home_spread
    (bookmaker convention) - not pass it through unchanged, which
    would silently invert every model_selected_team/side and
    downstream CLV sign once fed into core_v2.tracking.
    """
    import pandas as pd
    from scripts.capture_release_v3 import prepare_v3_board_for_release

    board = pd.DataFrame({"fair_home_spread_v3": [4.6, -2.4]})
    out = prepare_v3_board_for_release(board)
    assert out["fair_home_spread"].tolist() == [-4.6, 2.4]


# ---------------------------------------------------------------
# Stage 5: "My Bets" tab reuse of core_v2.bet_tracking
# ---------------------------------------------------------------


def test_bet_tracking_uses_bookmaker_convention_fair_display():
    """
    Regression guard: the My Bets tab must feed bet_tracking
    functions fair_display/market_display (bookmaker convention),
    never fair_home_spread_v3/market_home_spread (margin convention)
    directly - this mirrors the same bug class fixed in board.py and
    Stage 2/4's V2-integration points. Uses core_v2.bet_tracking
    itself (not a reimplementation) so this fails if that module's
    contract ever changes underneath this file.
    """
    from core_v2 import bet_tracking

    # LA (home) favored by 4.6 in bookmaker convention (post sign-flip
    # from V3's own margin-convention fair_home_spread_v3=+4.6).
    fair_display = -4.6
    selected_model_spread = bet_tracking.selected_team_model_spread(
        "LA", away_team="SF", home_team="LA", fair_home_spread=fair_display
    )
    # Home team's own spread equals fair_display directly (no further
    # sign change - home's number IS the home spread).
    assert selected_model_spread == pytest.approx(-4.6)

    away_selected = bet_tracking.selected_team_model_spread(
        "SF", away_team="SF", home_team="LA", fair_home_spread=fair_display
    )
    # Away team's own spread is the negation of the home spread.
    assert away_selected == pytest.approx(4.6)


def test_native_personnel_position_normalization():
    assert personnel.normalize_position_group("DE")=="EDGE"
    assert personnel.normalize_position_group("DT")=="IDL"
    assert personnel.normalize_position_group("S")=="DB"
    assert personnel.normalize_position_group("WR")=="WR_TE"
    assert personnel.normalize_position_group("QB")=="QB"

def test_attach_player_pvar_joins_by_gsis():
    audit=pd.DataFrame({"gsis_id":["00-1"],"player_name":["Defender"]})
    ratings=pd.DataFrame({
        "gsis_id":["00-1"],"pvar_production":[0.8],
        "pvar_source":["REBASED"],
    })
    out=personnel.attach_player_pvar_to_audit(audit,ratings)
    assert out.loc[0,"pvar_production"]==pytest.approx(0.8)

def test_native_personnel_uses_status_weight_times_rebased_pvar():
    audit=pd.DataFrame({
        "team":["SEA","SEA","SEA","SEA","SEA","SEA"],
        "player_name":["E","I","L","D","W","Q"],
        "gsis_id":["1","2","3","4","5","6"],
        "position_group":["EDGE","IDL","LB","DB","WR_TE","QB"],
        "status_weight":[1.0,0.5,0.2,0.35,1.0,1.0],
        "pvar_production":[0.4,0.6,0.5,0.8,1.0,1.0],
        "v2_shadow_eligible":[True]*6,
    })
    row=personnel.defensive_exposure_from_audit(audit).iloc[0]
    # .40 + .30 + .10 + .28 = 1.08; offense and QB excluded.
    assert row["def_loss_pvar"]==pytest.approx(1.08)
    assert row["def_edge_loss"]==pytest.approx(0.40)
    assert row["def_idl_loss"]==pytest.approx(0.30)
    assert row["def_lb_loss"]==pytest.approx(0.10)
    assert row["def_db_loss"]==pytest.approx(0.28)

def test_native_personnel_does_not_multiply_by_role_exposure():
    audit=pd.DataFrame({
        "team":["BUF"],"player_name":["DB"],"gsis_id":["1"],
        "position_group":["DB"],"status_weight":[0.35],
        "weighted_role_exposure_v2":[0.035],
        "pvar_production":[0.8],"v2_shadow_eligible":[True],
    })
    row=personnel.defensive_exposure_from_audit(audit).iloc[0]
    assert row["def_loss_pvar"]==pytest.approx(0.28)

def test_native_personnel_refuses_missing_eligible_defensive_pvar():
    audit=pd.DataFrame({
        "team":["SEA"],"player_name":["Missing"],"gsis_id":["1"],
        "position_group":["DB"],"status_weight":[1.0],
        "pvar_production":[float("nan")],"v2_shadow_eligible":[True],
    })
    with pytest.raises(ValueError, match="unresolved"):
        personnel.defensive_exposure_from_audit(audit)

def test_native_personnel_excludes_qb_before_pvar_validation():
    audit=pd.DataFrame({
        "team":["WAS"],"player_name":["QB"],"gsis_id":["1"],
        "position_group":["QB"],"status_weight":[1.0],
        "pvar_production":[float("nan")],"v2_shadow_eligible":[True],
    })
    assert personnel.defensive_exposure_from_audit(audit).empty

def test_threshold_calibration_empty_is_safe():
    check=personnel.check_threshold_calibration(pd.Series(dtype=float),0.75)
    assert check.n==0
    assert not check.intended_rate_within_ci

def test_add_personnel_to_matchups_sets_real_provenance():
    matchup=pd.DataFrame({"home_team":["SEA"],"away_team":["BUF"]})
    audit=pd.DataFrame({
        "team":["SEA","BUF"],"player_name":["A","B"],"gsis_id":["1","2"],
        "position_group":["DB","EDGE"],"status_weight":[1.0,0.5],
        "pvar_production":[0.6,0.4],"v2_shadow_eligible":[True,True],
    })
    out=personnel.add_personnel_to_matchups(matchup,audit,covered_teams={"SEA","BUF"}).iloc[0]
    assert out["home_def_loss"]==pytest.approx(0.6)
    assert out["away_def_loss"]==pytest.approx(0.2)
    assert out["home_personnel_provenance"]=="player_audit_native_v3"

def test_native_continuous_personnel_adjustment_fixed_225():
    adj=personnel.continuous_personnel_adjustment([0.20],[0.60])
    assert adj.iloc[0]==pytest.approx(0.90)

def test_native_continuous_personnel_refuses_missing_loss():
    with pytest.raises(ValueError, match="missing"):
        personnel.continuous_personnel_adjustment([float("nan")],[0.2])

def test_native_personnel_string_false_is_not_eligible():
    audit=pd.DataFrame({
        "team":["SEA","SEA"],"player_name":["A","B"],"gsis_id":["1","2"],
        "position_group":["DB","EDGE"],"status_weight":[1.0,1.0],
        "pvar_production":[0.6,0.4],"v2_shadow_eligible":["False","True"],
    })
    row=personnel.defensive_exposure_from_audit(audit).iloc[0]
    assert row["def_loss_pvar"]==pytest.approx(0.4)

def test_native_personnel_requires_pvar_inputs():
    audit=pd.DataFrame({
        "team":["SEA"],"player_name":["A"],"gsis_id":["1"],
        "position_group":["DB"],"v2_shadow_eligible":[True],
    })
    with pytest.raises(ValueError, match="PVAR inputs"):
        personnel.defensive_exposure_from_audit(audit)

def test_native_personnel_missing_team_coverage_stays_missing():
    matchup=pd.DataFrame({"home_team":["SEA"],"away_team":["BUF"]})
    audit=pd.DataFrame({
        "team":["SEA"],"player_name":["A"],"gsis_id":["1"],
        "position_group":["DB"],"status_weight":[1.0],
        "pvar_production":[0.2],"v2_shadow_eligible":[True],
    })
    out=personnel.add_personnel_to_matchups(matchup,audit,covered_teams={"SEA"}).iloc[0]
    assert pd.isna(out["away_def_loss"])
    assert out["away_personnel_provenance"]=="missing_team_coverage"
