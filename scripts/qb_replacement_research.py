"""Pregame QB replacement research; never changes production projections."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
PBP_COLUMNS = ["season", "week", "season_type", "game_id", "posteam", "defteam",
               "passer_player_id", "passer_player_name", "rusher_player_id",
               "rusher_player_name", "pass_attempt", "sack", "qb_scramble", "qb_epa", "epa"]


def load_qb_games(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        with urlopen(PBP_URL.format(season=season), timeout=120) as response:
            payload = response.read()
        pbp = pd.read_csv(BytesIO(payload), compression="gzip",
                          usecols=PBP_COLUMNS, low_memory=False)
        frames.append(build_qb_games(pbp))
        print(f"{season}: {len(frames[-1])} QB-game rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def build_qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.loc[pbp.season_type.eq("REG")].copy()
    for col in ("season", "week", "pass_attempt", "sack", "qb_scramble", "qb_epa", "epa"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    scramble = x.qb_scramble.eq(1)
    x["qb_id"] = x.passer_player_id.where(~scramble, x.rusher_player_id)
    x["qb_name"] = x.passer_player_name.where(~scramble, x.rusher_player_name)
    x["value"] = x.qb_epa.where(~scramble, x.epa)
    x = x.loc[(x.pass_attempt.eq(1) | x.sack.eq(1) | scramble)
              & x.qb_id.notna() & x.value.notna() & x.posteam.notna()
              & x.defteam.notna() & x.game_id.notna()].copy()
    x = x.groupby(["season", "week", "game_id", "posteam", "defteam", "qb_id", "qb_name"],
                  as_index=False).agg(dropbacks=("value", "size"), epa_sum=("value", "sum"))
    return x.rename(columns={"posteam": "team", "defteam": "opponent"})


def qb_abilities(history: pd.DataFrame, prior_db: float = 200,
                 defense_prior_db: float = 350) -> pd.Series:
    """Opponent-adjusted EPA/dropback, shrunk toward the historical mean."""
    if history.empty:
        return pd.Series(dtype=float)
    h = history.copy()
    h["qb_id"] = h.qb_id.astype(str)
    h["rate"] = h.epa_sum / h.dropbacks
    mu = h.epa_sum.sum() / h.dropbacks.sum()
    qb_n = h.groupby("qb_id").dropbacks.sum()
    def_n = h.groupby("opponent").dropbacks.sum()
    ability = pd.Series(0., index=qb_n.index)
    defense = pd.Series(0., index=def_n.index)
    for _ in range(20):
        qb_value = h.rate - mu + h.opponent.map(defense).fillna(0.)
        numerator = (qb_value * h.dropbacks).groupby(h.qb_id).sum()
        ability = numerator.reindex(qb_n.index, fill_value=0.) / (qb_n + prior_db)
        def_value = mu + h.qb_id.map(ability).fillna(0.) - h.rate
        numerator = (def_value * h.dropbacks).groupby(h.opponent).sum()
        defense = numerator.reindex(def_n.index, fill_value=0.) / (def_n + defense_prior_db)
    return ability


def pregame_switch_signals(qb_games: pd.DataFrame, control: pd.DataFrame,
                           prior_db: float = 200) -> pd.DataFrame:
    """Use current starter identity, never current-game EPA, for each forecast."""
    games = qb_games.sort_values(["season", "week", "game_id", "team", "dropbacks"],
                                 ascending=[True, True, True, True, False]).copy()
    games["team_db"] = games.groupby(["game_id", "team"]).dropbacks.transform("sum")
    starters = games.drop_duplicates(["game_id", "team"]).copy()
    starters = starters.loc[starters.dropbacks.ge(10)
                            & starters.dropbacks.div(starters.team_db).ge(.5)].copy()
    starters = starters.sort_values(["season", "team", "week", "game_id"])
    starters["previous_qb_id"] = starters.groupby(["season", "team"]).qb_id.shift(1)
    starters["previous_week"] = starters.groupby(["season", "team"]).week.shift(1)
    # Games with a previous starter in the same season are the cleanest
    # retrospective proxy for an expected-starter replacement.
    starters = starters.loc[starters.previous_qb_id.notna()].copy()
    rows = []
    for (season, week), current in starters.groupby(["season", "week"], sort=True):
        history = qb_games.loc[(qb_games.season.lt(season))
                               | (qb_games.season.eq(season) & qb_games.week.lt(week))]
        ability = qb_abilities(history, prior_db=prior_db)
        cur = current.copy()
        cur["new_ability"] = cur.qb_id.astype(str).map(ability).fillna(0.)
        cur["old_ability"] = cur.previous_qb_id.astype(str).map(ability).fillna(0.)
        cur["qb_change_epa"] = cur.new_ability - cur.old_ability
        rows.append(cur[["season", "week", "team", "game_id", "qb_id", "qb_name",
                         "previous_qb_id", "dropbacks", "team_db", "new_ability",
                         "old_ability", "qb_change_epa"]])
    if not rows:
        raise ValueError("No eligible starter histories were found")
    signals = pd.concat(rows, ignore_index=True)
    home = signals.rename(columns={"team": "home_team", "qb_change_epa": "home_change"})
    away = signals.rename(columns={"team": "away_team", "qb_change_epa": "away_change"})
    joined = control.merge(home, on=["season", "week", "game_id", "home_team"], how="left")
    joined = joined.merge(away, on=["season", "week", "game_id", "away_team"], how="left",
                          suffixes=("_home", "_away"))
    joined["qb_delta_35"] = 35 * (joined.home_change.fillna(0.) - joined.away_change.fillna(0.))
    joined["switch_game"] = (joined.qb_id_home.ne(joined.previous_qb_id_home)
                             | joined.qb_id_away.ne(joined.previous_qb_id_away))
    joined["eligible"] = joined.home_change.notna() & joined.away_change.notna()
    joined["switch_game"] &= joined.eligible
    return joined


def evaluate_loso(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit point mapping on other seasons only; score the held-out season."""
    x = panel.loc[panel.eligible].copy()
    rows = []
    for season in sorted(x.season.unique()):
        train = x.loc[x.season.ne(season) & x.switch_game]
        test = x.loc[x.season.eq(season)]
        if len(train) < 30 or test.empty:
            continue
        residual = train.home_margin - train.pred_v30
        signal = train.qb_delta_35
        # Zero intercept: the control model already estimates the margin.
        coefficient = float((signal * residual).sum() / (signal.pow(2).sum() + 100.))
        for label, subset in (("all", test), ("switch", test.loc[test.switch_game])):
            if subset.empty:
                continue
            base_error = (subset.home_margin - subset.pred_v30).abs()
            adjusted_error = (subset.home_margin - subset.pred_v30
                              - coefficient * subset.qb_delta_35).abs()
            rows.append(dict(season=int(season), segment=label, games=len(subset),
                             coefficient=coefficient, base_mae=base_error.mean(),
                             adjusted_mae=adjusted_error.mean(),
                             improvement=base_error.mean() - adjusted_error.mean()))
    return pd.DataFrame(rows)


def live_pair_estimate(games: pd.DataFrame, old_name: str, new_name: str,
                       team: str, coefficient: float, season: int,
                       before_week: int) -> dict:
    """Shadow point difference for named QBs, using only prior games."""
    history = games.loc[(games.season.lt(season))
                        | (games.season.eq(season) & games.week.lt(before_week))]
    ability = qb_abilities(history)
    players = history.loc[history.team.eq(team)].sort_values(["season", "week"])
    result = {"team": team, "old_qb": old_name, "new_qb": new_name,
              "old_ability": np.nan, "new_ability": np.nan,
              "replacement_points": np.nan, "mapping_coefficient": coefficient}
    for role, name in (("old", old_name), ("new", new_name)):
        matches = players.loc[players.qb_name.str.contains(name, case=False, regex=False, na=False)]
        if matches.empty:
            return result
        qb_id = str(matches.iloc[-1].qb_id)
        result[f"{role}_ability"] = float(ability.get(qb_id, 0.))
    result["replacement_points"] = 35 * coefficient * (result["new_ability"] - result["old_ability"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seasons", default="2017,2018,2019,2020,2021,2022,2023,2024,2025,2026")
    args = parser.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]
    games = load_qb_games(seasons)
    control = pd.read_csv(args.control)
    panel = pregame_switch_signals(games, control)
    folds = evaluate_loso(panel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_dir / "qb_switch_oos_panel.csv", index=False)
    folds.to_csv(args.out_dir / "qb_switch_loso.csv", index=False)
    if folds.empty:
        raise ValueError("Insufficient switch games for leave-one-season-out evaluation")
    print(folds.to_string(index=False), flush=True)
    print("Switch games:", int(panel.switch_game.sum()), "of", len(panel), flush=True)
    current = games.loc[(games.season.lt(2026)) | (games.season.eq(2026) & games.week.lt(5))]
    ability = qb_abilities(current)
    names = games.drop_duplicates("qb_id", keep="last").set_index("qb_id").qb_name
    ratings = ability.rename("qb_epa_per_db").to_frame()
    ratings["qb_name"] = ratings.index.map(names)
    ratings["dropbacks"] = current.groupby("qb_id").dropbacks.sum()
    ratings.reset_index(names="qb_id").to_csv(args.out_dir / "qb_ratings_through_week4.csv", index=False)
    # These are research estimates, not applied to the production board.
    coefficient = float(folds.loc[folds.segment.eq("switch"), "coefficient"].median())
    pair = live_pair_estimate(games, "Jackson", "Huntley", "BAL", coefficient, 2026, 5)
    pd.DataFrame([pair]).to_csv(args.out_dir / "qb_live_pair_shadow.csv", index=False)
    print("BAL shadow replacement:", pair, flush=True)


if __name__ == "__main__":
    main()
