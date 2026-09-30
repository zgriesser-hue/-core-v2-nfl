"""Create a review queue from official NFL news and current-week injury reports.

This scanner never edits the reviewed availability ledger or a projection.
Headlines and practice designations are leads, not confirmed OUT decisions.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from core_v2.personnel import fetch_nfl_injury_report, normalize_player_name
from v3_model.personnel import normalize_position_group

NEWS_URL = "https://www.nfl.com/news"
INJURY_WORDS = re.compile(
    r"\b(injur(?:y|ed|ies)|out|miss(?:es|ing)?|ruled|doubtful|questionable|"
    r"concussion|surgery|fractur(?:e|ed)|torn|return(?:s|ing)?|inactive)\b", re.I
)
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}
COLUMNS = ["season", "week", "team", "player", "gsis_id", "position_group",
           "source_type", "signal", "headline", "source_url", "first_seen_utc",
           "pvar_production", "potential_def_points", "review_status"]


def parse_news_links(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    articles = {}
    for link in soup.find_all("a", href=True):
        url = urljoin(NEWS_URL, link["href"]).split("?")[0]
        parsed = urlparse(url)
        if parsed.hostname not in {"www.nfl.com", "nfl.com"} or not parsed.path.startswith("/news/"):
            continue
        title = " ".join(link.get_text(" ", strip=True).split())
        if len(title) < 16 or not INJURY_WORDS.search(title):
            continue
        if len(title) > len(articles.get(url, "")):
            articles[url] = title
    return sorted(articles.items())


def recent_news(news: list[tuple[str, str]], earliest: pd.Timestamp) -> list[tuple[str, str]]:
    """Exclude dated headlines from before the Sunday that starts this NFL week."""
    dated = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b")
    kept = []
    for url, title in news:
        dates = dated.findall(title)
        if dates:
            published = pd.to_datetime(dates[-1], errors="coerce")
            if pd.notna(published) and published.normalize() < earliest.normalize():
                continue
        kept.append((url, title))
    return kept


def _player_in_headline(title: str, team: str, name: str, team_names: dict[str, str]) -> bool:
    normalized = normalize_player_name(name)
    title_key = normalize_player_name(title)
    if normalized and re.search(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", title_key):
        return True
    surname = normalized.split()[-1] if normalized else ""
    nickname = team_names.get(team, "")
    return bool(len(surname) >= 4 and nickname
                and re.search(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", title_key)
                and re.search(r"\b" + re.escape(nickname) + r"\b", title, re.I))


def build_candidates(news: list[tuple[str, str]], report: pd.DataFrame,
                     ratings: pd.DataFrame, teams: set[str], season: int,
                     week: int, observed_at: str) -> pd.DataFrame:
    roster = ratings.loc[ratings.team.isin(teams), [
        "team", "player_name", "gsis_id", "model_position_v2", "pvar_production"
    ]].copy()
    roster = roster.rename(columns={"player_name": "player"})
    if roster.duplicated(["team", "player"]).any() or roster.gsis_id.duplicated().any():
        raise ValueError("Frozen PVAR roster has ambiguous player identity")
    roster["position_group"] = roster.model_position_v2.map(normalize_position_group)
    roster["pvar_production"] = pd.to_numeric(roster.pvar_production, errors="coerce")
    records = []
    for url, title in news:
        if not INJURY_WORDS.search(title):
            continue
        matches = roster.loc[roster.apply(
            lambda player: _player_in_headline(title, player.team, player.player, TEAM_NAMES), axis=1
        )]
        # Avoid guessing which player an ambiguous headline names.
        if len(matches) != 1:
            continue
        player = matches.iloc[0]
        records.append(dict(team=player.team, player=player.player,
                            gsis_id=player.gsis_id, position_group=player.position_group,
                            pvar_production=player.pvar_production,
                            source_type="NFL_NEWS_HEADLINE", signal="REVIEW_HEADLINE",
                            headline=title, source_url=url))
    if not report.empty:
        for row in report.itertuples(index=False):
            if row.team not in teams:
                continue
            status = str(row.game_status).strip().lower()
            practice = str(row.practice_status).strip().lower()
            if status not in {"out", "doubtful", "questionable"} and "did not participate" not in practice:
                continue
            matches = roster.loc[roster.team.eq(row.team) &
                                 roster.player.map(normalize_player_name).eq(
                                     normalize_player_name(row.player_name))]
            if len(matches) != 1:
                continue
            player = matches.iloc[0]
            signal = ("OFFICIAL_OUT" if status == "out" else
                      "OFFICIAL_DOUBTFUL" if status == "doubtful" else
                      "OFFICIAL_QUESTIONABLE" if status == "questionable" else
                      "PRACTICE_DNP")
            records.append(dict(team=player.team, player=player.player,
                                gsis_id=player.gsis_id, position_group=player.position_group,
                                pvar_production=player.pvar_production,
                                source_type="NFL_INJURY_REPORT", signal=signal,
                                headline=f"{row.player_name}: {signal.replace('_', ' ')}",
                                source_url=str(row.source_url)))
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    found = pd.DataFrame(records)
    found.insert(0, "season", season)
    found.insert(1, "week", week)
    found["first_seen_utc"] = observed_at
    found["potential_def_points"] = found.apply(
        lambda row: 2.25 * row.pvar_production
        if row.position_group in {"EDGE", "IDL", "LB", "DB"} and pd.notna(row.pvar_production)
        else pd.NA, axis=1
    )
    found["review_status"] = "PENDING_REVIEW"
    return found[COLUMNS].drop_duplicates(["season", "week", "gsis_id", "source_url", "signal"])


def merge_queue(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "gsis_id", "source_url", "signal"]
    if old.empty:
        return new.sort_values(keys).reset_index(drop=True)
    if old.duplicated(keys).any():
        raise ValueError("Existing review queue has duplicate source/player signals")
    old_keys = set(map(tuple, old[keys].itertuples(index=False, name=None)))
    fresh = new.loc[~new[keys].apply(tuple, axis=1).isin(old_keys)]
    return pd.concat([old[COLUMNS], fresh[COLUMNS]], ignore_index=True).sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--availability-ledger", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule = pd.read_csv(args.schedule)
    games = schedule.loc[schedule.season.eq(args.season) & schedule.week.eq(args.week)]
    teams = set(games.away_team) | set(games.home_team)
    if games.empty or len(teams) != 2 * len(games):
        raise ValueError("Expected unique teams in the target-week schedule")
    response = requests.get(NEWS_URL, headers={"User-Agent": "Mozilla/5.0 NFL injury research"},
                            timeout=30)
    response.raise_for_status()
    news = parse_news_links(response.text)
    first_game = pd.to_datetime(games.gameday, errors="raise").min().normalize()
    earliest = first_game - pd.Timedelta(days=first_game.weekday() + 1)
    news = recent_news(news, earliest)
    if not news:
        raise ValueError("Official NFL news index returned no injury headlines; source may have changed")
    try:
        report, _ = fetch_nfl_injury_report(args.season, args.week, teams)
    except ValueError as exc:
        if "refusing to reuse potentially stale prior-week designations" not in str(exc):
            raise
        report = pd.DataFrame()
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ratings = pd.read_csv(args.ratings, dtype={"gsis_id": "string"})
    fresh = build_candidates(news, report, ratings, teams, args.season, args.week, observed)
    old = pd.read_csv(args.out, dtype={"gsis_id": "string"}) if args.out.exists() else pd.DataFrame(columns=COLUMNS)
    if not old.empty:
        old_news = old.loc[old.source_type.eq("NFL_NEWS_HEADLINE")]
        still_recent = set(url for url, _ in recent_news(
            list(zip(old_news.source_url, old_news.headline)), earliest
        ))
        old = old.loc[~old.source_type.eq("NFL_NEWS_HEADLINE")
                      | old.source_url.isin(still_recent)].copy()
    combined = merge_queue(old, fresh)
    if args.availability_ledger and args.availability_ledger.exists() and not combined.empty:
        reviewed = pd.read_csv(args.availability_ledger)
        reviewed = reviewed.loc[reviewed.season.eq(args.season) & reviewed.week.eq(args.week)]
        reviewed_keys = set(zip(reviewed.team, reviewed.player))
        matched = combined.apply(lambda row: (row.team, row.player) in reviewed_keys
                                 and row.season == args.season and row.week == args.week, axis=1)
        combined.loc[matched, "review_status"] = "REVIEWED_IN_LEDGER"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Injury monitor: {len(news)} headlines; {len(fresh)} matching signals; "
          f"{len(combined) - len(old)} new review items for Week {args.week}")


if __name__ == "__main__":
    main()
