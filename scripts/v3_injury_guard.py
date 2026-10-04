"""Identify material injury inputs that still need a current-week decision."""
from __future__ import annotations

import pandas as pd


def unresolved_injuries(queue: pd.DataFrame, continuing: pd.DataFrame,
                        ledger: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Produce unique team/player review leads; never infer an injury value."""
    reviewed = (set(zip(ledger.team.astype(str), ledger.player.astype(str)))
                if not ledger.empty and {"team", "player"}.issubset(ledger) else set())
    found = []
    if not queue.empty:
        required = {"season", "week", "team", "player", "position_group", "signal", "review_status", "source_url"}
        if not required.issubset(queue):
            raise ValueError("Injury review queue schema incomplete")
        q = queue.loc[queue.season.eq(season) & queue.week.eq(week)
                      & queue.signal.eq("OFFICIAL_OUT")
                      & queue.position_group.eq("QB")]
        for row in q.itertuples(index=False):
            if (row.team, row.player) not in reviewed:
                found.append(dict(team=row.team, player=row.player, reason="QB OFFICIAL OUT pending review",
                                  source_url=row.source_url))
    if not continuing.empty:
        required = {"season", "team", "player", "first_week", "last_week", "source_url"}
        if not required.issubset(continuing):
            raise ValueError("Continuing injury register schema incomplete")
        c = continuing.loc[continuing.season.eq(season) & continuing.first_week.le(week)
                           & continuing.last_week.ge(week)]
        for row in c.itertuples(index=False):
            if (row.team, row.player) not in reviewed:
                found.append(dict(team=row.team, player=row.player,
                                  reason="Continuing absence needs current-week review",
                                  source_url=row.source_url))
    return pd.DataFrame(found, columns=["team", "player", "reason", "source_url"]).drop_duplicates(
        ["team", "player"], keep="first")
