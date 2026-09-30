import pandas as pd

from scripts.monitor_injuries import build_candidates, merge_queue, parse_news_links


def test_official_headline_and_report_create_review_leads_without_autopricing():
    html = """
    <a href="/news/giants-jaxson-dart-out">Giants' Jaxson Dart ruled out for season</a>
    <a href="/news/buccaneers-baker-mayfield">Buccaneers' Mayfield to miss three games</a>
    <a href="/news/general">Power rankings and playoff picture</a>
    """
    news = parse_news_links(html)
    assert len(news) == 2
    ratings = pd.DataFrame([
        dict(team="NYG", player_name="Jaxson Dart", gsis_id="00-1111111",
             model_position_v2="QB", pvar_production=0.),
        dict(team="TB", player_name="Baker Mayfield", gsis_id="00-2222222",
             model_position_v2="QB", pvar_production=0.),
        dict(team="TB", player_name="Defender B", gsis_id="00-3333333",
             model_position_v2="CB", pvar_production=.8),
    ])
    report = pd.DataFrame([dict(team="TB", player_name="Defender B",
                                game_status="Out", practice_status="Did Not Participate",
                                source_url="https://www.nfl.com/injuries/league/2026/reg4")])
    first = build_candidates(news, report, ratings, {"NYG", "TB"}, 2026, 4,
                             "2026-09-29T12:00:00Z")
    assert set(first.player) == {"Jaxson Dart", "Baker Mayfield", "Defender B"}
    defender = first.loc[first.player.eq("Defender B")].iloc[0]
    assert defender.potential_def_points == 1.8
    assert defender.review_status == "PENDING_REVIEW"
    assert defender.signal == "OFFICIAL_OUT"
    assert first.loc[first.player.eq("Baker Mayfield"), "potential_def_points"].isna().all()
    updated = merge_queue(first, build_candidates(news, report, ratings, {"NYG", "TB"},
                                                   2026, 4, "2026-09-29T13:00:00Z"))
    assert len(updated) == len(first)
    assert set(updated.first_seen_utc) == {"2026-09-29T12:00:00Z"}
