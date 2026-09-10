import pandas as pd

from core_v2.io import clean_bool
from parlay_live import normalize_existing_lines


def test_clean_bool_handles_provider_strings() -> None:
    assert clean_bool(True) is True
    assert clean_bool(False) is False
    assert clean_bool("true") is True
    assert clean_bool("False") is False
    assert clean_bool(1) is True
    assert clean_bool(0) is False


def test_dashboard_csv_boolean_is_normalized() -> None:
    frame = pd.DataFrame(
        {
            "sportsbook": ["draftkings", "fanduel"],
            "away_team": ["WAS", "WAS"],
            "home_team": ["PHI", "PHI"],
            "book_topped_up_clean": ["False", "True"],
        }
    )
    result = normalize_existing_lines(frame)
    assert result["book_topped_up_clean"].tolist() == [False, True]
