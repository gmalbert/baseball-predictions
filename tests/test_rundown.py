"""Tests for src/ingestion/therundown.py."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.ingestion.therundown import (
    fetch_events_for_date,
    fetch_current_odds,
    get_consensus_line,
    SENTINEL_OFF_BOARD,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

SAMPLE_EVENT = {
    "event_id": "abc123",
    "event_date": "2026-07-10T23:10:00Z",
    "teams": [
        {"name": "New York Yankees", "is_away": True, "is_home": False},
        {"name": "Boston Red Sox", "is_away": False, "is_home": True},
    ],
    "markets": [
        {
            "market_id": 1,
            "participants": [
                {
                    "name": "New York Yankees",
                    "lines": [
                        {
                            "value": "",
                            "prices": {
                                "19": {"price": -150, "is_main_line": True},
                                "22": {"price": -145, "is_main_line": True},
                                "23": {"price": -155, "is_main_line": True},
                            },
                        }
                    ],
                },
                {
                    "name": "Boston Red Sox",
                    "lines": [
                        {
                            "value": "",
                            "prices": {
                                "19": {"price": 130, "is_main_line": True},
                                "22": {"price": 125, "is_main_line": True},
                                "23": {"price": 135, "is_main_line": True},
                            },
                        }
                    ],
                },
            ],
        },
        {
            "market_id": 2,
            "participants": [
                {
                    "name": "New York Yankees",
                    "lines": [
                        {
                            "value": "-1.5",
                            "prices": {
                                "19": {"price": 120, "is_main_line": True},
                                "22": {"price": 115, "is_main_line": True},
                            },
                        },
                        {
                            "value": "-2.5",
                            "prices": {
                                "19": {"price": 180, "is_main_line": False},
                            },
                        },
                    ],
                },
                {
                    "name": "Boston Red Sox",
                    "lines": [
                        {
                            "value": "1.5",
                            "prices": {
                                "19": {"price": -140, "is_main_line": True},
                                "22": {"price": -135, "is_main_line": True},
                            },
                        },
                        {
                            "value": "2.5",
                            "prices": {
                                "19": {"price": -180, "is_main_line": False},
                            },
                        },
                    ],
                },
            ],
        },
        {
            "market_id": 3,
            "participants": [
                {
                    "name": "Over",
                    "lines": [
                        {
                            "value": "8.5",
                            "prices": {
                                "19": {"price": -110, "is_main_line": True},
                            },
                        }
                    ],
                },
                {
                    "name": "Under",
                    "lines": [
                        {
                            "value": "8.5",
                            "prices": {
                                "19": {"price": -110, "is_main_line": True},
                            },
                        }
                    ],
                },
            ],
        },
    ],
}


def _mock_response(events, headers=None):
    """Create a mock requests.Response with JSON body."""
    resp = MagicMock()
    resp.json.return_value = {"events": events}
    resp.raise_for_status = MagicMock()
    resp.headers = headers or {}
    return resp


# ── Tests: fetch_events_for_date ─────────────────────────────────────────

@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_returns_events(mock_config, mock_get):
    mock_config.therundown_api_key = "test-key"
    mock_get.return_value = _mock_response([SAMPLE_EVENT])

    events = fetch_events_for_date(date(2026, 7, 10))
    assert len(events) == 1
    assert events[0]["event_id"] == "abc123"


@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_raises_without_api_key(mock_config, mock_get):
    mock_config.therundown_api_key = ""
    with pytest.raises(ValueError, match="THERUNDOWN_API_KEY"):
        fetch_events_for_date()


@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_retries_without_main_line(mock_config, mock_get):
    """When main_line=true returns no markets, retries without filter."""
    mock_config.therundown_api_key = "test-key"
    empty_resp = _mock_response([{"event_id": "x", "markets": []}])
    full_resp = _mock_response([SAMPLE_EVENT])
    mock_get.side_effect = [empty_resp, full_resp]

    events = fetch_events_for_date(date(2026, 7, 10))
    assert len(events) == 1
    assert mock_get.call_count == 2


# ── Tests: fetch_current_odds ────────────────────────────────────────────

@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_schema(mock_config, mock_fetch):
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))

    expected_cols = {
        "game_id", "commence_time", "away_team", "home_team",
        "bookmaker", "market", "outcome_name", "outcome_price",
        "outcome_point", "fetched_at",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_filters_sentinel(mock_config, mock_fetch):
    """Off-the-board sentinel (0.0001) should be excluded."""
    event = {
        "event_id": "xyz",
        "event_date": "2026-07-10T23:10:00Z",
        "teams": [
            {"name": "Team A", "is_away": True},
            {"name": "Team B", "is_home": True},
        ],
        "markets": [
            {
                "market_id": 1,
                "participants": [
                    {
                        "name": "Team A",
                        "lines": [
                            {
                                "value": "",
                                "prices": {
                                    "19": {"price": SENTINEL_OFF_BOARD, "is_main_line": True},
                                    "22": {"price": -150, "is_main_line": True},
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [event]

    df = fetch_current_odds(date(2026, 7, 10))
    assert SENTINEL_OFF_BOARD not in df["outcome_price"].values
    assert -150 in df["outcome_price"].values


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_filters_alternate_lines(mock_config, mock_fetch):
    """Spreads/totals should keep only the main line, not alternates."""
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))

    spreads = df[df["market"] == "spreads"]
    # Yankees should only have -1.5, not -2.5
    yankees_spreads = spreads[spreads["outcome_name"] == "New York Yankees"]
    assert set(yankees_spreads["outcome_point"].unique()) == {-1.5}

    # Red Sox should only have +1.5, not +2.5
    sox_spreads = spreads[spreads["outcome_name"] == "Boston Red Sox"]
    assert set(sox_spreads["outcome_point"].unique()) == {1.5}


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_moneyline_no_filtering(mock_config, mock_fetch):
    """Moneyline should not be filtered (only one line per participant)."""
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))
    h2h = df[df["market"] == "h2h"]
    assert len(h2h) == 6  # 2 teams × 3 books


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_empty_events(mock_config, mock_fetch):
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = []

    df = fetch_current_odds(date(2026, 7, 10))
    assert df.empty


# ── Tests: get_consensus_line ────────────────────────────────────────────

def test_get_consensus_line_medians():
    df = pd.DataFrame([
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -150, "outcome_point": None, "bookmaker": "dk"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -140, "outcome_point": None, "bookmaker": "fd"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -160, "outcome_point": None, "bookmaker": "mgm"},
    ])

    consensus = get_consensus_line(df)
    row = consensus.iloc[0]
    assert row["median_price"] == -150.0
    assert row["num_books"] == 3
    assert row["mean_price"] == pytest.approx(-150.0)


def test_get_consensus_line_with_points():
    df = pd.DataFrame([
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "spreads",
         "outcome_name": "A", "outcome_price": -110, "outcome_point": -1.5, "bookmaker": "dk"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "spreads",
         "outcome_name": "A", "outcome_price": -115, "outcome_point": -1.5, "bookmaker": "fd"},
    ])

    consensus = get_consensus_line(df)
    assert consensus.iloc[0]["median_point"] == -1.5


# ── Tests: fallback behavior ─────────────────────────────────────────────

@patch("src.ingestion.therundown.fetch_current_odds")
def test_fallback_on_rundown_failure(mock_rundown):
    """Simulate the fallback pattern used in scheduler/pipeline."""
    from src.ingestion.odds import fetch_current_odds as fallback_fetch

    mock_rundown.side_effect = Exception("API down")

    with patch("src.ingestion.odds.fetch_current_odds") as mock_fallback:
        mock_fallback.return_value = pd.DataFrame({"col": [1]})

        # Replicate _fetch_odds_primary logic
        try:
            result = mock_rundown()
        except Exception:
            result = mock_fallback()

        assert len(result) == 1
        mock_fallback.assert_called_once()
