from datetime import date

import pandas as pd

from src.picks import afternoon_refresh, daily_pipeline


def test_daily_pipeline_publishes_model_incompatible_instead_of_crashing(
    monkeypatch, tmp_path
) -> None:
    schedule = pd.DataFrame([{"game_id": "g1", "away_team": "Away", "home_team": "Home"}])
    raw_odds = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "away_team": "Away",
                "home_team": "Home",
                "bookmaker": "book",
                "market": "h2h",
                "outcome_name": "Home",
                "outcome_price": -110,
                "outcome_point": None,
                "fetched_at": "2026-08-13T12:00:00+00:00",
            }
        ]
    )
    consensus = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "away_team": "Away",
                "home_team": "Home",
                "market": "h2h",
                "outcome_name": "Home",
                "median_price": -110,
                "median_point": None,
            }
        ]
    )
    monkeypatch.setattr(daily_pipeline, "fetch_todays_probable_pitchers", lambda _date: schedule)
    monkeypatch.setattr(daily_pipeline, "fetch_current_odds", lambda **_kwargs: raw_odds)
    monkeypatch.setattr(daily_pipeline, "get_consensus_line", lambda _frame: consensus)
    monkeypatch.setattr(daily_pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(daily_pipeline, "PICKS_TODAY_PATH", tmp_path / "picks_today.csv")
    monkeypatch.setattr(daily_pipeline, "PICKS_METADATA_PATH", tmp_path / "picks_today.meta.json")
    result = daily_pipeline.run_daily_pipeline(date(2026, 8, 13))
    metadata = __import__("json").loads(
        (tmp_path / "picks_today.meta.json").read_text(encoding="utf-8")
    )
    assert all(not picks for picks in result.values())
    assert metadata["status"] == "model_incompatible"


def test_afternoon_refresh_cannot_invoke_quarantined_models(monkeypatch, tmp_path) -> None:
    consensus = pd.DataFrame(
        [
            {
                "game_id": "provider-game",
                "away_team": "Away",
                "home_team": "Home",
                "market": "h2h",
                "outcome_name": "Home",
            }
        ]
    )
    statuses: list[str] = []
    monkeypatch.setattr(afternoon_refresh, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(afternoon_refresh, "fetch_current_odds", lambda **_kwargs: consensus)
    monkeypatch.setattr(afternoon_refresh, "get_consensus_line", lambda _frame: consensus)
    monkeypatch.setattr(afternoon_refresh, "_save_consensus_snapshot", lambda *_args, **_kw: None)
    monkeypatch.setattr(afternoon_refresh, "quarantine_reason", lambda *_args: "quarantined")
    monkeypatch.setattr(
        afternoon_refresh,
        "write_pipeline_status",
        lambda status, *_args: statuses.append(status),
    )
    monkeypatch.setattr(
        afternoon_refresh,
        "fetch_todays_probable_pitchers",
        lambda *_args: (_ for _ in ()).throw(AssertionError("model gate must run first")),
    )

    assert afternoon_refresh.afternoon_picks_refresh(date(2026, 8, 13)) == {}
    assert statuses == ["model_incompatible"]
