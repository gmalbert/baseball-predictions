from __future__ import annotations

import pandas as pd

from src.ingestion import savant_leaderboard


class _Response:
    headers = {"Content-Type": "text/html"}
    text = "<html><body>Access Denied</body></html>"

    def raise_for_status(self) -> None:
        return None


def test_multi_year_savant_request_falls_back_to_seasons(monkeypatch):
    calls: list[str] = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(params["year"])
        if "," in params["year"]:
            return _Response()
        response = _Response()
        response.headers = {"Content-Type": "text/csv"}
        response.text = '"last_name, first_name",player_id,year\n"Player",1,' + params["year"]
        return response

    monkeypatch.setattr(savant_leaderboard.requests, "get", fake_get)
    monkeypatch.setattr(savant_leaderboard, "sleep", lambda _: None)

    result = savant_leaderboard.download_savant_leaderboard([2024, 2025])

    assert calls == ["2024,2025", "2024", "2025"]
    assert result["year"].tolist() == [2024, 2025]
    assert isinstance(result, pd.DataFrame)


def test_single_year_invalid_savant_response_has_actionable_error(monkeypatch):
    monkeypatch.setattr(savant_leaderboard.requests, "get", lambda *args, **kwargs: _Response())

    try:
        savant_leaderboard.download_savant_leaderboard(2025)
    except RuntimeError as exc:
        assert "valid batter CSV" in str(exc)
        assert "Access Denied" in str(exc)
    else:
        raise AssertionError("expected invalid Savant response to fail clearly")


def test_existing_savant_file_is_reused_without_http_request(monkeypatch, tmp_path):
    monkeypatch.setattr(savant_leaderboard.config, "project_root", tmp_path)
    cached = tmp_path / "data_files" / "raw" / "batting" / "savant_batter_2020_2025.csv"
    cached.parent.mkdir(parents=True)
    cached.write_text('"last_name, first_name",player_id,year\n"Cached",1,2025\n')

    def fail_get(*args, **kwargs):
        raise AssertionError("cached leaderboard should not make an HTTP request")

    monkeypatch.setattr(savant_leaderboard.requests, "get", fail_get)

    result = savant_leaderboard.fetch_and_save_batter_leaderboard(
        [2020, 2021, 2022, 2023, 2024, 2025]
    )

    assert result.loc[0, "player_id"] == 1
    assert result.loc[0, "year"] == 2025
