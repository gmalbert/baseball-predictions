import json
import sys
import types
from datetime import date

import pandas as pd



def _stub_module(name, **functions):
    module = types.ModuleType(name)
    for function_name, function in functions.items():
        setattr(module, function_name, function)
    sys.modules.setdefault(name, module)


_stub_module("src.ingestion.mlb_stats", fetch_todays_probable_pitchers=lambda: None)
_stub_module("src.ingestion.odds", fetch_current_odds=lambda: None, get_consensus_line=lambda value: value)
_stub_module("src.ingestion.weather", fetch_weather_for_games=lambda schedule: None)
_stub_module("src.models.underdog_model", predict_moneyline=lambda **kwargs: None)
_stub_module("src.models.spread_model", predict_spread=lambda **kwargs: None)
_stub_module("src.models.totals_model", predict_totals=lambda **kwargs: None)

from scripts import export_best_bets
from src.picks import daily_pipeline


def test_empty_snapshot_writes_explicit_status(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(daily_pipeline, "PICKS_TODAY_PATH", tmp_path / "picks_today.csv")
    monkeypatch.setattr(daily_pipeline, "PICKS_METADATA_PATH", tmp_path / "picks_today.meta.json")

    daily_pipeline._store_picks({}, date(2026, 7, 22), status="no_qualifying_picks", notes="No edge.")

    assert pd.read_csv(tmp_path / "picks_today.csv").empty
    metadata = json.loads((tmp_path / "picks_today.meta.json").read_text())
    assert metadata["status"] == "no_qualifying_picks"
    assert metadata["picks_count"] == 0


def test_export_preserves_no_games_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data_files" / "processed"
    processed.mkdir(parents=True)
    (processed / "picks_today.csv").write_text("game_id,home_team\n")
    (processed / "picks_today.meta.json").write_text(json.dumps({
        "status": "no_games",
        "target_date": "2026-07-22",
        "picks_count": 0,
        "notes": "No games.",
    }))
    monkeypatch.setattr(export_best_bets, "OUT_PATH", tmp_path / "data_files" / "best_bets_today.json")
    monkeypatch.setattr(export_best_bets, "SRC_PATH", processed / "picks_today.csv")
    monkeypatch.setattr(export_best_bets, "META_PATH", processed / "picks_today.meta.json")

    export_best_bets.main()

    payload = json.loads((tmp_path / "data_files" / "best_bets_today.json").read_text())
    assert payload["bets"] == []
    assert payload["meta"]["status"] == "no_games"


def test_export_normalizes_pipeline_csv_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data_files" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame([{
        "game_id": "g1",
        "away_team": "Away",
        "home_team": "Home",
        "pick_type": "spread",
        "pick_value": "Home -1.5",
        "confidence_score": 0.72,
        "edge": 0.08,
        "date": "2026-07-22",
        "source": "morning",
    }]).to_csv(processed / "picks_today.csv", index=False)
    (processed / "picks_today.meta.json").write_text(json.dumps({
        "status": "ok",
        "target_date": "2026-07-22",
        "picks_count": 1,
    }))
    monkeypatch.setattr(export_best_bets, "OUT_PATH", tmp_path / "data_files" / "best_bets_today.json")
    monkeypatch.setattr(export_best_bets, "SRC_PATH", processed / "picks_today.csv")
    monkeypatch.setattr(export_best_bets, "META_PATH", processed / "picks_today.meta.json")

    export_best_bets.main()

    payload = json.loads((tmp_path / "data_files" / "best_bets_today.json").read_text())
    assert payload["bets"][0]["bet_type"] == "Spread"
    assert payload["bets"][0]["pick"] == "Home -1.5"
    assert payload["bets"][0]["confidence"] == 0.72
