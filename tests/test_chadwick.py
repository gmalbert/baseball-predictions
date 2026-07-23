import pandas as pd

from src.ingestion import chadwick


def test_chadwick_uses_split_people_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(chadwick, "_REGISTRY_PATH", tmp_path / "player_registry.parquet")
    monkeypatch.setattr(
        chadwick.pd,
        "read_csv",
        lambda url, **kwargs: pd.DataFrame([{
            "key_uuid": url.rsplit("/", 1)[-1],
            "key_retro": "player1",
            "key_mlbam": "123",
        }]),
    )

    result = chadwick.load_player_registry(force_refresh=True)

    assert len(chadwick.CHADWICK_URLS) == 16
    assert chadwick.CHADWICK_URLS[0].endswith("people-0.csv")
    assert chadwick.CHADWICK_URLS[-1].endswith("people-f.csv")
    assert len(result) == 16
    assert (tmp_path / "player_registry.parquet").exists()


def test_chadwick_fetch_failure_returns_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(chadwick, "_REGISTRY_PATH", tmp_path / "player_registry.parquet")
    monkeypatch.setattr(chadwick.pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("404")))

    result = chadwick.load_player_registry(force_refresh=True)

    assert result.empty
    assert list(result.columns) == chadwick._KEEP_COLS
