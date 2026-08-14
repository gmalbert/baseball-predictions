from pathlib import Path

from src.catalog.database import Catalog


def test_all_catalog_migrations_are_idempotent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = Catalog(tmp_path / "catalog.duckdb")
    first = catalog.migrate(root / "migrations")
    second = catalog.migrate(root / "migrations")
    assert first == [
        "001_catalog.sql",
        "002_canonical_facts.sql",
        "003_observation_and_feature_facts.sql",
    ]
    assert second == []
    with catalog.connect() as connection:
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert {"raw_observation", "fact_odds_quote", "feature_game_snapshot", "model_run"} <= names
