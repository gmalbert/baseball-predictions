from pathlib import Path

from src.config.settings import load_config


def test_production_config_is_typed_and_not_bound_to_a_fixed_season() -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "config" / "baseball.yaml")
    assert config.data_schema_version == "2.0.0"
    assert config.bankroll.max_factor_fraction <= config.bankroll.max_day_fraction
    assert all(snapshot.name for snapshot in config.snapshots)
