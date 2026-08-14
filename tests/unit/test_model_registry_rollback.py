from pathlib import Path

import pytest

from src.models.registry import ModelRegistry, RegistryEntry


def entry(run_id: str, status: str) -> RegistryEntry:
    return RegistryEntry(
        model_run_id=run_id,
        model_name="model",
        market_id="moneyline_full_game",
        artifact_uri=f"artifacts/{run_id}.joblib",
        manifest_uri=f"artifacts/{run_id}.manifest.json",
        status=status,
        registered_at="2026-08-10T00:00:00Z",
    )


def test_champion_promotion_and_rollback_are_atomic(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(entry("old", "champion"))
    registry.register(entry("new", "challenger"))
    registry.promote("new", gates_passed=True, reason="passed")
    assert registry.champion("moneyline_full_game").model_run_id == "new"
    restored = registry.rollback("moneyline_full_game", "old", reason="drift incident")
    assert restored.model_run_id == "old"


def test_quarantined_bundle_cannot_be_rollback_target(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(entry("bad", "quarantined"))
    with pytest.raises(ValueError, match="quarantined"):
        registry.rollback("moneyline_full_game", "bad", reason="invalid")
