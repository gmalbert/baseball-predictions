from pathlib import Path

import pandas as pd
import pytest

from src.quality.contracts import load_contract, validate_contract


def test_all_yaml_contracts_load_with_unique_dataset_names() -> None:
    root = Path(__file__).resolve().parents[2] / "config" / "contracts"
    contracts = [load_contract(path) for path in sorted(root.glob("*.yaml"))]
    assert len(contracts) >= 12
    assert len({contract.dataset for contract in contracts}) == len(contracts)


def test_contract_rejects_wrong_required_dtype() -> None:
    contract = load_contract(
        Path(__file__).resolve().parents[2] / "config" / "contracts" / "fact_game.yaml"
    )
    frame = pd.DataFrame(
        {
            "game_id": ["g"],
            "season": ["not-an-integer"],
            "game_type": ["R"],
            "scheduled_start_utc": ["2026-08-10T23:00:00Z"],
            "venue_id": ["v"],
            "home_team_id": ["h"],
            "away_team_id": ["a"],
            "ruleset_version": ["2026"],
        }
    )
    with pytest.raises(ValueError, match="dtype:season"):
        validate_contract(frame, contract)
