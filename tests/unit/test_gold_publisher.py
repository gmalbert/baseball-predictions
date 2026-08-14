"""Tests for the gold artifact publisher."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.contracts.domain import Decision, Prediction, Quote, Selection
from src.pipelines.gold_publisher import publish_gold_artifacts

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)

GAMES = pd.DataFrame(
    [
        {
            "game_id": "g1",
            "home_team_id": "H",
            "away_team_id": "A",
            "scheduled_start_utc": "2026-08-10T23:00:00Z",
        }
    ]
)


def _lineup() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team_id": "H",
                "player_id": "p1",
                "batting_order": 1,
                "defensive_position": "3B",
                "lineup_status": "confirmed",
                "observed_at": "2026-08-10T15:00:00Z",
                "projected_pa": 4.5,
                "talent_mean": 0.30,
                "talent_sd": 0.04,
                "availability_probability": 1.0,
            }
        ]
    )


def _pitcher() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team_id": "H",
                "player_id": "sp1",
                "event_time": "2026-08-08T23:00:00Z",
                "observed_at": "2026-08-09T03:00:00Z",
                "pitches": 98,
                "outs_recorded": 18,
                "role": "starter",
            },
            {
                "game_id": "g1",
                "team_id": "H",
                "player_id": "rp1",
                "event_time": "2026-08-09T23:00:00Z",
                "observed_at": "2026-08-10T03:00:00Z",
                "pitches": 22,
                "outs_recorded": 3,
                "role": "reliever",
            },
        ]
    )


def _reliever() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team_id": "H",
                "player_id": "rp1",
                "observed_at": "2026-08-10T03:00:00Z",
                "pitches_last_3d": 22,
                "consecutive_days": 1,
                "quality": 0.6,
                "leverage_weight": 1.0,
            }
        ]
    )


def _observations() -> dict[str, pd.DataFrame]:
    return {
        "lineup_snapshot": _lineup(),
        "pitcher_pitch": _pitcher(),
        "reliever_usage": _reliever(),
    }


def _prediction() -> Prediction:
    return Prediction(
        prediction_id="pred-1",
        snapshot_id="snap-1",
        game_id="g1",
        model_run_id="m1",
        market_id="moneyline_full_game",
        selection=Selection.AWAY,
        probability_raw=0.62,
        probability=0.62,
        probability_low=0.58,
        probability_high=0.66,
        predicted_at=NOW,
        feature_row_hash="h",
    )


def _quote() -> Quote:
    return Quote(
        quote_id="q1",
        game_id="g1",
        bookmaker_id="book",
        market_id="moneyline_full_game",
        selection=Selection.AWAY,
        price_decimal=Decimal("1.91"),
        observed_at=NOW,
    )


def _decision() -> Decision:
    return Decision(
        decision_id="d1",
        prediction_id="pred-1",
        quote_id="q1",
        game_id="g1",
        market_id="moneyline_full_game",
        selection=Selection.AWAY,
        decided_at=NOW,
        market_probability=0.52,
        fair_probability=0.62,
        break_even_probability=0.52,
        edge=0.10,
        expected_value=0.10,
        recommended_stake=Decimal("0"),
        bankroll_before=Decimal("1000"),
        policy_version="conservative_v1",
        action="abstain",
        reason_codes=("stale_quote",),
    )


def test_publish_writes_all_four_artifacts(tmp_path: Path) -> None:
    result = publish_gold_artifacts(
        games=GAMES,
        observations=_observations(),
        gold_root=tmp_path,
        as_of=NOW,
    )
    assert result.status == "published"
    assert result.distributions == 1
    assert (tmp_path / "game_distributions.parquet").is_file()
    assert (tmp_path / "bullpen_availability.parquet").is_file()
    assert (tmp_path / "lineup_scenarios.parquet").is_file()
    assert (tmp_path / "eligibility.parquet").is_file()

    distributions = pd.read_parquet(tmp_path / "game_distributions.parquet")
    assert len(distributions) == 1
    row = distributions.iloc[0]
    # Coherent: home + away + tie = 1
    assert abs(row["home_moneyline"] + row["away_moneyline"] + row["tie_probability"] - 1.0) < 1e-6
    # Over/under/push at 9 reconcile
    assert (
        abs(
            row["over_probability_9"] + row["under_probability_9"] + row["push_probability_9"] - 1.0
        )
        < 1e-6
    )

    bullpen = pd.read_parquet(tmp_path / "bullpen_availability.parquet")
    assert not bullpen.empty
    assert "team_id" in bullpen

    lineups = pd.read_parquet(tmp_path / "lineup_scenarios.parquet")
    assert len(lineups) == 1
    assert lineups.iloc[0]["lineup_status"] == "confirmed"


def test_publish_eligibility_includes_abstention_reasons(tmp_path: Path) -> None:
    result = publish_gold_artifacts(
        games=GAMES,
        observations=_observations(),
        gold_root=tmp_path,
        as_of=NOW,
        predictions=[_prediction()],
        quotes=[_quote()],
        decisions=[_decision()],
    )
    assert result.eligibility == 1
    eligibility = pd.read_parquet(tmp_path / "eligibility.parquet")
    assert bool(eligibility.iloc[0]["eligible"]) is False
    assert "stale_quote" in eligibility.iloc[0]["reason_codes"]


def test_publish_blocks_on_no_games(tmp_path: Path) -> None:
    result = publish_gold_artifacts(
        games=pd.DataFrame(columns=["game_id", "home_team_id", "away_team_id"]),
        observations=_observations(),
        gold_root=tmp_path,
        as_of=NOW,
    )
    assert result.status == "blocked"
    assert result.reason == "no_games"


def test_publish_fails_closed_on_missing_observations(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="pitcher_pitch"):
        publish_gold_artifacts(
            games=GAMES,
            observations={"lineup_snapshot": _lineup(), "reliever_usage": _reliever()},
            gold_root=tmp_path,
            as_of=NOW,
        )


def test_publish_is_deterministic(tmp_path: Path) -> None:
    first = publish_gold_artifacts(
        games=GAMES, observations=_observations(), gold_root=tmp_path / "a", as_of=NOW
    )
    second = publish_gold_artifacts(
        games=GAMES, observations=_observations(), gold_root=tmp_path / "b", as_of=NOW
    )
    assert first.status == second.status == "published"
    a = pd.read_parquet(tmp_path / "a" / "game_distributions.parquet")
    b = pd.read_parquet(tmp_path / "b" / "game_distributions.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_publish_writes_run_manifest(tmp_path: Path) -> None:
    import json

    result = publish_gold_artifacts(
        games=GAMES,
        observations=_observations(),
        gold_root=tmp_path,
        as_of=NOW,
    )
    assert result.status == "published"
    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "published"
    assert manifest["target_date"] == "2026-08-10"
    assert manifest["stage"] == "gold_publish"
    assert manifest["games"] == 1
    assert manifest["distributions"] == 1
    assert manifest["bullpen"] >= 1
    assert manifest["lineups"] == 1
    sources = {row["source"] for row in manifest["sources"]}
    assert {
        "lineup_snapshot",
        "bullpen_availability",
        "game_distributions",
        "eligibility",
    } <= sources
    for row in manifest["sources"]:
        assert row["available"] is True or row["rows"] == 0
