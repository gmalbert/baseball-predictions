"""Point-in-time and leakage tests for lineup, bullpen, and pitcher facts."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.contracts.domain import GameSnapshot
from src.features.builders import (
    LineupOffenseBuilder,
    PitcherAvailabilityBuilder,
)
from src.features.snapshots import build_game_snapshot_rows

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)
AS_OF = NOW

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


def _lineup_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team_id": "H",
                "player_id": "p1",
                "batting_order": 1,
                "lineup_status": "confirmed",
                "observed_at": "2026-08-10T15:00:00Z",
                "projected_pa": 4.5,
                "talent_mean": 0.30,
                "talent_sd": 0.04,
                "availability_probability": 1.0,
            }
        ]
    )


def _pitcher_observations() -> pd.DataFrame:
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


def _reliever_observations() -> pd.DataFrame:
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
        "lineup_snapshot": _lineup_observations(),
        "pitcher_pitch": _pitcher_observations(),
        "reliever_usage": _reliever_observations(),
    }


def test_snapshot_builder_emits_watermarked_game_snapshot() -> None:
    snapshots = build_game_snapshot_rows(
        GAMES,
        _observations(),
        as_of=AS_OF,
    )
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert isinstance(snapshot, GameSnapshot)
    assert snapshot.game_id == "g1"
    assert snapshot.as_of_time == AS_OF
    # Lineup offense contributes game-grain features
    assert "lineup_offense" in snapshot.features
    # Team-grain families are side-prefixed (home_* / away_*)
    assert "home_bullpen_availability_bullpen_available_arms" in snapshot.features
    assert "home_pitcher_availability_starter_availability" in snapshot.features
    assert "home_starter_projection_expected_innings" in snapshot.features
    # Watermarks present for each family
    assert snapshot.source_watermarks
    # Missingness indicators are preserved as explicit uncertainty signals
    assert "lineup_offense_missing" in snapshot.features
    assert snapshot.features["lineup_offense_missing"] is False


def test_pitcher_availability_builder_respects_cutoff() -> None:
    builder = PitcherAvailabilityBuilder(_pitcher_observations())
    frame = builder.build(GAMES, as_of=AS_OF)
    # Team-grain output: one row per team with observations (only H here)
    assert set(frame["team_id"]) == {"H"}
    # The H side has a starter in the observations
    home = frame.iloc[0]
    assert home["starter_availability"] > 0.5


def test_late_lineup_observation_is_excluded() -> None:
    late = _lineup_observations().copy()
    late.loc[0, "observed_at"] = "2026-08-10T16:00:01Z"  # after cutoff
    builder = LineupOffenseBuilder(late)
    frame = builder.build(GAMES, as_of=AS_OF)
    # The late observation is excluded; the feature is explicitly missing
    # rather than leaking a future value.
    assert pd.isna(frame.loc[0, "lineup_offense"])
    assert bool(frame.loc[0, "lineup_offense_missing"]) is True


def test_missing_source_fails_closed() -> None:
    obs = _observations()
    del obs["pitcher_pitch"]
    with pytest.raises(KeyError, match="pitcher_pitch"):
        build_game_snapshot_rows(GAMES, obs, as_of=AS_OF)


def test_lineup_offense_sums_eligible_player_weights() -> None:
    lineup = _lineup_observations()
    updated = lineup.copy()
    updated.loc[0, "projected_pa"] = 5.0
    updated.loc[0, "observed_at"] = "2026-08-10T15:30:00Z"
    builder = LineupOffenseBuilder(pd.concat([lineup, updated], ignore_index=True))
    frame = builder.build(GAMES, as_of=AS_OF)
    # Two eligible observations both contribute to the team total
    assert frame.loc[0, "lineup_expected_pa"] == 4.5 + 5.0
