import pandas as pd
import pytest

from src.features.asof import point_in_time_join, shifted_rolling_features
from src.features.model_matrix import build_game_snapshots


def test_future_observation_is_excluded():
    targets = pd.DataFrame({"team_id": [1], "as_of_time": ["2026-04-01T12:00:00Z"]})
    observations = pd.DataFrame(
        {
            "team_id": [1, 1],
            "observed_at": ["2026-04-01T11:00:00Z", "2026-04-01T12:00:01Z"],
            "rating": [0.50, 0.99],
        }
    )
    result = point_in_time_join(targets, observations, by="team_id")
    assert result.loc[0, "rating"] == 0.50


def test_current_game_is_shifted_out_of_rolling_feature():
    games = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "event_time": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"], utc=True),
            "runs": [2, 4, 100],
        }
    )
    result = shifted_rolling_features(
        games,
        entity_col="team_id",
        event_time_col="event_time",
        value_cols=["runs"],
        windows=[2],
        half_lives=[],
    )
    assert result.loc[2, "runs_mean_2"] == 3


def _fixture(result_runs: int = 4, add_late: bool = False):
    games = pd.DataFrame(
        [
            {
                "game_id": "target",
                "scheduled_start_utc": "2026-04-03T23:00:00Z",
                "home_team_id": "H",
                "away_team_id": "A",
            }
        ]
    )
    rows = [
        {
            "team_id": "H",
            "game_id": "h1",
            "event_time": "2026-04-01T23:00:00Z",
            "observed_at": "2026-04-02T03:00:00Z",
            "runs": 2,
        },
        {
            "team_id": "A",
            "game_id": "a1",
            "event_time": "2026-04-01T23:00:00Z",
            "observed_at": "2026-04-02T03:00:00Z",
            "runs": 3,
        },
        {
            "team_id": "H",
            "game_id": "target",
            "event_time": "2026-04-03T23:00:00Z",
            "observed_at": "2026-04-04T03:00:00Z",
            "runs": result_runs,
        },
    ]
    if add_late:
        rows.append(
            {
                "team_id": "H",
                "game_id": "late",
                "event_time": "2026-04-02T23:00:00Z",
                "observed_at": "2026-04-03T20:00:01Z",
                "runs": 999,
            }
        )
    return games, pd.DataFrame(rows)


def test_target_result_and_late_observation_cannot_change_snapshot():
    games, history = _fixture(result_runs=4)
    before = build_game_snapshots(games, history, value_columns=["runs"])
    _, mutated_history = _fixture(result_runs=99, add_late=True)
    after = build_game_snapshots(games, mutated_history, value_columns=["runs"])
    columns = [column for column in before if column not in {"snapshot_id"}]
    pd.testing.assert_frame_equal(before[columns], after[columns])


def test_point_in_time_join_rejects_target_event_if_event_contract_is_requested():
    targets = pd.DataFrame(
        {
            "team_id": [1],
            "as_of_time": ["2026-04-02T12:00:00Z"],
            "game_start": ["2026-04-02T10:00:00Z"],
        }
    )
    observations = pd.DataFrame(
        {
            "team_id": [1],
            "observed_at": ["2026-04-02T09:00:00Z"],
            "event_time": ["2026-04-02T10:00:00Z"],
            "value": [99],
        }
    )
    with pytest.raises(ValueError, match="target or future event"):
        point_in_time_join(
            targets,
            observations,
            by="team_id",
            event_time="event_time",
            target_event_time="game_start",
        )
