"""Tests for PA-level label construction."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.labels.plate_appearances import (
    label_plate_appearances,
    pa_outcome_target,
    validate_label_frame,
)
from src.models.plate_appearance import PaOutcome

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "player_id": "b1",
                "pitcher_id": "p1",
                "event": "single",
                "outs": 0,
                "on_1b": False,
                "on_2b": False,
                "on_3b": False,
                "inning": 1,
                "score_diff": 0,
                "runs_scored": 0,
                "observed_at": "2026-08-10T15:00:00Z",
            },
            {
                "game_id": "g1",
                "player_id": "b2",
                "pitcher_id": "p1",
                "event": "home_run",
                "outs": 0,
                "on_1b": True,
                "on_2b": False,
                "on_3b": False,
                "inning": 1,
                "score_diff": 0,
                "runs_scored": 2,
                "observed_at": "2026-08-10T15:01:00Z",
            },
            {
                "game_id": "g1",
                "player_id": "b3",
                "pitcher_id": "p1",
                "event": "strikeout",
                "outs": 1,
                "on_1b": False,
                "on_2b": False,
                "on_3b": False,
                "inning": 1,
                "score_diff": 2,
                "runs_scored": 0,
                "observed_at": "2026-08-10T15:02:00Z",
            },
        ]
    )


def test_label_plate_appearances_maps_outcomes() -> None:
    frame = label_plate_appearances(_events())
    assert len(frame) == 3
    assert set(frame["outcome"]) == {
        PaOutcome.SINGLE,
        PaOutcome.HOME_RUN,
        PaOutcome.STRIKEOUT,
    }
    home_run = frame[frame["outcome"] == PaOutcome.HOME_RUN].iloc[0]
    assert home_run["state"].on_1b is True
    assert home_run["runs_scored"] == 2


def test_label_drops_unknown_events() -> None:
    events = _events()
    events.loc[0, "event"] = "unknown_weird_thing"
    frame = label_plate_appearances(events)
    assert len(frame) == 2


def test_label_requires_columns() -> None:
    with pytest.raises(KeyError, match="pitcher_id"):
        label_plate_appearances(_events().drop(columns=["pitcher_id"]))


def test_pa_outcome_target_one_hot() -> None:
    frame = label_plate_appearances(_events())
    encoded = pa_outcome_target(frame)
    assert f"outcome_{PaOutcome.SINGLE.value}" in encoded
    assert encoded[f"outcome_{PaOutcome.SINGLE.value}"].sum() == 1
    assert encoded[f"outcome_{PaOutcome.HOME_RUN.value}"].sum() == 1
    assert encoded[f"outcome_{PaOutcome.STRIKEOUT.value}"].sum() == 1


def test_validate_label_frame_rejects_future() -> None:
    frame = label_plate_appearances(_events())
    with pytest.raises(ValueError, match="future"):
        validate_label_frame(frame, as_of=datetime(2026, 8, 10, 14, tzinfo=UTC))


def test_validate_label_frame_passes_on_time() -> None:
    frame = label_plate_appearances(_events())
    validate_label_frame(frame, as_of=NOW)  # should not raise
