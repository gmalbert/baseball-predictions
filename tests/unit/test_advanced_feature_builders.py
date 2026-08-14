from datetime import UTC, datetime

import pandas as pd

from src.features.builders import (
    PlayerParkInteractionBuilder,
    ProjectionPriorBuilder,
    SeriesTimesThroughOrderBuilder,
)

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


def test_advanced_provider_neutral_builders_preserve_cutoff_and_missingness() -> None:
    park = PlayerParkInteractionBuilder(
        pd.DataFrame(
            [
                {
                    "game_id": "g",
                    "player_id": "p",
                    "observed_at": "2026-08-10T15:00:00Z",
                    "spray_match_score": 0.4,
                    "expected_hr_park_delta": 0.03,
                }
            ]
        )
    ).build(pd.DataFrame({"game_id": ["g"]}), as_of=NOW)
    series = SeriesTimesThroughOrderBuilder(
        pd.DataFrame(
            [
                {
                    "game_id": "g",
                    "player_id": "p",
                    "observed_at": "2026-08-10T15:00:00Z",
                    "tto_penalty_third": 0.08,
                    "series_familiarity_pa": 12,
                }
            ]
        )
    ).build(pd.DataFrame({"game_id": ["g"]}), as_of=NOW)
    projection = ProjectionPriorBuilder(
        pd.DataFrame(
            [
                {
                    "player_id": "p",
                    "observed_at": "2026-08-09T15:00:00Z",
                    "projected_rate": 0.31,
                    "projection_sd": 0.04,
                }
            ]
        )
    ).build(pd.DataFrame({"player_id": ["p", "missing"]}), as_of=NOW)
    assert park.loc[0, "spray_match_score"] == 0.4
    assert series.loc[0, "tto_penalty_third"] == 0.08
    assert projection.set_index("player_id").loc["missing", "projection_prior_missing"]
