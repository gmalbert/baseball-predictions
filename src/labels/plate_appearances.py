"""PA-level labels derived from play-by-play events.

``label_plate_appearances`` maps a raw play-by-play event stream to one row
per plate appearance: the batter, pitcher, base/out state before the PA, the
``PaOutcome``, and the runs scored on the play.  Only events with an explicit
``description``/``event`` and a resolved batter count as a PA; provider
corrections create a new version rather than mutating prior rows.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.models.plate_appearance import PaOutcome, PaState

_REACHES_BASE = {
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "hit_by_pitch",
    "field_error",
    "fielder's_choice",
    "intent_walk",
    "catcher_interference",
}

_PA_OUTCOME_BY_EVENT: dict[str, PaOutcome] = {
    "single": PaOutcome.SINGLE,
    "double": PaOutcome.DOUBLE,
    "triple": PaOutcome.TRIPLE,
    "home_run": PaOutcome.HOME_RUN,
    "walk": PaOutcome.WALK,
    "intent_walk": PaOutcome.WALK,
    "hit_by_pitch": PaOutcome.HBP,
    "strikeout": PaOutcome.STRIKEOUT,
    "field_out": PaOutcome.OUT,
    "grounded_into_double_play": PaOutcome.OUT,
    "double_play": PaOutcome.OUT,
    "triple_play": PaOutcome.OUT,
    "sac_fly": PaOutcome.OUT,
    "sac_bunt": PaOutcome.OUT,
    "field_error": PaOutcome.SINGLE,
    "fielder's_choice": PaOutcome.OUT,
    "catcher_interference": PaOutcome.WALK,
}


def _base_out_state(row: pd.Series) -> PaState:
    return PaState(
        outs=int(row.get("outs", 0)),
        on_1b=bool(row.get("on_1b", False)),
        on_2b=bool(row.get("on_2b", False)),
        on_3b=bool(row.get("on_3b", False)),
        inning=int(row.get("inning", 1)),
        score_diff=int(row.get("score_diff", 0)),
    )


def label_plate_appearances(events: pd.DataFrame) -> pd.DataFrame:
    """Convert a play-by-play event frame to one row per plate appearance.

    Required event columns: ``game_id``, ``player_id`` (batter), ``pitcher_id``,
    ``event`` (or ``event_type``), ``outs``, ``on_1b``/``on_2b``/``on_3b``,
    ``inning``, ``score_diff`` (home minus away before the PA), ``runs_scored``,
    and ``observed_at``.  Unknown or malformed events are dropped; a PA with no
    resolved batter is never synthesized.
    """
    required = {
        "game_id",
        "player_id",
        "pitcher_id",
        "event",
        "outs",
        "on_1b",
        "on_2b",
        "on_3b",
        "inning",
        "score_diff",
        "runs_scored",
        "observed_at",
    }
    if missing := required - set(events):
        raise KeyError(f"Events missing columns: {sorted(missing)}")

    frame = events.copy()
    frame["event"] = frame["event"].astype(str).str.strip().str.lower()
    frame = frame[frame["event"].isin(_PA_OUTCOME_BY_EVENT)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "pa_id",
                "game_id",
                "player_id",
                "pitcher_id",
                "outcome",
                "state",
                "runs_scored",
                "observed_at",
            ]
        )

    frame["outcome"] = frame["event"].map(_PA_OUTCOME_BY_EVENT)
    frame["state"] = frame.apply(_base_out_state, axis=1)
    frame["pa_id"] = frame.apply(
        lambda row: (
            f"{row['game_id']}:{row['inning']}:{row['player_id']}:"
            f"{pd.to_datetime(row['observed_at'], utc=True).isoformat()}"
        ),
        axis=1,
    )
    return frame[
        [
            "pa_id",
            "game_id",
            "player_id",
            "pitcher_id",
            "outcome",
            "state",
            "runs_scored",
            "observed_at",
        ]
    ].reset_index(drop=True)


def pa_outcome_target(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the PA outcome as binary indicator columns.

    Each column ``outcome_single``, ``outcome_home_run``, etc. is the binary
    target for a per-outcome calibrated classifier, so the outcome mix can be
    predicted per PA and renormalized to sum to one.
    """
    result = frame.copy()
    for outcome in PaOutcome:
        result[f"outcome_{outcome.value}"] = (result["outcome"] == outcome).astype(int)
    return result


def validate_label_frame(frame: pd.DataFrame, *, as_of: datetime) -> None:
    """Fail-closed checks: no future observed_at, no missing outcome."""
    if frame.empty:
        raise ValueError("No plate-appearance labels")
    observed = pd.to_datetime(frame["observed_at"], utc=True)
    if (observed > pd.Timestamp(as_of)).any():
        raise ValueError("Plate-appearance labels contain future observations")
    if frame["outcome"].isna().any():
        raise ValueError("Plate-appearance labels contain unresolved outcomes")
