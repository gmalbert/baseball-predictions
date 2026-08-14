"""Expand model outputs to the canonical one-row-per-selection grain."""

from __future__ import annotations

import pandas as pd


def binary_selection_probabilities(
    frame: pd.DataFrame,
    *,
    positive_probability: str,
    positive_selection: str,
    negative_selection: str,
) -> pd.DataFrame:
    required = {"game_id", positive_probability}
    if missing := required - set(frame):
        raise KeyError(f"Missing columns: {sorted(missing)}")
    positive = frame[["game_id", positive_probability]].rename(
        columns={positive_probability: "probability"}
    )
    positive["selection"] = positive_selection
    negative = positive.copy()
    negative["selection"] = negative_selection
    negative["probability"] = 1.0 - negative["probability"]
    result = pd.concat([positive, negative], ignore_index=True)
    if not result["probability"].between(0, 1).all():
        raise ValueError("Probability outside [0, 1]")
    return result[["game_id", "selection", "probability"]]


def three_way_selection_probabilities(
    frame: pd.DataFrame,
    *,
    win_probability: str,
    push_probability: str,
    win_selection: str,
    lose_selection: str,
) -> pd.DataFrame:
    required = {"game_id", win_probability, push_probability}
    if missing := required - set(frame):
        raise KeyError(f"Missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for row in frame[list(required)].to_dict("records"):
        win = float(row[win_probability])
        push = float(row[push_probability])
        loss = 1.0 - win - push
        if min(win, push, loss) < 0:
            raise ValueError("Invalid win/push/loss probability distribution")
        rows.extend(
            [
                {
                    "game_id": row["game_id"],
                    "selection": win_selection,
                    "probability": win,
                    "push_probability": push,
                },
                {
                    "game_id": row["game_id"],
                    "selection": lose_selection,
                    "probability": loss,
                    "push_probability": push,
                },
            ]
        )
    return pd.DataFrame(rows)
