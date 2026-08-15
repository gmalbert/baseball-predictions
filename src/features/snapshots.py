"""Assemble point-in-time game snapshots from provider-neutral observation frames.

``build_game_snapshot_rows`` runs the lineup, bullpen, and pitcher-availability
feature families at one as-of cutoff and emits the canonical ``GameSnapshot``
records used by the replay pipeline and the catalog.  Each family passes the
shared watermark validator before it is merged, so a late or future observation
fails the build rather than leaking into a snapshot, and every family keeps its
own ``source_max_observed_at`` watermark in the snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable, ClassVar, Protocol

import pandas as pd

from src.contracts.domain import GameSnapshot, stable_id
from src.features.builders import (
    BullpenAvailabilityBuilder,
    LineupOffenseBuilder,
    PitcherAvailabilityBuilder,
    StarterProjectionBuilder,
)


class _Builder(Protocol):
    entity_keys: ClassVar[tuple[str, ...]]

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame: ...

    def validate(self, frame: pd.DataFrame, *, as_of: datetime) -> None: ...


_BUILDERS: dict[str, Callable[[dict[str, pd.DataFrame]], _Builder]] = {
    "lineup_offense": lambda observations: LineupOffenseBuilder(observations["lineup_snapshot"]),
    "bullpen_availability": lambda observations: BullpenAvailabilityBuilder(
        observations["reliever_usage"]
    ),
    "pitcher_availability": lambda observations: PitcherAvailabilityBuilder(
        observations["pitcher_pitch"]
    ),
    "starter_projection": lambda observations: StarterProjectionBuilder(
        observations["pitcher_pitch"]
    ),
}

# Observation-frame keys each builder factory actually reads, used for the
# fail-closed source check.
_REQUIRED_FRAMES = {
    "lineup_offense": ("lineup_snapshot",),
    "bullpen_availability": ("reliever_usage",),
    "pitcher_availability": ("pitcher_pitch",),
    "starter_projection": ("pitcher_pitch",),
}

_FAMILY_WATERMARK = "source_max_observed_at"


def _json_safe(value: object) -> float | int | str | bool | None:
    if value is None or isinstance(value, (bool, int, str, float)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()  # type: ignore[no-any-return]
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, (bool, int, float)):
            return item
        return str(item)
    return str(value)


def _watermark(value: object) -> datetime | None:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().astimezone(UTC)  # type: ignore[no-any-return]
    if isinstance(value, str):
        parsed = pd.Timestamp(value, tz="UTC")
        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime().astimezone(UTC)  # type: ignore[no-any-return]
    if value is None or pd.isna(value):
        return None
    return None


def build_game_snapshot_rows(
    games: pd.DataFrame,
    observations: dict[str, pd.DataFrame],
    *,
    as_of: datetime,
    feature_set_version: str = "mlb_game_v2",
    snapshot_type: str = "confirmed_lineup",
) -> list[GameSnapshot]:
    """Build one ``GameSnapshot`` per game at a single as-of cutoff.

    ``games`` needs ``game_id``, ``home_team_id``, and ``away_team_id``.
    ``observations`` maps an observation-frame name (``lineup_snapshot``,
    ``reliever_usage``, ``pitcher_pitch``) to its provider-neutral frame.
    A missing frame for a family raises, matching the fail-closed snapshot
    contract.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    required = {"game_id", "home_team_id", "away_team_id"}
    if missing := required - set(games):
        raise KeyError(f"Games missing columns: {sorted(missing)}")

    # Team-grain families consume one row per (game, team).
    team_base = pd.concat(
        [
            games[["game_id", "home_team_id", "away_team_id"]]
            .rename(columns={"home_team_id": "team_id"})
            .assign(side="home"),
            games[["game_id", "home_team_id", "away_team_id"]]
            .rename(columns={"away_team_id": "team_id"})
            .assign(side="away"),
        ],
        ignore_index=True,
    )
    team_base["as_of_time"] = pd.Timestamp(as_of)
    game_base = games[["game_id"]].copy()
    game_base["as_of_time"] = pd.Timestamp(as_of)

    # Build families independently at their natural grain; no cross-grain merge.
    families: dict[str, pd.DataFrame] = {}
    for name, factory in _BUILDERS.items():
        if missing := set(_REQUIRED_FRAMES[name]) - set(observations):
            raise KeyError(f"{name} requires frames {sorted(missing)}")
        builder = factory(observations)
        if "team_id" in builder.entity_keys:
            family = builder.build(team_base, as_of=as_of)
        else:
            family = builder.build(game_base, as_of=as_of)
        builder.validate(family, as_of=as_of)
        families[name] = family

    # Index team-grain families by game for fast per-game assembly.  Starter
    # projection is per-player; pick the probable starter (highest expected
    # innings) so the snapshot stays one row per (game, team).
    lineup = families["lineup_offense"].set_index("game_id")
    bullpen = families["bullpen_availability"].set_index(["game_id", "team_id"])
    pitcher = families["pitcher_availability"].set_index(["game_id", "team_id"])
    starter_index = families["starter_projection"].set_index(["game_id", "team_id"])
    starter = (
        starter_index.assign(_expected=starter_index.get("expected_innings", 0.0))
        .sort_values("_expected", ascending=False)
        .groupby(level=[0, 1], sort=False)
        .head(1)
        .drop(columns="_expected")
    )

    rows = []
    for game in games.to_dict("records"):
        game_id = game["game_id"]
        snapshot_id = stable_id("snapshot", game_id, as_of, snapshot_type, feature_set_version)
        row_hash = stable_id("row", game_id, as_of, snapshot_type, feature_set_version)
        features: dict[str, float | int | str | bool | None] = {}

        lineup_row = lineup.loc[game_id] if game_id in lineup.index else None
        if lineup_row is not None:
            for key, value in lineup_row.items():
                if key in {"as_of_time", _FAMILY_WATERMARK, "observed_at"}:
                    continue
                features[key] = _json_safe(value)

        watermarks: dict[str, datetime] = {}
        for side, team_id in (("home", game["home_team_id"]), ("away", game["away_team_id"])):
            for name, index in (
                ("bullpen_availability", bullpen),
                ("pitcher_availability", pitcher),
            ):
                key = (game_id, team_id)
                if key not in index.index:
                    continue
                row = index.loc[key]
                for column, value in row.items():
                    if column in {"as_of_time", _FAMILY_WATERMARK, "observed_at", "side"}:
                        continue
                    if column.endswith("_missing"):
                        continue
                    features[f"{side}_{name}_{column}"] = _json_safe(value)
            for name, index in (("starter_projection", starter),):
                key = (game_id, team_id)
                if key not in index.index:
                    continue
                row = index.loc[key]
                for column, value in row.items():
                    if column in {"as_of_time", _FAMILY_WATERMARK, "observed_at", "side"}:
                        continue
                    if column.endswith("_missing"):
                        continue
                    features[f"{side}_{name}_{column}"] = _json_safe(value)

        # Collect per-family watermarks.
        for name, index in (
            ("lineup_offense", lineup),
            ("bullpen_availability", bullpen),
            ("pitcher_availability", pitcher),
            ("starter_projection", starter),
        ):
            if name == "lineup_offense":
                if game_id not in index.index:
                    continue
                mark = _watermark(index.loc[game_id].get(_FAMILY_WATERMARK))
            else:
                home = (game_id, game["home_team_id"])
                away = (game_id, game["away_team_id"])
                marks = [
                    _watermark(index.loc[key].get(_FAMILY_WATERMARK))
                    for key in (home, away)
                    if key in index.index
                ]
                mark = max((m for m in marks if m), default=None)
            if mark:
                watermarks[name] = mark

        rows.append(
            GameSnapshot(
                snapshot_id=snapshot_id,
                game_id=game_id,
                as_of_time=as_of.astimezone(UTC),
                snapshot_type=snapshot_type,
                feature_set_version=feature_set_version,
                features=features,
                source_watermarks=watermarks,
                row_hash=row_hash,
                build_run_id=None,
                quality_status="passed",
            )
        )
    return rows
