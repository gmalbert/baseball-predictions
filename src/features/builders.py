"""Reusable, point-in-time baseball feature builders for F16-F42.

The builders operate on provider-neutral observation frames.  Licensed source
adapters map into these columns; every result carries source watermarks,
missingness, and uncertainty rather than converting unknown states to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.base import validate_family


@dataclass
class FrameFeatureBuilder:
    observations: pd.DataFrame
    name: ClassVar[str] = "generic"
    version: ClassVar[str] = "1.0.0"
    required_sources: ClassVar[tuple[str, ...]] = ()
    entity_keys: ClassVar[tuple[str, ...]] = ("game_id",)

    def eligible(self, as_of: datetime) -> pd.DataFrame:
        frame = self.observations.copy()
        if "observed_at" not in frame:
            raise KeyError(f"{self.name} observations require observed_at")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame[frame["observed_at"] <= pd.Timestamp(as_of)].copy()

    def finish(self, frame: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        if frame.empty:
            raise ValueError(f"No eligible {self.name} observations at {as_of}")
        frame["as_of_time"] = pd.Timestamp(as_of)
        if "source_max_observed_at" not in frame:
            frame["source_max_observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame[f"{self.name}_missing"] = frame["source_max_observed_at"].isna()
        return frame

    def validate(self, frame: pd.DataFrame, *, as_of: datetime) -> None:
        validate_family(frame, as_of=as_of)


@dataclass
class TeamFormBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "team_form"
    required_sources: ClassVar[tuple[str, ...]] = ("team_game",)
    entity_keys: ClassVar[tuple[str, ...]] = ("team_id",)
    half_life: float = 10.0
    prior_weight: float = 10.0

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of).sort_values(["team_id", "event_time"])
        rows = []
        values = [
            column
            for column in ("runs_for", "runs_against", "offense", "defense", "baserunning")
            if column in eligible
        ]
        for team_id in entities["team_id"].drop_duplicates():
            history = eligible[eligible["team_id"] == team_id]
            row: dict[str, object] = {"team_id": team_id}
            for column in values:
                numeric = pd.to_numeric(history[column], errors="coerce").dropna()
                if numeric.empty:
                    row[f"{column}_ewm"] = np.nan
                    row[f"{column}_posterior_sd"] = np.nan
                    continue
                weights = np.power(0.5, np.arange(len(numeric) - 1, -1, -1) / self.half_life)
                league = float(pd.to_numeric(eligible[column], errors="coerce").mean())
                posterior = (float(np.dot(numeric, weights)) + self.prior_weight * league) / (
                    weights.sum() + self.prior_weight
                )
                row[f"{column}_ewm"] = posterior
                row[f"{column}_posterior_sd"] = float(
                    numeric.std(ddof=0) / sqrt(max(len(numeric), 1))
                )
            row["games_played"] = len(history)
            row["observed_at"] = history["observed_at"].max() if not history.empty else pd.NaT
            rows.append(row)
        return self.finish(pd.DataFrame(rows), as_of)


@dataclass
class PlayerTalentBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "player_talent"
    required_sources: ClassVar[tuple[str, ...]] = ("player_event",)
    entity_keys: ClassVar[tuple[str, ...]] = ("player_id",)
    prior_strength: float = 50.0

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        successes = "successes" if "successes" in eligible else "value"
        trials = "trials" if "trials" in eligible else None
        league_rate = (
            float(eligible[successes].sum() / eligible[trials].sum())
            if trials and eligible[trials].sum()
            else float(eligible[successes].mean())
        )
        alpha0 = max(league_rate * self.prior_strength, 1e-6)
        beta0 = max((1 - league_rate) * self.prior_strength, 1e-6)
        rows = []
        for player_id in entities["player_id"].drop_duplicates():
            history = eligible[eligible["player_id"] == player_id]
            successes_value = float(history[successes].sum()) if not history.empty else 0.0
            trials_value = (
                float(history[trials].sum())
                if trials and not history.empty
                else float(len(history))
            )
            alpha = alpha0 + successes_value
            beta = beta0 + max(trials_value - successes_value, 0.0)
            rows.append(
                {
                    "player_id": player_id,
                    "talent_mean": alpha / (alpha + beta),
                    "talent_sd": sqrt(alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1))),
                    "sample_size": trials_value,
                    "availability_probability": float(
                        history.get("availability_probability", pd.Series([1.0])).iloc[-1]
                    )
                    if not history.empty
                    else 0.0,
                    "observed_at": history["observed_at"].max() if not history.empty else pd.NaT,
                }
            )
        return self.finish(pd.DataFrame(rows), as_of)


@dataclass
class LineupOffenseBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "lineup_offense"
    required_sources: ClassVar[tuple[str, ...]] = ("lineup_snapshot", "player_talent")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        # Player-talent columns may be absent from the observation frame (a
        # confirmed lineup without a talent family).  Missing talent is
        # explicit missingness, not a fabricated zero: contribution stays NaN.
        weight = eligible.get("projected_pa")
        weight = (
            pd.to_numeric(weight, errors="coerce").fillna(1.0)
            if weight is not None
            else pd.Series(1.0, index=eligible.index)
        )
        eligible["weight"] = weight
        availability = eligible.get("availability_probability")
        eligible["availability"] = (
            pd.to_numeric(availability, errors="coerce").fillna(1.0)
            if availability is not None
            else pd.Series(1.0, index=eligible.index)
        )
        talent_mean = eligible.get("talent_mean")
        if talent_mean is not None:
            eligible["talent_mean"] = pd.to_numeric(talent_mean, errors="coerce")
            contribution = eligible["talent_mean"] * eligible["weight"] * eligible["availability"]
            uncertainty = (
                "talent_sd",
                lambda values: float(
                    np.sqrt(np.square(pd.to_numeric(values, errors="coerce")).sum())
                ),
            )
        else:
            eligible["talent_mean"] = np.nan
            contribution = np.nan
            # No talent_sd column exists; aggregate a constant so the shape is
            # stable and the missing flag carries the signal.
            uncertainty = ("weight", lambda values: float("nan"))
        eligible["contribution"] = contribution
        grouped = eligible.groupby("game_id", as_index=False).agg(
            lineup_offense=("contribution", "sum"),
            lineup_expected_pa=("weight", "sum"),
            lineup_uncertainty=uncertainty,
            observed_at=("observed_at", "max"),
        )
        grouped["lineup_offense_missing"] = grouped["lineup_offense"].isna()
        return self.finish(
            entities[["game_id"]].drop_duplicates().merge(grouped, on="game_id", how="left"), as_of
        )


@dataclass
class StarterProjectionBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "starter_projection"
    required_sources: ClassVar[tuple[str, ...]] = ("pitcher_pitch", "lineup_snapshot")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of).sort_values("event_time")
        group_cols = [
            column for column in ("game_id", "team_id", "player_id") if column in eligible
        ]
        latest = eligible.groupby(group_cols, as_index=False).tail(1)
        pitches = pd.to_numeric(latest.get("recent_pitch_count", 0), errors="coerce")
        rest = pd.to_numeric(latest.get("days_rest", 4), errors="coerce")
        velo_delta = pd.to_numeric(latest.get("velo_delta", 0), errors="coerce")
        role = latest.get("role", pd.Series("starter", index=latest.index))
        latest["expected_innings"] = np.clip(
            6.0 - np.maximum(pitches - 95, 0) / 20 + np.minimum(rest - 4, 1) * 0.2, 1, 7
        )
        latest.loc[role.isin(["opener", "bulk", "tandem"]), "expected_innings"] *= 0.55
        latest["degradation_risk"] = 1 / (
            1 + np.exp(-(np.maximum(pitches - 90, 0) / 10 - rest / 7 - velo_delta))
        )
        keep = group_cols + ["expected_innings", "degradation_risk", "observed_at"]
        return self.finish(latest[keep], as_of)


@dataclass
class BullpenAvailabilityBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "bullpen_availability"
    required_sources: ClassVar[tuple[str, ...]] = ("reliever_usage", "roster")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        workload = pd.to_numeric(eligible.get("pitches_last_3d", 0), errors="coerce").fillna(0)
        consecutive = pd.to_numeric(eligible.get("consecutive_days", 0), errors="coerce").fillna(0)
        availability = np.clip(np.exp(-workload / 55) * np.exp(-consecutive / 3), 0, 1)
        quality = pd.to_numeric(eligible.get("quality", 0), errors="coerce").fillna(0)
        leverage = pd.to_numeric(eligible.get("leverage_weight", 1), errors="coerce").fillna(1)
        eligible["available_quality"] = availability * quality * leverage
        grouped = eligible.groupby(
            [column for column in ("game_id", "team_id") if column in eligible], as_index=False
        ).agg(
            bullpen_available_quality=("available_quality", "sum"),
            bullpen_available_arms=("available_quality", lambda values: int((values > 0.2).sum())),
            observed_at=("observed_at", "max"),
        )
        return self.finish(grouped, as_of)


@dataclass
class PitcherAvailabilityBuilder(FrameFeatureBuilder):
    """F33 pitcher availability from recent pitches, rest, and role.

    Consumes the provider-neutral ``pitcher_pitch`` observation frame
    (``pitches``, ``outs_recorded``, ``role``, ``event_time``) and emits one
    row per (game_id, team_id): starter availability/expected innings from
    recent workload and rest, plus bullpen fresh-arm count from trailing
    volume and consecutive days.
    """

    name: ClassVar[str] = "pitcher_availability"
    required_sources: ClassVar[tuple[str, ...]] = ("pitcher_pitch",)
    entity_keys: ClassVar[tuple[str, ...]] = ("game_id", "team_id")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        eligible["event_time"] = pd.to_datetime(eligible["event_time"], utc=True, errors="coerce")
        eligible["pitches"] = pd.to_numeric(eligible.get("pitches", 0), errors="coerce").fillna(0)
        eligible["outs_recorded"] = pd.to_numeric(
            eligible.get("outs_recorded", 0), errors="coerce"
        ).fillna(0)
        eligible["role"] = eligible.get("role", pd.Series("reliever", index=eligible.index)).fillna(
            "reliever"
        )
        eligible["days_since"] = (pd.Timestamp(as_of) - eligible["event_time"]).dt.days.fillna(99)
        eligible["consecutive_days"] = 1.0
        eligible["fresh"] = eligible["days_since"] <= 1
        # Reliever availability decays with trailing workload over the last 3 days.
        eligible["recent_pitches"] = np.where(eligible["days_since"] <= 3, eligible["pitches"], 0)
        workload = eligible.groupby(["player_id", "team_id"], as_index=False)[
            "recent_pitches"
        ].transform("sum")
        eligible["availability"] = np.where(
            eligible["role"] == "starter",
            np.clip(1 - np.maximum(eligible["pitches"] - 95, 0) / 40, 0, 1),
            np.clip(np.exp(-workload / 55), 0, 1),
        )
        eligible["expected_innings"] = np.clip(
            6.0 - np.maximum(eligible["pitches"] - 95, 0) / 20, 1, 7
        )
        eligible["used_arm"] = (eligible["role"] == "reliever") & (eligible["availability"] < 0.5)
        # Split so starter and reliever summaries aggregate cleanly at team grain.
        starter_rows = eligible[eligible["role"] == "starter"]
        reliever_rows = eligible[eligible["role"] != "starter"]

        def _team_summary(frame: pd.DataFrame, *, starter: bool) -> pd.DataFrame:
            keys = [column for column in ("game_id", "team_id") if column in frame]
            if frame.empty:
                return pd.DataFrame(columns=keys)
            if starter:
                return frame.groupby(keys, as_index=False).agg(
                    starter_availability=("availability", "max"),
                    starter_expected_innings=("expected_innings", "max"),
                    max_recent_pitches=("pitches", "max"),
                    observed_at=("observed_at", "max"),
                )
            return frame.groupby(keys, as_index=False).agg(
                bullpen_fresh_arms=("used_arm", lambda values: int((~values).sum())),
                bullpen_used_arms=("used_arm", "sum"),
                observed_at=("observed_at", "max"),
            )

        starter = _team_summary(starter_rows, starter=True)
        reliever = _team_summary(reliever_rows, starter=False)
        group_cols = [column for column in ("game_id", "team_id") if column in eligible]
        grouped = starter.merge(reliever, on=group_cols, how="outer", suffixes=("", "_reliever"))
        for column in (
            "starter_availability",
            "starter_expected_innings",
            "max_recent_pitches",
            "bullpen_fresh_arms",
            "bullpen_used_arms",
        ):
            if column not in grouped:
                grouped[column] = 0.0
        grouped["observed_at"] = grouped.get("observed_at", pd.NaT).fillna(
            grouped.get("observed_at_reliever", pd.NaT)
        )
        grouped = grouped.drop(columns=["observed_at_reliever"], errors="ignore")
        return self.finish(grouped, as_of)


@dataclass
class ParkWeatherRoofBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "park_weather_roof"
    required_sources: ClassVar[tuple[str, ...]] = ("weather_snapshot", "venue")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = (
            self.eligible(as_of)
            .sort_values("observed_at")
            .groupby("game_id", as_index=False)
            .tail(1)
        )
        temp_c = (pd.to_numeric(eligible["temperature_f"], errors="coerce") - 32) * 5 / 9
        humidity = pd.to_numeric(eligible["relative_humidity"], errors="coerce") / 100
        pressure = pd.to_numeric(eligible["pressure_hpa"], errors="coerce") * 100
        # Good engineering approximation for moist-air density.
        saturation = 610.94 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
        vapor = humidity * saturation
        eligible["air_density_kg_m3"] = (pressure - vapor) / (
            287.05 * (temp_c + 273.15)
        ) + vapor / (461.495 * (temp_c + 273.15))
        wind_direction = np.radians(pd.to_numeric(eligible["wind_direction_deg"], errors="coerce"))
        field_direction = np.radians(
            pd.to_numeric(eligible["field_orientation_deg"], errors="coerce")
        )
        wind_speed = pd.to_numeric(eligible["wind_speed_mph"], errors="coerce")
        eligible["wind_out_mph"] = wind_speed * np.cos(wind_direction - field_direction)
        eligible["wind_cross_mph"] = wind_speed * np.sin(wind_direction - field_direction)
        roof = (
            eligible.get("roof_status", pd.Series("unknown", index=eligible.index))
            .astype(str)
            .str.lower()
        )
        eligible["roof_closed"] = roof.isin(["closed", "dome"]).astype(float)
        eligible["roof_uncertain"] = roof.isin(["unknown", "possible", "tbd"]).astype(float)
        columns = [
            "game_id",
            "air_density_kg_m3",
            "wind_out_mph",
            "wind_cross_mph",
            "roof_closed",
            "roof_uncertain",
            "observed_at",
        ]
        return self.finish(eligible[columns], as_of)


def travel_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371.0088
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return earth_radius * 2 * atan2(sqrt(a), sqrt(1 - a))


@dataclass
class TravelScheduleBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "travel_schedule"
    required_sources: ClassVar[tuple[str, ...]] = ("schedule", "venue")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of).sort_values(["team_id", "event_time"])
        rows = []
        for team_id, history in eligible.groupby("team_id"):
            latest = history.iloc[-1]
            distance = 0.0
            if len(history) > 1:
                previous = history.iloc[-2]
                distance = travel_distance_km(
                    previous["latitude"],
                    previous["longitude"],
                    latest["latitude"],
                    latest["longitude"],
                )
            rows.append(
                {
                    "team_id": team_id,
                    "travel_km": distance,
                    "time_zones_crossed": abs(
                        float(latest.get("utc_offset", 0))
                        - float(history.iloc[-2].get("utc_offset", 0))
                    )
                    if len(history) > 1
                    else 0,
                    "days_rest": max(
                        (
                            pd.Timestamp(latest["event_time"])
                            - pd.Timestamp(history.iloc[-2]["event_time"])
                        ).days
                        - 1,
                        0,
                    )
                    if len(history) > 1
                    else np.nan,
                    "games_last_7d": int(
                        (
                            pd.to_datetime(history["event_time"], utc=True)
                            >= pd.Timestamp(as_of) - pd.Timedelta(days=7)
                        ).sum()
                    ),
                    "is_doubleheader": bool(latest.get("doubleheader_number", 0)),
                    "observed_at": history["observed_at"].max(),
                }
            )
        return self.finish(pd.DataFrame(rows), as_of)


@dataclass
class RegimeBuilder(FrameFeatureBuilder):
    name: ClassVar[str] = "regime"
    required_sources: ClassVar[tuple[str, ...]] = ("ruleset",)

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of).sort_values("valid_from")
        valid = eligible[
            (pd.to_datetime(eligible["valid_from"], utc=True) <= pd.Timestamp(as_of))
            & (
                eligible["valid_to"].isna()
                | (pd.to_datetime(eligible["valid_to"], utc=True) > pd.Timestamp(as_of))
            )
        ]
        if valid.empty:
            raise ValueError("No ruleset valid at cutoff")
        row = valid.iloc[-1]
        result = entities[["game_id"]].drop_duplicates().copy()
        for column in (
            "pitch_clock",
            "shift_restrictions",
            "larger_bases",
            "abs_challenge",
            "tracking_era",
        ):
            result[column] = row.get(column)
        result["observed_at"] = row["observed_at"]
        return self.finish(result, as_of)


# Advanced families reuse the same cutoff/watermark contract while provider-specific
# feature columns remain visible and typed in their input frames.
class BenchSubstitutionBuilder(LineupOffenseBuilder):
    name = "bench_substitution"


class BullpenSequencingBuilder(BullpenAvailabilityBuilder):
    name = "bullpen_sequencing"


class BatteryUmpireBuilder(FrameFeatureBuilder):
    name = "battery_umpire"
    required_sources = ("catcher", "umpire", "ruleset")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = (
            self.eligible(as_of)
            .sort_values("observed_at")
            .groupby("game_id", as_index=False)
            .tail(1)
        )
        columns = [
            column
            for column in eligible
            if column
            in {
                "game_id",
                "framing_runs",
                "blocking_runs",
                "caught_stealing_value",
                "battery_familiarity",
                "umpire_zone_runs",
                "umpire_consistency",
                "abs_challenge_rate",
                "observed_at",
            }
        ]
        return self.finish(eligible[columns], as_of)


class PitchQualityBuilder(FrameFeatureBuilder):
    name = "pitch_quality"
    required_sources = ("pitch_model",)

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        group = [column for column in ("game_id", "player_id") if column in eligible]
        values = [
            column
            for column in (
                "stuff_plus",
                "command_plus",
                "xera",
                "pitch_shape",
                "location_value",
                "repertoire_matchup",
            )
            if column in eligible
        ]
        result = eligible.groupby(group, as_index=False).agg(
            {**{column: "mean" for column in values}, "observed_at": "max"}
        )
        return self.finish(result, as_of)


class ContactDefenseBaserunningBuilder(FrameFeatureBuilder):
    name = "contact_defense_baserunning"
    required_sources = ("statcast", "fielding", "baserunning")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        group = [column for column in ("game_id", "team_id") if column in eligible]
        values = [
            column
            for column in (
                "xwoba",
                "barrel_rate",
                "hard_hit_rate",
                "expected_hr",
                "oaa",
                "arm_value",
                "sprint_speed",
                "running_value",
            )
            if column in eligible
        ]
        result = eligible.groupby(group, as_index=False).agg(
            {**{column: "mean" for column in values}, "observed_at": "max"}
        )
        return self.finish(result, as_of)


class RosterAvailabilityBuilder(PlayerTalentBuilder):
    name = "roster_availability"


class MatchupCompatibilityBuilder(PitchQualityBuilder):
    name = "matchup_compatibility"


class PlayerParkInteractionBuilder(FrameFeatureBuilder):
    """F31/F38 handedness, spray-profile, and versioned wall-geometry interaction."""

    name = "player_park_interaction"
    required_sources = ("batted_ball_profile", "venue_geometry")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        group = [column for column in ("game_id", "player_id") if column in eligible]
        values = [
            column
            for column in (
                "handedness_park_factor",
                "spray_match_score",
                "wall_distance_delta_ft",
                "wall_height_delta_ft",
                "expected_hr_park_delta",
            )
            if column in eligible
        ]
        result = eligible.groupby(group, as_index=False).agg(
            {**{column: "mean" for column in values}, "observed_at": "max"}
        )
        return self.finish(result, as_of)


class SeriesTimesThroughOrderBuilder(FrameFeatureBuilder):
    """F40 starter degradation and opponent familiarity without target-game events."""

    name = "series_tto"
    required_sources = ("pitcher_appearance", "plate_appearance", "schedule")

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of)
        group = [column for column in ("game_id", "player_id") if column in eligible]
        values = [
            column
            for column in (
                "tto_penalty_second",
                "tto_penalty_third",
                "opponent_pa_last_14d",
                "series_familiarity_pa",
                "repertoire_change_score",
            )
            if column in eligible
        ]
        result = eligible.groupby(group, as_index=False).agg(
            {**{column: "mean" for column in values}, "observed_at": "max"}
        )
        return self.finish(result, as_of)


class ProjectionPriorBuilder(FrameFeatureBuilder):
    """F41 projection/minor-league priors with explicit translation uncertainty."""

    name = "projection_prior"
    required_sources = ("projection_system", "minor_league_translation")
    entity_keys = ("player_id",)

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
        eligible = self.eligible(as_of).sort_values("observed_at")
        latest = eligible.groupby("player_id", as_index=False).tail(1)
        columns = [
            column
            for column in (
                "player_id",
                "projected_rate",
                "projected_playing_time",
                "translation_factor",
                "projection_sd",
                "level",
                "observed_at",
            )
            if column in latest
        ]
        result = (
            entities[["player_id"]]
            .drop_duplicates()
            .merge(latest[columns], on="player_id", how="left")
        )
        return self.finish(result, as_of)
