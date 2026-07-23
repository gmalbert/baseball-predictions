"""Daily picks generation pipeline.

Run the full morning pipeline (8 AM – 11:25 AM window):
    1. Fetch schedule + probable pitchers
    2. Fetch consensus odds
    3. Fetch weather
    4. Build feature matrix
    5. Run all 3 models
    6. Filter by edge / confidence thresholds
    7. Store picks to CSV  (parquet append coming later)

The morning consensus odds snapshot is saved so the 4 PM afternoon_refresh
job can compare lines and detect significant movement.
"""
from __future__ import annotations

import logging
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.odds import fetch_current_odds, get_consensus_line
from src.ingestion.weather import fetch_weather_for_games
from src.models.underdog_model import predict_moneyline
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals
from src.models.features import MONEYLINE_FEATURES, SPREAD_FEATURES, TOTALS_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
TOTALS_MODEL_FILENAME = "totals_xgb_v1.joblib"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data_files" / "processed"
PICKS_TODAY_PATH = PROCESSED_DIR / "picks_today.csv"
PICKS_METADATA_PATH = PROCESSED_DIR / "picks_today.meta.json"
MODEL_FEATURES_PATH = PROCESSED_DIR / "model_features.parquet"
PICK_COLUMNS = [
    "game_id", "away_team", "home_team", "pick_type", "pick_value",
    "predicted_prob", "confidence_score", "edge", "game_date", "source",
]

# Minimum thresholds to publish a pick
MIN_EDGE_UNDERDOG = 0.03
MIN_EDGE_SPREAD = 0.03
MIN_EDGE_TOTALS = 0.025
MIN_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_daily_pipeline(target_date: Optional[date] = None) -> dict:
    """Run the full morning picks-generation pipeline.

    Returns:
        dict with keys ``underdog``, ``spread``, ``over_under``, each a
        list of pick dicts.
    """
    target_date = target_date or date.today()
    logger.info("Running daily pipeline for %s", target_date)

    # ---- 1. Schedule --------------------------------------------------------
    schedule = fetch_todays_probable_pitchers()
    if schedule.empty:
        logger.warning("No games found for today")
        _write_pick_artifacts({}, target_date, status="no_games", notes="No games were found on the target date.")
        return {"underdog": [], "spread": [], "over_under": []}
    logger.info("Found %d games today", len(schedule))

    # ---- 2. Odds ------------------------------------------------------------
    odds_raw = fetch_current_odds()
    consensus = get_consensus_line(odds_raw)

    # Persist morning snapshot so afternoon_refresh can detect movement
    _save_consensus_snapshot(consensus, target_date, label="morning")

    game_odds = _pivot_odds(consensus)

    # ---- 3. Weather ---------------------------------------------------------
    weather = fetch_weather_for_games(schedule)

    # ---- 4. Feature matrix --------------------------------------------------
    features = _build_todays_features(schedule, game_odds, weather)

    # ---- 5 & 6. Models + filtering ------------------------------------------
    picks: dict = {}

    underdog_preds = predict_moneyline(
        model_or_path=MODEL_DIR / "moneyline_xgb_v1.joblib",
        game_features=features,
        home_ml_col="home_moneyline",
        away_ml_col="away_moneyline",
    )
    picks["underdog"] = _format_picks(
        _filter_picks(underdog_preds, MIN_EDGE_UNDERDOG, MIN_CONFIDENCE), "underdog"
    )

    spread_preds = predict_spread(
        model_or_path=MODEL_DIR / "spread_xgb_v1.joblib",
        game_features=features,
        spread_price_col="home_spread_price",
    )
    picks["spread"] = _format_picks(
        _filter_picks(spread_preds, MIN_EDGE_SPREAD, MIN_CONFIDENCE), "spread"
    )

    totals_preds = predict_totals(
        model_or_path=MODEL_DIR / TOTALS_MODEL_FILENAME,
        game_features=features,
        over_price_col="over_price",
        under_price_col="under_price",
    )
    picks["over_under"] = _format_picks(
        _filter_picks(totals_preds, MIN_EDGE_TOTALS, MIN_CONFIDENCE), "over_under"
    )

    # ---- 7. Storage ---------------------------------------------------------
    total = sum(len(v) for v in picks.values())
    status = "ok" if total else "no_qualifying_picks"
    notes = "" if total else "Games and odds were processed; no pick cleared the configured edge threshold."
    _store_picks(picks, target_date, source="morning", status=status, notes=notes)
    logger.info(
        "Generated %d picks: %d underdog, %d spread, %d O/U",
        total,
        len(picks["underdog"]),
        len(picks["spread"]),
        len(picks["over_under"]),
    )
    return picks


# ---------------------------------------------------------------------------
# Internal helpers (also imported by afternoon_refresh)
# ---------------------------------------------------------------------------

def _pivot_odds(consensus: pd.DataFrame) -> pd.DataFrame:
    """Pivot consensus odds into one wide row per game."""
    games = consensus[["game_id", "away_team", "home_team"]].drop_duplicates().copy()

    for _, row in consensus[consensus["market"] == "h2h"].iterrows():
        mask = games["game_id"] == row["game_id"]
        col = "home_moneyline" if row["outcome_name"] == row.get("home_team", "") else "away_moneyline"
        games.loc[mask, col] = row["median_price"]

    for _, row in consensus[consensus["market"] == "spreads"].iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == row.get("home_team", ""):
            games.loc[mask, "home_spread_point"] = row["median_point"]
            games.loc[mask, "home_spread_price"] = row["median_price"]
        else:
            games.loc[mask, "away_spread_point"] = row["median_point"]
            games.loc[mask, "away_spread_price"] = row["median_price"]

    for _, row in consensus[consensus["market"] == "totals"].iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == "Over":
            games.loc[mask, "posted_total"] = row["median_point"]
            games.loc[mask, "over_price"] = row["median_price"]
        else:
            games.loc[mask, "under_price"] = row["median_price"]

    return games


def _build_todays_features(
    schedule: pd.DataFrame,
    odds: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build today's rows with the same feature schema used during training.

    Current-season precomputed rows provide the latest available team-level
    statistics. Features that are game-specific or unavailable live use the
    corresponding historical median, with a neutral zero fallback.
    """
    features = schedule.merge(odds, on=["away_team", "home_team"], how="left")
    if "game_id" not in features.columns:
        if "game_id_x" in features.columns:
            features = features.rename(columns={"game_id_x": "game_id"})
            if "game_id_y" in features.columns:
                features = features.drop(columns=["game_id_y"])
        elif "game_id_y" in features.columns:
            features = features.rename(columns={"game_id_y": "game_id"})

    if not weather.empty:
        weather_cols = [
            c for c in ["game_id", "temp_f", "wind_mph", "wind_dir_deg",
                         "precip_prob_pct", "is_dome"]
            if c in weather.columns
        ]
        if "game_id" in weather_cols:
            features = features.merge(weather[weather_cols], on="game_id", how="left")

    if "temp_f" in features.columns:
        features["temp"] = pd.to_numeric(features["temp_f"], errors="coerce")
    if "wind_mph" in features.columns:
        features["windspeed"] = pd.to_numeric(features["wind_mph"], errors="coerce")
    if "is_day" not in features.columns:
        features["is_day"] = 1.0

    if MODEL_FEATURES_PATH.exists():
        baseline = pd.read_parquet(MODEL_FEATURES_PATH)
        _merge_latest_team_features(features, baseline)
    else:
        logger.warning("%s not found; live features will use neutral defaults", MODEL_FEATURES_PATH)

    required = list(dict.fromkeys(MONEYLINE_FEATURES + SPREAD_FEATURES + TOTALS_FEATURES))
    missing_before_fill = [col for col in required if col not in features.columns]
    baseline_medians = {
        col: pd.to_numeric(baseline[col], errors="coerce").median()
        for col in required
        if MODEL_FEATURES_PATH.exists() and col in baseline.columns
    }
    missing_values = {
        col: baseline_medians.get(col, 0.0)
        for col in missing_before_fill
    }
    if missing_values:
        features = pd.concat(
            [features, pd.DataFrame(missing_values, index=features.index)], axis=1
        )
    features = features.copy()
    for col in required:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(
                baseline_medians.get(col, 0.0)
            )

    # Recompute direct matchup fields after team snapshots are attached.
    derived = {}
    for col, home_col, away_col in (
        ("WPct_diff", "home_WPct", "away_WPct"),
        ("PythWPct_diff", "home_PythWPct", "away_PythWPct"),
        ("ERA_diff", "away_ERA", "home_ERA"),
        ("WHIP_diff", "away_WHIP", "home_WHIP"),
        ("RS_advantage", "home_RS_G", "away_RS_G"),
        ("RA_advantage", "away_RA_G", "home_RA_G"),
    ):
        if home_col in features and away_col in features:
            derived[col] = features[home_col] - features[away_col]
    if derived:
        features = features.assign(**derived)
    default_total = features["home_RS_G"] + features["away_RS_G"]
    features = features.assign(
        exp_total=features.get("posted_total", default_total).fillna(default_total),
        hometeam=features["home_team"],
        visteam=features["away_team"],
        date=features.get("date", pd.Timestamp.now().date()),
    )
    _validate_live_feature_schema(features)
    return features


def _normalise_team_name(value: object) -> str:
    """Map MLB API names such as 'New York Yankees' to Retrosheet names."""
    name = " ".join(str(value or "").lower().replace(".", "").split())
    aliases = {
        "los angeles angels": "angels", "la angels": "angels",
        "los angeles dodgers": "dodgers", "la dodgers": "dodgers",
        "arizona diamondbacks": "diamondbacks", "atlanta braves": "braves",
        "baltimore orioles": "orioles", "boston red sox": "red sox",
        "chicago cubs": "cubs", "chicago white sox": "white sox",
        "cincinnati reds": "reds", "cleveland guardians": "guardians",
        "colorado rockies": "rockies", "detroit tigers": "tigers",
        "houston astros": "astros", "kansas city royals": "royals",
        "miami marlins": "marlins", "milwaukee brewers": "brewers",
        "minnesota twins": "twins", "new york mets": "mets",
        "new york yankees": "yankees", "oakland athletics": "athletics",
        "philadelphia phillies": "phillies", "pittsburgh pirates": "pirates",
        "san diego padres": "padres", "san francisco giants": "giants",
        "seattle mariners": "mariners", "st louis cardinals": "cardinals",
        "tampa bay rays": "rays", "texas rangers": "rangers",
        "toronto blue jays": "blue jays", "washington nationals": "nationals",
    }
    return aliases.get(name, name)


def _merge_latest_team_features(features: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Attach the latest season team-side features to live game rows."""
    if baseline.empty or not {"hometeam", "visteam"}.issubset(baseline.columns):
        return
    baseline = baseline.copy()
    if "date" in baseline:
        baseline["date"] = pd.to_datetime(baseline["date"], errors="coerce")
    if "season" in baseline:
        current = baseline[baseline["season"] == baseline["season"].max()]
        if not current.empty:
            baseline = current

    side_frames = []
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        value_cols = [c for c in baseline.columns if c.startswith(f"{side}_")]
        if not value_cols:
            continue
        team_rows = baseline[[team_col] + value_cols + (["date"] if "date" in baseline else [])].copy()
        team_rows["team_key"] = team_rows[team_col].map(_normalise_team_name)
        team_rows = team_rows.sort_values("date" if "date" in team_rows else team_col)
        team_rows = team_rows.drop_duplicates("team_key", keep="last")
        team_rows = team_rows.rename(columns={c: f"team_{c[len(side) + 1:]}" for c in value_cols})
        side_frames.append(team_rows.set_index("team_key"))

    if side_frames:
        team_snapshot = pd.concat(side_frames, axis=0).groupby(level=0).last()
    else:
        team_snapshot = pd.DataFrame()

    for target_side, live_col in (("home", "home_team"), ("away", "away_team")):
        if team_snapshot.empty:
            continue
        keys = features[live_col].map(_normalise_team_name)
        team_values = team_snapshot.reindex(keys).reset_index(drop=True)
        team_values = team_values.reset_index(drop=True)
        for col in team_values.columns:
            if col.startswith("team_"):
                features[f"{target_side}_{col[5:]}"] = team_values[col].to_numpy()


def _validate_live_feature_schema(features: pd.DataFrame) -> None:
    required = list(dict.fromkeys(MONEYLINE_FEATURES + SPREAD_FEATURES + TOTALS_FEATURES))
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(f"Live feature schema incomplete; missing {len(missing)} columns: {missing}")


def _filter_picks(predictions: pd.DataFrame, min_edge: float, min_confidence: float) -> pd.DataFrame:
    """Return only rows that clear edge and confidence thresholds."""
    edge_col = next((c for c in ("edge", "edge_home", "edge_away", "edge_over") if c in predictions.columns), None)
    conf_col = next((c for c in ("confidence_score", "pick_prob") if c in predictions.columns), None)

    mask = pd.Series([True] * len(predictions), index=predictions.index)
    if edge_col:
        mask &= predictions[edge_col].fillna(0) >= min_edge
    if conf_col:
        mask &= predictions[conf_col].fillna(0) >= min_confidence

    sort_col = conf_col or edge_col
    filtered = predictions[mask].copy()
    if sort_col:
        filtered = filtered.sort_values(sort_col, ascending=False)
    return filtered


def _format_picks(df: pd.DataFrame, pick_type: str) -> list[dict]:
    """Serialize a predictions DataFrame to a list of pick dicts."""
    picks = []
    for _, row in df.iterrows():
        d = row.to_dict()
        picks.append({
            "game_id": d.get("game_id"),
            "away_team": d.get("away_team") or d.get("visteam"),
            "home_team": d.get("home_team") or d.get("hometeam"),
            "pick_type": pick_type,
            "pick_value": d.get("pick_side") or d.get("pick") or "",
            "predicted_prob": round(float(d.get("pred_home_win_prob") or d.get("pick_prob") or d.get("pred_cover_prob") or 0), 3),
            "confidence_score": round(float(d.get("pick_prob") or d.get("pred_cover_prob") or d.get("pred_home_win_prob") or 0), 3),
            "edge": round(float(d.get("edge") or d.get("edge_home") or d.get("edge_away") or d.get("edge_over") or 0), 3),
        })
    return picks


def _write_pick_artifacts(picks: dict, target_date: date, status: str, notes: str = "", source: str = "morning") -> None:
    """Write the dated history file, canonical snapshot, and status metadata."""
    all_picks = [p for pick_list in picks.values() for p in pick_list]
    df = pd.DataFrame(all_picks, columns=PICK_COLUMNS)
    df["date"] = target_date.isoformat()
    df["source"] = source

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / f"picks_{target_date.isoformat()}.csv"
    df.to_csv(outpath, index=False)
    df.to_csv(PICKS_TODAY_PATH, index=False)

    metadata = {
        "status": status,
        "target_date": target_date.isoformat(),
        "picks_count": len(df),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "notes": notes,
    }
    PICKS_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Pick artifacts saved → %s and %s", outpath, PICKS_TODAY_PATH)


def _store_picks(picks: dict, target_date: date, source: str = "morning", status: str = "ok", notes: str = "") -> None:
    """Write pick artifacts, including a valid empty snapshot."""
    _write_pick_artifacts(picks, target_date, status=status, notes=notes, source=source)


def write_pipeline_status(status: str, target_date: Optional[date] = None, notes: str = "") -> None:
    """Persist an explicit pipeline status when orchestration catches a failure."""
    _write_pick_artifacts({}, target_date or date.today(), status=status, notes=notes, source="pipeline")


def _save_consensus_snapshot(consensus: pd.DataFrame, target_date: date, label: str) -> None:
    """Persist a consensus odds snapshot for later comparison."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"consensus_{target_date.isoformat()}_{label}.parquet"
    consensus.to_parquet(path, index=False)
    logger.info("Consensus snapshot (%s) saved → %s", label, path)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    run_daily_pipeline()
