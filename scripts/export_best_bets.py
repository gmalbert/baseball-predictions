"""
scripts/export_best_bets.py — MLB (baseball-predictions)
Reads data_files/processed/picks_today.csv and its metadata (written by src/picks/daily_pipeline.py)
and writes data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

SPORT = "MLB"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
SRC_PATH = Path("data_files/processed/picks_today.csv")
META_PATH = Path("data_files/processed/picks_today.meta.json")


def _write(bets: list, notes: str = "", status: str | None = None, target_date: str | None = None, picks_count: int | None = None) -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    if status:
        payload["meta"]["status"] = status
    if target_date:
        payload["meta"]["target_date"] = target_date
    if picks_count is not None:
        payload["meta"]["picks_count"] = picks_count
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _tier_from_badge(badge: str, confidence: float | None) -> str:
    if badge == "BET" and (confidence or 0) >= 0.60:
        return "Elite"
    elif badge == "BET":
        return "Strong"
    elif badge == "LEAN":
        return "Good"
    return "Standard"


def main() -> None:
    today = date.today()

    # Baseball season: March–November
    month = today.month
    if not (3 <= month <= 11):
        _write([], "MLB off-season", status="no_games", target_date=str(today), picks_count=0)
        return

    if not SRC_PATH.exists():
        _write([], "Canonical pick snapshot not found — daily pipeline may not have run yet", status="pipeline_pending", target_date=str(today), picks_count=0)
        return

    metadata = {}
    if META_PATH.exists():
        try:
            metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _write([], f"Failed to read {META_PATH}: {e}", status="pipeline_failed", target_date=str(today), picks_count=0)
            return

    status = metadata.get("status", "ok")
    target_date = metadata.get("target_date", str(today))
    if status in {"pipeline_failed", "no_games", "no_qualifying_picks"}:
        _write([], metadata.get("notes", status), status=status, target_date=target_date, picks_count=metadata.get("picks_count", 0))
        return

    try:
        import pandas as pd
        df = pd.read_csv(SRC_PATH)
    except Exception as e:
        _write([], f"Failed to read {SRC_PATH}: {e}", status="pipeline_failed", target_date=target_date, picks_count=0)
        return

    if df.empty:
        _write([], f"No MLB picks for {today}", status="no_qualifying_picks", target_date=target_date, picks_count=0)
        return

    # Filter to today
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        df = df[df["game_date"] == today]

    # Normalize the daily pipeline's legacy CSV fields at the contract boundary.
    if "pick_type" in df.columns and "bet_type" not in df.columns:
        type_map = {"underdog": "Moneyline", "spread": "Spread", "over_under": "Over/Under"}
        df["bet_type"] = df["pick_type"].map(type_map).fillna(df["pick_type"])
    if "pick_value" in df.columns and "pick" not in df.columns:
        df["pick"] = df["pick_value"]
    if "confidence_score" in df.columns and "confidence" not in df.columns:
        df["confidence"] = df["confidence_score"]

    # Exclude PASS signals
    if "badge" in df.columns:
        df = df[df["badge"].isin(["BET", "LEAN"])]

    if df.empty:
        _write([], f"No qualifying MLB picks for {today}", status="no_qualifying_picks", target_date=target_date, picks_count=0)
        return

    bets = []
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game = f"{away} @ {home}"
        badge = str(row.get("badge", "LEAN"))
        conf = _safe_float(row.get("prob_home_win", row.get("confidence")))
        edge = _safe_float(row.get("edge"))
        tier = _tier_from_badge(badge, conf)

        bet_type_raw = str(row.get("bet_type", "Moneyline"))
        bt_map = {"Moneyline": "Moneyline", "moneyline": "Moneyline",
                  "Run Line": "Spread", "Spread": "Spread",
                  "Over/Under": "Over/Under", "Total": "Over/Under"}
        bet_type = bt_map.get(bet_type_raw, bet_type_raw)

        bet: dict = {
            "game_date": str(today),
            "game_time": str(row.get("game_time", "")) or None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": bet_type,
            "pick": str(row.get("pick", home)),
            "confidence": conf,
            "edge": edge,
            "tier": tier,
            "odds": int(row["odds"]) if "odds" in row and _safe_float(row.get("odds")) is not None else None,
            "line": _safe_float(row.get("line")),
            "notes": str(row.get("notes", "")) or None,
        }
        bets.append(bet)

    _write(bets, status="ok", target_date=target_date, picks_count=len(bets))


if __name__ == "__main__":
    main()
