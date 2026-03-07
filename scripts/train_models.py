"""Simple CLI to train all three betting models.

This is intended for automated workflows (e.g. GitHub Actions) where we
want to re-train the models on fresh data without using the Streamlit UI.

The script mirrors the logic in the ``Models`` tab of the dashboard.
"""
from datetime import datetime
import sys
from pathlib import Path

# allow imports from the repository root (so "src" is on sys.path)
ROOT = Path(__file__).parent.parent
# ensure src/ is checked before ROOT so src/models shadows the root models/ (joblib) folder
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

from models.features import build_model_features
from models.underdog_model import train_moneyline_model
from models.spread_model import train_spread_model
from models.totals_model import train_totals_model


def main(start_year: int = 2020, end_year: int = None) -> None:
    if end_year is None:
        end_year = datetime.utcnow().year

    print(f"Building feature matrix for {start_year}-{end_year}…")
    feats = build_model_features(start_year, end_year)

    print("Training moneyline model…")
    ml_res = train_moneyline_model(feats)
    print(f"  -> ROC-AUC {ml_res['metrics']['roc_auc']:.4f}")

    print("Training spread model…")
    sp_res = train_spread_model(feats)
    print(f"  -> ROC-AUC {sp_res['metrics']['roc_auc']:.4f}")

    print("Training totals model…")
    ou_res = train_totals_model(feats)
    print(f"  -> ROC-AUC {ou_res['metrics']['roc_auc']:.4f}")

    print("Training complete. Models written to /models directory.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all betting models")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="first season to include in the feature matrix",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="last season to include (default: current year)",
    )
    args = parser.parse_args()

    main(start_year=args.start_year, end_year=args.end_year)
