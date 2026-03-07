# src/evaluation/clv.py
"""Closing Line Value (CLV) analysis.

CLV is the #1 indicator of long-term betting success.
If you consistently beat the closing line, you will be profitable.
"""

from __future__ import annotations

import pandas as pd

from ..models.features import implied_probability


def calculate_clv(
    picks_df: pd.DataFrame,
    opening_odds_col: str = "opening_odds",
    closing_odds_col: str = "closing_odds",
) -> pd.DataFrame:
    """Calculate Closing Line Value for each pick.

    CLV = implied_prob(closing_line) − implied_prob(opening_line)

    Positive CLV means the line moved in your direction after you picked it,
    indicating you identified value before the market corrected.

    Args:
        picks_df:          DataFrame containing at least an opening and a
                           closing odds column (American format).
        opening_odds_col:  Column name for the opening line.
        closing_odds_col:  Column name for the closing line.

    Returns:
        Copy of ``picks_df`` with added columns:
        ``opening_implied``, ``closing_implied``, ``clv``, ``clv_cents``.
    """
    df = picks_df.copy()

    df["opening_implied"] = df[opening_odds_col].apply(implied_probability)
    df["closing_implied"] = df[closing_odds_col].apply(implied_probability)
    df["clv"] = df["closing_implied"] - df["opening_implied"]
    df["clv_cents"] = (df["clv"] * 100).round(1)

    return df


def clv_report(df: pd.DataFrame) -> dict:
    """Summarize CLV performance.

    Args:
        df: DataFrame output from :func:`calculate_clv` (must have a
            ``clv`` column).

    Returns:
        Dict with summary statistics.
    """
    avg_clv = float(df["clv"].mean())
    pct_positive = float((df["clv"] > 0).mean())

    report = {
        "total_picks": len(df),
        "avg_clv": round(avg_clv, 4),
        "avg_clv_cents": round(avg_clv * 100, 1),
        "pct_positive_clv": round(pct_positive, 3),
        "median_clv_cents": round(float(df["clv"].median()) * 100, 1),
    }

    print("\n=== Closing Line Value Report ===")
    print(f"Avg CLV:      {report['avg_clv_cents']:+.1f} cents")
    print(f"Positive CLV: {report['pct_positive_clv']:.1%} of picks")
    print(f"Median CLV:   {report['median_clv_cents']:+.1f} cents")

    return report
