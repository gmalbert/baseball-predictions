"""Ensemble / meta-model utilities.

Combines predictions from multiple models via weighted probability averaging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_predictions(
    model_predictions: list[pd.DataFrame],
    weights: list[float] | None = None,
    prob_col: str = "predicted_prob",
    id_col: str = "game_id",
) -> pd.DataFrame:
    """Weighted average of multiple model predictions.

    Args:
        model_predictions: List of DataFrames, each containing id_col and prob_col.
        weights: Optional list of weights (must sum to 1.0).
                 Defaults to equal weighting across all models.
        prob_col: Column name holding each model's probability output.
        id_col:   Column uniquely identifying each game.

    Returns:
        DataFrame with columns [id_col, 'ensemble_prob'].

    Raises:
        ValueError: If weights don't sum to 1.0 (within tolerance).
    """
    if weights is None:
        weights = [1.0 / len(model_predictions)] * len(model_predictions)

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0 — got {sum(weights):.6f}"
        )

    if len(weights) != len(model_predictions):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of model prediction DataFrames ({len(model_predictions)})"
        )

    # Use first DataFrame's id column as the base
    result = model_predictions[0][[id_col]].copy()

    weighted_prob = np.zeros(len(result))
    for df, w in zip(model_predictions, weights):
        weighted_prob += df[prob_col].values * w

    result["ensemble_prob"] = weighted_prob.round(4)
    return result


def confidence_label(score: float) -> str:
    """Map a 0–1 confidence score to a human-readable tier.

    Tiers align with the bankroll strategy in 10-bankroll-strategy.md:
      HIGH   (score ≥ 0.65)  → half-Kelly sizing
      MEDIUM (score ≥ 0.35)  → quarter-Kelly sizing
      LOW    (score < 0.35)  → minimal sizing / tracked only
    """
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def compute_confidence_score(
    prob: "pd.Series | float",
    edge: "pd.Series | float",
    edge_clip: float = 0.20,
    prob_weight: float = 0.4,
    edge_weight: float = 0.6,
) -> "pd.Series | float":
    """Combine model probability strength and betting edge into a 0–1 score.

    Higher confidence when:
      - The model's predicted probability diverges significantly from 50 %
      - The edge over the implied probability is large

    Args:
        prob:        Predicted probability (scalar or Series).
        edge:        Betting edge = pred_prob − implied_prob (scalar or Series).
        edge_clip:   Maximum edge value considered (anything above is treated
                     the same as edge_clip).
        prob_weight: Weight given to the probability-strength component.
        edge_weight: Weight given to the edge component.

    Returns:
        Confidence score clipped to [0, 1].
    """
    prob_strength = (prob - 0.5).__abs__() * 2        # 0 at 50 %, 1 at 0/100 %
    edge_strength = (edge if hasattr(edge, "clip") else pd.Series([edge])).clip(
        0, edge_clip
    ) / edge_clip

    confidence = prob_weight * prob_strength + edge_weight * edge_strength
    if hasattr(confidence, "clip"):
        return confidence.clip(0, 1)
    return float(min(max(confidence, 0.0), 1.0))
