"""Mandatory low-complexity and market baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DynamicElo:
    k_factor: float = 20.0
    home_advantage: float = 35.0
    preseason_rating: float = 1500.0
    season_regression: float = 0.25

    def __post_init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.last_season: int | None = None

    def rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.preseason_rating)

    def probability(
        self, home_team_id: str, away_team_id: str, starter_adjustment: float = 0.0
    ) -> float:
        difference = (
            self.rating(home_team_id)
            + self.home_advantage
            + starter_adjustment
            - self.rating(away_team_id)
        )
        return 1.0 / (1.0 + 10 ** (-difference / 400.0))

    def update(
        self, home_team_id: str, away_team_id: str, home_score: float, *, season: int
    ) -> float:
        if self.last_season is not None and season != self.last_season:
            for team, rating in self.ratings.items():
                self.ratings[team] = self.preseason_rating + (rating - self.preseason_rating) * (
                    1 - self.season_regression
                )
        self.last_season = season
        predicted = self.probability(home_team_id, away_team_id)
        actual = 1.0 if home_score > 0.5 else 0.0
        change = self.k_factor * (actual - predicted)
        self.ratings[home_team_id] = self.rating(home_team_id) + change
        self.ratings[away_team_id] = self.rating(away_team_id) - change
        return predicted


def fit_regularized_logistic(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=0.25, max_iter=2_000, random_state=random_state)),
        ]
    )
    model.fit(X, y)
    model.feature_cols_ = list(X.columns)
    return model


@dataclass
class MarketResidualModel:
    correction_model: Pipeline
    feature_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        market_probability: str,
        target: str,
        feature_columns: list[str],
    ) -> MarketResidualModel:
        market = np.clip(frame[market_probability].to_numpy(float), 1e-6, 1 - 1e-6)
        offset = np.log(market / (1 - market))
        residual_target = frame[target].to_numpy(float) - market
        # A strongly regularized linear correction keeps the market as anchor.
        from sklearn.linear_model import Ridge

        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(frame[feature_columns], residual_target)
        model.feature_cols_ = feature_columns
        model.market_logit_offset_ = offset.mean()
        return cls(model, tuple(feature_columns))

    def predict(self, frame: pd.DataFrame, market_probability: pd.Series) -> np.ndarray:
        correction = self.correction_model.predict(frame[list(self.feature_columns)])
        return np.clip(market_probability.to_numpy(float) + correction, 1e-6, 1 - 1e-6)
