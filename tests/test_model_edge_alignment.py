import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

# test_pick_contract stubs these optional-ingestion modules; this file tests
# the real predictor implementations and must not inherit those stubs.
for _module_name in (
    "src.models.underdog_model",
    "src.models.spread_model",
    "src.models.totals_model",
):
    _module = sys.modules.get(_module_name)
    if _module is not None and not getattr(_module, "__file__", None):
        del sys.modules[_module_name]

from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals
from src.models.underdog_model import predict_moneyline


def _model():
    model = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
    model.fit([[0], [1], [2]], [0, 1, 1])
    model.feature_cols_ = ["feature"]
    return model


def _games():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": ["2026-07-23", "2026-07-23"],
            "hometeam": ["Home 1", "Home 2"],
            "visteam": ["Away 1", "Away 2"],
            "feature": [0.1, 0.2],
            "home_moneyline": [-110, 120],
            "away_moneyline": [100, -130],
            "home_spread_price": [-110, -105],
            "over_price": [-110, -115],
        }
    )


def test_moneyline_edges_align_one_probability_per_game():
    result = predict_moneyline(
        _model(), _games(), home_ml_col="home_moneyline", away_ml_col="away_moneyline"
    )
    assert len(result["edge_home"]) == 2
    assert result["edge_home"].notna().all()
    assert result["edge_away"].notna().all()


def test_spread_and_totals_edges_align_one_probability_per_game():
    games = _games()
    spread = predict_spread(_model(), games, spread_price_col="home_spread_price")
    totals = predict_totals(_model(), games, over_price_col="over_price")
    assert len(spread["edge"]) == 2
    assert len(totals["edge_over"]) == 2
    assert np.isfinite(spread["edge"]).all()
    assert np.isfinite(totals["edge_over"]).all()
