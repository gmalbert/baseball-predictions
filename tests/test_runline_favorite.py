import sys
import types
import pandas as pd

# Provide a minimal stub for external dependency `statsapi` so tests can import
# `predictions`/`page_utils` without installing all optional packages.
sys.modules.setdefault("statsapi", types.ModuleType("statsapi"))

from predictions import _build_game_recs


def test_runline_favorite_from_moneyline_away_favorite():
    g = {"home_name": "San Francisco Giants", "away_name": "New York Yankees"}
    espn_game = {"ml_home": "+113", "ml_away": "-136", "spread_home": "+100", "spread_away": "-120"}
    recs = _build_game_recs(g, espn_game, {}, pd.DataFrame())
    assert "rl" in recs
    assert recs["rl"]["away"]["pick"] == "Yankees −1.5"
    assert recs["rl"]["home"]["pick"] == "Giants +1.5"


def test_runline_favorite_from_moneyline_home_favorite():
    g = {"home_name": "Los Angeles Dodgers", "away_name": "Arizona Diamondbacks"}
    espn_game = {"ml_home": "-150", "ml_away": "+120", "spread_home": "-120", "spread_away": "+100"}
    recs = _build_game_recs(g, espn_game, {}, pd.DataFrame())
    assert "rl" in recs
    assert recs["rl"]["home"]["pick"] == "Dodgers −1.5"
    assert recs["rl"]["away"]["pick"] == "Diamondbacks +1.5"


def test_runline_favorite_missing_ml_uses_spread_heuristic():
    g = {"home_name": "San Francisco Giants", "away_name": "New York Yankees"}
    espn_game = {"ml_home": None, "ml_away": None, "spread_home": "+130", "spread_away": "-150"}
    recs = _build_game_recs(g, espn_game, {}, pd.DataFrame())
    assert "rl" in recs
    assert recs["rl"]["home"]["pick"] == "Giants −1.5"
    assert recs["rl"]["away"]["pick"] == "Yankees +1.5"


def test_runline_favorite_only_spread_away_minus_one_five():
    g = {"home_name": "San Francisco Giants", "away_name": "New York Yankees"}
    espn_game = {"ml_home": None, "ml_away": None, "spread_home": "-150", "spread_away": "+130"}
    recs = _build_game_recs(g, espn_game, {}, pd.DataFrame())
    assert "rl" in recs
    assert recs["rl"]["home"]["pick"] == "Giants +1.5"
    assert recs["rl"]["away"]["pick"] == "Yankees −1.5"


def test_runline_favorite_malformed_ml_falls_back_to_spread():
    g = {"home_name": "San Francisco Giants", "away_name": "New York Yankees"}
    espn_game = {"ml_home": "+abc", "ml_away": "-140", "spread_home": "+120", "spread_away": "-130"}
    recs = _build_game_recs(g, espn_game, {}, pd.DataFrame())
    assert "rl" in recs
    assert recs["rl"]["home"]["pick"] == "Giants −1.5"
    assert recs["rl"]["away"]["pick"] == "Yankees +1.5"
