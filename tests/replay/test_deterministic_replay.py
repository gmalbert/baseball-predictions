from datetime import UTC, date, datetime
from decimal import Decimal

from src.contracts.domain import GameSnapshot, Prediction, Quote, Selection
from src.decisions.policy import Policy
from src.pipelines.replay import replay

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


class Catalog:
    def __init__(self):
        self.predictions = []
        self.eligibility = []
        self.decisions = []

    def build_snapshots(self, target_date, as_of):
        return [
            GameSnapshot(
                snapshot_id="s",
                game_id="g",
                as_of_time=as_of,
                snapshot_type="morning",
                feature_set_version="v2",
                features={"x": 1},
                source_watermarks={"source": as_of},
                row_hash="h",
            )
        ]

    def quotes(self, game_ids, as_of):
        return [
            Quote(
                quote_id="q",
                game_id="g",
                bookmaker_id="book",
                market_id="moneyline_full_game",
                selection=Selection.AWAY,
                price_decimal=Decimal("2.10"),
                observed_at=as_of,
            )
        ]

    def save_predictions(self, rows):
        self.predictions = rows

    def save_eligibility(self, rows):
        self.eligibility = rows

    def save_decisions(self, rows):
        self.decisions = rows


class Predictor:
    def predict(self, snapshots, predicted_at):
        return [
            Prediction(
                prediction_id="p-away",
                snapshot_id="s",
                game_id="g",
                model_run_id="m",
                market_id="moneyline_full_game",
                selection=Selection.AWAY,
                probability_raw=0.62,
                probability=0.62,
                probability_low=0.58,
                probability_high=0.66,
                predicted_at=predicted_at,
                feature_row_hash="h",
            ),
            Prediction(
                prediction_id="p-under",
                snapshot_id="s",
                game_id="g",
                model_run_id="m",
                market_id="total_full_game",
                selection=Selection.UNDER,
                probability_raw=0.60,
                probability=0.60,
                probability_low=0.55,
                probability_high=0.65,
                predicted_at=predicted_at,
                feature_row_hash="h",
            ),
        ]


def test_replay_records_missing_quote_and_is_deterministic():
    first = Catalog()
    second = Catalog()
    result_one = replay(
        target_date=date(2026, 8, 10),
        as_of=NOW,
        catalog=first,
        predictor=Predictor(),
        policy=Policy(),
        bankroll=Decimal("1000"),
    )
    result_two = replay(
        target_date=date(2026, 8, 10),
        as_of=NOW,
        catalog=second,
        predictor=Predictor(),
        policy=Policy(),
        bankroll=Decimal("1000"),
    )
    assert result_one == result_two
    assert result_one.matched_quotes == 1
    assert result_one.no_quote == 1
    assert len(first.eligibility) == 2
    assert first.eligibility[1].reason_codes == ("missing_quote",)
    assert first.decisions[0].quote_id == "q"
