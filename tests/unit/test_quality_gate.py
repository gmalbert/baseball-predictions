import pandas as pd
import pytest

from src.quality.gate import known_values, max_missing, run_gate, schema_exact, valid_decimal_odds


def test_gate_accepts_canonical_quote_shape() -> None:
    frame = pd.DataFrame({"selection": ["home"], "decimal_odds": [1.91], "line": [None]})
    results = run_gate(
        frame,
        [
            schema_exact("selection", "decimal_odds", "line"),
            known_values("selection", {"home", "away"}),
            valid_decimal_odds(),
            max_missing("line", 1.0),
        ],
    )
    assert all(result.passed for result in results)


def test_gate_rejects_unknown_selection() -> None:
    with pytest.raises(ValueError, match="known_values"):
        run_gate(pd.DataFrame({"selection": ["favorite"]}), [known_values("selection", {"home"})])
