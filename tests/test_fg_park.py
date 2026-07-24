import pandas as pd

from src.ingestion import fg_park


def test_guts_park_factor_parser_preserves_handedness(monkeypatch):
    overall = pd.DataFrame(
        [[2026, "Angels", 98, 97, 96, 99, 100, 101, 102, 100, 99, 100, 100, 100, 100, 100]],
        columns=range(16),
    )
    handed = pd.DataFrame(
        [[2026, "Angels", 98, 101, 99, 102, 100, 103, 104, 108]],
        columns=range(10),
    )
    tables = iter([[overall], [handed]])
    monkeypatch.setattr(fg_park.pd, "read_html", lambda *args, **kwargs: next(tables))

    result = fg_park._fetch_fg_guts_park_factors(2026)

    assert result.to_dict("records") == [
        {"team": "Angels", "hand": "L", "season": 2026, "pf_basic": 98,
         "pf_hr": 104, "pf_1b": 98, "pf_2b": 99, "pf_3b": 100},
        {"team": "Angels", "hand": "R", "season": 2026, "pf_basic": 98,
         "pf_hr": 108, "pf_1b": 101, "pf_2b": 102, "pf_3b": 103},
    ]
