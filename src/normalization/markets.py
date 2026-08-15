"""Provider market/selection values mapped to stable canonical identifiers."""

from __future__ import annotations

from src.contracts.domain import Selection

MARKET_ALIASES = {
    "h2h": "moneyline_full_game",
    "moneyline": "moneyline_full_game",
    "spreads": "run_line_full_game",
    "run_line": "run_line_full_game",
    "totals": "total_full_game",
    "total": "total_full_game",
    "h2h_h1": "moneyline_first_5",
    "spreads_h1": "run_line_first_5",
    "totals_h1": "total_first_5",
    "nrfi_yrfi": "nrfi_yrfi",
}


def canonical_market_id(provider_market: str) -> str:
    key = provider_market.strip().lower()
    if key not in MARKET_ALIASES:
        raise KeyError(f"Unknown market: {provider_market}")
    return MARKET_ALIASES[key]


def canonical_selection(
    outcome_name: str,
    *,
    home_team: str | None = None,
    away_team: str | None = None,
) -> Selection:
    value = outcome_name.strip().lower()
    if home_team and value == home_team.strip().lower():
        return Selection.HOME
    if away_team and value == away_team.strip().lower():
        return Selection.AWAY
    mapping = {
        "home": Selection.HOME,
        "away": Selection.AWAY,
        "over": Selection.OVER,
        "under": Selection.UNDER,
        "yes": Selection.YES,
        "yrfi": Selection.YES,
        "no": Selection.NO,
        "nrfi": Selection.NO,
    }
    if value not in mapping:
        raise KeyError(f"Unknown selection: {outcome_name}")
    return mapping[value]
