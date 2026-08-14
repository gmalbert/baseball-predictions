"""Pure price, de-vig, expected-value, and stake math."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite, log

from scipy.optimize import brentq


def american_to_decimal(price: int) -> float:
    if price == 0:
        raise ValueError("American price cannot be zero")
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def decimal_to_american(price: float) -> int:
    if price <= 1.0 or not isfinite(price):
        raise ValueError("Decimal price must be finite and greater than 1")
    return round((price - 1.0) * 100 if price >= 2.0 else -100 / (price - 1.0))


def implied_probability(decimal_price: float) -> float:
    if decimal_price <= 1.0 or not isfinite(decimal_price):
        raise ValueError("Decimal price must be finite and greater than 1")
    return 1.0 / decimal_price


def _validate(raw: Iterable[float]) -> list[float]:
    values = list(raw)
    if len(values) < 2 or any(not 0 < value < 1 for value in values):
        raise ValueError("Need at least two raw implied probabilities in (0, 1)")
    if sum(values) <= 1.0:
        raise ValueError("Market does not contain positive overround")
    return values


def devig_multiplicative(raw: Iterable[float]) -> list[float]:
    values = _validate(raw)
    total = sum(values)
    return [value / total for value in values]


def devig_additive(raw: Iterable[float]) -> list[float]:
    values = _validate(raw)
    adjustment = (sum(values) - 1.0) / len(values)
    fair = [value - adjustment for value in values]
    if any(value <= 0 for value in fair):
        raise ValueError("Additive method produced non-positive probability")
    return fair


def devig_power(raw: Iterable[float]) -> list[float]:
    values = _validate(raw)
    exponent = brentq(lambda k: sum(value**k for value in values) - 1.0, 0.01, 100.0)
    return [value**exponent for value in values]


def devig_shin(raw: Iterable[float]) -> list[float]:
    """Shin de-vig probabilities for a mutually exclusive market.

    The insider fraction is solved numerically.  At effectively zero insider
    share, this reduces to multiplicative normalization.
    """
    values = _validate(raw)
    total = sum(values)

    def probabilities(z: float) -> list[float]:
        denominator = 2.0 * (1.0 - z)
        return [
            (((z * z) + 4.0 * (1.0 - z) * (value * value) / total) ** 0.5 - z) / denominator
            for value in values
        ]

    at_zero = sum(probabilities(0.0)) - 1.0
    if abs(at_zero) < 1e-12:
        return probabilities(0.0)
    z = brentq(lambda candidate: sum(probabilities(candidate)) - 1.0, 0.0, 0.999999)
    result = probabilities(z)
    normalizer = sum(result)
    return [value / normalizer for value in result]


def devig_ensemble(
    raw: Iterable[float], methods: Sequence[str] = ("multiplicative", "power", "shin")
) -> list[float]:
    values = list(raw)
    functions = {
        "multiplicative": devig_multiplicative,
        "additive": devig_additive,
        "power": devig_power,
        "shin": devig_shin,
    }
    if not methods:
        raise ValueError("At least one de-vig method is required")
    estimates = []
    for method in methods:
        if method not in functions:
            raise ValueError(f"Unknown de-vig method: {method}")
        estimates.append(functions[method](values))
    averaged = [
        sum(row[index] for row in estimates) / len(estimates) for index in range(len(values))
    ]
    total = sum(averaged)
    return [value / total for value in averaged]


def expected_value(
    win_probability: float, decimal_price: float, push_probability: float = 0.0
) -> float:
    if not 0 <= win_probability <= 1 or decimal_price <= 1 or not isfinite(decimal_price):
        raise ValueError("Invalid probability or price")
    loss_probability = 1.0 - win_probability - push_probability
    if not 0 <= push_probability <= 1 or loss_probability < 0:
        raise ValueError("Invalid outcome probabilities")
    return win_probability * (decimal_price - 1.0) - loss_probability


def kelly_fraction(
    win_probability: float, decimal_price: float, push_probability: float = 0.0
) -> float:
    if not 0 <= win_probability <= 1 or decimal_price <= 1 or not isfinite(decimal_price):
        raise ValueError("Invalid probability or price")
    if not 0 <= push_probability <= 1 - win_probability:
        raise ValueError("Invalid push probability")
    # Pushes return stake and therefore disappear from the growth derivative.
    loss_probability = 1.0 - win_probability - push_probability
    return max(
        0.0, (win_probability * (decimal_price - 1.0) - loss_probability) / (decimal_price - 1.0)
    )


def log_price_clv(bet_decimal: float, close_decimal: float) -> float:
    if min(bet_decimal, close_decimal) <= 1.0:
        raise ValueError("Decimal prices must exceed 1")
    return log(bet_decimal) - log(close_decimal)


@dataclass(frozen=True)
class ConsensusPrice:
    fair_probability: float
    best_decimal_price: float
    median_decimal_price: float
    dispersion: float
    book_count: int
