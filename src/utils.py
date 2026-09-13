"""
Generic, dependency-light helper functions used across the modeling
pipeline: American-odds conversion, empirical-Bayes shrinkage, and the
individual-carry "5+ yard" rule that defines the betting market.
"""

from __future__ import annotations

from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Odds conversion (Sections 21-22)
# ---------------------------------------------------------------------------
def american_odds_to_implied_probability(odds: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def probability_to_fair_american_odds(p: float) -> float:
    """Convert a model probability into fair (no-vig) American odds."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    if p > 0.5:
        return -100.0 * p / (1.0 - p)
    if p < 0.5:
        return 100.0 * (1.0 - p) / p
    return 100.0  # p == 0.5, +100/-100 both fair; report +100


def expected_value_per_100(p: float, american_odds: float) -> float:
    """Expected profit on a $100 stake at the given American odds, using
    model probability p as the true win probability."""
    odds = float(american_odds)
    if odds < 0:
        profit_if_win = 100.0 * 100.0 / abs(odds)
    else:
        profit_if_win = odds
    return p * profit_if_win - (1.0 - p) * 100.0


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage (Section 12)
# ---------------------------------------------------------------------------
def shrink_rate(observed_rate: float, n: float, prior_rate: float, prior_strength: float) -> float:
    """
    Empirical-Bayes shrinkage toward a prior rate.

    adjusted_rate = (n * observed_rate + prior_strength * prior_rate)
                    / (n + prior_strength)

    `n` is the number of observations (e.g. games) backing observed_rate,
    and `prior_strength` is the number of "pseudo-observations" of weight
    given to the prior. Small samples get pulled hard toward the prior;
    large samples are barely moved.
    """
    if n < 0 or prior_strength < 0:
        raise ValueError("n and prior_strength must be non-negative")
    denom = n + prior_strength
    if denom == 0:
        return prior_rate
    return (n * observed_rate + prior_strength * prior_rate) / denom


def weighted_average(values_weights: Iterable[tuple[Optional[float], float]]) -> Optional[float]:
    """Weighted average that ignores (skips + renormalizes) any None values."""
    total_w = 0.0
    total_v = 0.0
    for value, weight in values_weights:
        if value is None or weight is None:
            continue
        total_v += value * weight
        total_w += weight
    if total_w == 0:
        return None
    return total_v / total_w


def renormalize_weights(weights: dict) -> dict:
    """Renormalize a season -> weight dict so the values sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Market definition (Sections 7, 34): individual-carry 5+ rule
# ---------------------------------------------------------------------------
def got_five_plus(rush_yards: Iterable[float], threshold: float = 5.0) -> bool:
    """
    True if ANY individual rushing attempt in `rush_yards` is >= threshold.

    This is the exact rule the sportsbook market uses: it is NOT whether the
    total of all carries sums to >= threshold.

        [2, 3, 4, 4]  -> False (max single carry is 4)
        [2, 6]        -> True  (one carry of 6)
    """
    yards = [y for y in rush_yards if y is not None]
    if not yards:
        return False
    return max(yards) >= threshold


def max_carry(rush_yards: Iterable[float]) -> Optional[float]:
    yards = [y for y in rush_yards if y is not None]
    if not yards:
        return None
    return max(yards)


def confidence_label(sample_size: int, high_threshold: int, medium_threshold: int) -> str:
    if sample_size >= high_threshold:
        return "HIGH"
    if sample_size >= medium_threshold:
        return "MEDIUM"
    return "LOW"
