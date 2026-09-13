"""Unit tests for src/utils.py (Section 41)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (
    american_odds_to_implied_probability,
    probability_to_fair_american_odds,
    expected_value_per_100,
    shrink_rate,
    got_five_plus,
    max_carry,
    confidence_label,
)


class TestAmericanOddsConversion:
    def test_negative_odds(self):
        # -125 -> 125/225
        assert american_odds_to_implied_probability(-125) == pytest.approx(125 / 225)

    def test_positive_odds(self):
        # +150 -> 100/250 = 0.40
        assert american_odds_to_implied_probability(150) == pytest.approx(0.40)

    def test_even_money_negative(self):
        assert american_odds_to_implied_probability(-100) == pytest.approx(0.5)

    def test_even_money_positive(self):
        assert american_odds_to_implied_probability(100) == pytest.approx(0.5)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            american_odds_to_implied_probability(0)


class TestFairOddsConversion:
    def test_favorite(self):
        # p=0.6364 corresponds to fair odds of about -175
        p = 0.6363636363636364
        assert probability_to_fair_american_odds(p) == pytest.approx(-175, abs=0.5)

    def test_underdog(self):
        # p=0.4 -> fair odds = 100 * 0.6 / 0.4 = 150
        assert probability_to_fair_american_odds(0.4) == pytest.approx(150)

    def test_roundtrip_negative_favorite(self):
        odds = -125
        p = american_odds_to_implied_probability(odds)
        fair = probability_to_fair_american_odds(p)
        assert fair == pytest.approx(odds)

    def test_roundtrip_positive_underdog(self):
        odds = 150
        p = american_odds_to_implied_probability(odds)
        fair = probability_to_fair_american_odds(p)
        assert fair == pytest.approx(odds)

    def test_boundary_raises(self):
        with pytest.raises(ValueError):
            probability_to_fair_american_odds(0.0)
        with pytest.raises(ValueError):
            probability_to_fair_american_odds(1.0)


class TestExpectedValue:
    def test_ev_negative_odds_breakeven(self):
        # betting at exactly the fair price should have ~0 EV
        odds = -125
        p = american_odds_to_implied_probability(odds)
        assert expected_value_per_100(p, odds) == pytest.approx(0, abs=1e-6)

    def test_ev_positive_when_model_favors_bettor(self):
        # true probability higher than implied -> positive EV
        odds = -110
        implied = american_odds_to_implied_probability(odds)
        ev = expected_value_per_100(implied + 0.10, odds)
        assert ev > 0

    def test_ev_underdog_payout(self):
        # +200 odds, p=0.5 -> win $200 half the time, lose $100 half the time -> EV=+50
        assert expected_value_per_100(0.5, 200) == pytest.approx(50.0)


class TestShrinkage:
    def test_large_sample_barely_moves(self):
        # 100 observed carries at 90% rate, prior of 30%, weak prior strength -> stays close to observed
        adjusted = shrink_rate(observed_rate=0.9, n=100, prior_rate=0.3, prior_strength=8)
        assert adjusted > 0.8

    def test_small_sample_pulled_to_prior(self):
        # 1 game, 100% observed rate -- should NOT stay at 100%, must be pulled toward prior
        adjusted = shrink_rate(observed_rate=1.0, n=1, prior_rate=0.3, prior_strength=8)
        assert adjusted < 0.5
        assert adjusted > 0.3  # still pulled up somewhat by the 1 observation

    def test_zero_observations_returns_prior(self):
        assert shrink_rate(observed_rate=0.9, n=0, prior_rate=0.35, prior_strength=8) == pytest.approx(0.35)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            shrink_rate(observed_rate=0.5, n=-1, prior_rate=0.3, prior_strength=8)


class TestFivePlusDetection:
    """
    got_five_plus is the market rule: total accumulated rushing yards on
    the first drive, an Over/Under-style prop -- NOT whether any single
    carry individually reached the threshold.
    """

    def test_multiple_small_carries_summing_past_threshold_qualifies(self):
        # 2 + 3 + 4 = 9 >= 5 -> qualifies, even though no single carry hit 5
        assert got_five_plus([2, 3, 4]) is True

    def test_carries_summing_below_threshold_does_not_qualify(self):
        # 1 + 1 = 2 < 5 -> does not qualify
        assert got_five_plus([1, 1]) is False

    def test_single_big_carry_qualifies(self):
        # [2, 6] sums to 8 >= 5 -> qualifies
        assert got_five_plus([2, 6]) is True

    def test_exactly_threshold_qualifies(self):
        assert got_five_plus([5]) is True

    def test_just_under_threshold_does_not_qualify(self):
        assert got_five_plus([4]) is False

    def test_empty_list_does_not_qualify(self):
        assert got_five_plus([]) is False

    def test_none_values_ignored(self):
        assert got_five_plus([None, 2, None, 6]) is True
        assert got_five_plus([None, None]) is False

    def test_negative_yards_reduce_the_total(self):
        # a loss on one carry offsets gains on others
        assert got_five_plus([-2, 6]) is False   # sums to 4
        assert got_five_plus([-2, 8]) is True    # sums to 6

    def test_custom_threshold(self):
        assert got_five_plus([8, 9], threshold=20) is False
        assert got_five_plus([8, 12], threshold=20) is True

    def test_max_carry(self):
        # max_carry (longest single carry) is a separate diagnostic stat,
        # unaffected by the target-metric definition.
        assert max_carry([2, 6, -1]) == 6
        assert max_carry([]) is None


class TestConfidenceLabel:
    def test_high(self):
        assert confidence_label(20, high_threshold=16, medium_threshold=8) == "HIGH"

    def test_medium(self):
        assert confidence_label(10, high_threshold=16, medium_threshold=8) == "MEDIUM"

    def test_low(self):
        assert confidence_label(2, high_threshold=16, medium_threshold=8) == "LOW"
