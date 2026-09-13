"""Unit tests for src/matchup_analysis.py: composite opponent defense factor."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matchup_analysis import compute_composite_defense_factor


def _defense_metrics(rows):
    return pd.DataFrame(rows)


class TestCompositeDefenseFactor:
    def test_softest_defense_gets_factor_above_one(self):
        rows = [
            {"team": "SOFT", "five_plus_allowed_rate_shrunk": 0.70, "yards_per_rush_allowed": 5.5,
             "ten_plus_allowed_rate": 0.30, "success_rate_allowed": 0.55, "avg_epa_allowed": 0.10},
            {"team": "AVG", "five_plus_allowed_rate_shrunk": 0.58, "yards_per_rush_allowed": 4.5,
             "ten_plus_allowed_rate": 0.20, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0},
            {"team": "TOUGH", "five_plus_allowed_rate_shrunk": 0.45, "yards_per_rush_allowed": 3.5,
             "ten_plus_allowed_rate": 0.10, "success_rate_allowed": 0.35, "avg_epa_allowed": -0.10},
        ]
        result = compute_composite_defense_factor(_defense_metrics(rows)).set_index("team")
        assert result.loc["SOFT", "composite_defense_factor"] > 1.0
        assert result.loc["TOUGH", "composite_defense_factor"] < 1.0
        assert result.loc["SOFT", "composite_defense_factor"] > result.loc["AVG", "composite_defense_factor"]
        assert result.loc["AVG", "composite_defense_factor"] > result.loc["TOUGH", "composite_defense_factor"]

    def test_factor_is_capped(self):
        # an extreme outlier on every metric should still be clipped
        rows = [
            {"team": "EXTREME", "five_plus_allowed_rate_shrunk": 0.99, "yards_per_rush_allowed": 20.0,
             "ten_plus_allowed_rate": 0.99, "success_rate_allowed": 0.99, "avg_epa_allowed": 5.0},
            {"team": "NORMAL", "five_plus_allowed_rate_shrunk": 0.58, "yards_per_rush_allowed": 4.5,
             "ten_plus_allowed_rate": 0.20, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0},
        ]
        result = compute_composite_defense_factor(_defense_metrics(rows)).set_index("team")
        import config
        assert result.loc["EXTREME", "composite_defense_factor"] <= config.DEFENSE_ADJUSTMENT_MAX_FACTOR

    def test_identical_teams_get_neutral_factor(self):
        rows = [
            {"team": "A", "five_plus_allowed_rate_shrunk": 0.58, "yards_per_rush_allowed": 4.5,
             "ten_plus_allowed_rate": 0.20, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0},
            {"team": "B", "five_plus_allowed_rate_shrunk": 0.58, "yards_per_rush_allowed": 4.5,
             "ten_plus_allowed_rate": 0.20, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0},
        ]
        result = compute_composite_defense_factor(_defense_metrics(rows))
        assert result["composite_defense_factor"].apply(lambda x: x == pytest.approx(1.0)).all()

    def test_weights_must_sum_to_one(self):
        rows = [{"team": "A", "five_plus_allowed_rate_shrunk": 0.58, "yards_per_rush_allowed": 4.5,
                 "ten_plus_allowed_rate": 0.20, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0}]
        with pytest.raises(ValueError):
            compute_composite_defense_factor(_defense_metrics(rows), weights={"five_plus_allowed_rate_shrunk": 0.5})

    def test_missing_metric_column_is_skipped_not_fatal(self):
        rows = [
            {"team": "A", "five_plus_allowed_rate_shrunk": 0.70},
            {"team": "B", "five_plus_allowed_rate_shrunk": 0.45},
        ]
        result = compute_composite_defense_factor(_defense_metrics(rows),
                                                    weights={"five_plus_allowed_rate_shrunk": 1.0})
        assert result.set_index("team").loc["A", "composite_defense_factor"] > 1.0
