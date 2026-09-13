"""Unit tests for src/model.py (Section 41): role adjustment and confidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import apply_role_adjustment, apply_qb_competition_adjustment, compute_confidence


class TestRoleAdjustment:
    def test_rb1_boosts_probability(self):
        adjusted, _ = apply_role_adjustment(0.5, "RB1", "active")
        assert adjusted > 0.5

    def test_rb2_reduces_probability(self):
        adjusted, _ = apply_role_adjustment(0.5, "RB2", "active")
        assert adjusted < 0.5

    def test_committee_reduces_probability(self):
        adjusted, _ = apply_role_adjustment(0.5, "RB1", "committee")
        assert adjusted < 0.5

    def test_out_status_zeroes_probability(self):
        adjusted, note = apply_role_adjustment(0.9, "RB1", "out")
        assert adjusted == 0.0
        assert "OUT" in note

    def test_missing_role_keeps_historical_rate(self):
        adjusted, note = apply_role_adjustment(0.42, None, None)
        assert adjusted == 0.42
        assert "no current-role entry" in note


class TestQbCompetitionAdjustment:
    def test_reduces_carry_probability(self):
        adjusted = apply_qb_competition_adjustment(0.5, 0.10)
        assert adjusted == pytest.approx(0.45)

    def test_none_leaves_unchanged(self):
        assert apply_qb_competition_adjustment(0.5, None) == 0.5


class TestConfidence:
    def test_high_sample_clean_role_is_high(self):
        assert compute_confidence(20, "RB1", "active", 17) == "HIGH"

    def test_high_sample_but_committee_downgrades(self):
        assert compute_confidence(20, "RB1", "committee", 17) == "MEDIUM"

    def test_low_sample_is_low(self):
        assert compute_confidence(2, "RB1", "active", 17) == "LOW"

    def test_high_probability_alone_does_not_imply_high_confidence(self):
        # 3 games of history with a committee role must not be HIGH confidence
        # regardless of how favorable the raw probability looks.
        label = compute_confidence(3, "RB1", "committee", 17)
        assert label != "HIGH"
