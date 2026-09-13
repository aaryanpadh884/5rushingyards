"""Unit tests for src/rb_analysis.py (Section 41): RB carry share and the
individual-carry 5+ rule as applied through the real aggregation pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rb_analysis import attach_rb_positions, compute_team_game_rb_carries


PLAYERS = pd.DataFrame([
    {"player_id": "00-001", "player_name": "Back A", "position": "RB", "position_group": "RB"},
    {"player_id": "00-002", "player_name": "Back B", "position": "RB", "position_group": "RB"},
    {"player_id": "00-003", "player_name": "QB Test", "position": "QB", "position_group": "QB"},
])


def _row(rusher_id, yards, **overrides):
    row = {
        "season": 2024, "game_id": "G1", "posteam": "ARI", "drive": 1,
        "rush_attempt": 1, "is_real_rush_attempt": True, "is_offensive_scrimmage_play": True,
        "rusher_player_id": rusher_id, "rushing_yards": yards,
    }
    row.update(overrides)
    return row


class TestCarryShare:
    def test_carry_share_matches_spec_example(self):
        """
        Section 8 example: team has 4 RB carries, RB A=3, RB B=1 ->
        RB A carry share = 75%.
        """
        fd = pd.DataFrame([
            _row("00-001", 2), _row("00-001", 3), _row("00-001", 4),
            _row("00-002", 5),
        ])
        fd_pos = attach_rb_positions(fd, PLAYERS)
        per_player_game = compute_team_game_rb_carries(fd_pos)

        a = per_player_game[per_player_game["rusher_player_id"] == "00-001"].iloc[0]
        b = per_player_game[per_player_game["rusher_player_id"] == "00-002"].iloc[0]
        assert a["carries"] == 3
        assert a["carry_share"] == pytest.approx(0.75)
        assert b["carries"] == 1
        assert b["carry_share"] == pytest.approx(0.25)

    def test_qb_scramble_excluded_from_rb_carries(self):
        """A QB's carry (even if rush_attempt=1) must not count toward RB
        carry totals or carry share -- Section 10/19."""
        fd = pd.DataFrame([
            _row("00-001", 4),
            _row("00-003", 12),  # QB scramble, should be excluded from RB totals
        ])
        fd_pos = attach_rb_positions(fd, PLAYERS)
        per_player_game = compute_team_game_rb_carries(fd_pos)
        assert len(per_player_game) == 1
        assert per_player_game.iloc[0]["rusher_player_id"] == "00-001"
        assert per_player_game.iloc[0]["team_rb_carries"] == 1


class TestFivePlusFlagInAggregation:
    def test_got_5plus_true_when_one_big_carry(self):
        # Section 34 example: [2, 3, 4, 4] does NOT qualify
        fd = pd.DataFrame([_row("00-001", 2), _row("00-001", 3), _row("00-001", 4), _row("00-001", 4)])
        fd_pos = attach_rb_positions(fd, PLAYERS)
        per_player_game = compute_team_game_rb_carries(fd_pos)
        assert per_player_game.iloc[0]["got_5plus"] == False

    def test_got_5plus_true_with_single_qualifying_carry(self):
        # [2, 6] DOES qualify
        fd = pd.DataFrame([_row("00-001", 2), _row("00-001", 6)])
        fd_pos = attach_rb_positions(fd, PLAYERS)
        per_player_game = compute_team_game_rb_carries(fd_pos)
        assert per_player_game.iloc[0]["got_5plus"] == True
