"""Unit tests for src/roles.py: data-derived RB1/RB2/committee assignment."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.roles import derive_rb_roles


def _player(player_id, name, team, last_season=2026, position_group="RB"):
    return {"player_id": player_id, "player_name": name, "latest_team": team,
            "last_season": last_season, "position_group": position_group}


class TestDeriveRbRoles:
    def test_dominant_back_is_clear_rb1(self):
        # 80/20 split -> clear RB1/RB2, not a committee
        rb_metrics = pd.DataFrame([
            {"player_id": "A", "total_carries": 80},
            {"player_id": "B", "total_carries": 20},
        ])
        players = pd.DataFrame([_player("A", "Back A", "DET"), _player("B", "Back B", "DET")])
        roles = derive_rb_roles(rb_metrics, players)
        by_id = roles.set_index("player_id")
        assert by_id.loc["A", "role"] == "RB1"
        assert by_id.loc["B", "role"] == "RB2"
        assert (roles["status"] == "active").all()

    def test_even_split_is_committee(self):
        # 55/45 split -> both clear the committee threshold (>=30% each)
        # and neither clears the RB1 threshold (>=60%)
        rb_metrics = pd.DataFrame([
            {"player_id": "A", "total_carries": 55},
            {"player_id": "B", "total_carries": 45},
        ])
        players = pd.DataFrame([_player("A", "Back A", "NYG"), _player("B", "Back B", "NYG")])
        roles = derive_rb_roles(rb_metrics, players)
        by_id = roles.set_index("player_id")
        assert by_id.loc["A", "role"] == "committee"
        assert by_id.loc["B", "role"] == "committee"

    def test_single_back_with_data_is_rb1(self):
        rb_metrics = pd.DataFrame([{"player_id": "A", "total_carries": 40}])
        players = pd.DataFrame([_player("A", "Back A", "PHI")])
        roles = derive_rb_roles(rb_metrics, players)
        assert len(roles) == 1
        assert roles.iloc[0]["role"] == "RB1"

    def test_rookie_with_zero_history_gets_no_role(self):
        """A true rookie with no historical carries anywhere must not be
        assigned a guessed role -- he should simply be absent from the
        output."""
        rb_metrics = pd.DataFrame([{"player_id": "VET", "total_carries": 50}])
        players = pd.DataFrame([
            _player("VET", "Veteran Back", "LV"),
            _player("ROOKIE", "Rookie Back", "LV"),
        ])
        roles = derive_rb_roles(rb_metrics, players)
        assert "ROOKIE" not in roles["player_id"].values
        assert "VET" in roles["player_id"].values

    def test_player_not_on_current_roster_excluded(self):
        """A player whose last_season is behind CURRENT_SEASON (not on an
        active 2026 roster) must not receive a role even if he has a huge
        historical carry total."""
        rb_metrics = pd.DataFrame([{"player_id": "OLD", "total_carries": 200}])
        players = pd.DataFrame([_player("OLD", "Retired Back", "HOU", last_season=2024)])
        roles = derive_rb_roles(rb_metrics, players, current_season=config.CURRENT_SEASON)
        assert roles.empty

    def test_traded_veteran_ranked_against_new_teammates(self):
        """A player's historical volume (wherever he earned it) is used to
        rank him against his CURRENT team's other backs."""
        rb_metrics = pd.DataFrame([
            {"player_id": "TRADED", "total_carries": 90},   # earned elsewhere historically
            {"player_id": "INCUMBENT", "total_carries": 10},
        ])
        players = pd.DataFrame([
            _player("TRADED", "Traded Vet", "HOU"),
            _player("INCUMBENT", "Incumbent Back", "HOU"),
        ])
        roles = derive_rb_roles(rb_metrics, players)
        by_id = roles.set_index("player_id")
        assert by_id.loc["TRADED", "role"] == "RB1"
        assert by_id.loc["INCUMBENT", "role"] == "RB2"
