"""Unit tests for filtering out players no longer on an active NFL roster
(src.model.build_model_table) -- the Joe Mixon case: a player with real
historical usage but who is off the 2026 roster must not keep appearing
under his old team just because current_rb_roles.csv has no entry for him.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model import build_model_table, REQUIRED_ROLE_COLUMNS


def _rb_row(player_id, player_name, team, **overrides):
    row = {
        "player_id": player_id, "player_name": player_name, "team": team,
        "p_carry_1plus_shrunk": 0.6, "p_five_given_carry_shrunk": 0.5,
        "games_with_carry": 20, "avg_carry_share": 0.6, "total_carries": 60,
        "total_rushing_yards": 300, "longest_carry": 40,
        "games_with_5plus": 10, "games_with_10plus": 4,
        "team_total_first_drive_games": 30,
    }
    row.update(overrides)
    return row


def _team_metrics_row(team):
    return {"team": team, "first_drive_run_pct": 0.5, "games": 30}


def _defense_row(team):
    return {
        "team": team, "five_plus_allowed_rate_shrunk": 0.5, "yards_per_rush_allowed": 4.5,
        "ten_plus_allowed_rate": 0.2, "success_rate_allowed": 0.45, "avg_epa_allowed": 0.0,
        "composite_defense_factor": 1.0, "games": 30,
    }


class TestInactivePlayerFilter:
    def test_inactive_player_with_no_override_is_dropped(self):
        rb_metrics = pd.DataFrame([_rb_row("MIXON", "Joe Mixon", "HOU")])
        team_metrics = pd.DataFrame([_team_metrics_row("HOU")])
        defense_metrics = pd.DataFrame([_defense_row("BUF")])
        qb_competition = pd.DataFrame([{"season": 2025, "team": "HOU", "qb_rush_competition": 0.0}])
        current_roles = pd.DataFrame(columns=REQUIRED_ROLE_COLUMNS)
        matchups = pd.DataFrame([{"team": "HOU", "opponent": "BUF"}])
        players = pd.DataFrame([{"player_id": "MIXON", "latest_team": "HOU", "last_season": 2025}])

        result = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition,
                                    current_roles, matchups, players=players)
        assert result.empty

    def test_active_player_is_kept(self):
        rb_metrics = pd.DataFrame([_rb_row("MONTY", "David Montgomery", "HOU")])
        team_metrics = pd.DataFrame([_team_metrics_row("HOU")])
        defense_metrics = pd.DataFrame([_defense_row("BUF")])
        qb_competition = pd.DataFrame([{"season": 2025, "team": "HOU", "qb_rush_competition": 0.0}])
        current_roles = pd.DataFrame(columns=REQUIRED_ROLE_COLUMNS)
        matchups = pd.DataFrame([{"team": "HOU", "opponent": "BUF"}])
        players = pd.DataFrame([{"player_id": "MONTY", "latest_team": "HOU", "last_season": 2026}])

        result = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition,
                                    current_roles, matchups, players=players)
        assert len(result) == 1

    def test_manual_role_override_beats_inactive_flag(self):
        """A user who explicitly lists a player in current_rb_roles.csv
        vouches for him being relevant, even if the roster feed's
        last_season looks stale."""
        rb_metrics = pd.DataFrame([_rb_row("MIXON", "Joe Mixon", "HOU")])
        team_metrics = pd.DataFrame([_team_metrics_row("HOU")])
        defense_metrics = pd.DataFrame([_defense_row("BUF")])
        qb_competition = pd.DataFrame([{"season": 2025, "team": "HOU", "qb_rush_competition": 0.0}])
        current_roles = pd.DataFrame([{"team": "HOU", "player_name": "Joe Mixon", "player_id": "MIXON",
                                        "role": "RB1", "status": "active"}])
        matchups = pd.DataFrame([{"team": "HOU", "opponent": "BUF"}])
        players = pd.DataFrame([{"player_id": "MIXON", "latest_team": "HOU", "last_season": 2025}])

        result = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition,
                                    current_roles, matchups, players=players)
        assert len(result) == 1

    def test_no_players_arg_skips_the_check(self):
        """Backtest path: players=None means no filtering happens at all."""
        rb_metrics = pd.DataFrame([_rb_row("MIXON", "Joe Mixon", "HOU")])
        team_metrics = pd.DataFrame([_team_metrics_row("HOU")])
        defense_metrics = pd.DataFrame([_defense_row("BUF")])
        qb_competition = pd.DataFrame([{"season": 2025, "team": "HOU", "qb_rush_competition": 0.0}])
        current_roles = pd.DataFrame(columns=REQUIRED_ROLE_COLUMNS)
        matchups = pd.DataFrame([{"team": "HOU", "opponent": "BUF"}])

        result = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition,
                                    current_roles, matchups, players=None)
        assert len(result) == 1
