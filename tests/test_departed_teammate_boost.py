"""Unit tests for the departed-teammate carry boost (src.rb_analysis)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rb_analysis import compute_departed_teammate_boost


def _rb_metrics_row(team, player_id, total_carries):
    return {"team": team, "player_id": player_id, "total_carries": total_carries}


class TestDepartedTeammateBoost:
    def test_boost_when_backfield_mate_leaves(self):
        """
        DET had Gibbs (57 carries) and Montgomery (40 carries) historically.
        Montgomery has since left for HOU; Gibbs stayed at DET. Gibbs should
        get a boost roughly equal to Montgomery's share of DET's carries.
        """
        rb_metrics = pd.DataFrame([
            _rb_metrics_row("DET", "GIBBS", 57),
            _rb_metrics_row("DET", "MONTY", 40),
        ])
        players = pd.DataFrame([
            {"player_id": "GIBBS", "latest_team": "DET"},
            {"player_id": "MONTY", "latest_team": "HOU"},
        ])
        boost = compute_departed_teammate_boost(rb_metrics, players, max_boost=0.30)
        gibbs_boost = boost[boost["player_id"] == "GIBBS"]["departed_teammate_boost"].iloc[0]
        # 40 / 97 = 0.412, capped at 0.30
        assert gibbs_boost == pytest.approx(0.30)

    def test_no_boost_when_teammate_still_present(self):
        rb_metrics = pd.DataFrame([
            _rb_metrics_row("DET", "GIBBS", 57),
            _rb_metrics_row("DET", "MONTY", 40),
        ])
        players = pd.DataFrame([
            {"player_id": "GIBBS", "latest_team": "DET"},
            {"player_id": "MONTY", "latest_team": "DET"},  # still there
        ])
        boost = compute_departed_teammate_boost(rb_metrics, players, max_boost=0.30)
        assert boost[boost["player_id"] == "GIBBS"].empty

    def test_no_boost_for_player_who_also_left(self):
        """A player who himself changed teams should not get a boost tied
        to his OLD team's departures -- that framing doesn't apply to him."""
        rb_metrics = pd.DataFrame([
            _rb_metrics_row("KC", "PACHECO", 80),
            _rb_metrics_row("KC", "HUNT", 60),
        ])
        players = pd.DataFrame([
            {"player_id": "PACHECO", "latest_team": "DET"},  # Pacheco moved
            {"player_id": "HUNT", "latest_team": "KC"},
        ])
        boost = compute_departed_teammate_boost(rb_metrics, players, max_boost=0.30)
        assert boost[boost["player_id"] == "PACHECO"].empty

    def test_uncapped_small_departure(self):
        rb_metrics = pd.DataFrame([
            _rb_metrics_row("XX", "A", 90),
            _rb_metrics_row("XX", "B", 10),
        ])
        players = pd.DataFrame([
            {"player_id": "A", "latest_team": "XX"},
            {"player_id": "B", "latest_team": "YY"},
        ])
        boost = compute_departed_teammate_boost(rb_metrics, players, max_boost=0.30)
        a_boost = boost[boost["player_id"] == "A"]["departed_teammate_boost"].iloc[0]
        assert a_boost == pytest.approx(0.10)

    def test_missing_latest_team_column_returns_empty(self):
        rb_metrics = pd.DataFrame([_rb_metrics_row("DET", "GIBBS", 57)])
        players = pd.DataFrame([{"player_id": "GIBBS"}])
        boost = compute_departed_teammate_boost(rb_metrics, players, max_boost=0.30)
        assert boost.empty
