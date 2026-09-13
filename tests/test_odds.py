"""Unit tests for src/odds.py (Section 41): edge calculation end-to-end."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.odds import attach_odds_and_edge, load_sportsbook_odds


def test_attach_odds_and_edge_matches_spec_example(tmp_path):
    """
    Section 23 example: model_probability=0.64, sportsbook implied
    probability=0.524 (from -110-ish odds) -> edge should be +0.116ish.
    We construct odds so implied probability is exactly 0.5238 (-110).
    """
    odds_path = tmp_path / "sportsbook_odds.csv"
    odds_path.write_text(
        "player_name,team,market,line,american_odds,sportsbook,timestamp\n"
        "Test Back,DET,first_drive_rushing,5+,-110,DraftKings,2026-09-01T00:00:00\n"
    )
    odds = load_sportsbook_odds(odds_path)

    model_df = pd.DataFrame([{
        "player_name": "Test Back", "team": "DET", "model_probability": 0.64,
    }])

    result = attach_odds_and_edge(model_df, odds)
    row = result.iloc[0]

    assert row["sportsbook_implied_probability"] == pytest.approx(110 / 210, abs=1e-4)
    expected_edge = 0.64 - (110 / 210)
    assert row["edge"] == pytest.approx(expected_edge, abs=1e-4)
    assert row["odds_available"] == True


def test_no_matching_odds_marked_unavailable():
    odds = pd.DataFrame(columns=["player_name", "team", "market", "line", "american_odds",
                                  "sportsbook", "timestamp", "implied_probability"])
    model_df = pd.DataFrame([{"player_name": "Nobody", "team": "XYZ", "model_probability": 0.5}])
    result = attach_odds_and_edge(model_df, odds)
    row = result.iloc[0]
    assert row["odds_available"] == False
    assert pd.isna(row["edge"])
    assert pd.isna(row["sportsbook_odds"])
