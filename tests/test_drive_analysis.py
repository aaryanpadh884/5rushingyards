"""Unit tests for src/drive_analysis.py (Section 41): first-drive
identification, data cleaning exclusions, and team run-percentage math."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drive_analysis import clean_pbp, compute_first_drive_ids, build_first_drive_plays, compute_team_first_drive_metrics


def _base_row(**overrides):
    row = {
        "game_id": "2024_01_ARI_BUF", "season": 2024, "posteam": "ARI", "defteam": "BUF",
        "drive": 1, "play_type": "run", "rush_attempt": 1, "pass_attempt": 0, "sack": 0,
        "qb_scramble": 0, "two_point_attempt": 0, "aborted_play": 0, "touchdown": 0,
        "field_goal_attempt": 0, "extra_point_attempt": 0, "penalty": 0,
        "rusher_player_id": "00-001", "rusher_player_name": "R.Test", "rushing_yards": 4,
        "yards_gained": 4, "passer_player_name": None,
    }
    row.update(overrides)
    return row


def make_df(rows):
    return pd.DataFrame(rows)


class TestCleanPbp:
    def test_drops_null_posteam(self):
        df = make_df([_base_row(posteam=None)])
        cleaned = clean_pbp(df)
        assert len(cleaned) == 0

    def test_drops_null_drive(self):
        df = make_df([_base_row(drive=None)])
        cleaned = clean_pbp(df)
        assert len(cleaned) == 0

    def test_excludes_qb_kneel_from_scrimmage(self):
        df = make_df([_base_row(play_type="qb_kneel")])
        cleaned = clean_pbp(df)
        assert not cleaned.iloc[0]["is_offensive_scrimmage_play"]

    def test_excludes_two_point_attempt(self):
        df = make_df([_base_row(two_point_attempt=1)])
        cleaned = clean_pbp(df)
        assert not cleaned.iloc[0]["is_offensive_scrimmage_play"]

    def test_excludes_special_teams_play(self):
        df = make_df([_base_row(play_type="punt", rush_attempt=0)])
        cleaned = clean_pbp(df)
        assert not cleaned.iloc[0]["is_offensive_scrimmage_play"]

    def test_sack_counts_as_scrimmage_and_pass_not_rush(self):
        df = make_df([_base_row(play_type="pass", rush_attempt=0, sack=1, rusher_player_id=None)])
        cleaned = clean_pbp(df)
        assert cleaned.iloc[0]["is_offensive_scrimmage_play"]
        assert cleaned.iloc[0]["is_pass_play"]
        assert not cleaned.iloc[0]["is_real_rush_attempt"]

    def test_normal_run_is_real_rush_attempt(self):
        df = make_df([_base_row()])
        cleaned = clean_pbp(df)
        assert cleaned.iloc[0]["is_real_rush_attempt"]
        assert cleaned.iloc[0]["is_offensive_scrimmage_play"]


class TestFirstDriveIdentification:
    def test_first_drive_is_not_always_drive_one(self):
        """
        A team's first offensive drive is the MIN drive number where
        posteam == team, not a hardcoded drive==1. Simulate a team whose
        earliest offensive snap is drive 2 (e.g. drive 1 belonged to a
        muffed-punt/defensive possession quirk in the raw data).
        """
        rows = [
            _base_row(game_id="G1", posteam="ARI", defteam="BUF", drive=2, rusher_player_id="00-001"),
            _base_row(game_id="G1", posteam="BUF", defteam="ARI", drive=1, rusher_player_id="00-002"),
        ]
        df = make_df(rows)
        cleaned = clean_pbp(df)
        first_drive_ids = compute_first_drive_ids(cleaned)
        ari_first = first_drive_ids[first_drive_ids["posteam"] == "ARI"]["first_drive"].iloc[0]
        assert ari_first == 2

    def test_build_first_drive_plays_excludes_later_drives(self):
        rows = [
            _base_row(game_id="G1", posteam="ARI", drive=1, rusher_player_id="00-001", rushing_yards=3),
            _base_row(game_id="G1", posteam="ARI", drive=2, rusher_player_id="00-001", rushing_yards=99),
        ]
        df = make_df(rows)
        fd = build_first_drive_plays(df)
        assert len(fd) == 1
        assert fd.iloc[0]["drive"] == 1
        assert fd.iloc[0]["rushing_yards"] == 3


class TestTeamRunPercentage:
    def test_run_percentage_formula(self):
        # 3 rush, 2 pass, 1 sack -> run% = 3 / (3+2+1) = 0.5
        rows = [
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="run", rush_attempt=1),
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="run", rush_attempt=1),
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="run", rush_attempt=1),
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="pass", rush_attempt=0, pass_attempt=1),
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="pass", rush_attempt=0, pass_attempt=1),
            _base_row(game_id="G1", posteam="ARI", drive=1, play_type="pass", rush_attempt=0, sack=1),
        ]
        df = make_df(rows)
        fd = build_first_drive_plays(df)
        team_metrics = compute_team_first_drive_metrics(fd)
        row = team_metrics[team_metrics["team"] == "ARI"].iloc[0]
        assert row["first_drive_run_pct"] == pytest.approx(0.5)
