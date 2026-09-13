"""
Weekly rankings and betting filters (Sections 27-28).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

DISPLAY_COLUMNS = [
    "rank", "player_name", "team", "opponent",
    "p_carry_final", "p_five_given_carry_final", "model_probability",
    "sportsbook_odds", "sportsbook_implied_probability", "edge", "fair_odds",
    "confidence", "bet_category",
    "first_drive_run_pct", "avg_carry_share", "p_carry_1plus_shrunk",
    "total_carries", "total_rushing_yards", "opp_five_plus_allowed_rate",
    "games_with_carry", "games_with_5plus", "role", "status", "role_note", "odds_available",
    "expected_value_per_100", "current_role_source",
]


def label_bet_category(row: pd.Series) -> str:
    """
    Section 28: label PASS when a candidate doesn't clear the configured
    thresholds. Never force a bet.
    """
    if not row.get("odds_available", False):
        return "NO ODDS"
    if row["games_with_carry"] < config.MIN_SAMPLE_SIZE:
        return "PASS (small sample)"
    if row["model_probability"] < config.MIN_MODEL_PROBABILITY:
        return "PASS (low probability)"
    if row["edge"] is None or pd.isna(row["edge"]) or row["edge"] < config.MIN_EDGE:
        return "PASS (insufficient edge)"
    return "BEST BET"


def rank_candidates(model_with_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by: (1) positive EV, (2) model probability, (3) confidence
    (Section 27). PASS candidates sort to the bottom but are still shown.
    """
    df = model_with_odds.copy()
    df["bet_category"] = df.apply(label_bet_category, axis=1)

    confidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_confidence_rank"] = df["confidence"].map(confidence_rank).fillna(3)
    df["_is_best_bet"] = (df["bet_category"] == "BEST BET").astype(int)
    df["_ev_sort"] = df["expected_value_per_100"].fillna(-1e9)

    df = df.sort_values(
        by=["_is_best_bet", "_ev_sort", "model_probability", "_confidence_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    df = df.drop(columns=["_confidence_rank", "_is_best_bet", "_ev_sort"])
    return df


FILTER_DEFINITIONS = {
    "all": lambda df: df,
    "best_bets": lambda df: df[df["bet_category"] == "BEST BET"],
    "high_probability": lambda df: df[df["model_probability"] >= config.MIN_MODEL_PROBABILITY],
    "best_value": lambda df: df[df["edge"].fillna(-1) >= config.MIN_EDGE].sort_values("edge", ascending=False),
    "passes": lambda df: df[df["bet_category"].str.startswith("PASS", na=False)],
}


def filter_rankings(ranked_df: pd.DataFrame, filter_name: str = "all") -> pd.DataFrame:
    if filter_name not in FILTER_DEFINITIONS:
        raise ValueError(f"Unknown filter '{filter_name}'. Choose from: {list(FILTER_DEFINITIONS)}")
    return FILTER_DEFINITIONS[filter_name](ranked_df)


def get_weekly_rankings(rb_metrics: pd.DataFrame, team_metrics: pd.DataFrame,
                         defense_metrics: pd.DataFrame, qb_competition_by_season: pd.DataFrame,
                         current_roles: pd.DataFrame, matchups: pd.DataFrame,
                         odds: pd.DataFrame, filter_name: str = "all") -> pd.DataFrame:
    """
    Section 27 entry point: build the model table, attach odds/edge, rank,
    and apply the requested filter. This is what main.py's `rankings`
    command and the Streamlit dashboard call.
    """
    from src.model import build_model_table
    from src.odds import attach_odds_and_edge

    model_df = build_model_table(
        rb_metrics, team_metrics, defense_metrics, qb_competition_by_season, current_roles, matchups
    )
    with_odds = attach_odds_and_edge(model_df, odds)
    ranked = rank_candidates(with_odds)
    return filter_rankings(ranked, filter_name)
