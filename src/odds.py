"""
Sportsbook odds handling (Sections 20-23).

Two optional CSVs feed this module:

  data/week1_odds.csv       -- game environment (spread, team_total) per
                                Section 20. Optional context, not required
                                for the baseline model.

  data/sportsbook_odds.csv  -- actual first-drive-rushing-prop odds per
                                Section 21, one row per (RB, market, line,
                                sportsbook).

Neither file is invented if missing: callers get an explicit "unavailable"
marker rather than a silently guessed number (Section 45).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from src.utils import (
    american_odds_to_implied_probability,
    probability_to_fair_american_odds,
    expected_value_per_100,
)

logger = logging.getLogger(__name__)

ODDS_COLUMNS = ["player_name", "team", "market", "line", "american_odds", "sportsbook", "timestamp"]
GAME_ENV_COLUMNS = ["game_id", "team", "spread", "team_total"]


def load_sportsbook_odds(path: Optional[Path] = None) -> pd.DataFrame:
    """Load manually entered sportsbook odds (Section 21). Returns an empty
    frame with the right columns if the file doesn't exist -- callers must
    treat a player with no matching row as 'odds unavailable', never guess."""
    path = path or config.SPORTSBOOK_ODDS_FILE
    if not path.exists():
        logger.warning("sportsbook odds file not found at %s -- edge/fair-odds "
                        "columns will be marked unavailable.", path)
        return pd.DataFrame(columns=ODDS_COLUMNS)

    odds = pd.read_csv(path)
    missing = [c for c in ODDS_COLUMNS if c not in odds.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    odds["implied_probability"] = odds["american_odds"].apply(american_odds_to_implied_probability)
    return odds


def load_game_environment(path: Optional[Path] = None) -> pd.DataFrame:
    """Load optional Week-1 game environment data (Section 20)."""
    path = path or config.WEEK1_ODDS_FILE
    if not path.exists():
        logger.warning("game environment file not found at %s -- spread/team_total "
                        "context will be marked unavailable.", path)
        return pd.DataFrame(columns=GAME_ENV_COLUMNS)
    return pd.read_csv(path)


def pick_best_line(odds: pd.DataFrame, player_name: str, team: str, market: str = "first_drive_rushing", line: str = "5+") -> Optional[pd.Series]:
    """Return the best (highest payout / lowest implied probability) available
    sportsbook line for a player+market, or None if no odds are on file."""
    matches = odds[
        (odds["player_name"] == player_name)
        & (odds["team"] == team)
        & (odds["market"] == market)
        & (odds["line"].astype(str) == str(line))
    ]
    if matches.empty:
        return None
    return matches.sort_values("implied_probability").iloc[0]


def attach_odds_and_edge(model_df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Section 22-23: attach sportsbook implied probability, fair odds, edge,
    and $100-stake EV to each row of the model table. Rows without a
    matching sportsbook line get explicit NaN/'unavailable' markers rather
    than an assumed value.
    """
    df = model_df.copy()

    fair_odds, sb_odds, implied_p, edge, ev100, sportsbook_name = [], [], [], [], [], []
    for _, r in df.iterrows():
        fo = probability_to_fair_american_odds(r["model_probability"])
        fair_odds.append(fo)

        best = pick_best_line(odds, r["player_name"], r["team"])
        if best is None:
            sb_odds.append(None)
            implied_p.append(None)
            edge.append(None)
            ev100.append(None)
            sportsbook_name.append(None)
        else:
            sb_odds.append(best["american_odds"])
            implied_p.append(best["implied_probability"])
            edge.append(r["model_probability"] - best["implied_probability"])
            ev100.append(expected_value_per_100(r["model_probability"], best["american_odds"]))
            sportsbook_name.append(best["sportsbook"])

    df["fair_odds"] = fair_odds
    df["sportsbook_odds"] = sb_odds
    df["sportsbook_implied_probability"] = implied_p
    df["edge"] = edge
    df["expected_value_per_100"] = ev100
    df["sportsbook"] = sportsbook_name
    df["odds_available"] = df["sportsbook_odds"].notna()
    return df
