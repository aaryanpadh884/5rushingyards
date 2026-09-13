"""
The core probability model (Sections 13-16, 24, 44).

BASELINE MODEL (Section 44) -- deliberately simple and transparent:

    P(5+ on first drive) = P(RB gets >=1 first-drive carry)
                           x P(RB gets 5+ | RB gets >=1 first-drive carry)

Both factors start from shrunk, cross-season-weighted historical rates
(src/rb_analysis.py) and are then adjusted by a small number of documented,
transparent multipliers -- current depth-chart role, QB rush competition,
and opponent first-drive run defense. We deliberately do NOT fit a
black-box ML model on a few hundred team-games of first-drive data; see the
README "Known Limitations" section for why (overfitting risk given sample
size, and Section 44's explicit instruction to start with a transparent
baseline).

CURRENT ROLE OVERRIDE (Section 9, 14): historical tendencies describe how a
player USED to be used. The current depth chart (data/current_rb_roles.csv)
describes who is getting the ball NOW. We use history to estimate a
player's skill/opportunity rate, but the ROLE MULTIPLIER (config.py) is
what lets an accurate current-role assignment override a stale historical
read -- e.g. a backup who inherits a starting job after an offseason trade.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import confidence_label

logger = logging.getLogger(__name__)

REQUIRED_ROLE_COLUMNS = ["team", "player_name", "player_id", "role", "status"]


def load_current_rb_roles(path=None) -> pd.DataFrame:
    """
    Load the manually maintained current-depth-chart file (Section 9).

    This is the AUTHORITATIVE source for "who is this team's starting RB
    right now" -- historical carry share is informative but must not
    override an explicit current-role assignment (Section 35).
    """
    path = path or config.CURRENT_RB_ROLES_FILE
    if not path.exists():
        logger.warning(
            "current RB roles file not found at %s -- role-based adjustments "
            "will be skipped and every player will be flagged "
            "'CURRENT ROLE: unavailable'.", path,
        )
        return pd.DataFrame(columns=REQUIRED_ROLE_COLUMNS)

    roles = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED_ROLE_COLUMNS if c not in roles.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. Expected header: "
            f"{','.join(REQUIRED_ROLE_COLUMNS)}"
        )
    roles["role"] = roles["role"].str.strip()
    roles["status"] = roles["status"].str.strip().str.lower()
    return roles


def apply_role_adjustment(p_carry_raw: float, role: Optional[str], status: Optional[str]) -> tuple[float, str]:
    """
    Apply the current-role multiplier (Section 9, 14) to a historical
    P(carry) estimate. Returns (adjusted_probability, note).
    """
    if role is None or (isinstance(role, float) and np.isnan(role)):
        return p_carry_raw, "no current-role entry; using historical rate only"

    if status == "out":
        return 0.0, "player listed OUT -- removed"

    key = status if status in config.ROLE_CARRY_MULTIPLIER else role
    multiplier = config.ROLE_CARRY_MULTIPLIER.get(key, config.ROLE_CARRY_MULTIPLIER.get(role, 1.0))

    adjusted = float(np.clip(p_carry_raw * multiplier, 0.0, 0.99))
    note = f"role={role}, status={status}, multiplier={multiplier:.2f}"
    return adjusted, note


def apply_qb_competition_adjustment(p_carry: float, qb_rush_competition: Optional[float]) -> float:
    """Small negative adjustment to carry probability for high QB rush
    competition (Section 19). Capped by config.QB_RUSH_COMPETITION_MAX_PENALTY."""
    if qb_rush_competition is None or pd.isna(qb_rush_competition):
        return p_carry
    return float(np.clip(p_carry * (1.0 - qb_rush_competition), 0.0, 0.99))


def apply_defense_adjustment(p_five_given_carry: float, opp_five_plus_allowed_rate: Optional[float],
                              league_avg_five_plus_allowed_rate: Optional[float],
                              max_factor: float = 1.30, min_factor: float = 0.70) -> float:
    """
    Matchup adjustment to P(5+ | carry) (Section 15, 17): scale the
    player's own conditional rate by how much more/less likely THIS
    opponent is to allow a 5+ run on the first drive, relative to league
    average. Capped so a single opponent-sample outlier can't dominate the
    player's own established skill/role signal.
    """
    if (
        opp_five_plus_allowed_rate is None or pd.isna(opp_five_plus_allowed_rate)
        or league_avg_five_plus_allowed_rate is None or pd.isna(league_avg_five_plus_allowed_rate)
        or league_avg_five_plus_allowed_rate == 0
    ):
        return p_five_given_carry
    factor = opp_five_plus_allowed_rate / league_avg_five_plus_allowed_rate
    factor = float(np.clip(factor, min_factor, max_factor))
    return float(np.clip(p_five_given_carry * factor, 0.01, 0.99))


def compute_confidence(games_with_carry: float, role: Optional[str], status: Optional[str],
                        opponent_games: Optional[float]) -> str:
    """
    Section 24: confidence is about how much we TRUST the number, not
    whether the number is favorable. A 90% probability built on 3 games and
    a "committee" role is LOW confidence; a 60% probability built on 30
    games with a clean RB1/healthy role is HIGH confidence.
    """
    sample = games_with_carry if pd.notna(games_with_carry) else 0
    base = confidence_label(sample, config.CONFIDENCE_HIGH_SAMPLE, config.CONFIDENCE_MED_SAMPLE)

    role_uncertain = (role is None or (isinstance(role, float) and pd.isna(role)) or status in ("committee", "questionable"))
    opp_thin = opponent_games is not None and pd.notna(opponent_games) and opponent_games < config.MIN_SAMPLE_SIZE

    if base == "HIGH" and (role_uncertain or opp_thin):
        return "MEDIUM"
    if base == "MEDIUM" and role_uncertain and opp_thin:
        return "LOW"
    return base


def build_model_table(rb_metrics: pd.DataFrame, team_metrics: pd.DataFrame,
                       defense_metrics: pd.DataFrame, qb_competition_by_season: pd.DataFrame,
                       current_roles: pd.DataFrame, matchups: pd.DataFrame,
                       players: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Assemble the final model table: one row per RB with a scheduled
    opponent in `matchups` (columns: team, opponent [, week, game_id]).

    This is the function get_weekly_rankings() calls; it does NOT invent
    an opponent for a player if one isn't supplied.

    `players` (optional, the nflverse players roster with a `latest_team`
    column) enables the departed-teammate carry boost -- pass None to skip
    it. It must be omitted when this is called from the backtest, since
    `latest_team` is a live snapshot and using it against a past season
    would leak future roster information into a historical prediction.
    """
    # Resolve each RB's CURRENT team from the depth-chart file BEFORE joining
    # matchups/team/defense metrics. Historical play-by-play team assignment
    # (rb_metrics["team"]) reflects where a player used to play; a player
    # who was traded/signed elsewhere in the offseason must be matched
    # against his new team's opponent, not his old one (Section 9, 35).
    df = rb_metrics.rename(columns={"team": "historical_team"}).merge(
        current_roles[["team", "player_id", "role", "status"]],
        on="player_id", how="left",
    )
    df["team"] = df["team"].fillna(df["historical_team"])

    # Drop historical RBs who are no longer on an active NFL roster (per
    # nflverse's `last_season` field) and have no explicit current_roles
    # entry. Without this, a player with real 2024-2025 usage but who
    # retired, was released, or is on reserve/suspended (e.g. last_season
    # short of config.CURRENT_SEASON) would otherwise keep showing up as a
    # live candidate under his old team forever, purely because "no
    # current-role entry" falls back to his historical team rather than
    # checking whether he's still in the league. A manual current_roles
    # entry always overrides this check (role.notna()), so you can force a
    # specific player back in if the roster feed is behind reality.
    if players is not None and "last_season" in players.columns:
        last_season_lookup = players.set_index("player_id")["last_season"]
        df["_last_season"] = df["player_id"].map(last_season_lookup)
        is_inactive = df["_last_season"].notna() & (df["_last_season"] < config.CURRENT_SEASON)
        has_manual_override = df["role"].notna()
        dropped = int((is_inactive & ~has_manual_override).sum())
        if dropped:
            logger.info(
                "dropping %d player(s) with no current 2026 roster spot (last_season < %s) "
                "and no current_rb_roles.csv override", dropped, config.CURRENT_SEASON,
            )
        df = df[~(is_inactive & ~has_manual_override)].drop(columns=["_last_season"])

    if players is not None:
        from src.rb_analysis import compute_departed_teammate_boost
        boost_table = compute_departed_teammate_boost(rb_metrics, players, config.DEPARTED_TEAMMATE_MAX_BOOST)
        df = df.merge(
            boost_table.rename(columns={"team": "historical_team"}),
            on=["historical_team", "player_id"], how="left",
        )
        df["departed_teammate_boost"] = df["departed_teammate_boost"].fillna(0.0)
    else:
        df["departed_teammate_boost"] = 0.0

    df = df.merge(matchups, on="team", how="inner")

    latest_season = int(qb_competition_by_season["season"].max()) if not qb_competition_by_season.empty else None
    qb_latest = (
        qb_competition_by_season[qb_competition_by_season["season"] == latest_season]
        if latest_season is not None else qb_competition_by_season
    )
    df = df.merge(qb_latest[["team", "qb_rush_competition"]], on="team", how="left")

    df = df.merge(
        team_metrics[["team", "first_drive_run_pct", "games"]].rename(
            columns={"games": "team_first_drive_games"}
        ),
        on="team", how="left",
    )

    league_avg_five_allowed = defense_metrics["five_plus_allowed_rate_shrunk"].mean()
    df = df.merge(
        defense_metrics.rename(columns={"team": "opponent"})[
            ["opponent", "five_plus_allowed_rate_shrunk", "yards_per_rush_allowed", "games"]
        ].rename(columns={"games": "opponent_games", "five_plus_allowed_rate_shrunk": "opp_five_plus_allowed_rate"}),
        on="opponent", how="left",
    )

    p_carry_role_adj = []
    role_notes = []
    for _, r in df.iterrows():
        adj, note = apply_role_adjustment(r["p_carry_1plus_shrunk"], r.get("role"), r.get("status"))
        boost = r.get("departed_teammate_boost", 0.0) or 0.0
        if boost > 0:
            adj = float(np.clip(adj * (1 + boost), 0.0, 0.99))
            note = f"{note}; departed-teammate boost=+{boost:.0%}"
        p_carry_role_adj.append(adj)
        role_notes.append(note)
    df["p_carry_role_adjusted"] = p_carry_role_adj
    df["role_note"] = role_notes

    df["p_carry_final"] = df.apply(
        lambda r: apply_qb_competition_adjustment(r["p_carry_role_adjusted"], r.get("qb_rush_competition")), axis=1
    )

    df["p_five_given_carry_final"] = df.apply(
        lambda r: apply_defense_adjustment(
            r["p_five_given_carry_shrunk"], r.get("opp_five_plus_allowed_rate"), league_avg_five_allowed
        ),
        axis=1,
    )

    df["model_probability"] = (df["p_carry_final"] * df["p_five_given_carry_final"]).clip(0.01, 0.99)

    df["confidence"] = df.apply(
        lambda r: compute_confidence(r["games_with_carry"], r.get("role"), r.get("status"), r.get("opponent_games")),
        axis=1,
    )

    df["current_role_source"] = config.CURRENT_ROLE_SOURCE
    return df
