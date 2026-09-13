"""
Defensive first-drive metrics and matchup adjustments (Sections 17-19).

Everything here is computed on the SAME first_drive_plays frame used for
team/RB metrics, but grouped by `defteam` instead of `posteam` -- i.e. "how
does this defense perform when it is the opponent facing the ball on the
other team's opening possession."
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import got_five_plus, shrink_rate

logger = logging.getLogger(__name__)

EXPLOSIVE_RUN_THRESHOLD = 10.0


def compute_defense_first_drive_metrics(first_drive_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Section 17: opponent first-drive rushing-defense metrics, one row per
    (season, defteam).
    """
    df = first_drive_plays.copy()
    rush = df[df["is_real_rush_attempt"]].copy()
    rush["rushing_yards"] = pd.to_numeric(rush["rushing_yards"], errors="coerce")
    if "epa" not in rush.columns:
        rush["epa"] = np.nan
    if "success" not in rush.columns:
        rush["success"] = np.nan

    per_game = rush.groupby(["season", "game_id", "defteam"]).agg(
        rush_attempts=("rushing_yards", "size"),
        rushing_yards=("rushing_yards", "sum"),
        five_plus=("rushing_yards", lambda s: got_five_plus(s, config.BIG_RUN_THRESHOLD_YARDS)),
        ten_plus=("rushing_yards", lambda s: got_five_plus(s, EXPLOSIVE_RUN_THRESHOLD)),
        avg_epa=("epa", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()

    all_plays_per_game = df.groupby(["season", "game_id", "defteam"]).agg(
        total_plays=("is_offensive_scrimmage_play", "sum"),
        pass_or_sack=("is_pass_play", "sum"),
    ).reset_index()

    per_game = per_game.merge(all_plays_per_game, on=["season", "game_id", "defteam"], how="right")
    per_game["rush_attempts"] = per_game["rush_attempts"].fillna(0)
    per_game["rushing_yards"] = per_game["rushing_yards"].fillna(0)
    per_game["five_plus"] = per_game["five_plus"].fillna(False)
    per_game["ten_plus"] = per_game["ten_plus"].fillna(False)

    def_metrics = per_game.groupby(["season", "defteam"]).agg(
        games=("game_id", "nunique"),
        total_rush_attempts=("rush_attempts", "sum"),
        total_rushing_yards=("rushing_yards", "sum"),
        five_plus_allowed_games=("five_plus", "sum"),
        ten_plus_allowed_games=("ten_plus", "sum"),
        avg_epa_allowed=("avg_epa", "mean"),
        success_rate_allowed=("success_rate", "mean"),
        total_plays=("total_plays", "sum"),
        total_pass_or_sack=("pass_or_sack", "sum"),
    ).reset_index()

    def_metrics["run_pct_faced"] = def_metrics["total_rush_attempts"] / (
        def_metrics["total_rush_attempts"] + def_metrics["total_pass_or_sack"]
    ).replace(0, np.nan)
    def_metrics["yards_per_rush_allowed"] = (
        def_metrics["total_rushing_yards"] / def_metrics["total_rush_attempts"].replace(0, np.nan)
    )
    def_metrics["five_plus_allowed_rate"] = def_metrics["five_plus_allowed_games"] / def_metrics["games"]
    def_metrics["ten_plus_allowed_rate"] = def_metrics["ten_plus_allowed_games"] / def_metrics["games"]

    return def_metrics.rename(columns={"defteam": "team"})


def apply_defense_shrinkage(defense_metrics: pd.DataFrame) -> pd.DataFrame:
    """Shrink each defense's 5+-allowed rate toward the league average for
    that season, weighted by games (small-sample defenses get pulled to
    league average -- Section 12 applies here too)."""
    df = defense_metrics.copy()
    out = []
    for season, group in df.groupby("season"):
        league_avg = group["five_plus_allowed_rate"].mean()
        group = group.copy()
        group["five_plus_allowed_rate_shrunk"] = group.apply(
            lambda r: shrink_rate(
                observed_rate=r["five_plus_allowed_rate"],
                n=r["games"],
                prior_rate=league_avg,
                prior_strength=config.PRIOR_STRENGTH_FIVE_PLUS_RATE,
            ),
            axis=1,
        )
        out.append(group)
    return pd.concat(out, ignore_index=True)


def blend_defense_seasons(defense_metrics_shrunk: pd.DataFrame, season_weights: Optional[dict] = None) -> pd.DataFrame:
    """Cross-season weighted blend, same approach as RB metrics (Section 11)."""
    from src.utils import renormalize_weights

    weights = dict(season_weights or config.SEASON_WEIGHTS)
    rows = []
    for team, group in defense_metrics_shrunk.groupby("team"):
        w = renormalize_weights({s: wt for s, wt in weights.items() if s in group["season"].values})
        out = {"team": team, "most_recent_season": int(group["season"].max())}
        for col in ["five_plus_allowed_rate_shrunk", "ten_plus_allowed_rate", "yards_per_rush_allowed",
                    "run_pct_faced", "avg_epa_allowed", "success_rate_allowed"]:
            total_v, total_w = 0.0, 0.0
            for _, r in group.iterrows():
                wt = w.get(r["season"], 0.0)
                if pd.notna(r[col]):
                    total_v += r[col] * wt
                    total_w += wt
            out[col] = (total_v / total_w) if total_w > 0 else np.nan
        out["games"] = group["games"].sum()
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QB rushing competition (Section 19)
# ---------------------------------------------------------------------------
def compute_qb_rush_competition(first_drive_plays: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, posteam), estimate how much the QB's own first-drive
    rushing tendency competes with the RB for carries. This is used as a
    SMALL negative adjustment (config.QB_RUSH_COMPETITION_MAX_PENALTY caps
    it) to expected RB first-drive carry probability -- it is not meant to
    dominate the model.
    """
    df = first_drive_plays.copy()
    df["qb_scramble"] = pd.to_numeric(df.get("qb_scramble", 0), errors="coerce").fillna(0)
    is_qb_designed_or_scramble = df["is_real_rush_attempt"] & (
        (df["qb_scramble"] == 1) | (df["passer_player_name"].notna() & (df["rusher_player_id"] == df.get("passer_player_id")))
    )

    per_game = df.groupby(["season", "game_id", "posteam"]).agg(
        team_plays=("is_offensive_scrimmage_play", "sum"),
    ).reset_index()
    qb_rush_per_game = df[is_qb_designed_or_scramble].groupby(["season", "game_id", "posteam"]).size().reset_index(name="qb_rush_attempts")

    per_game = per_game.merge(qb_rush_per_game, on=["season", "game_id", "posteam"], how="left")
    per_game["qb_rush_attempts"] = per_game["qb_rush_attempts"].fillna(0)

    team_season = per_game.groupby(["season", "posteam"]).agg(
        games=("game_id", "nunique"),
        qb_rush_rate=("qb_rush_attempts", "mean"),  # avg QB rushes per first drive
    ).reset_index().rename(columns={"posteam": "team"})

    max_rate = team_season["qb_rush_rate"].max() or 1.0
    team_season["qb_rush_competition"] = (
        (team_season["qb_rush_rate"] / max_rate) * config.QB_RUSH_COMPETITION_MAX_PENALTY
    )
    return team_season


def build_defense_and_qb_metrics(first_drive_plays: pd.DataFrame) -> dict:
    defense_raw = compute_defense_first_drive_metrics(first_drive_plays)
    defense_shrunk = apply_defense_shrinkage(defense_raw)
    defense_blended = blend_defense_seasons(defense_shrunk, config.SEASON_WEIGHTS)

    qb_competition = compute_qb_rush_competition(first_drive_plays)

    return {
        "defense_by_season": defense_shrunk,
        "defense_metrics": defense_blended,
        "qb_competition_by_season": qb_competition,
    }
