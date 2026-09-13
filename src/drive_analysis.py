"""
Data cleaning and first-offensive-drive identification (Sections 5, 6, 32, 33).

METHODOLOGY
-----------
A team's "first offensive drive" is NOT `drive == 1` in the raw nflfastR
data. The `drive` column is a game-level sequential counter for BOTH teams'
possessions and can carry NaNs, or be relative to something other than
"this team's own first possession" in games with overtime, defensive/special
teams scores that flip possession count, etc.

Instead we compute, independently for every (game_id, team):

    first_drive = min(drive) over all offensive plays where posteam == team

and merge that back onto the full play-by-play so we can select exactly the
rows belonging to a team's opening possession
(game_id, posteam, drive == first_drive).

A team's first offensive drive begins when it first gains possession and
ends when that possession ends (Section 33). We only count OFFENSIVE plays
that belong to that possession.

DATA CLEANING / EXCLUSIONS (Section 32) — applied before first-drive plays
are selected, and documented here so the methodology is auditable:

  * Rows with a null `posteam` are dropped (no offense on the play — e.g.
    timeouts, some administrative rows).
  * Rows with a null `drive` are dropped (can't be assigned to a possession).
  * Special teams snaps unrelated to a scrimmage down (kickoff, punt,
    field_goal, extra_point) are excluded from "offensive play" counts —
    they are not a rushing/passing play and do not represent the offense's
    play-calling.
  * Two-point conversion attempts (`two_point_attempt == 1`) are excluded —
    they are a distinct, non-standard-down snap.
  * Plays whose `play_type` is "no_play" (typically a pre-snap penalty with
    no down consumed on offense, e.g. delay of game before a snap, false
    start) are excluded from rushing/passing play counts, UNLESS the play
    has a `rush_attempt`/`pass_attempt`/`sack` flag set to 1, in which case
    the underlying play (e.g. an accepted/declined penalty on a real rush)
    is kept because nflfastR still marks it as a rush/pass/sack.
  * Aborted snaps (`aborted_play == 1`, e.g. a botched exchange) are kept
    for drive-boundary purposes but excluded from rushing-attempt counts
    since no rush was actually attempted.
  * QB kneels (`play_type == "qb_kneel"`) are excluded from rushing metrics:
    they are clock-killing plays, not real rushing opportunities, and would
    badly distort first-drive run/pass rates and RB carry rates (they are
    almost always end-of-half/game situations that can't occur on a first
    drive anyway, but we exclude defensively).
  * Sacks (`sack == 1`) are COUNTED AS PASSING PLAYS for run/pass-mix
    purposes (Section 5): "A sack should count as an offensive passing play
    for the purposes of measuring opening-drive play-calling." They are
    excluded from rushing-attempt/rushing-yards aggregates for RBs (a sack
    is credited to no rusher).
  * Field goal and extra point plays are excluded entirely from offensive
    play-calling counts.
  * Overtime plays are dropped for the FIRST-DRIVE dataset only in the
    sense that they cannot be a team's *first* drive of the game (a team's
    first drive, by construction, always occurs in regulation), so no
    special filtering is needed — the min-drive-number logic naturally
    never selects an OT drive as `first_drive`.

The resulting `first_drive_plays` frame contains every real offensive
scrimmage play (rush attempt, pass attempt, or sack) run by each team on
its own opening possession, for every team in every game in the loaded
data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exclusion rules documented in the module docstring.

    Returns a copy with a boolean `is_offensive_scrimmage_play` column
    marking rows that count toward play-calling / rushing metrics, rather
    than dropping rows outright — callers that need drive boundaries (e.g.
    to find where a possession ends) still need non-scrimmage rows like
    punts.
    """
    df = df.copy()

    # Drop position-less / unassignable rows entirely -- they can never be
    # part of any team's offensive drive.
    before = len(df)
    df = df[df["posteam"].notna()]
    df = df[df["drive"].notna()]
    logger.info("clean_pbp: dropped %d rows with null posteam/drive", before - len(df))

    for col in ["rush_attempt", "pass_attempt", "sack", "qb_scramble",
                "two_point_attempt", "aborted_play", "touchdown",
                "field_goal_attempt", "extra_point_attempt", "penalty"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    if "play_type" not in df.columns:
        df["play_type"] = np.nan

    is_kneel = df["play_type"].eq("qb_kneel")
    is_special_teams = df["play_type"].isin(["kickoff", "punt", "field_goal", "extra_point"])
    is_two_point = df["two_point_attempt"].eq(1)
    is_no_play_and_not_scrimmage = df["play_type"].eq("no_play") & (
        (df["rush_attempt"] == 0) & (df["pass_attempt"] == 0) & (df["sack"] == 0)
    )
    is_aborted_no_rush = (df["aborted_play"] == 1) & (df["rush_attempt"] == 1)

    is_scrimmage = (
        ((df["rush_attempt"] == 1) | (df["pass_attempt"] == 1) | (df["sack"] == 1))
        & ~is_kneel
        & ~is_special_teams
        & ~is_two_point
        & ~is_no_play_and_not_scrimmage
        & ~is_aborted_no_rush
    )

    df["is_offensive_scrimmage_play"] = is_scrimmage
    # A "real" rushing attempt for RB/rushing-yardage purposes excludes
    # sacks (counted as pass plays) and qb kneels.
    df["is_real_rush_attempt"] = (
        (df["rush_attempt"] == 1) & ~is_kneel & ~is_aborted_no_rush
    )
    df["is_pass_play"] = ((df["pass_attempt"] == 1) | (df["sack"] == 1)) & ~is_two_point

    return df


def compute_first_drive_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for every (game_id, posteam), the number of that team's first
    offensive drive.

    Returns a DataFrame with columns: game_id, posteam, first_drive.
    """
    scrimmage = df[df["is_offensive_scrimmage_play"]]
    first_drive_ids = (
        scrimmage.groupby(["game_id", "posteam"])["drive"]
        .min()
        .reset_index()
        .rename(columns={"drive": "first_drive"})
    )
    return first_drive_ids


def build_first_drive_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw pbp, identify each team's first offensive drive, and
    return every offensive scrimmage play belonging to that drive.
    """
    cleaned = clean_pbp(df)
    first_drive_ids = compute_first_drive_ids(cleaned)

    merged = cleaned.merge(first_drive_ids, on=["game_id", "posteam"], how="inner")
    first_drive_plays = merged[
        (merged["drive"] == merged["first_drive"]) & (merged["is_offensive_scrimmage_play"])
    ].copy()

    logger.info(
        "build_first_drive_plays: %d first-drive scrimmage plays across %d team-games",
        len(first_drive_plays), first_drive_ids.shape[0],
    )
    return first_drive_plays


# ---------------------------------------------------------------------------
# Team first-drive metrics (Section 6)
# ---------------------------------------------------------------------------
def compute_team_first_drive_metrics(first_drive_plays: pd.DataFrame, all_cleaned_pbp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate to one row per (season, posteam) with the metrics required by
    Section 6. Also emits a per-team-game table's derived rates.
    """
    fd = first_drive_plays.copy()

    fd["is_rush"] = fd["is_real_rush_attempt"].astype(int)
    fd["is_pass_or_sack"] = fd["is_pass_play"].astype(int)
    fd["is_sack"] = fd["sack"].astype(int)
    fd["yards_gained"] = pd.to_numeric(fd.get("yards_gained", np.nan), errors="coerce")
    fd["rushing_yards"] = pd.to_numeric(fd.get("rushing_yards", np.nan), errors="coerce")

    game_group_cols = ["season", "game_id", "posteam"]
    per_game = fd.groupby(game_group_cols).agg(
        plays=("is_offensive_scrimmage_play", "sum"),
        rush_attempts=("is_rush", "sum"),
        pass_attempts=("pass_attempt", "sum"),
        sacks=("is_sack", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        total_yards=("yards_gained", "sum"),
        touchdown=("touchdown", "max"),
    ).reset_index()

    per_game["run_pct"] = per_game["rush_attempts"] / (
        per_game["rush_attempts"] + per_game["pass_attempts"] + per_game["sacks"]
    ).replace(0, np.nan)
    per_game["pass_pct"] = 1 - per_game["run_pct"]
    per_game["is_three_and_out"] = (
        (per_game["rush_attempts"] + per_game["pass_attempts"] + per_game["sacks"] <= 3)
        & (per_game["touchdown"] == 0)
    ).astype(int)
    # "scoring" here means the first drive itself ended in a touchdown
    # (nflfastR's per-play `touchdown` flag is drive-scoped for the scoring
    # play). Field goals are not attributable to a single drive's offensive
    # snaps in this simplified pass and are left for a future OL/scoring
    # enhancement (documented as a known limitation in the README).
    per_game["is_scoring_drive"] = per_game["touchdown"]

    team_metrics = per_game.groupby(["season", "posteam"]).agg(
        games=("game_id", "nunique"),
        total_plays=("plays", "sum"),
        total_rush_attempts=("rush_attempts", "sum"),
        total_pass_attempts=("pass_attempts", "sum"),
        total_sacks=("sacks", "sum"),
        total_rushing_yards=("rushing_yards", "sum"),
        total_yards=("total_yards", "sum"),
        avg_plays=("plays", "mean"),
        avg_rush_attempts=("rush_attempts", "mean"),
        avg_rushing_yards=("rushing_yards", "mean"),
        avg_pass_attempts=("pass_attempts", "mean"),
        avg_yards_gained=("total_yards", "mean"),
        touchdown_rate=("touchdown", "mean"),
        scoring_rate=("is_scoring_drive", "mean"),
        three_and_out_rate=("is_three_and_out", "mean"),
    ).reset_index()

    denom = (team_metrics["total_rush_attempts"] + team_metrics["total_pass_attempts"] + team_metrics["total_sacks"])
    team_metrics["first_drive_run_pct"] = team_metrics["total_rush_attempts"] / denom.replace(0, np.nan)
    team_metrics["first_drive_pass_pct"] = 1 - team_metrics["first_drive_run_pct"]

    return team_metrics.rename(columns={"posteam": "team"})


def blend_team_seasons(team_metrics_by_season: pd.DataFrame, season_weights: Optional[dict] = None) -> pd.DataFrame:
    """
    Cross-season weighted blend of per-season team metrics into one row per
    team (Section 11) -- the same treatment given to RB and defense metrics,
    so every table build_model_table joins on "team" has exactly one row
    per team and cannot fan out a merge.
    """
    import config
    from src.utils import renormalize_weights

    weights = dict(season_weights or config.SEASON_WEIGHTS)
    rate_cols = [
        "first_drive_run_pct", "first_drive_pass_pct", "avg_plays", "avg_rush_attempts",
        "avg_rushing_yards", "avg_pass_attempts", "avg_yards_gained",
        "touchdown_rate", "scoring_rate", "three_and_out_rate",
    ]
    rows = []
    for team, group in team_metrics_by_season.groupby("team"):
        w = renormalize_weights({s: wt for s, wt in weights.items() if s in group["season"].values})
        out = {"team": team, "most_recent_season": int(group["season"].max())}
        for col in rate_cols:
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
