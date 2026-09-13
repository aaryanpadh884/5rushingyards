"""
RB-level first-drive rushing metrics (Sections 7-12, 34).

MARKET DEFINITION (Sections 7, 34) — this is the single most important rule
in this module:

    "RB records >=5 rushing yards on first drive" means the RB had AT LEAST
    ONE INDIVIDUAL rushing attempt of 5+ yards during the team's first
    offensive drive. It is NOT the sum of all his carries on the drive.

    [2, 3, 4, 4] -> does NOT qualify (max single carry = 4)
    [2, 6]       -> DOES qualify (one carry of 6)

RB IDENTIFICATION: rushing plays are attributed to `rusher_player_id`
(gsis_id). We restrict to players whose `position_group == "RB"` in the
nflverse players roster (see data_loader.load_player_positions) so that
quarterback scrambles/designed QB runs and WR jet-sweep carries do not get
counted as "RB carries" or pollute a real RB's per-game splits. Names are
carried along for display only (Section 10).

HISTORICAL WEIGHTING (Section 11): recent-season metrics are blended with
older-season metrics using config.SEASON_WEIGHTS, renormalized to whatever
seasons are actually present.

SHRINKAGE (Section 12): every rate metric (P(carry), P(5+|carry), etc.) is
passed through empirical-Bayes shrinkage toward a league-average prior,
with the prior weighted more heavily when the player's own sample size is
small. This prevents a rookie with 1 carry and 1 big run from being scored
as a 100% lock.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import got_five_plus, max_carry, shrink_rate, renormalize_weights

logger = logging.getLogger(__name__)


def attach_rb_positions(first_drive_plays: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Tag each first-drive rushing play with the rusher's position_group."""
    df = first_drive_plays.copy()
    df = df.merge(
        players.rename(columns={"player_id": "rusher_player_id"}),
        on="rusher_player_id",
        how="left",
        suffixes=("", "_roster"),
    )
    return df


def compute_team_game_rb_carries(fd_with_positions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, game_id, posteam, rusher_player_id) real rushing
    attempt aggregate on the first drive, restricted to RBs.

    Also computes each team-game's total RB carries so per-player carry
    share (Section 8) can be derived.
    """
    df = fd_with_positions[fd_with_positions["is_real_rush_attempt"]].copy()
    df["rushing_yards"] = pd.to_numeric(df["rushing_yards"], errors="coerce")

    is_rb = df["position_group"] == "RB"
    rb_plays = df[is_rb].copy()

    # Group on the roster's full display name (e.g. "Jahmyr Gibbs"), not
    # nflfastR's abbreviated `rusher_player_name` (e.g. "J.Gibbs") -- the
    # full name is what data/current_rb_roles.csv and
    # data/sportsbook_odds.csv key on (Section 21 has no player_id column),
    # so display must match it exactly for odds/role lookups to succeed.
    per_player_game = rb_plays.groupby(
        ["season", "game_id", "posteam", "rusher_player_id", "player_name"], dropna=False
    ).agg(
        carries=("rushing_yards", "size"),
        rushing_yards=("rushing_yards", "sum"),
        longest_carry=("rushing_yards", "max"),
    ).reset_index()

    got5 = (
        rb_plays.groupby(["season", "game_id", "posteam", "rusher_player_id"])["rushing_yards"]
        .apply(lambda s: got_five_plus(s, config.BIG_RUN_THRESHOLD_YARDS))
        .reset_index(name="got_5plus")
    )
    got10 = (
        rb_plays.groupby(["season", "game_id", "posteam", "rusher_player_id"])["rushing_yards"]
        .apply(lambda s: got_five_plus(s, 10))
        .reset_index(name="got_10plus")
    )

    per_player_game = per_player_game.merge(
        got5, on=["season", "game_id", "posteam", "rusher_player_id"], how="left"
    ).merge(
        got10, on=["season", "game_id", "posteam", "rusher_player_id"], how="left"
    )

    team_game_rb_totals = rb_plays.groupby(["season", "game_id", "posteam"]).agg(
        team_rb_carries=("rushing_yards", "size"),
    ).reset_index()

    per_player_game = per_player_game.merge(
        team_game_rb_totals, on=["season", "game_id", "posteam"], how="left"
    )
    per_player_game["carry_share"] = (
        per_player_game["carries"] / per_player_game["team_rb_carries"].replace(0, np.nan)
    )

    return per_player_game


def compute_league_priors(per_player_game: pd.DataFrame, season: int) -> dict:
    """
    League-wide first-drive RB rates for a given season, used as the
    empirical-Bayes prior for shrinkage (Section 12).
    """
    season_rows = per_player_game[per_player_game["season"] == season]
    if season_rows.empty:
        return {"p_five_plus_given_carry": 0.30, "avg_carry_share": 0.5}

    p5 = season_rows["got_5plus"].mean()
    return {
        "p_five_plus_given_carry": float(p5) if pd.notna(p5) else 0.30,
        "avg_carry_share": float(season_rows["carry_share"].mean()),
    }


def compute_player_season_metrics(per_player_game: pd.DataFrame, first_drive_team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per_player_game up to one row per (season, player_id), with
    the raw counts and rates described in Section 7 -- but NOT yet shrunk
    or cross-season weighted (that happens in build_rb_metrics).

    `first_drive_team_games` must have one row per (season, game_id,
    posteam) so we can compute P(RB gets >=1 carry) as
    (games with >=1 carry) / (team's total first-drive games), i.e. using
    the correct denominator of ALL that team's games, not just games where
    the RB happened to touch the ball.
    """
    g = per_player_game.groupby(["season", "posteam", "rusher_player_id", "player_name"])

    player_season = g.agg(
        games_with_carry=("game_id", "nunique"),
        total_carries=("carries", "sum"),
        total_rushing_yards=("rushing_yards", "sum"),
        longest_carry=("longest_carry", "max"),
        games_with_2plus=("carries", lambda s: (s >= 2).sum()),
        games_with_3plus=("carries", lambda s: (s >= 3).sum()),
        games_with_5plus=("got_5plus", "sum"),
        games_with_10plus=("got_10plus", "sum"),
        avg_carry_share=("carry_share", "mean"),
    ).reset_index()

    team_games = first_drive_team_games.groupby(["season", "posteam"])["game_id"].nunique().reset_index(
        name="team_total_first_drive_games"
    )
    player_season = player_season.merge(team_games, on=["season", "posteam"], how="left")

    player_season["yards_per_carry"] = (
        player_season["total_rushing_yards"] / player_season["total_carries"].replace(0, np.nan)
    )
    player_season["p_carry_1plus_raw"] = (
        player_season["games_with_carry"] / player_season["team_total_first_drive_games"].replace(0, np.nan)
    )
    player_season["p_carry_2plus_raw"] = (
        player_season["games_with_2plus"] / player_season["team_total_first_drive_games"].replace(0, np.nan)
    )
    player_season["p_carry_3plus_raw"] = (
        player_season["games_with_3plus"] / player_season["team_total_first_drive_games"].replace(0, np.nan)
    )
    # P(5+ | carry) uses the games-with-a-carry denominator, since the
    # conditional probability is meaningless without a carry.
    player_season["p_five_given_carry_raw"] = (
        player_season["games_with_5plus"] / player_season["games_with_carry"].replace(0, np.nan)
    )
    player_season["p_ten_given_carry_raw"] = (
        player_season["games_with_10plus"] / player_season["games_with_carry"].replace(0, np.nan)
    )
    # Unconditional P(5+) purely from history (no role/matchup adjustment).
    player_season["p_five_unconditional_raw"] = (
        player_season["games_with_5plus"] / player_season["team_total_first_drive_games"].replace(0, np.nan)
    )

    return player_season.rename(columns={"posteam": "team", "rusher_player_id": "player_id"})


def apply_shrinkage(player_season: pd.DataFrame, league_priors_by_season: dict) -> pd.DataFrame:
    """
    Empirical-Bayes shrinkage (Section 12) of the per-season conditional
    5+ rate and the carry-share rate toward that season's league average,
    weighted by games_with_carry (sample size backing the rate).
    """
    df = player_season.copy()

    def prior_for(row):
        return league_priors_by_season.get(row["season"], {"p_five_plus_given_carry": 0.30, "avg_carry_share": 0.5})

    df["_prior_p5"] = df.apply(lambda r: prior_for(r)["p_five_plus_given_carry"], axis=1)

    df["p_five_given_carry_shrunk"] = df.apply(
        lambda r: shrink_rate(
            observed_rate=r["p_five_given_carry_raw"] if pd.notna(r["p_five_given_carry_raw"]) else r["_prior_p5"],
            n=r["games_with_carry"],
            prior_rate=r["_prior_p5"],
            prior_strength=config.PRIOR_STRENGTH_FIVE_PLUS_RATE,
        ),
        axis=1,
    )

    league_avg_carry_prob = df["p_carry_1plus_raw"].mean()
    df["p_carry_1plus_shrunk"] = df.apply(
        lambda r: shrink_rate(
            observed_rate=r["p_carry_1plus_raw"] if pd.notna(r["p_carry_1plus_raw"]) else league_avg_carry_prob,
            n=r["team_total_first_drive_games"],
            prior_rate=league_avg_carry_prob,
            prior_strength=config.PRIOR_STRENGTH_CARRY_RATE,
        ),
        axis=1,
    )

    df = df.drop(columns=["_prior_p5"])
    return df


def blend_seasons(player_season_shrunk: pd.DataFrame, season_weights: Optional[dict] = None) -> pd.DataFrame:
    """
    Combine each player's per-season shrunk metrics into one blended,
    historically-weighted row per (team, player_id), per Section 11.

    Weights come from config.SEASON_WEIGHTS and are renormalized to
    whichever seasons the player actually has data for -- e.g. a rookie
    with only a 2025 row gets weight 1.0 on 2025, not 0.70.
    """
    weights = dict(season_weights or config.SEASON_WEIGHTS)
    df = player_season_shrunk.copy()

    rate_cols = [
        "p_carry_1plus_shrunk", "p_carry_2plus_raw", "p_carry_3plus_raw",
        "p_five_given_carry_shrunk", "p_ten_given_carry_raw",
        "yards_per_carry", "avg_carry_share",
    ]
    sum_cols = [
        "games_with_carry", "total_carries", "total_rushing_yards",
        "games_with_5plus", "games_with_10plus", "team_total_first_drive_games",
    ]

    rows = []
    for (player_id,), group in df.groupby(["player_id"]):
        player_weights = renormalize_weights({s: w for s, w in weights.items() if s in group["season"].values})
        out = {"player_id": player_id}
        # display fields: use the most recent season's team/name
        latest = group.sort_values("season").iloc[-1]
        out["player_name"] = latest["player_name"]
        out["team"] = latest["team"]
        out["most_recent_season"] = int(latest["season"])

        for col in rate_cols:
            total_w, total_v = 0.0, 0.0
            for _, r in group.iterrows():
                w = player_weights.get(r["season"], 0.0)
                if pd.notna(r[col]):
                    total_v += r[col] * w
                    total_w += w
            out[col] = (total_v / total_w) if total_w > 0 else np.nan

        for col in sum_cols:
            out[col] = group[col].sum()

        out["longest_carry"] = group["longest_carry"].max()
        rows.append(out)

    blended = pd.DataFrame(rows)
    blended["p_five_unconditional_blended"] = (
        blended["p_carry_1plus_shrunk"] * blended["p_five_given_carry_shrunk"]
    )
    return blended


def compute_departed_teammate_boost(rb_metrics: pd.DataFrame, players: pd.DataFrame, max_boost: float) -> pd.DataFrame:
    """
    Historical carry rates are earned *while splitting touches with whoever
    else was on the team at the time*. When a back's historical backfield
    mate leaves the team (trade, free agency, release) and the back himself
    stays, that back's true current carry probability is higher than his
    raw historical rate implies -- the model just has no way to know that
    from the play-by-play alone, since the pbp only records what happened,
    not who left.

    This estimates that effect directly from real roster data (the
    nflverse players roster's `latest_team` field) rather than guessing a
    number: for a player who is STILL on the same team he accumulated his
    historical first-drive carries with, we sum the historical carries of
    any teammates at that same team who have SINCE left (their
    `latest_team` no longer matches), express that as a fraction of the
    team's total historical RB carry volume, and cap it
    (`max_boost`) so a single departure can't dominate the model.

    A player whose OWN latest_team differs from his historical team is
    excluded entirely -- he changed teams too, so "his teammates left" is
    not a coherent framing; his historical rate transferring imperfectly to
    a new team is a separate, documented limitation (see README).

    This must NEVER be used in the backtest: `players.parquet` is a live
    snapshot taken at run time, so applying "who has left since" to a past
    season would leak future roster information into a historical
    prediction. Only the live rankings pipeline should call this.
    """
    if "player_id" not in players.columns or "latest_team" not in players.columns:
        return pd.DataFrame(columns=["team", "player_id", "departed_teammate_boost"])

    current_team_lookup = players.set_index("player_id")["latest_team"]

    rows = []
    for hist_team, group in rb_metrics.groupby("team"):
        carries = group.set_index("player_id")["total_carries"].fillna(0)
        team_total = carries.sum()
        if team_total <= 0:
            continue
        current_team_of = {pid: current_team_lookup.get(pid) for pid in carries.index}

        for pid in carries.index:
            if current_team_of.get(pid) != hist_team:
                continue  # this player himself is no longer on this team
            departed_carries = sum(
                c for other_pid, c in carries.items()
                if other_pid != pid and current_team_of.get(other_pid) != hist_team
            )
            boost = min(departed_carries / team_total, max_boost)
            if boost > 0:
                rows.append({"team": hist_team, "player_id": pid, "departed_teammate_boost": boost})

    return pd.DataFrame(rows, columns=["team", "player_id", "departed_teammate_boost"])


def build_rb_metrics(first_drive_plays: pd.DataFrame, players: pd.DataFrame) -> dict:
    """
    Full pipeline entry point for Sections 7-12: returns a dict with the
    intermediate per-player-game table, the per-player-season table, and
    the final blended (cross-season-weighted, shrunk) RB metrics table.
    """
    fd_pos = attach_rb_positions(first_drive_plays, players)
    per_player_game = compute_team_game_rb_carries(fd_pos)

    first_drive_team_games = first_drive_plays[["season", "game_id", "posteam"]].drop_duplicates()

    player_season = compute_player_season_metrics(per_player_game, first_drive_team_games)

    league_priors_by_season = {
        int(s): compute_league_priors(per_player_game, s) for s in player_season["season"].unique()
    }

    player_season_shrunk = apply_shrinkage(player_season, league_priors_by_season)
    blended = blend_seasons(player_season_shrunk, config.SEASON_WEIGHTS)

    return {
        "per_player_game": per_player_game,
        "player_season": player_season_shrunk,
        "rb_metrics": blended,
    }
