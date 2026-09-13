"""
Walk-forward backtesting (Sections 29-31).

CRITICAL RULE: no future information may leak into a historical prediction.
For every (season, week) in the loaded data, we:

  1. Take a strict TRAINING CUTOFF: every play from a strictly earlier
     (season, week) than the one being predicted.
  2. Rebuild RB / team / defense / QB-competition metrics from ONLY that
     training slice (src/rb_analysis.py, src/drive_analysis.py,
     src/matchup_analysis.py are re-run on the truncated frame).
  3. For every team playing that week, pick a "presumed starter": the RB
     with the most first-drive carries in the training window so far
     (highest `games_with_carry`, tie-broken by `avg_carry_share`). This
     stands in for data/current_rb_roles.csv, which by definition cannot
     exist for a past week -- see the README's "Known Limitations" section.
  4. Predict P(5+) for that presumed starter against that week's real
     opponent, using src/model.py exactly as the live pipeline does (minus
     the current-role-file override, since there is no such file for a
     past week).
  5. Compare to the ACTUAL outcome read off that week's real plays: if the
     presumed starter had zero first-drive carries that game, the outcome
     is automatically "no" (he can't record a 5+ carry without a carry).

The very first week of the very first loaded season is skipped: there is no
prior data to train on, so no prediction is made for it (rather than
falling back to a league-average guess dressed up as a real prediction).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.drive_analysis import compute_team_first_drive_metrics
from src.rb_analysis import attach_rb_positions, compute_team_game_rb_carries, build_rb_metrics
from src.matchup_analysis import build_defense_and_qb_metrics
from src.model import build_model_table, REQUIRED_ROLE_COLUMNS

logger = logging.getLogger(__name__)


def _sorted_season_weeks(fd: pd.DataFrame) -> list[tuple[int, int]]:
    sw = fd[["season", "week"]].drop_duplicates()
    sw = sw.dropna()
    sw["season"] = sw["season"].astype(int)
    sw["week"] = sw["week"].astype(int)
    return sorted(map(tuple, sw.values.tolist()))


def run_backtest(first_drive_plays: pd.DataFrame, players: pd.DataFrame,
                  min_train_games: int = 20) -> pd.DataFrame:
    """
    Returns one row per (season, week, team, presumed_starter) prediction
    with columns: season, week, game_id, team, opponent, player_id,
    player_name, model_probability, actual_got_5plus, games_with_carry
    (the sample size the prediction was built on).
    """
    fd_all = first_drive_plays.copy()
    fd_pos_all = attach_rb_positions(fd_all, players)
    ppg_all = compute_team_game_rb_carries(fd_pos_all)

    season_weeks = _sorted_season_weeks(fd_all)
    empty_roles = pd.DataFrame(columns=REQUIRED_ROLE_COLUMNS)

    records = []
    for season, week in season_weeks:
        train_mask = (fd_all["season"] < season) | ((fd_all["season"] == season) & (fd_all["week"] < week))
        fd_train = fd_all[train_mask]
        if len(fd_train) < min_train_games:
            continue  # not enough history yet -- e.g. week 1 of the first season

        test_mask = (fd_all["season"] == season) & (fd_all["week"] == week)
        fd_test = fd_all[test_mask]
        if fd_test.empty:
            continue

        matchups_test = (
            fd_test[["game_id", "posteam", "defteam"]]
            .drop_duplicates()
            .rename(columns={"posteam": "team", "defteam": "opponent"})
        )

        try:
            rb_out = build_rb_metrics(fd_train, players)
            rb_metrics_train = rb_out["rb_metrics"]
            team_metrics_train = compute_team_first_drive_metrics(fd_train)
            dq = build_defense_and_qb_metrics(fd_train)
        except Exception:
            logger.exception("failed to build training metrics for %s wk%s, skipping", season, week)
            continue

        if rb_metrics_train.empty or team_metrics_train.empty:
            continue

        model_df = build_model_table(
            rb_metrics_train, team_metrics_train, dq["defense_metrics"],
            dq["qb_competition_by_season"], empty_roles, matchups_test,
        )
        if model_df.empty:
            continue

        # presumed starter per team = most first-drive carries in training window
        model_df = model_df.sort_values(
            ["team", "games_with_carry", "avg_carry_share"], ascending=[True, False, False]
        )
        presumed_starters = model_df.groupby("team", as_index=False).first()

        for _, row in presumed_starters.iterrows():
            # match on team + player_id directly against this week's actuals
            match = ppg_all[
                (ppg_all["season"] == season)
                & (ppg_all["posteam"] == row["team"])
                & (ppg_all["rusher_player_id"] == row["player_id"])
                & (ppg_all["game_id"].isin(fd_test.loc[fd_test["posteam"] == row["team"], "game_id"].unique()))
            ]
            actual_got_5plus = bool(match["got_5plus"].iloc[0]) if not match.empty else False

            game_ids_this_team = fd_test.loc[fd_test["posteam"] == row["team"], "game_id"].unique()
            game_id = game_ids_this_team[0] if len(game_ids_this_team) else None

            records.append({
                "season": season,
                "week": week,
                "game_id": game_id,
                "team": row["team"],
                "opponent": row["opponent"],
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "model_probability": row["model_probability"],
                "actual_got_5plus": actual_got_5plus,
                "games_with_carry": row["games_with_carry"],
            })

    return pd.DataFrame.from_records(records)


def compute_backtest_metrics(records: pd.DataFrame) -> dict:
    """Section 29-31: accuracy, Brier score, log loss, calibration table."""
    if records.empty:
        return {"error": "no backtest predictions were generated (insufficient history)"}

    p = records["model_probability"].to_numpy(dtype=float)
    y = records["actual_got_5plus"].to_numpy(dtype=float)

    predicted_class = (p >= 0.5).astype(float)
    accuracy = float((predicted_class == y).mean())
    brier = float(np.mean((p - y) ** 2))

    eps = 1e-9
    p_clipped = np.clip(p, eps, 1 - eps)
    log_loss = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))

    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["<50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
    records = records.copy()
    records["bucket"] = pd.cut(records["model_probability"], bins=bins, labels=labels, right=False)
    calibration = records.groupby("bucket", observed=False).agg(
        n=("actual_got_5plus", "size"),
        predicted_avg=("model_probability", "mean"),
        actual_rate=("actual_got_5plus", "mean"),
    ).reset_index()

    return {
        "n_predictions": int(len(records)),
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": log_loss,
        "calibration": calibration,
    }


def simulate_betting_strategy(records: pd.DataFrame, historical_odds: Optional[pd.DataFrame] = None,
                               min_probability: float = None, min_edge: float = None) -> dict:
    """
    Section 30: bet-simulation on top of the backtest predictions.

    Requires HISTORICAL sportsbook odds for the same games (player, week,
    american_odds) to compute real ROI/win-rate. We do not fabricate
    historical odds -- if `historical_odds` is None or empty, this returns
    an explicit "unavailable" result rather than a fake number (Section 45).
    """
    min_probability = config.MIN_MODEL_PROBABILITY if min_probability is None else min_probability
    min_edge = config.MIN_EDGE if min_edge is None else min_edge

    if historical_odds is None or historical_odds.empty:
        return {
            "status": "unavailable",
            "reason": (
                "No historical sportsbook odds on file for the backtest window. "
                "ROI/win-rate/closing-line backtesting requires actual historical "
                "first_drive_rushing odds per game -- provide a CSV with columns "
                "[season, week, player_id, american_odds] to enable this."
            ),
        }

    from src.utils import american_odds_to_implied_probability, expected_value_per_100

    df = records.merge(historical_odds, on=["season", "week", "player_id"], how="inner")
    df["implied_probability"] = df["american_odds"].apply(american_odds_to_implied_probability)
    df["edge"] = df["model_probability"] - df["implied_probability"]

    bets = df[(df["model_probability"] >= min_probability) & (df["edge"] >= min_edge)].copy()
    if bets.empty:
        return {"status": "ok", "total_bets": 0, "reason": "no historical bets cleared the thresholds"}

    def payout(row):
        odds = row["american_odds"]
        stake = 100.0
        if not row["actual_got_5plus"]:
            return -stake
        return stake * (100.0 / abs(odds)) if odds < 0 else stake * (odds / 100.0)

    bets["profit"] = bets.apply(payout, axis=1)
    cumulative = bets["profit"].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max).min()

    wins = int(bets["actual_got_5plus"].sum())
    total = len(bets)
    return {
        "status": "ok",
        "total_bets": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total,
        "profit": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / (total * 100.0)),
        "max_drawdown": float(drawdown),
        "avg_odds": float(bets["american_odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
    }
