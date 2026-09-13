#!/usr/bin/env python3
"""
CLI entry point for the NFL first-drive RB rushing model (Section 36).

Commands:
    python main.py download-data
    python main.py build-first-drives
    python main.py calculate-rb-metrics
    python main.py derive-rb-roles
    python main.py rankings [--filter all|best_bets|high_probability|best_value|passes]
    python main.py backtest
    python main.py full-pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config
from src.data_loader import load_nfl_pbp, load_player_positions, DataLoadError
from src.drive_analysis import build_first_drive_plays, compute_team_first_drive_metrics, blend_team_seasons
from src.rb_analysis import build_rb_metrics
from src.matchup_analysis import build_defense_and_qb_metrics
from src.model import load_current_rb_roles, build_model_table
from src.roles import derive_rb_roles
from src.odds import load_sportsbook_odds, attach_odds_and_edge
from src.rankings import rank_candidates, filter_rankings
from src.backtest import run_backtest, compute_backtest_metrics, simulate_betting_strategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

QB_COMPETITION_CACHE = config.PROCESSED_DATA_DIR / "qb_competition_by_season.csv"


def _get_first_drive_plays(seasons, force_download=False) -> pd.DataFrame:
    if config.FIRST_DRIVE_PLAYS_OUT.exists() and not force_download:
        logger.info("loading cached first-drive plays from %s", config.FIRST_DRIVE_PLAYS_OUT)
        return pd.read_parquet(config.FIRST_DRIVE_PLAYS_OUT)
    pbp = load_nfl_pbp(seasons, force_download=force_download)
    fd = build_first_drive_plays(pbp)
    fd.to_parquet(config.FIRST_DRIVE_PLAYS_OUT, index=False)
    return fd


def cmd_download_data(args):
    try:
        load_nfl_pbp(args.seasons, force_download=args.force)
        load_player_positions(force_download=args.force)
    except DataLoadError as exc:
        logger.error("download failed: %s", exc)
        sys.exit(1)
    print(f"Downloaded and cached play-by-play for seasons: {args.seasons}")


def cmd_build_first_drives(args):
    fd = _get_first_drive_plays(args.seasons, force_download=args.force)
    print(f"Built {len(fd)} first-drive offensive plays -> {config.FIRST_DRIVE_PLAYS_OUT}")


def cmd_calculate_rb_metrics(args):
    fd = _get_first_drive_plays(args.seasons)
    players = load_player_positions()

    team_metrics = compute_team_first_drive_metrics(fd)
    team_metrics.to_csv(config.TEAM_METRICS_OUT, index=False)
    print(f"Wrote team first-drive metrics (per-season detail) -> {config.TEAM_METRICS_OUT} ({len(team_metrics)} rows)")

    team_metrics_blended = blend_team_seasons(team_metrics, config.SEASON_WEIGHTS)
    team_metrics_blended.to_csv(config.TEAM_METRICS_BLENDED_OUT, index=False)
    print(f"Wrote cross-season-weighted team metrics -> {config.TEAM_METRICS_BLENDED_OUT} ({len(team_metrics_blended)} rows)")

    rb_out = build_rb_metrics(fd, players)
    rb_out["rb_metrics"].to_csv(config.RB_METRICS_OUT, index=False)
    print(f"Wrote RB first-drive metrics -> {config.RB_METRICS_OUT} ({len(rb_out['rb_metrics'])} rows)")

    dq = build_defense_and_qb_metrics(fd)
    dq["defense_metrics"].to_csv(config.DEFENSE_METRICS_OUT, index=False)
    dq["qb_competition_by_season"].to_csv(QB_COMPETITION_CACHE, index=False)
    print(f"Wrote defense first-drive metrics -> {config.DEFENSE_METRICS_OUT} ({len(dq['defense_metrics'])} rows)")


def cmd_derive_rb_roles(args):
    if not config.RB_METRICS_OUT.exists():
        logger.error("missing %s -- run `python main.py calculate-rb-metrics` first", config.RB_METRICS_OUT)
        sys.exit(1)
    rb_metrics = pd.read_csv(config.RB_METRICS_OUT)
    players = load_player_positions()

    old_roles = load_current_rb_roles() if config.CURRENT_RB_ROLES_FILE.exists() else pd.DataFrame()

    derived = derive_rb_roles(rb_metrics, players)
    if derived.empty:
        logger.error("could not derive any roles -- check that calculate-rb-metrics has been run "
                      "and the players roster download succeeded")
        sys.exit(1)

    if not old_roles.empty:
        backup_path = config.CURRENT_RB_ROLES_FILE.with_suffix(".before_derivation.csv")
        old_roles.to_csv(backup_path, index=False)
        print(f"Backed up previous roles file -> {backup_path}")

        old_by_team = old_roles.groupby("team")["player_name"].apply(list).to_dict()
        new_by_team = derived.groupby("team")["player_name"].apply(list).to_dict()
        changed_teams = [t for t in sorted(set(old_by_team) | set(new_by_team))
                          if set(old_by_team.get(t, [])) != set(new_by_team.get(t, []))]
        if changed_teams:
            print(f"\nRosters changed for {len(changed_teams)} team(s):")
            for t in changed_teams:
                print(f"  {t}: {old_by_team.get(t, [])} -> {new_by_team.get(t, [])}")

    derived.to_csv(config.CURRENT_RB_ROLES_FILE, index=False)
    print(f"\nWrote data-derived RB roles -> {config.CURRENT_RB_ROLES_FILE} ({len(derived)} rows)")
    print("Note: role (RB1/RB2/committee) is derived from real carry-share data. "
          "Injury status (questionable/injured/out) is NOT derived -- no injury feed is "
          "wired in, so every row defaults to status=active. Edit the CSV by hand for injuries.")


def _load_metrics_csvs():
    missing = [p for p in [config.TEAM_METRICS_BLENDED_OUT, config.RB_METRICS_OUT, config.DEFENSE_METRICS_OUT, QB_COMPETITION_CACHE] if not p.exists()]
    if missing:
        logger.error(
            "missing metrics file(s): %s -- run `python main.py calculate-rb-metrics` first",
            [str(p) for p in missing],
        )
        sys.exit(1)
    return (
        pd.read_csv(config.TEAM_METRICS_BLENDED_OUT),
        pd.read_csv(config.RB_METRICS_OUT),
        pd.read_csv(config.DEFENSE_METRICS_OUT),
        pd.read_csv(QB_COMPETITION_CACHE),
    )


def cmd_rankings(args):
    team_metrics, rb_metrics, defense_metrics, qb_competition = _load_metrics_csvs()
    current_roles = load_current_rb_roles()

    if not config.WEEK1_MATCHUPS_FILE.exists():
        logger.error(
            "no matchups file at %s -- provide a CSV with columns [team, opponent] "
            "(optionally game_id) for the slate you want rankings for.",
            config.WEEK1_MATCHUPS_FILE,
        )
        sys.exit(1)
    matchups = pd.read_csv(config.WEEK1_MATCHUPS_FILE)

    odds = load_sportsbook_odds()
    players = load_player_positions()

    model_df = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition, current_roles, matchups, players=players)
    with_odds = attach_odds_and_edge(model_df, odds)
    ranked = rank_candidates(with_odds)
    ranked.to_csv(config.RANKINGS_OUT, index=False)

    filtered = filter_rankings(ranked, args.filter)
    print(f"Wrote {len(ranked)} ranked candidates -> {config.RANKINGS_OUT}")
    print(f"\n=== Filter: {args.filter} ({len(filtered)} rows) ===")
    display_cols = ["rank", "player_name", "team", "opponent", "model_probability",
                     "games_with_carry", "games_with_5plus",
                     "sportsbook_odds", "sportsbook_implied_probability", "edge",
                     "fair_odds", "confidence", "bet_category"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        print(filtered[display_cols].to_string(index=False))


def cmd_backtest(args):
    fd = _get_first_drive_plays(args.seasons)
    players = load_player_positions()

    print("Running walk-forward backtest (no future information is used for any prediction)...")
    records = run_backtest(fd, players, min_train_games=args.min_train_games)
    records.to_csv(config.BACKTEST_OUT, index=False)
    print(f"Wrote {len(records)} backtest predictions -> {config.BACKTEST_OUT}")

    metrics = compute_backtest_metrics(records)
    if "error" in metrics:
        print(metrics["error"])
        return

    print(f"\nn_predictions: {metrics['n_predictions']}")
    print(f"accuracy:      {metrics['accuracy']:.3f}")
    print(f"brier_score:   {metrics['brier_score']:.4f}")
    print(f"log_loss:      {metrics['log_loss']:.4f}")
    print("\ncalibration (predicted vs actual):")
    print(metrics["calibration"].to_string(index=False))

    strategy = simulate_betting_strategy(records)
    print("\nbetting-strategy backtest:")
    if strategy.get("status") == "unavailable":
        print(f"  UNAVAILABLE: {strategy['reason']}")
    else:
        print(strategy)


def cmd_full_pipeline(args):
    print("=== STEP 1-2: download + clean data ===")
    cmd_download_data(args)

    print("\n=== STEP 3: build first offensive drives ===")
    cmd_build_first_drives(args)

    print("\n=== STEP 4-6: team / RB / defense first-drive metrics ===")
    cmd_calculate_rb_metrics(args)

    print("\n=== STEP 7: derive current RB roles from real roster + carry-share data ===")
    cmd_derive_rb_roles(args)

    print("\n=== STEP 8-12: projections, odds, edge, rankings ===")
    args.filter = getattr(args, "filter", "all")
    cmd_rankings(args)

    print(f"\nCURRENT ROLE SOURCE: {config.CURRENT_ROLE_SOURCE}")
    print("Full pipeline complete.")


def build_parser():
    p = argparse.ArgumentParser(description="NFL first-drive RB rushing prop model")
    p.add_argument("--seasons", type=int, nargs="+", default=config.DEFAULT_SEASONS,
                    help=f"seasons to load (default: {config.DEFAULT_SEASONS})")
    p.add_argument("--force", action="store_true", help="force re-download of cached data")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download-data").set_defaults(func=cmd_download_data)
    sub.add_parser("build-first-drives").set_defaults(func=cmd_build_first_drives)
    sub.add_parser("calculate-rb-metrics").set_defaults(func=cmd_calculate_rb_metrics)
    sub.add_parser("derive-rb-roles").set_defaults(func=cmd_derive_rb_roles)

    rankings_p = sub.add_parser("rankings")
    rankings_p.add_argument("--filter", choices=["all", "best_bets", "high_probability", "best_value", "passes"],
                             default="all")
    rankings_p.set_defaults(func=cmd_rankings)

    backtest_p = sub.add_parser("backtest")
    backtest_p.add_argument("--min-train-games", type=int, default=20, dest="min_train_games")
    backtest_p.set_defaults(func=cmd_backtest)

    full_p = sub.add_parser("full-pipeline")
    full_p.add_argument("--filter", choices=["all", "best_bets", "high_probability", "best_value", "passes"],
                         default="all")
    full_p.set_defaults(func=cmd_full_pipeline)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
