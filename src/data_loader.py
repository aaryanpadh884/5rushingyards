"""
Download and load nflverse/nflfastR play-by-play Parquet files.

Data source (Section 1/3):
    https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet

Files are cached in data/raw/ and never re-downloaded once present unless
`force=True` is passed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# Columns we rely on somewhere in the pipeline (Section 4). We verify these
# exist (or degrade gracefully) right after loading each season file.
EXPECTED_COLUMNS = [
    "game_id", "season", "week", "game_date", "posteam", "defteam",
    "drive", "play_type", "rush_attempt", "pass_attempt", "sack",
    "rusher_player_name", "rusher_player_id", "rushing_yards",
    "passer_player_name", "qb_scramble", "touchdown", "down", "ydstogo",
    "yardline_100", "qtr", "quarter_seconds_remaining",
    "half_seconds_remaining", "game_seconds_remaining", "score_differential",
    "home_team", "away_team", "posteam_type", "season_type", "desc",
    "game_type", "two_point_attempt", "field_goal_attempt", "extra_point_attempt",
    "penalty", "aborted_play", "play_type_nfl", "wp",
]

# The spec (and older nflfastR docs) call the regular/postseason flag
# `game_type`. Current nflverse-data release files (verified against the
# 2024/2025 play_by_play parquet files) use `season_type` instead, with
# values "REG"/"POST". We check both names and prefer whichever is present
# so the pipeline works against the real schema, and we log which one we
# used so a future schema change is easy to diagnose (Section 4).
REG_SEASON_COLUMN_CANDIDATES = ["game_type", "season_type"]


class DataLoadError(RuntimeError):
    """Raised when a season file cannot be downloaded or read."""


def _raw_path(season: int) -> Path:
    return config.RAW_DATA_DIR / f"play_by_play_{season}.parquet"


def download_season(season: int, force: bool = False, timeout: int = 180) -> Path:
    """Download one season's play-by-play parquet file if not already cached."""
    dest = _raw_path(season)
    if dest.exists() and not force:
        logger.info("season %s already cached at %s, skipping download", season, dest)
        return dest

    url = config.NFLVERSE_PBP_URL_TEMPLATE.format(season=season)
    logger.info("downloading %s -> %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp_path = dest.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.rename(dest)
    except requests.RequestException as exc:
        raise DataLoadError(
            f"Failed to download play-by-play data for season {season} from {url}: {exc}"
        ) from exc
    return dest


def _verify_columns(df: pd.DataFrame, season: int) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "Season %s parquet is missing expected columns: %s. "
            "Downstream metrics relying on these will be skipped or set to NaN. "
            "This usually means nflverse renamed a field upstream — check "
            "https://nflreadr.nflverse.com/articles/dictionary_pbp.html and "
            "update EXPECTED_COLUMNS / the affected src module.",
            season, missing,
        )


def load_nfl_pbp(seasons: Iterable[int] = None, force_download: bool = False) -> pd.DataFrame:
    """
    Load combined regular-season play-by-play for the given seasons.

    Steps (Section 3):
      1. Download the parquet file if missing (cached under data/raw/).
      2. Load with pandas/pyarrow.
      3. Filter to game_type == "REG".
      4. Concatenate all seasons into one DataFrame.

    Raises DataLoadError if a season cannot be downloaded or parsed; callers
    (CLI/pipeline) should catch this and report which season/field failed
    rather than silently proceeding with partial data.
    """
    seasons = list(seasons) if seasons is not None else list(config.DEFAULT_SEASONS)
    if not seasons:
        raise ValueError("seasons list is empty")

    frames: List[pd.DataFrame] = []
    for season in seasons:
        path = download_season(season, force=force_download)
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pyarrow/parquet errors of various types
            raise DataLoadError(f"Failed to read parquet for season {season} at {path}: {exc}") from exc

        _verify_columns(df, season)

        reg_col = next((c for c in REG_SEASON_COLUMN_CANDIDATES if c in df.columns), None)
        if reg_col is not None:
            before = len(df)
            df = df[df[reg_col] == "REG"].copy()
            logger.info(
                "season %s: kept %d/%d REG-season plays (filtered on '%s' column)",
                season, len(df), before, reg_col,
            )
        else:
            raise DataLoadError(
                f"Season {season} parquet has neither a 'game_type' nor a "
                f"'season_type' column, so regular-season plays cannot be "
                f"isolated. The nflverse schema may have changed — inspect "
                f"the file's columns and update REG_SEASON_COLUMN_CANDIDATES "
                f"in src/data_loader.py."
            )

        if "season" not in df.columns:
            df["season"] = season

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("loaded %d total REG-season plays across seasons %s", len(combined), seasons)
    return combined


# ---------------------------------------------------------------------------
# Player position lookup (needed to tell RB rushing attempts apart from QB
# scrambles/designed QB runs and WR jet sweeps -- nflfastR's play-by-play
# file itself carries no position field for the rusher).
# ---------------------------------------------------------------------------
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"


def load_player_positions(force_download: bool = False, timeout: int = 120) -> pd.DataFrame:
    """
    Download (once, cached) the nflverse players roster file and return a
    DataFrame of [player_id, player_name, position, position_group,
    latest_team] keyed by `gsis_id`, which is the same ID space as
    `rusher_player_id` in the play-by-play data.

    `latest_team` reflects the roster file's live snapshot at download
    time -- it is what lets the live rankings pipeline know a player has
    changed teams since the historical play-by-play was recorded. It must
    NOT be used inside the backtest (see
    rb_analysis.compute_departed_teammate_boost's docstring for why).
    """
    dest = config.RAW_DATA_DIR / "players.parquet"
    if not dest.exists() or force_download:
        try:
            resp = requests.get(PLAYERS_URL, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except requests.RequestException as exc:
            raise DataLoadError(f"Failed to download players roster file: {exc}") from exc

    players = pd.read_parquet(dest)
    if "gsis_id" not in players.columns or "position_group" not in players.columns:
        raise DataLoadError(
            "players.parquet is missing 'gsis_id' or 'position_group' columns; "
            "the nflverse players schema may have changed."
        )
    cols = ["player_id", "player_name", "position", "position_group"]
    if "latest_team" in players.columns:
        cols.append("latest_team")
    return players.rename(columns={"gsis_id": "player_id", "display_name": "player_name"})[cols]
