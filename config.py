"""
Central configuration for the NFL first-drive RB rushing model.

Every tunable knob referenced by src/*.py lives here so the model can be
re-parameterized without touching logic code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_RB_ROLES_FILE = DATA_DIR / "current_rb_roles.csv"
WEEK1_ODDS_FILE = DATA_DIR / "week1_odds.csv"
WEEK1_MATCHUPS_FILE = DATA_DIR / "week1_matchups.csv"
SPORTSBOOK_ODDS_FILE = DATA_DIR / "sportsbook_odds.csv"

TEAM_METRICS_OUT = PROCESSED_DATA_DIR / "team_first_drive_metrics.csv"
TEAM_METRICS_BLENDED_OUT = PROCESSED_DATA_DIR / "team_first_drive_metrics_blended.csv"
RB_METRICS_OUT = PROCESSED_DATA_DIR / "rb_first_drive_metrics.csv"
DEFENSE_METRICS_OUT = PROCESSED_DATA_DIR / "defense_first_drive_metrics.csv"
RANKINGS_OUT = PROCESSED_DATA_DIR / "week1_rankings.csv"
BACKTEST_OUT = PROCESSED_DATA_DIR / "backtest_results.csv"
FIRST_DRIVE_PLAYS_OUT = PROCESSED_DATA_DIR / "first_drive_plays.parquet"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
NFLVERSE_PBP_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)

DEFAULT_SEASONS = [2024, 2025]

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
BIG_RUN_THRESHOLD_YARDS = 5  # the market: >=1 individual rush of this many yards

# ---------------------------------------------------------------------------
# Historical weighting (Section 11)
# ---------------------------------------------------------------------------
# season -> weight. Most recent season gets more weight. Renormalized
# automatically if a configured season is missing from the loaded data.
SEASON_WEIGHTS = {
    2025: 0.70,
    2024: 0.30,
}
RECENT_SEASON_WEIGHT = 0.70
OLDER_SEASON_WEIGHT = 0.30

# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage (Section 12)
# ---------------------------------------------------------------------------
# adjusted_rate = (n * observed_rate + PRIOR_STRENGTH * prior_rate) / (n + PRIOR_STRENGTH)
# PRIOR_STRENGTH is expressed in "pseudo-games" of the league prior.
PRIOR_STRENGTH_CARRY_RATE = 6.0
PRIOR_STRENGTH_FIVE_PLUS_RATE = 8.0
PRIOR_STRENGTH_TEAM_RUN_RATE = 6.0

# ---------------------------------------------------------------------------
# Sample-size / betting thresholds (Sections 12, 28, 40)
# ---------------------------------------------------------------------------
MIN_SAMPLE_SIZE = 8            # minimum historical first-drive games for a "real" number
MIN_MODEL_PROBABILITY = 0.55
MIN_EDGE = 0.03

# ---------------------------------------------------------------------------
# Confidence scoring thresholds (Section 24)
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH_SAMPLE = 16
CONFIDENCE_MED_SAMPLE = 8

# ---------------------------------------------------------------------------
# QB rush competition adjustment (Section 19) — small negative multiplier
# applied to expected RB first-drive carries based on QB first-drive rush rate.
# ---------------------------------------------------------------------------
QB_RUSH_COMPETITION_MAX_PENALTY = 0.12  # max fractional reduction to carry prob

# ---------------------------------------------------------------------------
# Role-based multipliers applied to the raw historical carry probability to
# reflect the CURRENT depth chart (data/current_rb_roles.csv). These are
# transparent, documented adjustments — not fitted coefficients.
# ---------------------------------------------------------------------------
ROLE_CARRY_MULTIPLIER = {
    "RB1": 1.10,
    "RB2": 0.55,
    "committee": 0.80,
    "questionable": 0.65,
    "injured": 0.0,
    "out": 0.0,
}

CURRENT_ROLE_SOURCE = "data/current_rb_roles.csv"

RANDOM_SEED = 42
