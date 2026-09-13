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

# The season the live rankings pipeline is projecting for. Used to filter
# out historical RBs who are no longer on an active NFL roster (per the
# nflverse players feed's `last_season` field) before they can show up as
# a live candidate under their old team.
CURRENT_SEASON = 2026

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
BIG_RUN_THRESHOLD_YARDS = 5  # the market: total rushing yards on the first drive >= this many

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
# Composite opponent run-defense adjustment (Section 17). Combines every
# first-drive run-defense signal computed in matchup_analysis.py into one
# bounded multiplier, instead of using only the 5+ allowed rate in
# isolation. Each metric is standardized (z-score vs league average for
# that metric) and combined with these weights (must sum to 1.0); the
# combined z-score is then scaled and clipped into a multiplier applied to
# P(5+ | carry).
# ---------------------------------------------------------------------------
DEFENSE_FACTOR_WEIGHTS = {
    "five_plus_allowed_rate_shrunk": 0.40,  # direct analog of the target metric
    "yards_per_rush_allowed": 0.25,
    "ten_plus_allowed_rate": 0.15,
    "success_rate_allowed": 0.10,
    "avg_epa_allowed": 0.10,
}
DEFENSE_COMPOSITE_SCALE = 0.15   # how much a 1-standard-deviation composite z-score shifts the multiplier
DEFENSE_ADJUSTMENT_MIN_FACTOR = 0.70
DEFENSE_ADJUSTMENT_MAX_FACTOR = 1.30

# ---------------------------------------------------------------------------
# QB rush competition adjustment (Section 19) — small negative multiplier
# applied to expected RB first-drive carries based on QB first-drive rush rate.
# ---------------------------------------------------------------------------
QB_RUSH_COMPETITION_MAX_PENALTY = 0.12  # max fractional reduction to carry prob

# ---------------------------------------------------------------------------
# Departed-teammate boost: when a back's historical backfield competition
# has since left the team (per the live nflverse roster feed) and the back
# himself stayed, his carry probability is bumped up by the departed
# teammates' share of the team's historical RB carry volume, capped here.
# Live rankings only -- never applied inside the backtest (see
# rb_analysis.compute_departed_teammate_boost).
# ---------------------------------------------------------------------------
DEPARTED_TEAMMATE_MAX_BOOST = 0.30

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

# ---------------------------------------------------------------------------
# Data-derived role inference (src/roles.py). Replaces hand-picked RB1/RB2/
# committee judgment calls with a rule based on real current-roster
# membership and real historical carry-volume share between a team's top
# two backs.
# ---------------------------------------------------------------------------
ROLE_INFERENCE_RB1_SHARE_THRESHOLD = 0.60   # top back needs >=60% of the pair's volume to be a clear RB1
ROLE_INFERENCE_COMMITTEE_SHARE_THRESHOLD = 0.30  # both backs need >=30% each to be called a committee

RANDOM_SEED = 42
