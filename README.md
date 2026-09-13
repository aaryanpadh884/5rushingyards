# NFL First-Drive RB Rushing Model

A data-driven model for one specific sportsbook prop: **"Starting RB to
record 5+ rushing yards on his team's first offensive drive."**

This is not a general fantasy-football model. Every metric in this
repository is scoped to a team's opening possession of the game, and the
target is defined precisely as the market defines it (see "The 5+ Rule"
below).

## What the model does

1. Downloads nflverse/nflfastR play-by-play data (2024-2025 by default).
2. Identifies each team's real first offensive drive in every game (not
   `drive == 1` -- see "First-Drive Definition").
3. Computes team-level, RB-level, and defense-level first-drive metrics.
4. Blends 2025 and 2024 data (70/30) and applies empirical-Bayes shrinkage
   so small samples don't produce overconfident probabilities.
5. Combines historical RB performance with the CURRENT depth chart
   (`data/current_rb_roles.csv`), opponent run defense, and QB rushing
   competition into one baseline probability:

   ```
   P(5+ on first drive) = P(RB gets >=1 first-drive carry)
                          x P(RB gets 5+ | RB gets >=1 first-drive carry)
   ```

6. Compares that probability to sportsbook odds (when supplied) to compute
   edge, fair odds, and expected value.
7. Backtests the whole approach walk-forward on real historical games, with
   no future information leaking into any prediction.

## Data source

Play-by-play: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`

Player positions (to tell RB carries apart from QB scrambles / WR jet
sweeps -- the pbp file itself has no position column):
`https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet`

Both are downloaded once and cached under `data/raw/`. Files are never
re-downloaded unless `--force` is passed.

**Schema note:** the spec and older nflfastR docs call the regular/postseason
flag `game_type`. The actual 2024/2025 release files use `season_type`
instead (values `"REG"`/`"POST"`). `src/data_loader.py` checks both column
names and logs which one it used, so a future nflverse rename is easy to
diagnose rather than silently breaking.

## First-drive definition

A team's first offensive drive is **not** `drive == 1`. It is computed
independently for every `(game_id, team)` as:

```
first_drive = min(drive) over all offensive plays where posteam == team
```

That value is merged back onto the play-by-play, and every play where
`drive == first_drive` and `posteam == team` is kept. See the module
docstring in `src/drive_analysis.py` for the full, line-by-line exclusion
list (special teams plays, two-point attempts, QB kneels, no-play
penalties, aborted snaps). Sacks are counted as **passing** plays for
run/pass-mix purposes (per the spec) and excluded from RB rushing
aggregates (a sack has no rusher).

## The 5+ rule

The market is: *did the RB have at least one individual rushing attempt of
5+ yards on the first drive* -- **not** whether his carries summed to 5+.

```
[2, 3, 4, 4]  -> False  (no single carry reaches 5)
[2, 6]        -> True   (one carry of 6)
```

Implemented as `src.utils.got_five_plus()`, which takes the max of the
individual carries, not the sum.

## Historical weighting & shrinkage

- **Cross-season weighting** (`config.SEASON_WEIGHTS`, default 2025=0.70 /
  2024=0.30): every player/team/defense rate is blended across seasons,
  renormalized to whichever seasons that entity actually has data for (a
  rookie with only 2025 data gets 100% weight on 2025, not 70%).
- **Empirical-Bayes shrinkage** (`src.utils.shrink_rate`): each rate is
  pulled toward a league-average prior, weighted by sample size:

  ```
  adjusted_rate = (n * observed_rate + prior_strength * prior_rate)
                  / (n + prior_strength)
  ```

  A rookie with 1 game and a single 5+ run does **not** get scored as a
  100% lock -- his rate is dominated by the prior until he accumulates a
  real sample (`config.PRIOR_STRENGTH_*` controls how fast that happens).

## Current RB roles

`data/current_rb_roles.csv` is the **authoritative** source for who is
getting the ball right now (columns: `team,player_name,player_id,role,status`;
roles: `RB1`/`RB2`/`committee`; status: `active`/`questionable`/`injured`/`out`).
Historical play-by-play tells you how a player has been *used*; this file
tells the model who is *currently* the starter, and a player's role
multiplier (`config.ROLE_CARRY_MULTIPLIER`) can override a stale historical
read entirely (e.g. `status=out` zeroes the probability).

Critically, this file also resolves **which team a player is on now** --
if a player was traded or signed elsewhere in the offseason, his historical
play-by-play team is stale, and the model uses this file's team to look up
his real Week 1 opponent, current team's run rate, and that opponent's
defense. Without an entry, the player falls back to his historical team and
is flagged `"no current-role entry; using historical rate only"`.

**To update it:** edit the CSV directly, one row per player. Remove a
player entirely if they're not part of your relevant slate, or set
`status=out` to keep them in the table but zero their probability.

The seed file included here was cross-checked against the live
nflverse players roster feed (`latest_team` field) to catch known 2026
offseason moves (e.g. Isiah Pacheco to DET, David Montgomery to HOU,
Kenneth Walker III to KC, Travis Etienne to NO, Rachaad White to WAS, Najee
Harris to NYG, Rico Dowdle to PIT). It reflects a best-effort snapshot as of
build time, not a live feed -- **verify and update it weekly** before
relying on it; this sandbox's outbound network access does not reach
fantasy/depth-chart sites (ESPN, CBS Sports, RotoWire, etc.) to auto-verify
beyond the nflverse roster data.

## Adding sportsbook odds

`data/sportsbook_odds.csv` (columns: `player_name,team,market,line,american_odds,sportsbook,timestamp`).
Add one row per line you want evaluated. Without a matching row, a
player's odds/edge/EV columns are explicitly `NaN`/"unavailable" -- never
invented.

American-odds -> implied probability:

```
negative odds:  abs(odds) / (abs(odds) + 100)
positive odds:  100 / (odds + 100)
```

Model probability -> fair American odds:

```
p > 0.5:  -100 * p / (1 - p)
p < 0.5:   100 * (1 - p) / p
```

`edge = model_probability - sportsbook_implied_probability`. Rankings sort
by (1) positive EV, (2) model probability, (3) confidence -- never by raw
probability alone.

## Game environment (optional)

`data/week1_odds.csv` (spread/team_total per team) and
`data/week1_matchups.csv` (team/opponent pairs the model needs to know who
plays whom) are pre-populated from the **real, official 2026 Week 1 NFL
schedule** (nflverse's `nfldata` schedules feed), so spreads/totals here are
live market data, not placeholders. Regenerate for a future week with your
own matchup/odds data in the same two-column (`team,opponent`) /
four-column (`game_id,team,spread,team_total`) format.

## Known limitations

- **RB position filtering** relies on the nflverse players roster's
  `position_group == "RB"`. A play-by-play file with a player ID not in
  that roster snapshot will be silently excluded from RB metrics.
- **Scoring-drive rate** uses the per-play `touchdown` flag scoped to the
  drive; field-goal-scoring first drives are not currently counted as
  "scoring" (documented as a gap, not silently wrong -- extend
  `compute_team_first_drive_metrics` if you need it).
- **Backtesting has no historical sportsbook odds**, so ROI/win-rate
  simulation (Section 30) is architecturally implemented
  (`src.backtest.simulate_betting_strategy`) but reports "unavailable"
  until you supply a CSV of real historical lines
  (`season,week,player_id,american_odds`). Accuracy, Brier score, log loss,
  and calibration are all computed for real, since they don't require odds.
- **Backtest "presumed starter"** cannot use `current_rb_roles.csv` (that
  file describes right now, not the past). Instead it uses the RB with the
  most first-drive carries in the training window up to that point in time
  as a proxy for "who was the guy." This is a legitimate proxy but not
  identical to what a real depth chart would have said in that moment.
- **Offensive-line metrics** (Section 18) are architecturally supported
  (the model has room for an OL adjustment) but not populated -- no
  reliable public OL-grade dataset is wired in. Do not assume any OL signal
  is currently affecting the probabilities.
- The model is a transparent baseline (`P(carry) x P(5+|carry)`) plus a
  small number of documented multiplicative adjustments -- not a
  fitted ML model. This is deliberate (see Section 44 of the original
  spec): with a few hundred team-games of first-drive data, a complex model
  would overfit. Extend cautiously, and always compare any new model
  against this baseline out-of-sample before trusting it.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py download-data                 # cache 2024/2025 pbp + player roster
python main.py build-first-drives             # identify each team's first offensive drive
python main.py calculate-rb-metrics           # team/RB/defense first-drive metrics -> data/processed/
python main.py rankings --filter best_bets    # build Week 1 rankings (all|best_bets|high_probability|best_value|passes)
python main.py backtest                       # walk-forward backtest + calibration
python main.py full-pipeline                  # everything above, end to end
```

Add `--seasons 2023 2024 2025` to any command to change the historical
window, or `--force` to re-download cached files.

Dashboard:

```bash
streamlit run app.py
```

Tests:

```bash
python -m pytest tests/ -v
```

## Project structure

```
nfl_first_drive_model/ (this repo)
├── data/
│   ├── raw/                          # cached parquet downloads
│   ├── processed/                    # generated CSVs (see below)
│   ├── current_rb_roles.csv          # AUTHORITATIVE current depth chart
│   ├── sportsbook_odds.csv           # manually entered prop lines
│   ├── week1_odds.csv                # game environment (spread/team_total)
│   └── week1_matchups.csv            # team -> opponent for the target slate
├── src/
│   ├── data_loader.py                # download/cache/load pbp + roster
│   ├── drive_analysis.py             # cleaning, first-drive ID, team metrics
│   ├── rb_analysis.py                # RB metrics, carry share, shrinkage
│   ├── matchup_analysis.py           # defense metrics, QB rush competition
│   ├── model.py                      # baseline probability model
│   ├── odds.py                       # odds conversion, edge, EV
│   ├── rankings.py                   # ranking + betting filters
│   ├── backtest.py                   # walk-forward backtest + calibration
│   └── utils.py                      # odds math, shrinkage, 5+ rule
├── tests/                            # pytest unit tests
├── app.py                            # Streamlit dashboard
├── config.py                         # all tunable parameters
├── main.py                           # CLI
└── requirements.txt
```

## Output files (`data/processed/`)

| File | Contents |
|---|---|
| `team_first_drive_metrics.csv` | Per-season team first-drive tendencies |
| `team_first_drive_metrics_blended.csv` | Cross-season-weighted team metrics (used by the model) |
| `rb_first_drive_metrics.csv` | Cross-season-weighted, shrunk RB metrics |
| `defense_first_drive_metrics.csv` | Cross-season-weighted defensive first-drive metrics |
| `qb_competition_by_season.csv` | QB rush-competition adjustment inputs |
| `week1_rankings.csv` | Final ranked model output with odds/edge/confidence |
| `backtest_results.csv` | Per-game walk-forward backtest predictions |

## Modeling principle

Model probabilities are estimates, not ground truth. The point of this
system is to find spots where the model's probability is *meaningfully
higher* than the sportsbook's implied probability, and to be honest about
confidence and sample size along the way -- not to claim certainty. Judge
it on calibration and out-of-sample backtest performance, not on how
confident any single number sounds.
