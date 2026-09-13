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

The market is: *did the RB's total rushing yards on the first drive add up
to 5+* -- a standard accumulated-total prop, same as any other rushing-yards
Over/Under, evaluated over just the first drive.

```
[2, 3, 4]     -> True   (sums to 9)
[1, 1]        -> False  (sums to 2)
[2, 6]        -> True   (sums to 8)
```

Implemented as `src.utils.got_five_plus()`, which sums the individual
carries.

**Revision note:** an earlier version of this project used a different rule
-- true only if a single individual carry reached 5+ yards, false no matter
how the total added up. That was changed after confirming what a real
"X+ rushing yards on the first drive" sportsbook prop actually measures.
The switch is a meaningful behavior change, not a tweak: hit rates jump
substantially across the board, since a back who gets 2-3 carries on a
drive clears "5 total yards" far more often than he breaks one specific
5+ yard run. Every historical rate, and the whole backtest, was recomputed
under the new definition -- nothing here is a mix of the two rules.

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

## Opponent defense adjustment

`matchup_analysis.py` computes five separate first-drive run-defense
signals per team: 5+ allowed rate, yards per rush allowed, 10+ allowed
rate, success rate allowed, and average EPA allowed. All five are combined
into one **composite** adjustment (`compute_composite_defense_factor`)
rather than using the 5+ allowed rate on its own:

1. Each metric is standardized against the league (z-score: how many
   standard deviations above/below average that defense is on that one
   metric). All five point the same direction -- higher always means a
   softer defense -- so no sign flips are needed.
2. The five z-scores are combined with fixed weights
   (`config.DEFENSE_FACTOR_WEIGHTS`, summing to 1.0 -- 40% on the 5+
   allowed rate since it's the closest direct analog of the target,
   25% yards per rush, 15% explosive-run rate, 10% each on success
   rate/EPA).
3. The combined score is scaled (`config.DEFENSE_COMPOSITE_SCALE`) and
   clipped to `[DEFENSE_ADJUSTMENT_MIN_FACTOR, DEFENSE_ADJUSTMENT_MAX_FACTOR]`
   (0.70x-1.30x by default), then multiplied into the player's own
   conditional `P(5+ | carry)`.

The cap matters more here than in a single-metric version: five real but
noisy signals combined could otherwise compound into an extreme swing for
a defense that happens to be bad on all five in a small sample. The clip
keeps a single opponent read from ever dominating the player's own
established rate. All five raw metrics are still visible in
`defense_first_drive_metrics.csv` even though only the composite factor
feeds the model.

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
defense. Without an entry, the player falls back to his historical team, is
checked against `last_season` (see below), and is flagged
`"no current-role entry; using historical rate only"`.

### Generating it from data instead of guessing

```bash
python main.py derive-rb-roles
```

This **replaces hand-picked RB1/RB2/committee judgment calls with a rule
derived from real data** (`src/roles.py`):

1. Take each team's actual current roster of RBs (nflverse players feed,
   `position_group == "RB"` and `last_season == config.CURRENT_SEASON`).
2. Rank them by career first-drive carry volume (`total_carries`, wherever
   he earned it -- a traded veteran's volume at his old team is an
   imperfect but real signal of how much a team trusts him).
3. One back with >=60% of the top-two's combined volume -> clear RB1/RB2.
   Both with >=30% each -> `committee`. Thresholds are
   `config.ROLE_INFERENCE_RB1_SHARE_THRESHOLD` /
   `..._COMMITTEE_SHARE_THRESHOLD`.
4. A player with **zero** historical carries anywhere (a true rookie who
   entered the league this year) gets **no role at all** -- guessing one
   would be exactly the kind of invented number this project refuses to
   produce. The command logs every such player so you can add them by hand
   if you have reason to believe otherwise.

It automatically backs up whatever was in `current_rb_roles.csv` to
`current_rb_roles.before_derivation.csv` first, and prints which teams'
top-two changed. It's part of `full-pipeline` by default.

**What it does NOT do:** there's no injury feed wired in, so every derived
row defaults to `status=active`. Questionable/injured/out designations
still require manually editing the CSV after generation -- and if you
believe the data is wrong for a specific player (a rookie you expect to
start Week 1, a role change the roster feed hasn't caught up to yet), edit
the row by hand; a manual edit is never overwritten except by re-running
`derive-rb-roles` again.

This must never run inside the backtest -- `last_season`/`latest_team` are
live snapshots, so deriving "today's" roles and applying them to a past
season would leak future information into a historical prediction
(confirmed: backtest metrics are identical before/after this feature).

The seed file originally shipped with this repo was a hand-picked,
best-effort snapshot cross-checked against the live nflverse roster feed.
Running `derive-rb-roles` replaces it with a repeatable, data-driven
version -- e.g. it correctly demoted several of the original manual picks
once real carry-share data disagreed (Jacksonville's actual leading backs
turned out to be Chris Rodriguez Jr./Ameer Abdullah, not the rookies I'd
guessed; New York Giants' Cam Skattebo ranked below Najee Harris in real
volume; the Chargers' Omarion Hampton and Kimani Vidal split closely enough
to be a real `committee`, not a clean RB1/RB2).

## Departed-teammate carry boost

A back's historical carry rate was earned *while splitting touches with
whoever else was on the roster at the time*. If that backfield competition
has since left the team (trade, free agency, release) and the back himself
stayed, his real current carry probability is higher than his raw
historical rate implies -- the play-by-play alone has no way to know that.

The model corrects for this using the live nflverse players roster's
`latest_team` field: for any RB who is still on the same team he
accumulated his historical first-drive carries with, it sums the
historical carries of any teammates at that team who have since left,
expresses that as a fraction of the team's historical RB carry volume, and
applies it as a capped boost (`config.DEPARTED_TEAMMATE_MAX_BOOST`, default
30%) to his carry probability. It shows up transparently in the `role_note`
column (e.g. `"departed-teammate boost=+30%"`).

Example: Jahmyr Gibbs shared Detroit's backfield with David Montgomery in
2024-2025 (Montgomery took ~42% of DET's RB carries). Montgomery has since
signed with Houston; Gibbs remains DET's RB1. Without this adjustment,
Gibbs' carry probability is still implicitly priced as if that competition
exists. With it, his P(carry) is boosted from the raw ~60% up to ~86% (the
30% cap binds here), which moves his overall model probability from ~46%
to ~60%.

This is **live-rankings only** -- it is never applied inside the backtest.
`latest_team` is a snapshot taken at run time, so using "who has left
since" against a past season would leak future roster information into a
historical prediction (confirmed: backtest metrics are identical with and
without this feature wired in, since `run_backtest` never passes `players`
into `build_model_table`).

It also only fires when the RB himself didn't change teams -- a player who
was traded inherits a *new* team's committee dynamics that his own
historical rate says nothing about; that mismatch is a separate, documented
limitation, not something this boost tries to fix.

## Filtering out players no longer in the league

A related gap: a player with real 2024-2025 usage but no entry in
`data/current_rb_roles.csv` used to default to showing up forever under his
old historical team, with no check on whether he's still active. Example:
Joe Mixon has real Houston history in the loaded seasons, but the live
nflverse roster feed shows his `last_season` as 2025 (reserve/suspended
status) -- he is not on Houston's active 2026 roster, yet the model kept
ranking him as a live HOU candidate purely because nothing said otherwise.

`build_model_table` now drops any historical RB whose `last_season` (from
the live players roster) is behind `config.CURRENT_SEASON` **and** who has
no explicit `current_rb_roles.csv` entry. A manual roles-file entry always
overrides this -- if you know the roster feed is behind reality for a
specific player, add him to the CSV and he's back in regardless of what
`last_season` says. Like the departed-teammate boost, this only runs when
`players` is passed in, so it never touches the backtest.

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
python main.py derive-rb-roles                # regenerate current_rb_roles.csv from real carry-share data
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
