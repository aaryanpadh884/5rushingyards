"""
Data-derived current RB roles (replaces hand-picked role assignments).

Section 9 of the original spec calls for a "manually maintained" depth
chart file. In practice that means someone's opinion decides who is RB1 --
which is exactly the kind of silent, unverifiable judgment call this
project otherwise avoids. This module derives RB1/RB2/committee roles
directly from real data instead:

  1. Take each team's CURRENT roster of RBs (nflverse players roster,
     `position_group == "RB"` and `last_season == config.CURRENT_SEASON`
     -- i.e. actually on a 2026 roster, not a name lingering in old
     play-by-play).
  2. For every RB on that roster, look up his career first-drive carry
     volume (`total_carries` from the blended rb_metrics table) -- this is
     his own historical workload, WHEREVER he earned it. A traded veteran's
     volume at his old team is an imperfect but real signal of how much a
     team trusted him; that imperfection is documented, not hidden.
  3. Rank each team's RBs by that volume and classify the top two:
        - one back takes >= RB1_SHARE_THRESHOLD of the pair's combined
          volume  -> RB1 / RB2
        - both backs take >= COMMITTEE_SHARE_THRESHOLD each            -> committee / committee
        - otherwise                                                    -> RB1 / RB2 (share split unclear but not close enough to call a committee)
  4. A player with ZERO historical carries anywhere (a true rookie, e.g.
     drafted in 2026) generates no data-driven signal and is NOT assigned a
     role -- guessing one would be exactly the kind of invented number this
     project refuses to produce (Section 45). He's flagged separately so
     you know to add him by hand if you believe otherwise.

There is no injury-status feed wired in, so every derived row gets
`status = "active"`. Injury/questionable/out designations still require
manual editing of data/current_rb_roles.csv -- this module only replaces
the RB1/RB2/committee judgment call, not the injury report.

This must never be used inside the backtest, for the same reason
rb_analysis.compute_departed_teammate_boost can't be: `players` is a live
snapshot, so deriving "today's" roles and applying them to a past season
would leak future information into a historical prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

ROLE_COLUMNS = ["team", "player_name", "player_id", "role", "status"]


def derive_rb_roles(rb_metrics: pd.DataFrame, players: pd.DataFrame,
                     current_season: Optional[int] = None,
                     rb1_share_threshold: Optional[float] = None,
                     committee_share_threshold: Optional[float] = None,
                     max_backs_per_team: int = 2) -> pd.DataFrame:
    """
    Returns a DataFrame with the same shape as data/current_rb_roles.csv
    (team, player_name, player_id, role, status), derived from real
    current-roster membership and real historical carry volume.

    Also returns (via logging) the list of currently-rostered RBs who have
    zero historical carry data and therefore got no derived role -- these
    are exactly the cases (true rookies, practice-squad call-ups) where no
    amount of data analysis can substitute for a human decision.
    """
    current_season = current_season or config.CURRENT_SEASON
    rb1_share_threshold = rb1_share_threshold if rb1_share_threshold is not None else config.ROLE_INFERENCE_RB1_SHARE_THRESHOLD
    committee_share_threshold = committee_share_threshold if committee_share_threshold is not None else config.ROLE_INFERENCE_COMMITTEE_SHARE_THRESHOLD

    required = {"position_group", "last_season", "latest_team", "player_id", "player_name"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players roster is missing required columns for role derivation: {missing}")

    current_rbs = players[
        (players["position_group"] == "RB") & (players["last_season"] == current_season)
    ].copy()

    workload = rb_metrics.set_index("player_id")["total_carries"]

    rows = []
    no_data_players = []

    for team, group in current_rbs.groupby("latest_team"):
        candidates = []
        for _, p in group.iterrows():
            w = workload.get(p["player_id"])
            if w is None or pd.isna(w) or w <= 0:
                no_data_players.append((team, p["player_name"]))
                continue
            candidates.append({"player_id": p["player_id"], "player_name": p["player_name"], "workload": float(w)})

        if not candidates:
            continue

        candidates.sort(key=lambda c: c["workload"], reverse=True)
        top = candidates[:max_backs_per_team]
        total = sum(c["workload"] for c in top)
        shares = [c["workload"] / total for c in top]

        if len(top) == 1:
            roles = ["RB1"]
        elif shares[0] >= rb1_share_threshold:
            roles = ["RB1"] + ["RB2"] * (len(top) - 1)
        elif all(s >= committee_share_threshold for s in shares[:2]):
            roles = ["committee"] * len(top)
        else:
            roles = ["RB1"] + ["RB2"] * (len(top) - 1)

        for c, role in zip(top, roles):
            rows.append({
                "team": team, "player_name": c["player_name"], "player_id": c["player_id"],
                "role": role, "status": "active",
            })

    if no_data_players:
        logger.info(
            "no historical first-drive data for %d currently-rostered RB(s); no role could be "
            "derived for them (add manually to data/current_rb_roles.csv if you disagree): %s",
            len(no_data_players), no_data_players,
        )

    return pd.DataFrame(rows, columns=ROLE_COLUMNS)
