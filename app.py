"""
Streamlit dashboard for the NFL first-drive RB rushing model (Section 38-39).

Run with:  streamlit run app.py

Every page reads from the CSVs produced by `python main.py full-pipeline`.
If a file is missing, the page says so and tells you which CLI command to
run -- it never fabricates numbers to fill the gap (Section 45).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.data_loader import load_player_positions
from src.model import load_current_rb_roles
from src.odds import load_sportsbook_odds
from src.rankings import rank_candidates, filter_rankings
from src.model import build_model_table
from src.odds import attach_odds_and_edge
from src.backtest import compute_backtest_metrics

st.set_page_config(page_title="First-Drive RB Rushing Model", layout="wide")


@st.cache_data
def _read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _missing(path, cli_hint):
    st.warning(f"`{path}` not found. Run `{cli_hint}` first.")


def page_rankings():
    st.title("Best First-Drive RB Bets")
    st.caption(f"CURRENT ROLE SOURCE: {config.CURRENT_ROLE_SOURCE}")

    team_metrics = _read_csv(config.TEAM_METRICS_BLENDED_OUT)
    rb_metrics = _read_csv(config.RB_METRICS_OUT)
    defense_metrics = _read_csv(config.DEFENSE_METRICS_OUT)
    qb_competition = _read_csv(config.PROCESSED_DATA_DIR / "qb_competition_by_season.csv")

    if any(x is None for x in [team_metrics, rb_metrics, defense_metrics, qb_competition]):
        _missing(config.RB_METRICS_OUT, "python main.py calculate-rb-metrics")
        return

    if not config.WEEK1_MATCHUPS_FILE.exists():
        _missing(config.WEEK1_MATCHUPS_FILE, "add a matchups CSV with [team, opponent] columns")
        return
    matchups = pd.read_csv(config.WEEK1_MATCHUPS_FILE)

    current_roles = load_current_rb_roles()
    odds = load_sportsbook_odds()
    players = load_player_positions()

    model_df = build_model_table(rb_metrics, team_metrics, defense_metrics, qb_competition, current_roles, matchups, players=players)
    with_odds = attach_odds_and_edge(model_df, odds)
    ranked = rank_candidates(with_odds)

    filter_choice = st.selectbox("Filter", ["all", "best_bets", "high_probability", "best_value", "passes"])
    filtered = filter_rankings(ranked, filter_choice)

    display_cols = [
        "rank", "player_name", "team", "opponent", "model_probability",
        "games_with_carry", "games_with_5plus",
        "sportsbook_odds", "sportsbook_implied_probability", "edge", "fair_odds",
        "confidence", "bet_category",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Player Detail")
    names = filtered["player_name"].tolist()
    if names:
        selected = st.selectbox("Select an RB", names)
        row = filtered[filtered["player_name"] == selected].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Probability", f"{row['model_probability']:.1%}")
        col2.metric("P(carry)", f"{row['p_carry_final']:.1%}")
        col3.metric("P(5+ | carry)", f"{row['p_five_given_carry_final']:.1%}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Fair Odds", f"{row['fair_odds']:.0f}")
        sb = row["sportsbook_odds"]
        col5.metric("Sportsbook Odds", f"{sb:.0f}" if pd.notna(sb) else "unavailable")
        edge = row["edge"]
        col6.metric("Edge", f"{edge:+.1%}" if pd.notna(edge) else "unavailable")

        st.write(f"**Current role:** {row.get('role', 'unavailable')} ({row.get('role_note', '')})")
        st.write(f"**Confidence:** {row['confidence']}")
        st.write(f"**Historical first-drive games with a carry:** {int(row['games_with_carry'])}")
        st.write(f"**Team first-drive run%:** {row['first_drive_run_pct']:.1%}" if pd.notna(row.get("first_drive_run_pct")) else "")
        st.write(f"**Opponent first-drive 5+ allowed rate:** {row['opp_five_plus_allowed_rate']:.1%}" if pd.notna(row.get("opp_five_plus_allowed_rate")) else "")


def page_team_trends():
    st.title("Team First-Drive Trends")
    df = _read_csv(config.TEAM_METRICS_OUT)
    if df is None:
        _missing(config.TEAM_METRICS_OUT, "python main.py calculate-rb-metrics")
        return
    fig = px.bar(df.sort_values("first_drive_run_pct", ascending=False), x="team", y="first_drive_run_pct",
                 color="season", barmode="group", title="First-Drive Run % by Team/Season")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_rb_trends():
    st.title("RB First-Drive Trends")
    df = _read_csv(config.RB_METRICS_OUT)
    if df is None:
        _missing(config.RB_METRICS_OUT, "python main.py calculate-rb-metrics")
        return
    fig = px.scatter(df, x="p_carry_1plus_shrunk", y="p_five_given_carry_shrunk", size="games_with_carry",
                      hover_name="player_name", color="team",
                      title="P(carry) vs P(5+ | carry), sized by sample size")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_matchup_analysis():
    st.title("Matchup Analysis: Opponent First-Drive Run Defense")
    df = _read_csv(config.DEFENSE_METRICS_OUT)
    if df is None:
        _missing(config.DEFENSE_METRICS_OUT, "python main.py calculate-rb-metrics")
        return
    fig = px.bar(df.sort_values("five_plus_allowed_rate_shrunk", ascending=False),
                 x="team", y="five_plus_allowed_rate_shrunk",
                 title="Opponent First-Drive 5+ Allowed Rate (shrunk)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_sportsbook_value():
    st.title("Sportsbook Value")
    odds = load_sportsbook_odds()
    if odds.empty:
        st.info(f"No sportsbook odds on file. Add rows to {config.SPORTSBOOK_ODDS_FILE}.")
        return
    st.dataframe(odds, use_container_width=True, hide_index=True)


def page_backtesting():
    st.title("Backtesting")
    df = _read_csv(config.BACKTEST_OUT)
    if df is None:
        _missing(config.BACKTEST_OUT, "python main.py backtest")
        return
    metrics = compute_backtest_metrics(df)
    if "error" in metrics:
        st.error(metrics["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Brier Score", f"{metrics['brier_score']:.4f}")
    c3.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_calibration():
    st.title("Model Calibration")
    df = _read_csv(config.BACKTEST_OUT)
    if df is None:
        _missing(config.BACKTEST_OUT, "python main.py backtest")
        return
    metrics = compute_backtest_metrics(df)
    if "error" in metrics:
        st.error(metrics["error"])
        return
    cal = metrics["calibration"]
    fig = px.bar(cal, x="bucket", y=["predicted_avg", "actual_rate"], barmode="group",
                 title="Predicted vs Actual Hit Rate by Probability Bucket")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cal, use_container_width=True, hide_index=True)


PAGES = {
    "Week 1 Rankings": page_rankings,
    "Team First Drive Trends": page_team_trends,
    "RB First Drive Trends": page_rb_trends,
    "Matchup Analysis": page_matchup_analysis,
    "Sportsbook Value": page_sportsbook_value,
    "Backtesting": page_backtesting,
    "Model Calibration": page_calibration,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Page", list(PAGES.keys()))
PAGES[choice]()
