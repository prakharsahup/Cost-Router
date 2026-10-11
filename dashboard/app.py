"""Streamlit dashboard — Pareto curve + budget events view (REQ-BENCH-5).

Reads from the SQLite database populated by the benchmark harness and
displays:
1. **Pareto curve**: Cost (x) vs Accuracy (y), colored by condition.
2. **Budget events**: threshold-crossing events over time.
3. **Decision breakdown**: per-run model choice distribution.

Usage::

    streamlit run dashboard/app.py -- --db costrouter.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def default_db_path() -> str:
    """``--db`` CLI arg (after streamlit's ``--``), else env var, else default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db")
    args, _ = parser.parse_known_args()
    return args.db or os.environ.get("COSTROUTER_DB_PATH", "costrouter.db")


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open the DB read-only so viewing results never mutates them."""
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_summary(conn) -> pd.DataFrame:
    """Aggregate cost and accuracy per run."""
    query = """
        SELECT
            r.run_id,
            r.condition,
            r.task_set,
            r.policy_config_json,
            r.started_at,
            COALESCE(
                (SELECT COUNT(*) FROM task_results tr WHERE tr.run_id = r.run_id AND tr.correct = 1),
                0
            ) * 1.0 / NULLIF(
                (SELECT COUNT(*) FROM task_results tr WHERE tr.run_id = r.run_id),
                0
            ) AS accuracy,
            COALESCE(
                (SELECT SUM(tr.total_cost_usd) FROM task_results tr WHERE tr.run_id = r.run_id),
                0
            ) AS total_cost,
            (SELECT COUNT(*) FROM task_results tr WHERE tr.run_id = r.run_id) AS task_count
        FROM runs r
        WHERE r.finished_at IS NOT NULL
        ORDER BY r.started_at
    """
    return pd.read_sql(query, conn)


def load_decisions(conn) -> pd.DataFrame:
    """Load all routing decisions."""
    return pd.read_sql("SELECT * FROM routing_decisions ORDER BY id", conn)


def load_budget_events(conn) -> pd.DataFrame:
    """Load all budget-threshold crossing events."""
    return pd.read_sql("SELECT * FROM budget_events ORDER BY id", conn)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def pareto_chart(df: pd.DataFrame) -> go.Figure:
    """Cost vs Accuracy scatter plot colored by condition (SRS sample chart)."""
    color_map = {
        "always_cheap": "#2ecc71",
        "always_expensive": "#e74c3c",
        "router": "#3498db",
    }

    fig = px.scatter(
        df,
        x="total_cost",
        y="accuracy",
        color="condition",
        color_discrete_map=color_map,
        hover_data=["run_id", "task_set", "task_count"],
        title="Cost vs Accuracy (Pareto View)",
        labels={
            "total_cost": "Total Cost (USD)",
            "accuracy": "Accuracy",
            "condition": "Condition",
        },
    )

    # Connect router points with a line if multiple exist
    router_df = df[df["condition"] == "router"].sort_values("total_cost")
    if len(router_df) > 1:
        fig.add_trace(
            go.Scatter(
                x=router_df["total_cost"],
                y=router_df["accuracy"],
                mode="lines",
                name="Router Frontier",
                line=dict(color="#3498db", dash="dash", width=1),
                showlegend=True,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Total Cost (USD)",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
    )
    return fig


def model_choice_chart(decisions_df: pd.DataFrame) -> go.Figure:
    """Per-run bar chart of cheap vs expensive model choices."""
    if decisions_df.empty:
        return go.Figure()

    agg = (
        decisions_df.groupby(["run_id", "model_chosen"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        agg,
        x="run_id",
        y="count",
        color="model_chosen",
        color_discrete_map={"cheap": "#2ecc71", "expensive": "#e74c3c"},
        title="Model Choice Distribution per Run",
        labels={"count": "Decisions", "run_id": "Run ID", "model_chosen": "Model"},
        barmode="stack",
    )
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-45)
    return fig


def budget_events_chart(events_df: pd.DataFrame) -> go.Figure:
    """Timeline of budget-threshold crossings."""
    if events_df.empty:
        return go.Figure().update_layout(
            title="Budget Threshold Events (none recorded)"
        )

    fig = px.scatter(
        events_df,
        x="triggered_at",
        y="remaining_budget",
        color="run_id",
        size=[15] * len(events_df),
        hover_data=["threshold_pct"],
        title="Budget Threshold Crossings",
        labels={
            "triggered_at": "Time",
            "remaining_budget": "Remaining Budget (USD)",
        },
    )
    fig.update_layout(template="plotly_dark")
    return fig


def budget_forced_chart(decisions_df: pd.DataFrame) -> go.Figure:
    """Count of budget-forced decisions per run."""
    if decisions_df.empty:
        return go.Figure()

    agg = (
        decisions_df[decisions_df["budget_forced"] == 1]
        .groupby("run_id")
        .size()
        .reset_index(name="forced_count")
    )

    if agg.empty:
        return go.Figure().update_layout(
            title="Budget-Forced Decisions (none recorded)"
        )

    fig = px.bar(
        agg,
        x="run_id",
        y="forced_count",
        title="Budget-Forced Decision Count per Run",
        labels={"forced_count": "Forced Decisions", "run_id": "Run ID"},
        color_discrete_sequence=["#f39c12"],
    )
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-45)
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="CostRouter Dashboard",
        page_icon="$",
        layout="wide",
    )

    st.title("CostRouter Dashboard")
    st.markdown("Cost-aware LLM routing benchmark results")

    # ---- Sidebar: DB path ----
    db_path = st.sidebar.text_input(
        "SQLite Database Path",
        value=default_db_path(),
    )

    if not os.path.isfile(db_path):
        st.warning(
            f"Database not found at `{db_path}`. "
            "Run the benchmark harness first to generate data, "
            "or enter the correct path in the sidebar."
        )
        st.info(
            "Generate it with:\n```\npython -m benchmark.run --fresh\n```\n"
            "or rebuild it from the committed response cache, no API key "
            "needed:\n```\npython -m benchmark.run --offline --fresh\n```"
        )
        return

    conn = open_readonly(db_path)

    # ---- Load data ----
    run_summary = load_run_summary(conn)
    decisions = load_decisions(conn)
    budget_events = load_budget_events(conn)
    conn.close()

    if run_summary.empty:
        st.info("No completed runs found. Run the benchmark harness first.")
        return

    # ---- Filters ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    task_sets = sorted(run_summary["task_set"].unique())
    selected_task_set = st.sidebar.multiselect(
        "Task Set", task_sets, default=task_sets
    )

    conditions = sorted(run_summary["condition"].unique())
    selected_conditions = st.sidebar.multiselect(
        "Condition", conditions, default=conditions
    )

    filtered = run_summary[
        (run_summary["task_set"].isin(selected_task_set))
        & (run_summary["condition"].isin(selected_conditions))
    ]

    # ---- Main content ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(filtered))
    col2.metric(
        "Avg Accuracy",
        f"{filtered['accuracy'].mean():.1%}" if not filtered.empty else "N/A",
    )
    col3.metric(
        "Avg Cost",
        f"${filtered['total_cost'].mean():.4f}" if not filtered.empty else "N/A",
    )

    # ---- Pareto curve ----
    st.plotly_chart(pareto_chart(filtered), width="stretch")

    # ---- Two-column layout for secondary charts ----
    col_left, col_right = st.columns(2)

    with col_left:
        filtered_decisions = decisions[
            decisions["run_id"].isin(filtered["run_id"])
        ] if not decisions.empty else decisions
        st.plotly_chart(model_choice_chart(filtered_decisions), width="stretch")

    with col_right:
        st.plotly_chart(
            budget_forced_chart(
                decisions[decisions["run_id"].isin(filtered["run_id"])]
                if not decisions.empty
                else decisions
            ),
            width="stretch",
        )

    # ---- Budget events ----
    filtered_events = budget_events[
        budget_events["run_id"].isin(filtered["run_id"])
    ] if not budget_events.empty else budget_events
    st.plotly_chart(budget_events_chart(filtered_events), width="stretch")

    # ---- Raw data tables ----
    with st.expander("Run Summary Table"):
        st.dataframe(filtered, width="stretch")

    with st.expander("Routing Decisions Table"):
        if not decisions.empty:
            st.dataframe(
                decisions[decisions["run_id"].isin(filtered["run_id"])],
                width="stretch",
            )
        else:
            st.info("No decisions recorded.")


if __name__ == "__main__":
    main()
