"""SQLite persistence + DecisionLogger (SRS §6, REQ-ROUTER-4).

Stores every routing decision, budget event, run metadata, and task result
in a local SQLite database.  Every write is a single transaction per step
to satisfy "every routing decision must be traceable" (SRS §2.5) even if the
process crashes mid-run.

The dashboard reads from this DB via ``pandas.read_sql``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from costrouter.types import RoutingDecision


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    condition TEXT NOT NULL CHECK(condition IN ('always_cheap','always_expensive','router')),
    task_set TEXT NOT NULL,
    policy_config_json TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    step_id TEXT NOT NULL,
    task_id TEXT,
    prompt_length INTEGER NOT NULL,
    prompt_hash TEXT NOT NULL,
    difficulty_score REAL NOT NULL,
    budget_remaining_before REAL NOT NULL,
    budget_remaining_after REAL NOT NULL,
    model_chosen TEXT NOT NULL CHECK(model_chosen IN ('cheap','expensive')),
    budget_forced INTEGER NOT NULL DEFAULT 0,
    retried INTEGER NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    estimator_latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    threshold_pct REAL NOT NULL,
    remaining_budget REAL NOT NULL,
    triggered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    correct INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str = "costrouter.db") -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure schema exists.

    Returns a ``sqlite3.Connection`` with WAL mode enabled for better
    concurrent-read performance.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# DecisionLogger
# ---------------------------------------------------------------------------

class DecisionLogger:
    """Thin wrapper that persists ``RoutingDecision`` objects to SQLite.

    Each ``log()`` call is a single transaction — crash-safe by design.

    Args:
        conn: an open ``sqlite3.Connection`` (from ``init_db``).
        run_id: the current run's identifier (links decisions to a run row).
    """

    def __init__(self, conn: sqlite3.Connection, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id

    def log(self, decision: RoutingDecision) -> None:
        """Persist a single routing decision (REQ-ROUTER-4)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO routing_decisions
                    (run_id, step_id, task_id, prompt_length, prompt_hash, difficulty_score,
                     budget_remaining_before, budget_remaining_after,
                     model_chosen, budget_forced, retried,
                     actual_cost_usd, latency_ms, estimator_latency_ms,
                     error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._run_id,
                    decision.step_id,
                    decision.task_id,
                    decision.prompt_length,
                    decision.prompt_hash,
                    decision.difficulty_score,
                    decision.budget_remaining_before,
                    decision.budget_remaining_after,
                    decision.model_chosen.value,
                    int(decision.budget_forced),
                    int(decision.retried),
                    decision.actual_cost_usd,
                    decision.latency_ms,
                    decision.estimator_latency_ms,
                    decision.error,
                    now,
                ),
            )

    def log_budget_event(
        self, threshold_pct: float, remaining_budget: float
    ) -> None:
        """Persist a budget-threshold crossing event."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO budget_events
                    (run_id, threshold_pct, remaining_budget, triggered_at)
                VALUES (?, ?, ?, ?)
                """,
                (self._run_id, threshold_pct, remaining_budget, now),
            )

    def log_task_result(
        self, task_id: str, dataset: str, correct: bool, total_cost_usd: float
    ) -> None:
        """Persist a single task evaluation result."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO task_results
                    (run_id, task_id, dataset, correct, total_cost_usd)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._run_id, task_id, dataset, int(correct), total_cost_usd),
            )


# ---------------------------------------------------------------------------
# Run management helpers
# ---------------------------------------------------------------------------

def start_run(
    conn: sqlite3.Connection,
    run_id: str,
    condition: str,
    task_set: str,
    policy_config: Optional[dict] = None,
) -> None:
    """Insert a new run row."""
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, condition, task_set, policy_config_json, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                condition,
                task_set,
                json.dumps(policy_config) if policy_config else None,
                now,
            ),
        )


def finish_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Mark a run as finished."""
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?",
            (now, run_id),
        )
