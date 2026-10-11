"""Tests for SQLite persistence + DecisionLogger (Phase 4)."""

from __future__ import annotations

import sqlite3

import pytest

from costrouter.types import ModelTier, RoutingDecision
from costrouter.store import DecisionLogger, init_db, start_run, finish_run


@pytest.fixture
def db_conn():
    """In-memory SQLite database for test isolation."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


class TestSchema:
    def test_tables_created(self, db_conn):
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "runs" in tables
        assert "routing_decisions" in tables
        assert "budget_events" in tables
        assert "task_results" in tables

    def test_init_db_idempotent(self, db_conn):
        """Calling init_db twice doesn't crash (CREATE TABLE IF NOT EXISTS)."""
        init_db(":memory:")  # second call should be fine


class TestRunManagement:
    def test_start_and_finish_run(self, db_conn):
        start_run(db_conn, "run-1", "router", "triviaqa", {"threshold": 0.5})
        row = db_conn.execute("SELECT * FROM runs WHERE run_id = 'run-1'").fetchone()
        assert row is not None
        assert row[1] == "router"  # condition
        assert row[2] == "triviaqa"  # task_set
        assert row[5] is None  # finished_at

        finish_run(db_conn, "run-1")
        row = db_conn.execute("SELECT finished_at FROM runs WHERE run_id = 'run-1'").fetchone()
        assert row[0] is not None


class TestDecisionLogger:
    def _make_decision(self, step_id: str = "s1", **overrides) -> RoutingDecision:
        defaults = dict(
            step_id=step_id,
            task_id="t1",
            prompt_length=12,
            prompt_hash="abcdef",
            difficulty_score=0.42,
            budget_remaining_before=10.0,
            budget_remaining_after=9.5,
            model_chosen=ModelTier.CHEAP,
            budget_forced=False,
            retried=False,
            actual_cost_usd=0.5,
            latency_ms=120.0,
            estimator_latency_ms=1.5,
            error=None,
        )
        defaults.update(overrides)
        return RoutingDecision(**defaults)

    def test_log_decision(self, db_conn):
        start_run(db_conn, "run-1", "router", "triviaqa")
        logger = DecisionLogger(db_conn, "run-1")
        logger.log(self._make_decision("s1"))

        rows = db_conn.execute("SELECT * FROM routing_decisions").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row[2] == "s1"  # step_id
        assert row[9] == "cheap"  # model_chosen

    def test_log_multiple_decisions(self, db_conn):
        start_run(db_conn, "run-1", "router", "triviaqa")
        logger = DecisionLogger(db_conn, "run-1")
        for i in range(5):
            logger.log(self._make_decision(f"s{i}"))
        count = db_conn.execute("SELECT COUNT(*) FROM routing_decisions").fetchone()[0]
        assert count == 5

    def test_log_budget_event(self, db_conn):
        start_run(db_conn, "run-1", "router", "triviaqa")
        logger = DecisionLogger(db_conn, "run-1")
        logger.log_budget_event(0.5, 5.0)
        logger.log_budget_event(0.1, 1.0)

        rows = db_conn.execute("SELECT * FROM budget_events").fetchall()
        assert len(rows) == 2
        assert rows[0][2] == 0.5  # threshold_pct
        assert rows[1][2] == 0.1

    def test_log_task_result(self, db_conn):
        start_run(db_conn, "run-1", "router", "triviaqa")
        logger = DecisionLogger(db_conn, "run-1")
        logger.log_task_result("t1", "triviaqa", correct=True, total_cost_usd=0.05)
        logger.log_task_result("t2", "triviaqa", correct=False, total_cost_usd=0.03)

        rows = db_conn.execute("SELECT * FROM task_results").fetchall()
        assert len(rows) == 2
        assert rows[0][4] == 1  # correct
        assert rows[1][4] == 0  # incorrect

    def test_decision_fields_match(self, db_conn):
        """Verify all fields round-trip correctly through SQLite."""
        start_run(db_conn, "run-1", "router", "gsm8k")
        logger = DecisionLogger(db_conn, "run-1")
        d = self._make_decision(
            step_id="step-x",
            model_chosen=ModelTier.EXPENSIVE,
            budget_forced=True,
            retried=True,
            error="some error",
        )
        logger.log(d)

        row = db_conn.execute(
            "SELECT step_id, model_chosen, budget_forced, retried, error "
            "FROM routing_decisions WHERE step_id = 'step-x'"
        ).fetchone()
        assert row == ("step-x", "expensive", 1, 1, "some error")
