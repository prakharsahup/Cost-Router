"""Headless render test of the Streamlit dashboard against a harness-built DB."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from benchmark.datasets import load_triviaqa
from benchmark.graders import grade_triviaqa
from benchmark.harness import run_benchmark
from costrouter.store import init_db
from conftest import DEFAULT_RATE_CARD, MockChatModel

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_dashboard_renders_runs(tmp_path, monkeypatch):
    db = tmp_path / "dash.db"
    conn = init_db(str(db))
    run_benchmark(
        items=load_triviaqa(limit=3), dataset_name="triviaqa", grader=grade_triviaqa,
        cheap_model=MockChatModel(), expensive_model=MockChatModel(),
        rate_card=DEFAULT_RATE_CARD, conn=conn,
    )
    conn.close()
    monkeypatch.setenv("COSTROUTER_DB_PATH", str(db))

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert [m.value for m in at.metric][0] == "3"  # Total Runs
    assert len(at.get("plotly_chart")) >= 3


def test_dashboard_warns_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTROUTER_DB_PATH", str(tmp_path / "missing.db"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.warning
