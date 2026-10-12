"""Dashboard tests: engine (replay, live session) + headless render of every page.

The database is rebuilt from the committed response cache (``benchmark.run
--offline``), so these run without network or API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

import benchmark.run as run_mod

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def bench_db(tmp_path_factory):
    if not (ROOT / "data" / "response_cache" / "groq.jsonl").is_file():
        pytest.skip("committed response cache missing")
    tmp = tmp_path_factory.mktemp("dash")
    db = tmp / "bench.db"
    argv, reports = sys.argv, run_mod.REPORTS_DIR
    sys.argv = ["run", "--offline", "--provider", "groq", "--fresh", "--db", str(db)]
    run_mod.REPORTS_DIR = tmp / "reports"
    try:
        run_mod.main()
    finally:
        sys.argv, run_mod.REPORTS_DIR = argv, reports
    return str(db)


@pytest.fixture
def offline(monkeypatch, bench_db):
    monkeypatch.setenv("COSTROUTER_OFFLINE", "1")
    monkeypatch.setenv("COSTROUTER_DB_PATH", bench_db)
    return bench_db


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_replay_reproduces_stored_benchmark_runs(offline):
    from dashboard import engine

    runs = engine.load_runs(offline)
    meta = engine.load_meta(offline)
    for _, r in runs[runs.kind == "router"].iterrows():
        rep = engine.replay(meta, r.threshold, r.budget_usd, hard_stop=False)
        assert rep["accuracy"] == pytest.approx(r.accuracy)
        assert rep["total_cost"] == pytest.approx(r.total_cost)
    hard = runs[runs.kind == "capped_hard"].iloc[0]
    rep = engine.replay(meta, hard.threshold, hard.budget_usd, hard_stop=True)
    assert rep["total_cost"] <= hard.budget_usd
    assert rep["accuracy"] == pytest.approx(hard.accuracy)


def test_oracle_routes_each_dataset_to_the_more_accurate_model(offline):
    from dashboard import engine

    o = engine.oracle_point(offline)
    assert o["picks"] == {"gsm8k": "cheap", "triviaqa": "expensive"}
    exp = engine.load_runs(offline).query("kind == 'always_expensive'").iloc[0]
    assert o["accuracy"] == pytest.approx(exp.accuracy) and o["total_cost"] < exp.total_cost


def test_live_session_presets_work_offline(offline):
    from dashboard import engine

    sess = engine.LiveSession(engine.load_meta(offline), budget_usd=0.01, threshold=0.3)
    assert not sess.live_available
    by_label = {item["label"]: item for item in engine.SHOWCASE}

    def ask(label):
        item = by_label[label]
        return sess.route(engine.showcase_prompt(item), item["step_type"], item["question"])

    easy = ask("Capital of Australia")
    assert (easy["status"], easy["tier"], easy["source"]) == ("ok", "cheap", "cache")
    assert "Canberra" in easy["final_answer"]
    hard = ask("Cheapest way to buy 22 pens")
    assert hard["tier"] == "expensive" and hard["explain"]["score"] > 0.3
    tricky = ask("UK quiz-show trivia")  # cheap reply is empty in the cache
    assert tricky["retried"] and tricky["tier"] == "expensive"
    assert "Millionaire" in tricky["final_answer"]
    assert sess.budget.remaining_budget() == pytest.approx(
        0.01 - sum(e["cost"] for e in sess.history))


def test_live_session_custom_prompt_without_key_is_reported(offline):
    from dashboard import engine

    sess = engine.LiveSession(engine.load_meta(offline))
    e = sess.route(engine.CUSTOM_TEMPLATE.format(question="Never seen before?"), "general", "x")
    assert e["status"] == "error" and e["tier"] is None


# ---------------------------------------------------------------------------
# Headless render
# ---------------------------------------------------------------------------

def _app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_every_page_renders(offline):
    at = _app()
    assert not at.exception
    for label in ["⚡ Live Router", "🎛️ Playground", "📊 Results", "🛡️ Budget Safety", "🧾 Audit Log"]:
        at.session_state["nav"] = label
        at.run()
        assert not at.exception, label
    at.session_state["nav"] = "📊 Results"
    at.run()
    assert len(at.get("plotly_chart")) >= 3


def test_live_preset_click_keeps_page_and_updates_ledger(offline):
    at = _app()
    at.session_state["nav"] = "⚡ Live Router"
    at.run()
    at.button(key="preset-0").click().run()
    at.button(key="preset-2").click().run()
    assert not at.exception
    assert any(b.key == "preset-0" for b in at.button)  # still on the Live Router page
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Questions"] == "2" and metrics["Cheap / expensive"] == "1 / 1"


def test_playground_slider_replays(offline):
    at = _app()
    at.session_state["nav"] = "🎛️ Playground"
    at.run()
    at.slider(key="pg_threshold").set_value(0.0).run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sent to expensive"] == "100%"
    at.toggle(key="pg_capped").set_value(True).run()
    at.radio(key="pg_mode").set_value("HARD_STOP").run()
    assert not at.exception
    assert int({m.label: m.value for m in at.metric}["Refused / failed"]) > 0


def test_dashboard_warns_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTROUTER_DB_PATH", str(tmp_path / "missing.db"))
    at = _app()
    assert not at.exception
    assert at.warning
