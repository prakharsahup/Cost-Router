"""End-to-end smoke test of the benchmark runner (no network: offline cache)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

import benchmark.run as run_mod
from benchmark.prompts import CachedChatModel, has_final_answer, load_mixed
from costrouter.types import ModelResponse

ROOT = Path(__file__).resolve().parent.parent


def test_has_final_answer_validator():
    assert has_final_answer(ModelResponse(text="x\nFinal answer: 4"))
    assert not has_final_answer(ModelResponse(text="the answer is 4"))
    assert not has_final_answer(ModelResponse(text="Final answer:   "))
    assert not has_final_answer(ModelResponse(text="Final answer: 4", error="boom"))


def test_cache_round_trip_and_offline_miss(tmp_path):
    class Model:
        calls = 0

        def invoke(self, prompt, **kw):
            Model.calls += 1
            return ModelResponse(text=f"Final answer: {prompt}", prompt_tokens=3, completion_tokens=2)

    path = tmp_path / "cache.jsonl"
    m = CachedChatModel(Model(), "m", path)
    assert m.invoke("a").text == "Final answer: a"
    assert m.invoke("a").completion_tokens == 2 and Model.calls == 1

    offline = CachedChatModel(None, "m", path, offline=True)
    assert offline.invoke("a").prompt_tokens == 3
    assert offline.invoke("b").error  # miss, no call
    assert CachedChatModel(None, "other", path, offline=True).invoke("a").error  # key includes model


def test_mixed_loader_keeps_dataset_prefixes():
    items = load_mixed(None, None, limit=3)
    assert sorted(tid.rsplit("-", 1)[0] for tid, _, _ in items) == ["gsm8k"] * 3 + ["triviaqa"] * 3
    assert all("Final answer:" in p for _, p, _ in items)


def _fake_answer(tier, prompt, gold):
    """Expensive always right; cheap right on TriviaQA, and omits the line on GSM8K."""
    g = gold[0] if isinstance(gold, list) else gold
    if tier == "cheap" and prompt.startswith("Work out"):
        return "I think it is 0"
    return f"Final answer: {g}"


@pytest.mark.parametrize("n", [3])
def test_offline_run_writes_db_and_reports(tmp_path, monkeypatch, n):
    data = tmp_path / "data"
    data.mkdir()
    for f in ("triviaqa_sample.json", "gsm8k_sample.jsonl"):
        shutil.copy(ROOT / "data" / f, data / f)
    monkeypatch.setattr(run_mod, "DATA_DIR", data)
    monkeypatch.setattr(run_mod, "REPORTS_DIR", tmp_path / "reports")

    items = load_mixed(str(data / "triviaqa_sample.json"), str(data / "gsm8k_sample.jsonl"), limit=n)
    spec = run_mod.PROVIDERS["groq"]
    cache = data / "response_cache" / "groq.jsonl"
    cache.parent.mkdir()
    with cache.open("w", encoding="utf-8") as f:
        for tier in ("cheap", "expensive"):
            name = spec[tier]
            m = CachedChatModel(None, name, cache, offline=True)
            for _, prompt, gold in items:
                f.write(json.dumps({"key": m._key(prompt), "model": name,
                                    "text": _fake_answer(tier, prompt, gold),
                                    "prompt_tokens": 50, "completion_tokens": 20}) + "\n")

    db = tmp_path / "out.db"
    monkeypatch.setattr(sys, "argv", ["run", "--offline", "--provider", "groq",
                                      "--n", str(n), "--db", str(db), "--fresh"])
    run_mod.main()

    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in summary["runs"]}
    assert runs["always_cheap-mixed"]["accuracy"] == pytest.approx(0.5)
    assert runs["always_expensive-mixed"]["accuracy"] == 1.0
    router_runs = [r for r in runs.values() if r["condition"] == "router"]
    capped = next(r for r in router_runs if r["run_id"].endswith("-budgetcap-mixed"))
    hard = next(r for r in router_runs if "budgetcap-hardstop" in r["run_id"])
    uncapped = [r for r in router_runs if "budgetcap" not in r["run_id"]]
    assert len(uncapped) >= 2
    # Uncapped: malformed cheap GSM8K replies are retry-escalated → no errors.
    assert all(r["errors"] == 0 and r["accuracy"] == 1.0 for r in uncapped)
    # Capped: the budget forces cheap and blocks escalation, so the malformed
    # GSM8K replies surface as errors, and spend stays well below always_expensive.
    assert capped["budget_forced"] > 0
    assert capped["total_cost"] < runs["always_expensive-mixed"]["total_cost"]
    # HARD_STOP never spends past the cap; refused steps count as errors.
    assert hard["total_cost"] <= hard["budget_usd"]
    assert hard["errors"] > 0

    conn = sqlite3.connect(db)
    n_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL").fetchone()[0]
    n_results = conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
    conn.close()
    assert n_runs == len(runs) and n_results == len(runs) * 2 * n
    assert (tmp_path / "reports" / "spot_check.csv").is_file()
