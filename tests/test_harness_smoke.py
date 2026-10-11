"""Harness smoke tests (Phase 6).

3-condition run over a small synthetic task set completes and writes
expected rows to a temp SQLite database.
"""

from __future__ import annotations

import pytest

from costrouter.types import ModelResponse, ModelTier
from costrouter.store import init_db

from benchmark.datasets import load_triviaqa, load_gsm8k, _MINI_TRIVIAQA, _MINI_GSM8K
from benchmark.graders import grade_triviaqa, grade_gsm8k
from benchmark.harness import run_benchmark, run_condition, run_policy_sweep

from conftest import MockChatModel, DEFAULT_RATE_CARD, ok_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SmartMockModel:
    """A mock model that returns answers matching the gold for specific datasets."""

    def __init__(self, answers: dict[str, str], prefix: str = ""):
        self._answers = answers
        self._prefix = prefix

    def invoke(self, prompt: str, **kwargs) -> ModelResponse:
        # Try to find a matching answer based on prompt content
        for key, answer in self._answers.items():
            if key.lower() in prompt.lower():
                return ModelResponse(
                    text=f"{self._prefix}{answer}",
                    prompt_tokens=len(prompt) // 4,
                    completion_tokens=20,
                    cost_usd=0.001 if "cheap" in self._prefix.lower() else 0.01,
                )
        return ModelResponse(
            text=f"{self._prefix}I don't know",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=10,
            cost_usd=0.001,
        )


# Pre-built answer maps
TRIVIAQA_ANSWERS = {
    "capital of france": "Paris",
    "romeo and juliet": "William Shakespeare",
    "chemical symbol for water": "H2O",
    "world war ii": "1945",
    "red planet": "Mars",
    "mona lisa": "Leonardo da Vinci",
    "largest ocean": "Pacific Ocean",
    "plants absorb": "Carbon dioxide",
    "penicillin": "Alexander Fleming",
    "square root": "12",
}

GSM8K_ANSWERS = {
    "janet": "#### 18",
    "robe": "#### 3",
    "flipping a house": "#### 70000",
    "james writes": "#### 624",
    "wendi feeds": "#### 20",
}


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------

class TestDatasetLoaders:
    def test_triviaqa_mini_loads(self):
        items = load_triviaqa(limit=3)
        assert len(items) == 3
        assert all(len(item) == 3 for item in items)
        assert items[0][0] == "triviaqa-0"

    def test_gsm8k_mini_loads(self):
        items = load_gsm8k(limit=2)
        assert len(items) == 2
        assert items[0][0] == "gsm8k-0"

    def test_triviaqa_full_mini(self):
        items = load_triviaqa()
        assert len(items) == len(_MINI_TRIVIAQA)

    def test_gsm8k_full_mini(self):
        items = load_gsm8k()
        assert len(items) == len(_MINI_GSM8K)


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:
    def test_triviaqa_exact_match(self):
        assert grade_triviaqa("Paris", "Paris")

    def test_triviaqa_case_insensitive(self):
        assert grade_triviaqa("paris", "Paris")

    def test_triviaqa_with_extra_text(self):
        assert grade_triviaqa("The answer is Paris, the capital of France.", "Paris")

    def test_triviaqa_wrong_answer(self):
        assert not grade_triviaqa("London", "Paris")

    def test_gsm8k_numeric_match(self):
        assert grade_gsm8k("The answer is #### 18", "18")

    def test_gsm8k_number_in_text(self):
        assert grade_gsm8k("She makes 18 dollars", "18")

    def test_gsm8k_wrong_number(self):
        assert not grade_gsm8k("The answer is 20", "18")

    def test_gsm8k_with_commas(self):
        assert grade_gsm8k("The profit is $70,000", "70000")


# ---------------------------------------------------------------------------
# Harness smoke test
# ---------------------------------------------------------------------------

class TestHarnessSmoke:
    def test_three_condition_run(self):
        """3-condition run over mini TriviaQA completes and writes to DB."""
        cheap = SmartMockModel(TRIVIAQA_ANSWERS, prefix="[cheap] ")
        expensive = SmartMockModel(TRIVIAQA_ANSWERS, prefix="[expensive] ")
        items = load_triviaqa(limit=3)
        conn = init_db(":memory:")

        results = run_benchmark(
            items=items,
            dataset_name="triviaqa",
            grader=grade_triviaqa,
            cheap_model=cheap,
            expensive_model=expensive,
            rate_card=DEFAULT_RATE_CARD,
            budget_usd=1.0,
            conn=conn,
        )

        assert len(results) == 3
        conditions = {r["condition"] for r in results}
        assert conditions == {"always_cheap", "always_expensive", "router"}

        # Verify SQLite rows
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert run_count == 3

        decision_count = conn.execute("SELECT COUNT(*) FROM routing_decisions").fetchone()[0]
        assert decision_count == 9  # 3 conditions * 3 tasks

        result_count = conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
        assert result_count == 9

    def test_single_condition_returns_correct_fields(self):
        cheap = SmartMockModel(TRIVIAQA_ANSWERS, prefix="")
        expensive = SmartMockModel(TRIVIAQA_ANSWERS, prefix="")
        items = load_triviaqa(limit=2)
        conn = init_db(":memory:")

        r = run_condition(
            items=items,
            condition="always_cheap",
            dataset_name="triviaqa",
            grader=grade_triviaqa,
            cheap_model=cheap,
            expensive_model=expensive,
            rate_card=DEFAULT_RATE_CARD,
            conn=conn,
        )

        assert "run_id" in r
        assert "accuracy" in r
        assert "total_cost" in r
        assert r["total_count"] == 2
        assert 0 <= r["accuracy"] <= 1.0
        assert r["total_cost"] > 0

    def test_gsm8k_condition(self):
        cheap = SmartMockModel(GSM8K_ANSWERS, prefix="")
        expensive = SmartMockModel(GSM8K_ANSWERS, prefix="")
        items = load_gsm8k(limit=2)
        conn = init_db(":memory:")

        r = run_condition(
            items=items,
            condition="router",
            dataset_name="gsm8k",
            grader=grade_gsm8k,
            cheap_model=cheap,
            expensive_model=expensive,
            rate_card=DEFAULT_RATE_CARD,
            conn=conn,
        )

        assert r["condition"] == "router"
        assert r["dataset"] == "gsm8k"
        assert r["total_count"] == 2


class TestNoGoldLabelLeakage:
    def test_grades_do_not_feed_history_by_default(self):
        """A wrong-but-valid cheap answer must not raise later difficulty scores."""
        from benchmark.harness import run_condition
        items = [(f"t{i}", "What is the capital of France?", "Paris") for i in range(8)]
        common = dict(
            items=items, condition="router", dataset_name="qa", grader=grade_triviaqa,
            cheap_model=MockChatModel([ok_response(text="London")] * 8),
            expensive_model=MockChatModel([ok_response(text="Paris")] * 8),
            rate_card=DEFAULT_RATE_CARD, threshold=0.5,
        )
        default = run_condition(**common)
        assert {t["model_chosen"] for t in default["per_task"]} == {"cheap"}
        assert default["per_task"][-1]["difficulty_score"] < 0.5

        common["cheap_model"] = MockChatModel([ok_response(text="London")] * 8)
        learned = run_condition(**common, learn_from_grades=True)
        assert learned["per_task"][-1]["difficulty_score"] > default["per_task"][-1]["difficulty_score"]
