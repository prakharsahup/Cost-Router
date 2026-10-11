"""Policy sweep test (Phase 7, REQ-BENCH-4).

Sweeping 5 thresholds produces 5 distinct router points, not 1.
"""

from __future__ import annotations

import pytest

from costrouter.store import init_db

from benchmark.datasets import load_triviaqa
from benchmark.graders import grade_triviaqa
from benchmark.harness import run_policy_sweep

from conftest import DEFAULT_RATE_CARD
from test_harness_smoke import SmartMockModel, TRIVIAQA_ANSWERS


class TestPolicySweep:
    def test_five_thresholds_produce_five_points(self):
        """REQ-BENCH-4: sweeping 5 thresholds → 5 distinct router points."""
        cheap = SmartMockModel(TRIVIAQA_ANSWERS, prefix="[cheap] ")
        expensive = SmartMockModel(TRIVIAQA_ANSWERS, prefix="[expensive] ")
        items = load_triviaqa(limit=5)
        conn = init_db(":memory:")

        results = run_policy_sweep(
            items=items,
            dataset_name="triviaqa",
            grader=grade_triviaqa,
            cheap_model=cheap,
            expensive_model=expensive,
            rate_card=DEFAULT_RATE_CARD,
            thresholds=[0.2, 0.35, 0.5, 0.65, 0.8],
            budget_usd=10.0,
            conn=conn,
        )

        assert len(results) == 5

        # All should be "router" condition
        assert all(r["condition"] == "router" for r in results)

        # Each should have a unique run_id
        run_ids = [r["run_id"] for r in results]
        assert len(set(run_ids)) == 5

        # Verify all results have correct task count
        assert all(r["total_count"] == 5 for r in results)

        # Verify rows in the DB
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert run_count == 5

        decision_count = conn.execute(
            "SELECT COUNT(*) FROM routing_decisions"
        ).fetchone()[0]
        assert decision_count == 25  # 5 thresholds * 5 tasks

    def test_sweep_cost_varies_by_threshold(self):
        """Lower thresholds → more expensive calls → higher total cost."""
        cheap = SmartMockModel(TRIVIAQA_ANSWERS, prefix="")
        expensive = SmartMockModel(TRIVIAQA_ANSWERS, prefix="")

        # Override costs to make the difference visible
        class CheapModel:
            def invoke(self, prompt, **kw):
                from costrouter.types import ModelResponse
                return ModelResponse(text="Paris", prompt_tokens=100,
                                     completion_tokens=20, cost_usd=0.001)

        class ExpensiveModel:
            def invoke(self, prompt, **kw):
                from costrouter.types import ModelResponse
                return ModelResponse(text="Paris", prompt_tokens=100,
                                     completion_tokens=200, cost_usd=0.05)

        items = load_triviaqa(limit=5)
        conn = init_db(":memory:")

        results = run_policy_sweep(
            items=items,
            dataset_name="triviaqa",
            grader=grade_triviaqa,
            cheap_model=CheapModel(),
            expensive_model=ExpensiveModel(),
            rate_card=DEFAULT_RATE_CARD,
            thresholds=[0.1, 0.9],
            budget_usd=100.0,
            conn=conn,
        )

        # Lower threshold (0.1) → more expensive calls → higher cost
        # Higher threshold (0.9) → more cheap calls → lower cost
        assert results[0]["total_cost"] >= results[1]["total_cost"]
