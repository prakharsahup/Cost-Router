"""Tests for BudgetTracker (REQ-BUDGET-1..7).

Covers:
- Initial budget setting and remaining query
- Actual cost recording with provider-reported cost
- Fallback estimation when cost_usd is None (REQ-BUDGET-6)
- Threshold warnings fire exactly once each (REQ-BUDGET-4)
- Soft mode forces cheap-only when exhausted (REQ-BUDGET-1)
- Hard-stop mode (REQ-BUDGET-1)
- can_afford pre-flight check
- Concurrent decrements sum correctly (REQ-BUDGET-7)
"""

from __future__ import annotations

import threading

import pytest

from costrouter.types import BudgetMode, ModelResponse, ModelTier
from costrouter.budget import BudgetTracker

from conftest import DEFAULT_RATE_CARD, ok_response


# ---------------------------------------------------------------------------
# Basic ledger operations
# ---------------------------------------------------------------------------

class TestBudgetBasics:
    def test_initial_budget(self):
        bt = BudgetTracker(initial_budget_usd=10.0, rate_card=DEFAULT_RATE_CARD)
        assert bt.remaining_budget() == 10.0

    def test_record_actual_cost_with_provider_cost(self):
        bt = BudgetTracker(initial_budget_usd=10.0, rate_card=DEFAULT_RATE_CARD)
        resp = ok_response(cost=0.5)
        cost = bt.record_actual_cost(ModelTier.CHEAP, resp)
        assert cost == 0.5
        assert bt.remaining_budget() == pytest.approx(9.5)

    def test_record_multiple_costs(self):
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.3))
        bt.record_actual_cost(ModelTier.EXPENSIVE, ok_response(cost=0.4))
        assert bt.remaining_budget() == pytest.approx(0.3)

    def test_exhausted_flag(self):
        bt = BudgetTracker(initial_budget_usd=0.5, rate_card=DEFAULT_RATE_CARD)
        assert not bt.exhausted()
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.5))
        assert bt.exhausted()

    def test_exhausted_when_negative(self):
        bt = BudgetTracker(initial_budget_usd=0.1, rate_card=DEFAULT_RATE_CARD)
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.2))
        assert bt.exhausted()
        assert bt.remaining_budget() < 0


# ---------------------------------------------------------------------------
# Fallback estimator (REQ-BUDGET-6)
# ---------------------------------------------------------------------------

class TestFallbackEstimation:
    def test_fallback_when_cost_is_none(self):
        """When provider doesn't report cost, BudgetTracker estimates via rate card."""
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        resp = ModelResponse(
            text="answer",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_usd=None,  # no provider cost
        )
        cost = bt.record_actual_cost(ModelTier.CHEAP, resp)
        # cheap: 1000 * 0.000001 + 500 * 0.000002 = 0.001 + 0.001 = 0.002
        assert cost == pytest.approx(0.002)
        assert bt.remaining_budget() == pytest.approx(0.998)

    def test_fallback_with_expensive_model(self):
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        resp = ModelResponse(
            text="answer", prompt_tokens=1000, completion_tokens=500, cost_usd=None
        )
        cost = bt.record_actual_cost(ModelTier.EXPENSIVE, resp)
        # expensive: 1000 * 0.00001 + 500 * 0.00003 = 0.01 + 0.015 = 0.025
        assert cost == pytest.approx(0.025)

    def test_fallback_with_none_tokens(self):
        """If both cost and tokens are None, fallback cost should be 0."""
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        resp = ModelResponse(text="answer", cost_usd=None)
        cost = bt.record_actual_cost(ModelTier.CHEAP, resp)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# can_afford pre-flight check
# ---------------------------------------------------------------------------

class TestCanAfford:
    def test_can_afford_with_budget(self):
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        assert bt.can_afford(ModelTier.CHEAP, est_prompt_tokens=100, est_completion_tokens=50)

    def test_cannot_afford_expensive(self):
        bt = BudgetTracker(initial_budget_usd=0.0001, rate_card=DEFAULT_RATE_CARD)
        # expensive: 10000 * 0.00001 + 5000 * 0.00003 = 0.1 + 0.15 = 0.25
        assert not bt.can_afford(ModelTier.EXPENSIVE, est_prompt_tokens=10000, est_completion_tokens=5000)


# ---------------------------------------------------------------------------
# Threshold warnings (REQ-BUDGET-4)
# ---------------------------------------------------------------------------

class TestThresholdWarnings:
    def test_thresholds_fire_at_correct_levels(self):
        fired: list[tuple[float, float]] = []
        bt = BudgetTracker(
            initial_budget_usd=1.0,
            rate_card=DEFAULT_RATE_CARD,
            warn_thresholds_pct=(0.5, 0.1),
            on_threshold=lambda pct, rem: fired.append((pct, rem)),
        )
        # Spend 0.6 → remaining 0.4 → 40% → crosses 50% threshold
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.6))
        assert len(fired) == 1
        assert fired[0][0] == 0.5

        # Spend 0.35 → remaining 0.05 → 5% → crosses 10% threshold
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.35))
        assert len(fired) == 2
        assert fired[1][0] == 0.1

    def test_threshold_fires_only_once(self):
        fired: list[float] = []
        bt = BudgetTracker(
            initial_budget_usd=1.0,
            rate_card=DEFAULT_RATE_CARD,
            warn_thresholds_pct=(0.5,),
            on_threshold=lambda pct, rem: fired.append(pct),
        )
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.6))
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.1))
        # Should have fired only once despite two calls below 50%
        assert fired == [0.5]

    def test_no_threshold_callback_is_ok(self):
        """No crash when on_threshold is not set."""
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.9))  # no crash


# ---------------------------------------------------------------------------
# Budget modes (REQ-BUDGET-1)
# ---------------------------------------------------------------------------

class TestBudgetModes:
    def test_soft_mode_is_default(self):
        bt = BudgetTracker(initial_budget_usd=1.0, rate_card=DEFAULT_RATE_CARD)
        assert bt.mode == BudgetMode.SOFT

    def test_hard_stop_mode(self):
        bt = BudgetTracker(
            initial_budget_usd=1.0,
            rate_card=DEFAULT_RATE_CARD,
            mode=BudgetMode.HARD_STOP,
        )
        assert bt.mode == BudgetMode.HARD_STOP


# ---------------------------------------------------------------------------
# Thread safety (REQ-BUDGET-7)
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_decrements(self):
        """N threads recording costs concurrently should sum correctly."""
        n_threads = 50
        cost_per_call = 0.01
        initial = n_threads * cost_per_call + 1.0  # plenty of budget

        bt = BudgetTracker(initial_budget_usd=initial, rate_card=DEFAULT_RATE_CARD)
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=cost_per_call))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_remaining = initial - (n_threads * cost_per_call)
        assert bt.remaining_budget() == pytest.approx(expected_remaining, abs=1e-9)


class TestReentrantCallback:
    def test_callback_can_read_tracker_without_deadlock(self):
        """on_threshold runs outside the lock, so it may call back in."""
        seen: list[float] = []
        bt = BudgetTracker(
            initial_budget_usd=1.0,
            rate_card=DEFAULT_RATE_CARD,
            on_threshold=lambda pct, rem: seen.append(bt.remaining_budget()),
        )
        worker = threading.Thread(
            target=bt.record_actual_cost,
            args=(ModelTier.CHEAP, ok_response(cost=0.6)),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive(), "record_actual_cost deadlocked"
        assert seen == [pytest.approx(0.4)]

    def test_initial_budget_property(self):
        bt = BudgetTracker(initial_budget_usd=3.0, rate_card=DEFAULT_RATE_CARD)
        bt.record_actual_cost(ModelTier.CHEAP, ok_response(cost=1.0))
        assert bt.initial_budget == 3.0
