"""Performance/latency assertion tests (REQ-DIFF-3).

These tests enforce the NFR that estimator latency is under 200ms
and router overhead is under 50ms.
"""

from __future__ import annotations

import time

import pytest

from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode, RouterConfig
from costrouter.types import BudgetMode
from costrouter.budget import BudgetTracker

from conftest import MockChatModel, ok_response, DEFAULT_RATE_CARD


class TestLatency:
    def test_estimator_latency(self):
        """Estimator must return well under 200ms (REQ-DIFF-3)."""
        store = FailureRateStore(prior=0.5, min_samples=5)
        est = HeuristicDifficultyEstimator(failure_store=store)

        prompt = "This is a moderately long prompt " * 50
        
        # Warmup
        est.estimate(prompt, "default")
        
        # Test 100 iterations
        t0 = time.perf_counter()
        for _ in range(100):
            res = est.estimate(prompt, "default")
            # The individual recorded latency should be < 200ms
            assert res.latency_ms < 200.0
            
        t1 = time.perf_counter()
        
        # The average latency should be well under 10ms in reality
        avg_latency_ms = ((t1 - t0) * 1000) / 100
        assert avg_latency_ms < 50.0  # Safe threshold for CI


    def test_router_overhead(self):
        """Router overhead (minus model call) should be < 50ms."""
        cheap = MockChatModel([ok_response()] * 100)
        expensive = MockChatModel([ok_response()] * 100)
        
        store = FailureRateStore(prior=0.5, min_samples=5)
        est = HeuristicDifficultyEstimator(failure_store=store)
        
        budget = BudgetTracker(
            initial_budget_usd=100.0,
            rate_card=DEFAULT_RATE_CARD,
            mode=BudgetMode.SOFT,
        )
        
        config = RouterConfig(threshold=0.5)
        
        router = RouterNode(
            cheap_model=cheap,
            expensive_model=expensive,
            estimator=est,
            budget=budget,
            config=config,
        )
        
        state = {
            "step_id": "step_1",
            "prompt": "short prompt",
        }
        
        # Warmup
        router(state)
        
        # Since MockChatModel invoke is essentially instantaneous, 
        # the entire router call time is effectively overhead.
        t0 = time.perf_counter()
        for _ in range(50):
            router(state)
        t1 = time.perf_counter()
        
        avg_overhead_ms = ((t1 - t0) * 1000) / 50
        assert avg_overhead_ms < 50.0
