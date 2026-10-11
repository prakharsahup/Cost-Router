"""Tests for RouterNode + threshold_policy (REQ-ROUTER-1..6).

Covers:
- Basic routing: cheap for low difficulty, expensive for high (ROUTER-1, ROUTER-2)
- Budget-forced override: expensive downgraded to cheap when unaffordable (ROUTER-3)
- Fail-open: estimator exception → fall back to cheap (ROUTER-5)
- Retry-escalate: malformed cheap output → retry with expensive (ROUTER-6)
- No infinite retry: expensive failure is not retried (ROUTER-6)
- Hard-stop mode raises BudgetExhaustedError (BUDGET-1)
- Decision audit fields are populated correctly (ROUTER-4)
"""

from __future__ import annotations

import pytest

from costrouter.types import (
    BudgetMode,
    DifficultyEstimate,
    ModelResponse,
    ModelTier,
    RouterConfig,
    RoutingDecision,
)
from costrouter.budget import BudgetTracker
from costrouter.difficulty import DifficultyEstimator, FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode, threshold_policy
from costrouter.exceptions import BudgetExhaustedError, MalformedOutputError
from conftest import DEFAULT_RATE_CARD, MockChatModel, ok_response, error_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedScoreEstimator(DifficultyEstimator):
    """Always returns a fixed difficulty score — useful for deterministic tests."""
    def __init__(self, score: float):
        self._score = score

    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate:
        return DifficultyEstimate(score=self._score, latency_ms=0.1)


class ExplodingEstimator(DifficultyEstimator):
    """Always raises — used to test REQ-ROUTER-5 fail-open behaviour."""
    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate:
        raise RuntimeError("estimator crashed")


def _make_state(prompt: str = "test prompt", step_id: str = "step_1",
                step_type: str = "default", task_id: str = "task_1") -> dict:
    return {"prompt": prompt, "step_id": step_id, "step_type": step_type, "task_id": task_id}


def _build_router(
    cheap_responses=None, expensive_responses=None,
    estimator=None, budget_usd=10.0,
    config=None, budget_mode=BudgetMode.SOFT,
):
    cheap = MockChatModel(cheap_responses or [ok_response(text="cheap answer")])
    expensive = MockChatModel(expensive_responses or [ok_response(text="expensive answer")])
    est = estimator or FixedScoreEstimator(0.3)
    budget = BudgetTracker(
        initial_budget_usd=budget_usd,
        rate_card=DEFAULT_RATE_CARD,
        mode=budget_mode,
    )
    cfg = config or RouterConfig(budget_mode=budget_mode)
    router = RouterNode(
        cheap_model=cheap, expensive_model=expensive,
        estimator=est, budget=budget, config=cfg,
    )
    return router, cheap, expensive, budget


# ---------------------------------------------------------------------------
# Basic routing (ROUTER-1, ROUTER-2)
# ---------------------------------------------------------------------------

class TestBasicRouting:
    def test_low_difficulty_routes_to_cheap(self):
        router, cheap, expensive, _ = _build_router(estimator=FixedScoreEstimator(0.3))
        result = router(_make_state())
        assert cheap.call_count == 1
        assert expensive.call_count == 0
        assert result["response"] == "cheap answer"
        assert result["last_decision"].model_chosen == ModelTier.CHEAP

    def test_high_difficulty_routes_to_expensive(self):
        router, cheap, expensive, _ = _build_router(
            estimator=FixedScoreEstimator(0.8),
            expensive_responses=[ok_response(text="expensive answer")],
        )
        result = router(_make_state())
        assert expensive.call_count == 1
        assert cheap.call_count == 0
        assert result["response"] == "expensive answer"
        assert result["last_decision"].model_chosen == ModelTier.EXPENSIVE

    def test_threshold_boundary_exact(self):
        """Score exactly equal to threshold → should route to cheap (strictly > for expensive)."""
        router, cheap, expensive, _ = _build_router(
            estimator=FixedScoreEstimator(0.5),
            config=RouterConfig(threshold=0.5),
        )
        result = router(_make_state())
        assert result["last_decision"].model_chosen == ModelTier.CHEAP

    def test_custom_threshold(self):
        router, cheap, expensive, _ = _build_router(
            estimator=FixedScoreEstimator(0.3),
            config=RouterConfig(threshold=0.2),
            expensive_responses=[ok_response(text="expensive answer")],
        )
        result = router(_make_state())
        assert result["last_decision"].model_chosen == ModelTier.EXPENSIVE


# ---------------------------------------------------------------------------
# Budget-forced override (ROUTER-3)
# ---------------------------------------------------------------------------

class TestBudgetForced:
    def test_expensive_downgraded_when_unaffordable(self):
        """If policy says expensive but budget can't afford it → force cheap."""
        router, cheap, expensive, budget = _build_router(
            estimator=FixedScoreEstimator(0.9),  # wants expensive
            budget_usd=0.00001,  # barely any budget
        )
        result = router(_make_state())
        assert result["last_decision"].model_chosen == ModelTier.CHEAP
        assert result["last_decision"].budget_forced is True
        assert cheap.call_count == 1
        assert expensive.call_count == 0

    def test_not_budget_forced_when_affordable(self):
        router, cheap, expensive, _ = _build_router(
            estimator=FixedScoreEstimator(0.9),
            budget_usd=100.0,
            expensive_responses=[ok_response(text="expensive answer")],
        )
        result = router(_make_state())
        assert result["last_decision"].budget_forced is False


# ---------------------------------------------------------------------------
# Hard-stop mode (BUDGET-1)
# ---------------------------------------------------------------------------

class TestHardStop:
    def test_hard_stop_raises_when_exhausted(self):
        router, _, _, budget = _build_router(
            budget_usd=0.001,
            budget_mode=BudgetMode.HARD_STOP,
            estimator=FixedScoreEstimator(0.1),  # wants cheap
        )
        # Exhaust the budget
        budget.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.01))
        assert budget.exhausted()
        with pytest.raises(BudgetExhaustedError):
            router(_make_state())

    def test_soft_mode_does_not_raise(self):
        router, _, _, budget = _build_router(
            budget_usd=0.001,
            budget_mode=BudgetMode.SOFT,
        )
        budget.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.01))
        # Should not raise — just routes to cheap
        result = router(_make_state())
        assert result["last_decision"].model_chosen == ModelTier.CHEAP


# ---------------------------------------------------------------------------
# Fail-open (ROUTER-5)
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_estimator_exception_falls_back_to_cheap(self):
        router, cheap, expensive, _ = _build_router(
            estimator=ExplodingEstimator(),
        )
        result = router(_make_state())
        assert result["last_decision"].model_chosen == ModelTier.CHEAP
        assert result["last_decision"].difficulty_score == 0.0
        assert cheap.call_count == 1


# ---------------------------------------------------------------------------
# Retry-escalate (ROUTER-6)
# ---------------------------------------------------------------------------

class TestRetryEscalate:
    def test_malformed_cheap_retries_with_expensive(self):
        """Cheap model returns empty text → validator fails → retry with expensive."""
        router, cheap, expensive, _ = _build_router(
            cheap_responses=[ModelResponse(text="", error="bad output")],
            expensive_responses=[ok_response(text="good answer")],
            estimator=FixedScoreEstimator(0.3),
        )
        result = router(_make_state())
        assert result["last_decision"].retried is True
        assert result["last_decision"].model_chosen == ModelTier.EXPENSIVE
        assert result["response"] == "good answer"

    def test_expensive_failure_is_not_retried(self):
        """If expensive model fails, it should NOT retry — raise instead."""
        router, cheap, expensive, _ = _build_router(
            estimator=FixedScoreEstimator(0.9),
            expensive_responses=[ModelResponse(text="", error="bad")],
            budget_usd=100.0,
        )
        with pytest.raises(MalformedOutputError):
            router(_make_state())

    def test_retry_disabled(self):
        """With retry_on_error=False, malformed output raises immediately."""
        router, cheap, expensive, _ = _build_router(
            cheap_responses=[ModelResponse(text="", error="bad")],
            config=RouterConfig(retry_on_error=False),
        )
        with pytest.raises(MalformedOutputError):
            router(_make_state())

    def test_custom_validator(self):
        """Custom validator can reject responses that pass the default check."""
        # Validator requires the word "ANSWER"
        def strict_validator(r: ModelResponse) -> bool:
            return "ANSWER" in r.text

        router, cheap, expensive, _ = _build_router(
            cheap_responses=[ok_response(text="some text without keyword")],
            expensive_responses=[ok_response(text="ANSWER: 42")],
            config=RouterConfig(validator=strict_validator),
        )
        result = router(_make_state())
        assert result["last_decision"].retried is True
        assert result["response"] == "ANSWER: 42"


# ---------------------------------------------------------------------------
# Decision audit record (ROUTER-4)
# ---------------------------------------------------------------------------

class TestDecisionAudit:
    def test_decision_fields_populated(self):
        router, _, _, _ = _build_router(
            estimator=FixedScoreEstimator(0.3),
            budget_usd=10.0,
        )
        result = router(_make_state(step_id="s1", task_id="t1"))
        d: RoutingDecision = result["last_decision"]
        assert d.step_id == "s1"
        assert d.task_id == "t1"
        assert d.difficulty_score == pytest.approx(0.3)
        assert d.budget_remaining_before == 10.0
        assert d.actual_cost_usd > 0
        assert d.latency_ms >= 0
        assert d.estimator_latency_ms >= 0
        assert d.error is None

    def test_budget_remaining_decreases_after_call(self):
        router, _, _, _ = _build_router(budget_usd=10.0)
        result = router(_make_state())
        d = result["last_decision"]
        assert d.budget_remaining_after < d.budget_remaining_before


# ---------------------------------------------------------------------------
# threshold_policy
# ---------------------------------------------------------------------------

class TestThresholdPolicy:
    def test_below_threshold_returns_cheap(self):
        assert threshold_policy(0.3, 10.0, RouterConfig(threshold=0.5)) == ModelTier.CHEAP

    def test_above_threshold_returns_expensive(self):
        assert threshold_policy(0.7, 10.0, RouterConfig(threshold=0.5)) == ModelTier.EXPENSIVE

    def test_equal_to_threshold_returns_cheap(self):
        assert threshold_policy(0.5, 10.0, RouterConfig(threshold=0.5)) == ModelTier.CHEAP


# ---------------------------------------------------------------------------
# Regression tests: retry-escalate must respect cost accounting and budget
# ---------------------------------------------------------------------------

class TestRetryAccounting:
    def test_failed_cheap_call_cost_is_charged(self):
        """A rejected cheap response still cost money — both calls are charged."""
        router, _, _, budget = _build_router(
            cheap_responses=[ok_response(text="", cost=0.01)],
            expensive_responses=[ok_response(text="good", cost=0.02)],
            budget_usd=1.0,
        )
        d = router(_make_state())["last_decision"]
        assert d.actual_cost_usd == pytest.approx(0.03)
        assert budget.remaining_budget() == pytest.approx(0.97)

    def test_retry_records_cheap_failure_in_decision(self):
        router, _, _, _ = _build_router(
            cheap_responses=[ModelResponse(text="", error="bad output")],
            expensive_responses=[ok_response(text="good")],
        )
        d = router(_make_state())["last_decision"]
        assert d.retried is True
        assert "MalformedOutputError" in d.error

    def test_retry_does_not_escalate_when_unaffordable(self):
        """If the pre-flight says expensive is unaffordable, the retry must not call it."""
        router, _, expensive, budget = _build_router(
            cheap_responses=[error_response()],
            expensive_responses=[ok_response(cost=5.0)],
            budget_usd=0.00001,  # below the ~$0.00008 pre-flight estimate
        )
        with pytest.raises(MalformedOutputError):
            router(_make_state())
        assert expensive.call_count == 0
        assert budget.remaining_budget() >= 0

    def test_retry_hard_stop_when_cheap_call_exhausts_budget(self):
        router, _, expensive, _ = _build_router(
            cheap_responses=[ok_response(text="", cost=1.0)],
            budget_usd=0.5,
            budget_mode=BudgetMode.HARD_STOP,
        )
        with pytest.raises(BudgetExhaustedError):
            router(_make_state())
        assert expensive.call_count == 0

    def test_expensive_retry_output_is_validated(self):
        """If the escalated call also fails, raise — don't return an empty answer."""
        router, _, _, budget = _build_router(
            cheap_responses=[error_response()],
            expensive_responses=[ModelResponse(text="", error="boom", cost_usd=0.02)],
            budget_usd=1.0,
        )
        with pytest.raises(MalformedOutputError):
            router(_make_state())
        # The failed expensive call was still paid for.
        assert budget.remaining_budget() == pytest.approx(0.98)

    def test_cheap_exception_without_response_escalates(self):
        class RaisingModel:
            def invoke(self, prompt, **kwargs):
                raise ConnectionError("network down")

        budget = BudgetTracker(10.0, DEFAULT_RATE_CARD)
        router = RouterNode(
            cheap_model=RaisingModel(),
            expensive_model=MockChatModel([ok_response(text="rescued", cost=0.05)]),
            estimator=FixedScoreEstimator(0.1), budget=budget, config=RouterConfig(),
        )
        result = router(_make_state())
        assert result["response"] == "rescued"
        assert result["last_decision"].actual_cost_usd == pytest.approx(0.05)
        assert "ConnectionError" in result["last_decision"].error


class TestBudgetModeSource:
    def test_soft_mode_exhausted_forces_cheap(self):
        router, cheap, expensive, budget = _build_router(
            estimator=FixedScoreEstimator(0.9), budget_usd=1.0,
        )
        budget.record_actual_cost(ModelTier.CHEAP, ok_response(cost=2.0))
        d = router(_make_state())["last_decision"]
        assert d.model_chosen == ModelTier.CHEAP
        assert d.budget_forced is True
        assert expensive.call_count == 0

    def test_tracker_mode_is_used_when_config_mode_unset(self):
        budget = BudgetTracker(0.001, DEFAULT_RATE_CARD, mode=BudgetMode.HARD_STOP)
        budget.record_actual_cost(ModelTier.CHEAP, ok_response(cost=0.01))
        router = RouterNode(
            cheap_model=MockChatModel(), expensive_model=MockChatModel(),
            estimator=FixedScoreEstimator(0.1), budget=budget, config=RouterConfig(),
        )
        with pytest.raises(BudgetExhaustedError):
            router(_make_state())

    def test_mismatched_budget_mode_is_rejected(self):
        budget = BudgetTracker(1.0, DEFAULT_RATE_CARD, mode=BudgetMode.SOFT)
        with pytest.raises(ValueError, match="budget_mode"):
            RouterNode(
                cheap_model=MockChatModel(), expensive_model=MockChatModel(),
                estimator=FixedScoreEstimator(0.1), budget=budget,
                config=RouterConfig(budget_mode=BudgetMode.HARD_STOP),
            )


class TestWorstCasePreflight:
    def test_max_completion_tokens_blocks_expensive_that_guess_would_allow(self):
        """With a worst-case completion size the pre-flight downgrades to cheap."""
        prompt = "x" * 40  # ~10 prompt tokens
        # Guess (10 + 10 tokens): 10*1e-5 + 10*3e-5 = $0.0004 -> affordable
        # Worst case (10 + 1000): 10*1e-5 + 1000*3e-5 = $0.0301 -> not affordable
        for max_tokens, expected in [(None, ModelTier.EXPENSIVE), (1000, ModelTier.CHEAP)]:
            router, _, _, _ = _build_router(
                estimator=FixedScoreEstimator(0.9), budget_usd=0.01,
                config=RouterConfig(max_completion_tokens=max_tokens),
            )
            d = router(_make_state(prompt=prompt))["last_decision"]
            assert d.model_chosen == expected

    def test_hard_stop_refuses_unaffordable_cheap_call(self):
        router, cheap, _, _ = _build_router(
            estimator=FixedScoreEstimator(0.1), budget_usd=0.001,
            budget_mode=BudgetMode.HARD_STOP,
            config=RouterConfig(max_completion_tokens=1000),  # cheap worst case $0.002
        )
        with pytest.raises(BudgetExhaustedError):
            router(_make_state())
        assert cheap.call_count == 0

    def test_soft_mode_still_runs_cheap_when_unaffordable(self):
        router, cheap, _, _ = _build_router(
            estimator=FixedScoreEstimator(0.1), budget_usd=0.001,
            config=RouterConfig(max_completion_tokens=1000),
        )
        router(_make_state())
        assert cheap.call_count == 1


class TestFailureStoreFeedback:
    def _router(self, cheap_responses, store):
        return RouterNode(
            cheap_model=MockChatModel(cheap_responses),
            expensive_model=MockChatModel([ok_response(text="fixed")] * 5),
            estimator=FixedScoreEstimator(0.1),
            budget=BudgetTracker(10.0, DEFAULT_RATE_CARD),
            config=RouterConfig(),
            failure_store=store,
        )

    def test_first_attempt_outcomes_are_recorded(self):
        store = FailureRateStore(prior=0.5, min_samples=4)
        router = self._router(
            [ok_response(), error_response(), ok_response(), error_response()], store,
        )
        for i in range(4):
            router(_make_state(step_id=f"s{i}", step_type="qa"))
        assert store.failure_rate("qa") == pytest.approx(0.5)
        assert store.failure_rate("other") == 0.5  # untouched -> prior

    def test_no_store_means_no_recording(self):
        router = self._router([error_response()], None)
        assert router(_make_state())["last_decision"].retried is True
