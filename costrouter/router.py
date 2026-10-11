"""RouterNode — LangGraph node wiring difficulty + budget into routing (REQ-ROUTER-1..6).

``RouterNode`` is a callable class suitable for use as a LangGraph
``StateGraph`` node.  Dependencies (estimator, budget, models) are injected
at construction; the ``__call__`` signature matches LangGraph's
``(state, config?) → partial_state_update`` contract.

Register with ``builder.add_node("router", router_instance)`` — don't wrap
it in an extra lambda; a callable instance is a first-class node.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from costrouter.types import (
    BudgetMode,
    ChatModelLike,
    DifficultyEstimate,
    ModelTier,
    ModelResponse,
    PolicyFn,
    RouterConfig,
    RoutingDecision,
    Validator,
)
from costrouter.budget import BudgetTracker
from costrouter.difficulty import DifficultyEstimator, FailureRateStore
from costrouter.exceptions import BudgetExhaustedError, MalformedOutputError


# ---------------------------------------------------------------------------
# Built-in policy (REQ-ROUTER-2)
# ---------------------------------------------------------------------------

def threshold_policy(
    difficulty: float, remaining_budget: float, config: RouterConfig
) -> ModelTier:
    """Route to expensive only when difficulty *strictly exceeds* the threshold.

    This is the default policy satisfying REQ-ROUTER-2.
    """
    return ModelTier.EXPENSIVE if difficulty > config.threshold else ModelTier.CHEAP


# ---------------------------------------------------------------------------
# RouterNode
# ---------------------------------------------------------------------------

class RouterNode:
    """LangGraph-compatible node that routes each step to cheap or expensive.

    Args:
        cheap_model: model wrapper for the cheap tier.
        expensive_model: model wrapper for the expensive tier.
        estimator: difficulty scorer.
        budget: shared ``BudgetTracker`` instance.
        config: routing knobs (threshold, budget mode, validator, …).
        policy: ``(difficulty, remaining_budget, config) → ModelTier``.
        logger: optional ``DecisionLogger`` for persistence (Phase 4).
        failure_store: optional ``FailureRateStore`` (normally the one your
            estimator reads).  The router records whether each step's *first*
            attempt failed validation — a signal available in production,
            unlike graded correctness — so the history signal learns online.
    """

    def __init__(
        self,
        cheap_model: ChatModelLike,
        expensive_model: ChatModelLike,
        estimator: DifficultyEstimator,
        budget: BudgetTracker,
        config: RouterConfig,
        policy: PolicyFn = threshold_policy,
        logger: Optional[object] = None,  # DecisionLogger, typed loosely to avoid circular import
        failure_store: Optional[FailureRateStore] = None,
    ) -> None:
        if config.budget_mode is not None and config.budget_mode != budget.mode:
            raise ValueError(
                f"RouterConfig.budget_mode={config.budget_mode.value!r} disagrees "
                f"with BudgetTracker.mode={budget.mode.value!r}; the tracker "
                "owns the budget mode — set it there."
            )
        self._cheap = cheap_model
        self._expensive = expensive_model
        self._estimator = estimator
        self._budget = budget
        self._config = config
        self._policy = policy
        self._logger = logger
        self._failure_store = failure_store

    def __call__(self, state: dict, config: Optional["RunnableConfig"] = None) -> dict:
        """Execute one routing decision.

        Reads ``state["step_id"]``, ``state["prompt"]``, and optionally
        ``state["step_type"]`` and ``state["task_id"]``.

        Returns a partial state update with ``response`` (str) and
        ``last_decision`` (``RoutingDecision``).
        """
        step_id: str = state["step_id"]
        prompt: str = state["prompt"]
        step_type: str = state.get("step_type", "default")

        # ---- REQ-ROUTER-5: fail open to cheap if the estimator throws ----
        try:
            est = self._estimator.estimate(prompt, step_type)
        except Exception:
            est = DifficultyEstimate(score=0.0, latency_ms=0.0)

        # ---- REQ-BUDGET-1: hard-stop check ----
        self._check_hard_stop(step_id)

        remaining_before = self._budget.remaining_budget()
        tier = self._policy(est.score, remaining_before, self._config)

        # ---- REQ-ROUTER-3 / REQ-BUDGET-1 (SOFT): budget-forced override ----
        # SOFT mode: an exhausted budget forces cheap-only routing.  Otherwise
        # downgrade when the expensive call is (crudely) unaffordable.
        budget_forced = False
        if tier == ModelTier.EXPENSIVE and (
            self._budget.exhausted() or not self._can_afford(ModelTier.EXPENSIVE, prompt)
        ):
            tier = ModelTier.CHEAP
            budget_forced = True

        # HARD_STOP is a cap: refuse even a cheap call the budget can't cover.
        if self._budget.mode == BudgetMode.HARD_STOP and not self._can_afford(tier, prompt):
            raise BudgetExhaustedError(step_id)

        # ---- Model call + REQ-ROUTER-6 retry-escalate ----
        model = self._cheap if tier == ModelTier.CHEAP else self._expensive
        retried = False
        error: Optional[str] = None
        cost = 0.0
        validator: Validator = self._config.validator or (
            lambda r: bool(r.text) and not r.error
        )

        t0 = time.perf_counter()
        try:
            response = model.invoke(prompt)
            # Charge every call that returned, even one we then reject.
            cost += self._budget.record_actual_cost(tier, response)
            if not validator(response):
                raise MalformedOutputError(step_id)
            self._record_outcome(step_type, failed=False)
        except Exception as exc:
            self._record_outcome(step_type, failed=True)
            if not (self._config.retry_on_error and tier == ModelTier.CHEAP):
                raise
            # Escalation is still subject to the budget: never let a retry
            # overspend.  Surface the original cheap-tier failure instead.
            self._check_hard_stop(step_id)
            if self._budget.exhausted() or not self._can_afford(ModelTier.EXPENSIVE, prompt):
                raise
            error = f"{type(exc).__name__}: {exc}"
            tier = ModelTier.EXPENSIVE
            retried = True
            response = self._expensive.invoke(prompt)
            cost += self._budget.record_actual_cost(tier, response)
            # The expensive tier is never retried — a bad answer is an error.
            if not validator(response):
                raise MalformedOutputError(step_id) from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        decision = RoutingDecision(
            step_id=step_id,
            task_id=state.get("task_id"),
            prompt_length=len(prompt),
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            difficulty_score=est.score,
            budget_remaining_before=remaining_before,
            budget_remaining_after=self._budget.remaining_budget(),
            model_chosen=tier,
            budget_forced=budget_forced,
            retried=retried,
            actual_cost_usd=cost,
            latency_ms=latency_ms,
            estimator_latency_ms=est.latency_ms,
            error=error,
        )

        # ---- REQ-ROUTER-4: log the decision ----
        if self._logger and hasattr(self._logger, "log"):
            self._logger.log(decision)

        return {"response": response.text, "last_decision": decision}

    # -- helpers --------------------------------------------------------------

    def _check_hard_stop(self, step_id: str) -> None:
        """Raise ``BudgetExhaustedError`` if HARD_STOP mode and budget is gone."""
        if self._budget.mode == BudgetMode.HARD_STOP and self._budget.exhausted():
            raise BudgetExhaustedError(step_id)

    def _can_afford(self, tier: ModelTier, prompt: str) -> bool:
        """Pre-flight: ~4 chars/token for the prompt; completion is priced at
        ``config.max_completion_tokens`` (worst case) or, if unset, guessed
        as ≈ prompt size."""
        prompt_tokens = max(len(prompt) // 4, 1)
        completion_tokens = self._config.max_completion_tokens or prompt_tokens
        return self._budget.can_afford(tier, prompt_tokens, completion_tokens)

    def _record_outcome(self, step_type: str, *, failed: bool) -> None:
        if self._failure_store is not None:
            self._failure_store.record(step_type, failed=failed)
