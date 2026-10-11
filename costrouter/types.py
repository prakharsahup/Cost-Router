"""Core data types, enums, protocols, and configuration for CostRouter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol


class ModelTier(str, Enum):
    """Which model tier a step is routed to."""

    CHEAP = "cheap"
    EXPENSIVE = "expensive"


class BudgetMode(str, Enum):
    """What happens when the budget is exhausted.

    HARD_STOP: halt graph execution entirely.
    SOFT: force cheap-only routing for all remaining steps.
    """

    HARD_STOP = "hard_stop"
    SOFT = "soft"


@dataclass
class ModelResponse:
    """Uniform wrapper around any provider's chat completion response.

    ``cost_usd`` may be ``None`` when a provider doesn't report cost;
    in that case ``BudgetTracker`` falls back to the rate-card estimator
    (REQ-BUDGET-6).
    """

    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class ChatModelLike(Protocol):
    """Structural protocol any model wrapper must satisfy."""

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse: ...


@dataclass
class DifficultyEstimate:
    """Result of a difficulty estimation pass.

    ``score`` is normalised to [0, 1] (REQ-DIFF-4).
    ``latency_ms`` is wall-clock time of the estimation itself.
    """

    score: float
    confidence: Optional[float] = None
    latency_ms: float = 0.0


@dataclass
class RoutingDecision:
    """Complete audit record of a single routing decision (REQ-ROUTER-4)."""

    step_id: str
    task_id: Optional[str]
    prompt_length: int
    prompt_hash: str
    difficulty_score: float
    budget_remaining_before: float
    budget_remaining_after: float
    model_chosen: ModelTier
    budget_forced: bool
    retried: bool
    actual_cost_usd: float
    latency_ms: float
    estimator_latency_ms: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

PolicyFn = Callable[[float, float, "RouterConfig"], ModelTier]
"""(difficulty_score, remaining_budget, config) → model tier."""

Validator = Callable[[ModelResponse], bool]
"""Return True if the model response is acceptable."""


@dataclass
class RouterConfig:
    """Knobs exposed to the caller for tuning routing behaviour.

    Budget mode and warning thresholds are owned by ``BudgetTracker`` — the
    tracker is the single source of truth for budget behaviour.

    Attributes:
        threshold: difficulty score above which the expensive model is used.
        budget_mode: optional assertion of the expected budget mode.  If set,
            ``RouterNode`` raises ``ValueError`` when it disagrees with the
            ``BudgetTracker``'s mode.  ``None`` (default) uses the tracker's.
        retry_on_error: whether to retry-escalate on malformed output (REQ-ROUTER-6).
        validator: optional callable that checks model output quality.
        max_completion_tokens: the ``max_tokens`` limit you configure on your
            providers.  When set, the pre-flight affordability check prices
            this many completion tokens (a worst case) instead of guessing
            "completion ≈ prompt size", so a call can't overspend by more than
            the prompt-token estimation error.  ``None`` keeps the guess.
    """

    threshold: float = 0.5
    budget_mode: Optional[BudgetMode] = None
    retry_on_error: bool = True
    validator: Optional[Validator] = None
    max_completion_tokens: Optional[int] = None
