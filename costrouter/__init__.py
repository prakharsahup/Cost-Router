"""CostRouter — Cost-aware LLM routing for LangGraph agents."""

from costrouter.types import (
    ModelTier,
    BudgetMode,
    ModelResponse,
    DifficultyEstimate,
    RoutingDecision,
    RouterConfig,
)
from costrouter.exceptions import BudgetExhaustedError, MalformedOutputError

__all__ = [
    "ModelTier",
    "BudgetMode",
    "ModelResponse",
    "DifficultyEstimate",
    "RoutingDecision",
    "RouterConfig",
    "BudgetExhaustedError",
    "MalformedOutputError",
]


def __getattr__(name: str):
    """Lazy imports for modules built in later phases."""
    if name == "BudgetTracker":
        from costrouter.budget import BudgetTracker
        return BudgetTracker
    if name in ("DifficultyEstimator", "FailureRateStore", "HeuristicDifficultyEstimator"):
        import costrouter.difficulty as _d
        return getattr(_d, name)
    if name in ("RouterNode", "threshold_policy"):
        import costrouter.router as _r
        return getattr(_r, name)
    if name == "DecisionLogger":
        from costrouter.store import DecisionLogger
        return DecisionLogger
    if name in ("LangChainChatModel", "OpenAICompatibleChatModel"):
        import costrouter.adapters as _a
        return getattr(_a, name)
    raise AttributeError(f"module 'costrouter' has no attribute {name!r}")
