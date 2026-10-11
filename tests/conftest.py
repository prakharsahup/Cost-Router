"""Shared test fixtures — MockChatModel and helpers."""

from __future__ import annotations

from collections import deque
from typing import Any

from costrouter.types import ModelResponse, ModelTier


class MockChatModel:
    """A fake model that returns canned responses from a queue.

    Implements ``ChatModelLike`` without making any network calls,
    satisfying SRS §5.4's testability requirement.
    """

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._queue: deque[ModelResponse] = deque(responses or [])
        self.call_count = 0
        self.last_prompt: str | None = None

    def enqueue(self, *responses: ModelResponse) -> None:
        """Add one or more canned responses to the end of the queue."""
        self._queue.extend(responses)

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse:
        self.call_count += 1
        self.last_prompt = prompt
        if not self._queue:
            return ModelResponse(text="default mock response")
        return self._queue.popleft()


# ---------------------------------------------------------------------------
# Convenience fixtures
# ---------------------------------------------------------------------------

DEFAULT_RATE_CARD: dict[ModelTier, dict[str, float]] = {
    ModelTier.CHEAP: {"prompt": 0.000001, "completion": 0.000002},
    ModelTier.EXPENSIVE: {"prompt": 0.00001, "completion": 0.00003},
}
"""Per-token USD rates used across tests.  Cheap ≈ 10× cheaper than expensive."""


def ok_response(
    text: str = "ok",
    cost: float | None = 0.001,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> ModelResponse:
    """Build a simple successful ``ModelResponse``."""
    return ModelResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )


def error_response(error: str = "provider error") -> ModelResponse:
    """Build a ``ModelResponse`` with an error flag set."""
    return ModelResponse(text="", error=error)
