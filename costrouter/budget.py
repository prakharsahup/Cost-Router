"""BudgetTracker — USD-denominated ledger for LLM spend (REQ-BUDGET-1..7).

The ledger's source of truth is **USD**, decremented by the *actual* cost of
each call (from provider usage metadata, converted via a per-model rate card).
Token counting is used only as the REQ-BUDGET-6 fallback estimator when a
provider doesn't return usage metadata — never as the primary unit.

Thread-safety (REQ-BUDGET-7) is provided via ``threading.Lock`` for sync /
threaded concurrency.  This project targets synchronous LangGraph execution
for v1.0 (see Implementation Plan §0, Decision #5).
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from costrouter.types import BudgetMode, ModelResponse, ModelTier


class BudgetTracker:
    """Owns the USD budget ledger, thresholds, and hard/soft stop logic.

    Args:
        initial_budget_usd: Starting budget in USD.
        rate_card: Per-token USD rates keyed by ``ModelTier``.  Each entry is
            ``{"prompt": <float>, "completion": <float>}``.  Used **only** as
            the fallback when a provider response has ``cost_usd is None``.
        mode: ``HARD_STOP`` halts execution; ``SOFT`` forces cheap-only.
        warn_thresholds_pct: Budget-% levels at which to fire the callback
            (REQ-BUDGET-4).  E.g. ``(0.5, 0.1)`` fires at 50 % and 10 %.
        on_threshold: Optional callback ``(threshold_pct, remaining_usd) → None``
            invoked **once** per threshold crossing.  It runs *after* the
            internal lock is released, so it may safely call back into the
            tracker (e.g. ``remaining_budget()``).
    """

    def __init__(
        self,
        initial_budget_usd: float,
        rate_card: dict[ModelTier, dict[str, float]],
        mode: BudgetMode = BudgetMode.SOFT,
        warn_thresholds_pct: tuple[float, ...] = (0.5, 0.1),
        on_threshold: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        self._initial = initial_budget_usd
        self._remaining = initial_budget_usd
        self._rate_card = rate_card
        self._mode = mode
        self._warn_thresholds = sorted(warn_thresholds_pct, reverse=True)
        self._fired: set[float] = set()
        self._on_threshold = on_threshold
        self._lock = threading.Lock()

    # -- public read-only properties ----------------------------------------

    @property
    def mode(self) -> BudgetMode:
        """Return the configured budget mode (SOFT or HARD_STOP)."""
        return self._mode

    @property
    def initial_budget(self) -> float:
        """Return the starting budget in USD."""
        return self._initial

    def remaining_budget(self) -> float:
        """Return the current remaining budget in USD (thread-safe read)."""
        with self._lock:
            return self._remaining

    def exhausted(self) -> bool:
        """Return ``True`` when remaining budget ≤ 0."""
        return self.remaining_budget() <= 0

    # -- pre-flight affordability check -------------------------------------

    def can_afford(
        self,
        tier: ModelTier,
        est_prompt_tokens: int,
        est_completion_tokens: int,
    ) -> bool:
        """Estimate whether the budget can cover a call of the given tier.

        Uses the rate card to convert estimated tokens → USD and compares
        against the current remaining budget.  This is a *crude* pre-flight
        check (see Plan §4.1) — not meant to be precise.
        """
        rates = self._rate_card[tier]
        est_cost = (
            est_prompt_tokens * rates["prompt"]
            + est_completion_tokens * rates["completion"]
        )
        return self.remaining_budget() >= est_cost

    # -- recording actual spend ---------------------------------------------

    def record_actual_cost(
        self, tier: ModelTier, response: ModelResponse
    ) -> float:
        """Decrement the ledger by the actual cost of *response*.

        If ``response.cost_usd`` is provided by the model provider, that value
        is used directly.  Otherwise falls back to the rate-card estimator
        (REQ-BUDGET-6).

        Returns:
            The USD cost that was deducted.
        """
        with self._lock:
            cost = response.cost_usd
            if cost is None:  # REQ-BUDGET-6 fallback
                rates = self._rate_card[tier]
                cost = (
                    (response.prompt_tokens or 0) * rates["prompt"]
                    + (response.completion_tokens or 0) * rates["completion"]
                )
            self._remaining -= cost
            crossed = self._newly_crossed_thresholds()
            remaining = self._remaining

        # Fire callbacks outside the lock: a callback that re-enters the
        # tracker would otherwise deadlock on the non-reentrant lock.
        if self._on_threshold:
            for t in crossed:
                self._on_threshold(t, remaining)
        return cost

    # -- internal: threshold warnings ---------------------------------------

    def _newly_crossed_thresholds(self) -> list[float]:
        """Mark and return any newly crossed budget-% thresholds.

        Must be called while ``self._lock`` is held.
        """
        pct = self._remaining / self._initial if self._initial else 0
        crossed = []
        for t in self._warn_thresholds:
            if pct <= t and t not in self._fired:
                self._fired.add(t)
                crossed.append(t)
        return crossed
