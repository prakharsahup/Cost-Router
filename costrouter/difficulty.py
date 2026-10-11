"""DifficultyEstimator — score an upcoming step 0–1 (REQ-DIFF-1..4).

Pure local computation → effectively free and well under the 200 ms bound
by construction, since it never makes a network call (REQ-DIFF-3).

The ``DifficultyEstimator`` ABC is intentionally left open for a future
learned model (REQ-DIFF-2 restricts v1.0 to heuristics only).
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

from costrouter.types import DifficultyEstimate

# ---------------------------------------------------------------------------
# Reasoning-cue pattern (REQ-DIFF-1, signal #2)
# ---------------------------------------------------------------------------

REASONING_CUES = re.compile(
    r"\b(step by step|calculate|prove|derive|explain why|multi-step|compare"
    r"|analy[sz]e|justify|solve|optimi[sz]e|trade-?offs?)\b",
    re.IGNORECASE,
)

#: Number of distinct cues at which the cue signal saturates at 1.0.  A single
#: incidental keyword ("compare", "solve") only contributes half the signal,
#: so one word can't push an otherwise trivial prompt to the expensive tier.
CUE_SATURATION = 2


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DifficultyEstimator(ABC):
    """Interface for difficulty scoring.

    Subclasses must implement ``estimate()`` and return a ``DifficultyEstimate``
    with ``score`` normalised to [0, 1].
    """

    @abstractmethod
    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate:
        """Score the difficulty of *prompt* for the given *step_type*."""
        ...


# ---------------------------------------------------------------------------
# Failure-rate store (Decision #4 — cold-start handling)
# ---------------------------------------------------------------------------

class FailureRateStore:
    """Track per-step-type historical failure rates.

    Cold-start handled with a configurable *prior* (default 0.5) until
    *min_samples* observations are recorded for a step type, then switches
    to the observed rate.  Document this in benchmarking runs — early-run
    difficulty scores will look artificially uniform.
    """

    def __init__(self, prior: float = 0.5, min_samples: int = 5) -> None:
        self._prior = prior
        self._min_samples = min_samples
        # step_type → (successes, failures)
        self._counts: dict[str, tuple[int, int]] = {}

    def failure_rate(self, step_type: str) -> float:
        """Return the failure rate for *step_type*.

        Returns the prior if fewer than ``min_samples`` observations exist.
        """
        s, f = self._counts.get(step_type, (0, 0))
        total = s + f
        if total < self._min_samples or total == 0:
            return self._prior
        return f / total

    def record(self, step_type: str, *, failed: bool) -> None:
        """Record an observation for *step_type*."""
        s, f = self._counts.get(step_type, (0, 0))
        if failed:
            self._counts[step_type] = (s, f + 1)
        else:
            self._counts[step_type] = (s + 1, f)


# ---------------------------------------------------------------------------
# Heuristic estimator (v1.0 implementation)
# ---------------------------------------------------------------------------

class HeuristicDifficultyEstimator(DifficultyEstimator):
    """Three-signal heuristic scorer (REQ-DIFF-1).

    Signals (all normalised to [0, 1] independently):
        1. **Prompt length** — longer prompts are assumed harder.
        2. **Reasoning cues** — distinct keywords like "step by step",
           saturating at ``CUE_SATURATION`` cues.
        3. **Historical failure rate** — from ``FailureRateStore``.

    The raw score is a weighted sum clamped to [0, 1] (REQ-DIFF-4).

    Args:
        failure_store: shared ``FailureRateStore`` instance.
        weights: ``(w_length, w_cue, w_history)`` — must sum to ≤ 1.0 for
            the clamping guarantee to be meaningful, but this is not enforced.
        length_norm_chars: character count at which the length signal saturates
            at 1.0.
    """

    def __init__(
        self,
        failure_store: FailureRateStore,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
        length_norm_chars: int = 2000,
    ) -> None:
        self._store = failure_store
        self._w_len, self._w_cue, self._w_hist = weights
        self._length_norm_chars = length_norm_chars

    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate:
        """Score *prompt* using length, reasoning cues, and history."""
        t0 = time.perf_counter()

        len_score = min(len(prompt) / self._length_norm_chars, 1.0)
        distinct_cues = {m.lower() for m in REASONING_CUES.findall(prompt)}
        cue_score = min(len(distinct_cues) / CUE_SATURATION, 1.0)
        hist_score = self._store.failure_rate(step_type)

        raw = (
            self._w_len * len_score
            + self._w_cue * cue_score
            + self._w_hist * hist_score
        )
        score = max(0.0, min(raw, 1.0))

        latency_ms = (time.perf_counter() - t0) * 1000
        return DifficultyEstimate(score=score, latency_ms=latency_ms)
