"""Tests for DifficultyEstimator + FailureRateStore (REQ-DIFF-1..4).

Covers:
- Length signal moves score in the right direction (DIFF-1)
- Reasoning-cue signal moves score in the right direction (DIFF-1)
- Historical failure rate moves score in the right direction (DIFF-1)
- Cold-start prior returned below min_samples (Decision #4)
- Observed rate used after min_samples reached
- Score is normalised to [0, 1] (DIFF-4)
- Estimator returns within acceptable latency (DIFF-3)
- FailureRateStore.record bookkeeping
"""

from __future__ import annotations

import pytest

from costrouter.difficulty import (
    FailureRateStore,
    HeuristicDifficultyEstimator,
)
from costrouter.types import DifficultyEstimate


# ---------------------------------------------------------------------------
# FailureRateStore
# ---------------------------------------------------------------------------

class TestFailureRateStore:
    def test_cold_start_returns_prior(self):
        """Below min_samples, failure_rate returns the configurable prior."""
        store = FailureRateStore(prior=0.5, min_samples=5)
        assert store.failure_rate("unseen_type") == 0.5

    def test_cold_start_custom_prior(self):
        store = FailureRateStore(prior=0.3, min_samples=5)
        assert store.failure_rate("unseen_type") == 0.3

    def test_switches_to_observed_after_min_samples(self):
        store = FailureRateStore(prior=0.5, min_samples=5)
        # Record 4 successes, 1 failure → 5 total → meets min_samples
        for _ in range(4):
            store.record("qa", failed=False)
        store.record("qa", failed=True)
        # observed rate = 1 failure / 5 total = 0.2
        assert store.failure_rate("qa") == pytest.approx(0.2)

    def test_still_prior_below_min_samples(self):
        store = FailureRateStore(prior=0.5, min_samples=5)
        for _ in range(4):
            store.record("qa", failed=False)
        # only 4 observations — still below min_samples
        assert store.failure_rate("qa") == 0.5

    def test_all_failures(self):
        store = FailureRateStore(prior=0.5, min_samples=3)
        for _ in range(3):
            store.record("hard", failed=True)
        assert store.failure_rate("hard") == pytest.approx(1.0)

    def test_all_successes(self):
        store = FailureRateStore(prior=0.5, min_samples=3)
        for _ in range(3):
            store.record("easy", failed=False)
        assert store.failure_rate("easy") == pytest.approx(0.0)

    def test_independent_step_types(self):
        store = FailureRateStore(prior=0.5, min_samples=2)
        store.record("a", failed=True)
        store.record("a", failed=True)
        store.record("b", failed=False)
        store.record("b", failed=False)
        assert store.failure_rate("a") == pytest.approx(1.0)
        assert store.failure_rate("b") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HeuristicDifficultyEstimator
# ---------------------------------------------------------------------------

class TestHeuristicDifficultyEstimator:
    def _make_estimator(self, **kwargs) -> HeuristicDifficultyEstimator:
        store = FailureRateStore(prior=0.0, min_samples=0)
        return HeuristicDifficultyEstimator(failure_store=store, **kwargs)

    def test_short_prompt_low_score(self):
        est = self._make_estimator()
        result = est.estimate("Hi", step_type="default")
        assert result.score < 0.5

    def test_long_prompt_higher_score(self):
        est = self._make_estimator()
        short = est.estimate("Hi", step_type="default")
        long = est.estimate("x " * 2000, step_type="default")
        assert long.score > short.score

    def test_reasoning_cue_boosts_score(self):
        est = self._make_estimator()
        no_cue = est.estimate("What is the capital of France?", step_type="default")
        with_cue = est.estimate(
            "Explain step by step why the sky is blue.", step_type="default"
        )
        assert with_cue.score > no_cue.score

    def test_various_reasoning_cues(self):
        """All documented reasoning cues should be detected."""
        est = self._make_estimator()
        baseline = est.estimate("simple question", step_type="default")
        for cue in ["step by step", "calculate", "prove", "derive", "explain why",
                     "multi-step", "compare"]:
            result = est.estimate(f"Please {cue} this.", step_type="default")
            assert result.score > baseline.score, f"Cue '{cue}' did not boost score"

    def test_historical_failure_moves_score(self):
        store = FailureRateStore(prior=0.0, min_samples=2)
        est_low = HeuristicDifficultyEstimator(failure_store=store)
        # Type "easy": all successes
        store.record("easy", failed=False)
        store.record("easy", failed=False)
        score_easy = est_low.estimate("test prompt", step_type="easy").score

        # Type "hard": all failures
        store.record("hard", failed=True)
        store.record("hard", failed=True)
        score_hard = est_low.estimate("test prompt", step_type="hard").score

        assert score_hard > score_easy

    def test_score_normalised_to_0_1(self):
        """Score must always be in [0, 1] regardless of inputs (REQ-DIFF-4)."""
        est = self._make_estimator()
        # Very short prompt
        r1 = est.estimate("", step_type="default")
        assert 0.0 <= r1.score <= 1.0

        # Very long prompt with reasoning cues
        r2 = est.estimate("step by step " * 1000, step_type="default")
        assert 0.0 <= r2.score <= 1.0

    def test_latency_is_recorded(self):
        est = self._make_estimator()
        result = est.estimate("test", step_type="default")
        assert result.latency_ms >= 0

    def test_latency_under_200ms(self):
        """DifficultyEstimator must return well under 200ms (REQ-DIFF-3)."""
        est = self._make_estimator()
        # Run 100 estimations — all should be fast
        for _ in range(100):
            result = est.estimate("A moderately long prompt " * 50, step_type="qa")
            assert result.latency_ms < 200

    def test_custom_weights(self):
        """Custom weights should shift the relative importance of signals."""
        store = FailureRateStore(prior=0.0, min_samples=0)
        # All weight on length
        est_len = HeuristicDifficultyEstimator(
            failure_store=store, weights=(1.0, 0.0, 0.0)
        )
        # All weight on cue
        est_cue = HeuristicDifficultyEstimator(
            failure_store=store, weights=(0.0, 1.0, 0.0)
        )

        prompt = "step by step short"
        score_len = est_len.estimate(prompt, step_type="default").score
        score_cue = est_cue.estimate(prompt, step_type="default").score
        # Short prompt → length score low; has cue → cue score high
        assert score_cue > score_len


class TestCueSaturation:
    def test_two_cues_score_higher_than_one(self):
        est = HeuristicDifficultyEstimator(FailureRateStore(prior=0.0, min_samples=0))
        one = est.estimate("Please compare these.", step_type="d").score
        two = est.estimate("Please compare and justify these.", step_type="d").score
        assert two > one

    def test_repeated_cue_counts_once(self):
        est = HeuristicDifficultyEstimator(FailureRateStore(prior=0.0, min_samples=0))
        once = est.estimate("compare", step_type="d").score
        thrice = est.estimate("compare Compare COMPARE", step_type="d").score
        assert thrice == pytest.approx(once, abs=0.01)

    def test_single_incidental_cue_stays_cheap_by_default(self):
        """'Compare' alone must not cross the default 0.5 threshold."""
        est = HeuristicDifficultyEstimator(FailureRateStore())
        assert est.estimate("Compare", step_type="d").score <= 0.5
