"""3-condition benchmark harness + policy sweep (REQ-BENCH-1..6).

Conditions:
- ``always_cheap``: every task goes to the cheap model.
- ``always_expensive``: every task goes to the expensive model.
- ``router``: the ``RouterNode`` decides per task.

The harness writes every result to SQLite via ``DecisionLogger`` so the
dashboard can aggregate cost/accuracy per ``run_id``.
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

from costrouter.types import (
    BudgetMode,
    ModelResponse,
    ModelTier,
    RouterConfig,
    RoutingDecision,
)
from costrouter.budget import BudgetTracker
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode, threshold_policy
from costrouter.store import DecisionLogger, init_db, start_run, finish_run

from benchmark.datasets import BenchmarkItem


# Type alias for a grading function
GraderFn = Callable[[str, str], bool]


def item_dataset(task_id: str) -> str:
    """Dataset name encoded in a loader task id (``"gsm8k-7"`` → ``"gsm8k"``)."""
    return task_id.rsplit("-", 1)[0]


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    items: list[BenchmarkItem],
    condition: str,
    dataset_name: str,
    grader: GraderFn,
    cheap_model,
    expensive_model,
    rate_card: dict[ModelTier, dict[str, float]],
    budget_usd: float = 10.0,
    threshold: float = 0.5,
    db_path: str = ":memory:",
    conn=None,
    run_id: str | None = None,
    seed: int = 42,
    learn_from_grades: bool = False,
    validator: Optional[Callable[[ModelResponse], bool]] = None,
    max_completion_tokens: Optional[int] = None,
    run_config: Optional[dict] = None,
    budget_mode: BudgetMode = BudgetMode.SOFT,
) -> dict:
    """Run a single benchmark condition and return aggregated results.

    Args:
        items: list of ``(task_id, prompt, gold_answer)`` tuples.
        condition: ``"always_cheap"``, ``"always_expensive"``, or ``"router"``.
        dataset_name: e.g. ``"triviaqa"`` or ``"gsm8k"`` (the run's task set).
        grader: function ``(model_answer, gold_answer) -> bool``, or a dict
            ``{dataset: grader}`` for a mixed task set — each item's dataset
            is read from its task id (``item_dataset``) and also used as its
            ``step_type``, so failure history is kept per dataset.
        cheap_model: model wrapper for the cheap tier.
        expensive_model: model wrapper for the expensive tier.
        rate_card: per-token USD rates.
        budget_usd: initial budget for this run.
        threshold: difficulty threshold (only used for ``"router"`` condition).
        db_path: SQLite database path (ignored if *conn* is provided).
        conn: optional pre-opened ``sqlite3.Connection``.
        run_id: optional run identifier (auto-generated if omitted).
        seed: random seed for reproducibility (REQ-BENCH-6).
        learn_from_grades: if ``True``, also feed each task's *graded*
            correctness into the failure-rate history.  Off by default: gold
            labels aren't available in production, and since history is keyed
            by ``step_type`` (= dataset name here) it would mostly teach the
            router "this dataset is hard" from the answer key, flattering the
            router condition.  By default the router learns only from its own
            validator failures, as it would when deployed.
        validator: output validator for the ``"router"`` condition (drives
            retry-escalate and failure history).  The baselines always use the
            router's default validator — they never retry, so a stricter one
            would only turn answers into errors.
        max_completion_tokens: passed to ``RouterConfig`` for the budget
            pre-flight check; set it to the provider ``max_tokens``.
        run_config: extra metadata (models, provider, …) stored in the run's
            ``policy_config_json`` so a Pareto point can be regenerated.
        budget_mode: ``SOFT`` (default) keeps answering with the cheap model
            once the budget is gone; ``HARD_STOP`` refuses the remaining steps
            (each raises ``BudgetExhaustedError`` and counts as an error).

    A task whose step raises (e.g. a provider error with no retry left) is
    graded incorrect and charged whatever the budget ledger recorded for it;
    the run continues and ``errors`` counts such tasks.

    Returns:
        A dict with ``run_id``, ``condition``, ``accuracy``, ``total_cost``,
        ``correct_count``, ``total_count``, ``errors``, and ``per_task``.
    """
    if conn is None:
        conn = init_db(db_path)

    if run_id is None:
        run_id = f"{condition}-{dataset_name}-{uuid.uuid4().hex[:8]}"

    policy_config = {
        "condition": condition,
        "threshold": threshold,
        "budget_usd": budget_usd,
        "seed": seed,
        "learn_from_grades": learn_from_grades,
        "max_completion_tokens": max_completion_tokens,
        "budget_mode": budget_mode.value,
        **(run_config or {}),
    }
    start_run(conn, run_id, condition, dataset_name, policy_config)
    logger = DecisionLogger(conn, run_id)

    budget = BudgetTracker(
        initial_budget_usd=budget_usd,
        rate_card=rate_card,
        mode=budget_mode,
        on_threshold=lambda pct, rem: logger.log_budget_event(pct, rem),
    )

    # Build the router or a forced-model wrapper
    if condition == "always_cheap":
        # Force cheap by using a threshold of 1.0 (nothing exceeds it)
        cfg = RouterConfig(threshold=1.0, retry_on_error=False,
                           max_completion_tokens=max_completion_tokens)
    elif condition == "always_expensive":
        # Force expensive with a threshold below every score (scores are ≥ 0)
        cfg = RouterConfig(threshold=-1.0, retry_on_error=False,
                           max_completion_tokens=max_completion_tokens)
    else:
        cfg = RouterConfig(threshold=threshold, validator=validator,
                           max_completion_tokens=max_completion_tokens)

    failure_store = FailureRateStore(prior=0.5, min_samples=5)
    estimator = HeuristicDifficultyEstimator(failure_store=failure_store)

    router = RouterNode(
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        estimator=estimator,
        budget=budget,
        config=cfg,
        logger=logger,
        failure_store=failure_store,
    )

    # Run each task
    correct_count = 0
    total_cost = 0.0
    errors = 0
    per_task = []

    for task_id, prompt, gold_answer in items:
        if isinstance(grader, dict):
            task_dataset = item_dataset(task_id)
            grade = grader[task_dataset]
        else:
            task_dataset, grade = dataset_name, grader
        state = {
            "step_id": task_id,
            "task_id": task_id,
            "step_type": task_dataset,
            "prompt": prompt,
        }
        budget_before = budget.remaining_budget()
        try:
            result = router(state)
        except Exception as exc:
            # Keep the run going; the task counts as wrong and keeps its cost.
            errors += 1
            cost = budget_before - budget.remaining_budget()
            model_answer, decision = "", None
            error = f"{type(exc).__name__}: {exc}"
        else:
            model_answer = result["response"]
            decision = result["last_decision"]
            cost = decision.actual_cost_usd
            error = None

        is_correct = decision is not None and grade(model_answer, gold_answer)
        correct_count += int(is_correct)
        total_cost += cost

        logger.log_task_result(
            task_id=task_id,
            dataset=task_dataset,
            correct=is_correct,
            total_cost_usd=cost,
        )

        if learn_from_grades:
            failure_store.record(task_dataset, failed=not is_correct)

        per_task.append({
            "task_id": task_id,
            "dataset": task_dataset,
            "correct": is_correct,
            "cost_usd": cost,
            "model_chosen": decision.model_chosen.value if decision else None,
            "difficulty_score": decision.difficulty_score if decision else None,
            "retried": decision.retried if decision else False,
            "budget_forced": decision.budget_forced if decision else False,
            "answer": model_answer,
            "error": error,
        })

    finish_run(conn, run_id)

    accuracy = correct_count / len(items) if items else 0.0

    return {
        "run_id": run_id,
        "condition": condition,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total_cost": total_cost,
        "correct_count": correct_count,
        "total_count": len(items),
        "errors": errors,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# 3-condition benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    items: list[BenchmarkItem],
    dataset_name: str,
    grader: GraderFn,
    cheap_model,
    expensive_model,
    rate_card: dict[ModelTier, dict[str, float]],
    budget_usd: float = 10.0,
    threshold: float = 0.5,
    conn=None,
    learn_from_grades: bool = False,
    **condition_kwargs,
) -> list[dict]:
    """Run all three conditions on the same task set.

    Extra keyword arguments (``validator``, ``max_completion_tokens``,
    ``run_config``, …) are passed through to ``run_condition``.

    Returns a list of three result dicts (always_cheap, always_expensive, router).
    """
    if conn is None:
        conn = init_db(":memory:")

    results = []
    for condition in ["always_cheap", "always_expensive", "router"]:
        r = run_condition(
            items=items,
            condition=condition,
            dataset_name=dataset_name,
            grader=grader,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            rate_card=rate_card,
            budget_usd=budget_usd,
            threshold=threshold,
            conn=conn,
            learn_from_grades=learn_from_grades,
            **condition_kwargs,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Policy sweep (REQ-BENCH-4)
# ---------------------------------------------------------------------------

def run_policy_sweep(
    items: list[BenchmarkItem],
    dataset_name: str,
    grader: GraderFn,
    cheap_model,
    expensive_model,
    rate_card: dict[ModelTier, dict[str, float]],
    thresholds: list[float] | None = None,
    budget_usd: float = 10.0,
    conn=None,
    learn_from_grades: bool = False,
    **condition_kwargs,
) -> list[dict]:
    """Sweep multiple threshold values to produce multiple Pareto points.

    Args:
        thresholds: list of threshold values to test (default: 5 standard
            values).  Real prompts often score in a narrow band — use
            ``prompt_score_thresholds`` to place the thresholds inside it.

    Extra keyword arguments are passed through to ``run_condition``.

    Returns:
        A list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]

    if conn is None:
        conn = init_db(":memory:")

    results = []
    for t in thresholds:
        r = run_condition(
            items=items,
            condition="router",
            dataset_name=dataset_name,
            grader=grader,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            rate_card=rate_card,
            budget_usd=budget_usd,
            threshold=t,
            conn=conn,
            run_id=f"router-t{t:g}-{dataset_name}-{uuid.uuid4().hex[:8]}",
            learn_from_grades=learn_from_grades,
            **condition_kwargs,
        )
        results.append(r)

    return results


def prompt_score_thresholds(
    items: list[BenchmarkItem],
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    ndigits: int = 4,
) -> list[float]:
    """Sweep thresholds at quantiles of the estimator's prompt-only score.

    Scores each prompt with the default ``HeuristicDifficultyEstimator`` and
    no failure history (prior 0), i.e. the length + reasoning-cue part of the
    score, then returns the requested quantiles.  Uses no model calls and no
    gold labels, so choosing thresholds this way doesn't leak the answer key.
    At run time the history signal adds to these scores (the cold-start prior
    for the first steps of each type, then the observed failure rate).
    """
    estimator = HeuristicDifficultyEstimator(
        FailureRateStore(prior=0.0, min_samples=1 << 30)
    )
    scores = sorted(estimator.estimate(p, "sweep").score for _, p, _ in items)
    if not scores:
        return []
    out = []
    for q in quantiles:
        pos = q * (len(scores) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(scores) - 1)
        value = scores[lo] + (scores[hi] - scores[lo]) * (pos - lo)
        out.append(round(value, ndigits))
    return sorted(set(out))
