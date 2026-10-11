"""Minimal LangGraph + RouterNode example.

A 2-node graph (router → downstream) that runs end-to-end against
``MockChatModel`` — no API keys needed.  This doubles as the README
quickstart snippet.

Usage::

    python examples/minimal_graph.py
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Annotated, Optional

from langgraph.graph import StateGraph, START, END

from costrouter.types import (
    BudgetMode,
    ModelResponse,
    ModelTier,
    RouterConfig,
    RoutingDecision,
)
from costrouter.budget import BudgetTracker
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode
from costrouter.store import DecisionLogger, init_db, start_run, finish_run


# ---------------------------------------------------------------------------
# Mock models (no API keys required)
# ---------------------------------------------------------------------------

class MockCheapModel:
    """Simulates a cheap LLM that gives short, sometimes incomplete answers."""

    def invoke(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(
            text=f"[cheap] Quick answer to: {prompt[:50]}...",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=20,
            cost_usd=0.0001,
        )


class MockExpensiveModel:
    """Simulates an expensive LLM that gives detailed answers."""

    def invoke(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(
            text=f"[expensive] Detailed analysis of: {prompt[:50]}...",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=200,
            cost_usd=0.005,
        )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(dict):
    """Minimal state for the graph."""
    pass


# ---------------------------------------------------------------------------
# Downstream node (just prints the result)
# ---------------------------------------------------------------------------

def downstream_node(state: dict) -> dict:
    """Process the router's response."""
    decision: RoutingDecision = state["last_decision"]
    print(f"  |- Model used : {decision.model_chosen.value}")
    print(f"  |- Difficulty  : {decision.difficulty_score:.3f}")
    print(f"  |- Cost        : ${decision.actual_cost_usd:.6f}")
    print(f"  |- Budget left : ${decision.budget_remaining_after:.6f}")
    print(f"  |- Forced cheap: {decision.budget_forced}")
    print(f"  '- Response    : {state['response'][:80]}")
    return state


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def main():
    # ---- Configure components ----
    rate_card = {
        ModelTier.CHEAP: {"prompt": 0.000001, "completion": 0.000002},
        ModelTier.EXPENSIVE: {"prompt": 0.00001, "completion": 0.00003},
    }

    budget = BudgetTracker(
        initial_budget_usd=0.05,
        rate_card=rate_card,
        mode=BudgetMode.SOFT,
        on_threshold=lambda pct, rem: print(
            f"  [!] Budget warning: {pct*100:.0f}% threshold crossed, ${rem:.6f} remaining"
        ),
    )

    failure_store = FailureRateStore(prior=0.5, min_samples=5)
    estimator = HeuristicDifficultyEstimator(failure_store=failure_store)

    config = RouterConfig(threshold=0.5, budget_mode=BudgetMode.SOFT)

    # ---- Optional: persistence ----
    db_conn = init_db(":memory:")
    start_run(db_conn, "demo-run", "router", "demo", {"threshold": 0.5})
    logger = DecisionLogger(db_conn, "demo-run")

    router = RouterNode(
        cheap_model=MockCheapModel(),
        expensive_model=MockExpensiveModel(),
        estimator=estimator,
        budget=budget,
        config=config,
        logger=logger,
    )

    # ---- Build LangGraph ----
    builder = StateGraph(dict)
    builder.add_node("router", router)
    builder.add_node("downstream", downstream_node)
    builder.add_edge(START, "router")
    builder.add_edge("router", "downstream")
    builder.add_edge("downstream", END)
    graph = builder.compile()

    # ---- Run a few sample tasks ----
    tasks = [
        {
            "step_id": "q1",
            "task_id": "demo",
            "step_type": "qa",
            "prompt": "What is the capital of France?",
        },
        {
            "step_id": "q2",
            "task_id": "demo",
            "step_type": "math",
            "prompt": "Calculate step by step: if a train travels 120km in 2 hours, "
                      "and then 180km in 3 hours, what is its average speed for the "
                      "entire journey? Derive the formula first.",
        },
        {
            "step_id": "q3",
            "task_id": "demo",
            "step_type": "qa",
            "prompt": "Who wrote Hamlet?",
        },
    ]

    print("=" * 60)
    print("CostRouter - Minimal Graph Demo")
    print("=" * 60)

    for task in tasks:
        print(f"\n> Step '{task['step_id']}': {task['prompt'][:60]}...")
        graph.invoke(task)

    # ---- Summary ----
    finish_run(db_conn, "demo-run")
    rows = db_conn.execute(
        "SELECT step_id, model_chosen, actual_cost_usd FROM routing_decisions"
    ).fetchall()

    print("\n" + "=" * 60)
    print("Decision Log (from SQLite)")
    print("-" * 60)
    total_cost = 0.0
    for step_id, model, cost in rows:
        print(f"  {step_id}: {model:>9s}  ${cost:.6f}")
        total_cost += cost
    print(f"\n  Total cost: ${total_cost:.6f}")
    print(f"  Budget remaining: ${budget.remaining_budget():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
