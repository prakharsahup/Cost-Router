"""RouterNode inside a compiled LangGraph graph, as in the README example."""

from __future__ import annotations

import warnings
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from costrouter.budget import BudgetTracker
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode
from costrouter.types import ModelTier, RouterConfig
from conftest import DEFAULT_RATE_CARD, MockChatModel, ok_response


class State(TypedDict, total=False):
    step_id: str
    step_type: str
    task_id: str
    prompt: str
    response: str
    last_decision: Any
    answer: str


def test_router_state_update_merges_into_typed_state():
    cheap = MockChatModel([ok_response("Paris")])
    router = RouterNode(cheap, MockChatModel(), HeuristicDifficultyEstimator(FailureRateStore()),
                        BudgetTracker(1.0, DEFAULT_RATE_CARD), RouterConfig(threshold=0.9))
    builder = StateGraph(State)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. LangGraph's config-annotation warning
        builder.add_node("router", router)
    builder.add_node("downstream", lambda s: {"answer": s["response"].upper()})
    builder.add_edge(START, "router")
    builder.add_edge("router", "downstream")
    builder.add_edge("downstream", END)

    result = builder.compile().invoke(
        {"step_id": "q1", "prompt": "What is the capital of France?", "step_type": "qa"})

    assert result["response"] == "Paris"
    assert result["answer"] == "PARIS"
    assert result["last_decision"].model_chosen == ModelTier.CHEAP
    assert result["prompt"] == "What is the capital of France?"  # input kept
