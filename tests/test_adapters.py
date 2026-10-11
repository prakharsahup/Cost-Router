"""Tests for the LangChain chat-model adapter (no network calls)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from costrouter.adapters import LangChainChatModel
from costrouter.budget import BudgetTracker
from costrouter.types import ModelTier
from conftest import DEFAULT_RATE_CARD


class StubChatModel:
    def __init__(self, message=None, exc=None):
        self._message, self._exc = message, exc
        self.prompts = []

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self._exc:
            raise self._exc
        return self._message


def test_text_and_usage_are_extracted():
    msg = AIMessage(
        content="Paris",
        usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
    )
    resp = LangChainChatModel(StubChatModel(msg)).invoke("Capital of France?")
    assert resp.text == "Paris"
    assert (resp.prompt_tokens, resp.completion_tokens) == (1000, 500)
    assert resp.cost_usd is None and resp.error is None


def test_rate_card_prices_adapter_response():
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500})
    resp = LangChainChatModel(StubChatModel(msg)).invoke("q")
    bt = BudgetTracker(1.0, DEFAULT_RATE_CARD)
    assert bt.record_actual_cost(ModelTier.CHEAP, resp) == 0.002


def test_content_block_list_is_joined():
    msg = AIMessage(content=[{"type": "text", "text": "Hel"}, {"type": "image_url", "image_url": "u"}, "lo"])
    assert LangChainChatModel(StubChatModel(msg)).invoke("q").text == "Hello"


def test_provider_exception_becomes_error_response():
    resp = LangChainChatModel(StubChatModel(exc=TimeoutError("slow"))).invoke("q")
    assert resp.text == ""
    assert resp.error == "TimeoutError: slow"


# ---------------------------------------------------------------------------
# OpenAICompatibleChatModel
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from costrouter.adapters import OpenAICompatibleChatModel


class StubCompletions:
    def __init__(self, completion=None, exc=None):
        self._completion, self._exc = completion, exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._completion


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _completion(text, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_openai_compatible_extracts_text_and_usage():
    stub = StubCompletions(_completion("Paris", 12, 3))
    resp = OpenAICompatibleChatModel("m", client=_client(stub), max_tokens=64).invoke("q")
    assert (resp.text, resp.prompt_tokens, resp.completion_tokens) == ("Paris", 12, 3)
    assert resp.cost_usd is None and resp.error is None
    call = stub.calls[0]
    assert call["model"] == "m"
    assert call["messages"] == [{"role": "user", "content": "q"}]
    assert (call["temperature"], call["seed"], call["max_tokens"]) == (0.0, 42, 64)


def test_openai_compatible_error_becomes_error_response():
    stub = StubCompletions(exc=TimeoutError("slow"))
    resp = OpenAICompatibleChatModel("m", client=_client(stub)).invoke("q")
    assert resp.text == "" and resp.error == "TimeoutError: slow"


def test_openai_compatible_null_content_is_empty_text():
    stub = StubCompletions(_completion(None))
    assert OpenAICompatibleChatModel("m", client=_client(stub)).invoke("q").text == ""
