"""Adapters from real chat-model clients to ``ChatModelLike``.

``LangChainChatModel`` wraps any LangChain ``BaseChatModel`` (``ChatOpenAI``,
``ChatAnthropic``, …) so it can be passed to ``RouterNode`` directly::

    from langchain_openai import ChatOpenAI
    from costrouter.adapters import LangChainChatModel

    cheap = LangChainChatModel(ChatOpenAI(model="gpt-4o-mini"))
    expensive = LangChainChatModel(ChatOpenAI(model="gpt-4o"))

API keys are read by the provider package from its usual environment variable
(e.g. ``OPENAI_API_KEY``) — never pass them through CostRouter.

Token counts come from LangChain's ``usage_metadata``.  ``cost_usd`` is left
``None`` so ``BudgetTracker`` prices the call with its rate card
(REQ-BUDGET-6) — keep the rate card in sync with your provider's pricing.
"""

from __future__ import annotations

from typing import Any

from costrouter.types import ModelResponse


class LangChainChatModel:
    """Wrap a LangChain chat model as a ``ChatModelLike``.

    Provider exceptions are caught and returned as ``ModelResponse(error=...)``
    so the router can retry-escalate (REQ-ROUTER-6) and the failure is
    recorded rather than crashing the graph.

    Args:
        model: any object with LangChain's ``invoke(input) -> AIMessage`` API.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse:
        try:
            message = self._model.invoke(prompt, **kwargs)
        except Exception as exc:  # provider/network errors → error response
            return ModelResponse(text="", error=f"{type(exc).__name__}: {exc}")

        usage = getattr(message, "usage_metadata", None) or {}
        return ModelResponse(
            text=_message_text(message),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            cost_usd=None,
        )


def _message_text(message: Any) -> str:
    """Extract plain text from an ``AIMessage`` (str or content-block list)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class OpenAICompatibleChatModel:
    """Call any OpenAI-compatible chat-completions endpoint as a ``ChatModelLike``.

    Works with OpenAI, Groq, OpenRouter, and other providers that speak the
    same API, using the ``openai`` package (``pip install openai``)::

        cheap = OpenAICompatibleChatModel(
            "llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )

    Requests use ``temperature=0`` and a fixed ``seed`` by default for
    reproducibility (REQ-BENCH-6).  Like ``LangChainChatModel``, provider
    errors come back as ``ModelResponse(error=...)`` and ``cost_usd`` is left
    ``None`` so the rate card prices the call.

    Args:
        model: provider model name.
        client: optional pre-built ``openai.OpenAI``-like client (used in
            tests); otherwise one is created from *base_url* / *api_key*.
        max_tokens: completion cap sent with every request.
        max_retries: SDK-level retries with backoff (rate limits, 5xx).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
        max_retries: int = 8,
        timeout: float = 120.0,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url, api_key=api_key,
                max_retries=max_retries, timeout=timeout,
            )
        self.model = model
        self._client = client
        self._params: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            self._params["seed"] = seed
        if max_tokens is not None:
            self._params["max_tokens"] = max_tokens

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self._params, **kwargs},
            )
        except Exception as exc:  # provider/network errors → error response
            return ModelResponse(text="", error=f"{type(exc).__name__}: {exc}")

        choices = getattr(completion, "choices", None) or []
        text = (choices[0].message.content or "") if choices else ""
        usage = getattr(completion, "usage", None)
        return ModelResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=None,
        )
