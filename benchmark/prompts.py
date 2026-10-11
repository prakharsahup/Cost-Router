"""Benchmark prompt templates, format validator, mixed task set, response cache.

The templates ask every model to finish with a ``Final answer: ...`` line.
That gives the graders one unambiguous span to grade and gives the router a
label-free validator (``has_final_answer``): a cheap-model reply without the
line is "malformed output" in the REQ-ROUTER-6 sense, so it triggers
retry-escalate and is recorded in the failure-rate history.

Templates deliberately avoid the estimator's reasoning-cue words ("solve",
"step by step", …) so routing reacts to the *question*, not to boilerplate
that is identical for every item of a dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Any

from costrouter.types import ModelResponse

from benchmark.datasets import BenchmarkItem, load_gsm8k, load_triviaqa
from benchmark.graders import _FINAL_ANSWER

TRIVIAQA_TEMPLATE = (
    "Answer this trivia question. Reply with a single line of the form "
    "'Final answer: <answer>' and nothing else.\n\n"
    "Question: {question}"
)

GSM8K_TEMPLATE = (
    "Work out the answer to this math word problem. Show your working briefly, "
    "then end with a line of the form 'Final answer: <number>'.\n\n"
    "Problem: {question}"
)

TEMPLATES = {"triviaqa": TRIVIAQA_TEMPLATE, "gsm8k": GSM8K_TEMPLATE}


def has_final_answer(response: ModelResponse) -> bool:
    """Validator: no provider error and a non-empty ``Final answer:`` line."""
    if response.error or not response.text:
        return False
    return any(m.strip() for m in _FINAL_ANSWER.findall(response.text))


def apply_template(items: list[BenchmarkItem], dataset: str) -> list[BenchmarkItem]:
    template = TEMPLATES[dataset]
    return [(tid, template.format(question=q), gold) for tid, q, gold in items]


def load_mixed(
    triviaqa_path: str | None,
    gsm8k_path: str | None,
    limit: int | None = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Templated TriviaQA + GSM8K items, shuffled together with a fixed seed.

    *limit* applies per dataset.  Task ids keep their dataset prefix, which is
    how the harness picks each item's grader and ``step_type``.
    """
    items = (
        apply_template(load_triviaqa(triviaqa_path, limit=limit), "triviaqa")
        + apply_template(load_gsm8k(gsm8k_path, limit=limit), "gsm8k")
    )
    random.Random(seed).shuffle(items)
    return items


class CachedChatModel:
    """Wrap a ``ChatModelLike`` with a persistent (model, prompt) response cache.

    Every benchmark condition then sees the *same* answer for the same model
    and prompt, so differences between runs come from routing alone rather
    than sampling noise, and repeated runs cost nothing.  Cached hits keep the
    original token usage, so the budget ledger still charges the real cost of
    the call; their latency is not meaningful.  Error responses are never
    cached.  With ``offline=True`` a cache miss returns an error response
    instead of calling the provider.

    The cache is a JSONL file of ``{"key", "model", "text", "prompt_tokens",
    "completion_tokens"}`` rows; it stores a hash of the prompt, not the text.
    """

    def __init__(self, model: Any, model_name: str, cache_path: str | Path,
                 offline: bool = False) -> None:
        self._model = model
        self.model_name = model_name
        self._path = Path(cache_path)
        self._offline = offline
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self._path.is_file():
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model_name}\0{prompt}".encode("utf-8")).hexdigest()

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse:
        key = self._key(prompt)
        row = self._cache.get(key)
        if row is not None:
            self.hits += 1
            return ModelResponse(text=row["text"], prompt_tokens=row["prompt_tokens"],
                                 completion_tokens=row["completion_tokens"])
        self.misses += 1
        if self._offline or self._model is None:
            return ModelResponse(text="", error=f"cache miss for {self.model_name} (offline)")
        response = self._model.invoke(prompt, **kwargs)
        if not response.error:
            row = {"key": key, "model": self.model_name, "text": response.text,
                   "prompt_tokens": response.prompt_tokens,
                   "completion_tokens": response.completion_tokens}
            with self._lock:
                self._cache[key] = row
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response
