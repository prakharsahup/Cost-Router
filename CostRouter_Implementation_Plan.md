# CostRouter — Implementation Plan v1.0

Companion to `CostRouter_SRS_Final_v2`. Written as a build brief you can hand to Claude
Opus 4.6 (via Claude Code) and work through phase by phase. Each phase has a concrete
Definition of Done tied back to the SRS's `REQ-*` IDs so "done" is checkable, not vibes.

---

## 0. Read this first — 4 things in the SRS that don't fully resolve on their own

The SRS is solid, but four spots have either an internal contradiction or an
underspecified contract. Rather than let an implementer guess silently, here's what I
found and the decision I'm building the rest of this plan around. Push back on any of
these if you disagree — they're design calls, not facts.

**1. Benchmark dataset conflict (REQ-BENCH-1 vs. Appendix C).**
REQ-BENCH-1 names HotpotQA + GSM8K explicitly. The Appendix C resolution for the same
requirement swaps HotpotQA for TriviaQA/SQuAD, specifically *because* "metrics must be
exact-match gradable" and HotpotQA is normally graded with F1, not strict EM. Appendix C
is the resolution of the TBD, so it should win — but the parenthetical text in
REQ-BENCH-1 was never updated to match. **Decision: use TriviaQA (or SQuAD) for the easy
set and GSM8K for the hard set**, exact-match grading on both (GSM8K: numeric answer
match after normalization; TriviaQA/SQuAD: normalized string match against the
gold-answer list). Flag this to whoever signs off on the SRS — it's a one-line fix on
their end.

**2. Budget unit conflict (REQ-BUDGET-1 vs. Appendix C).**
REQ-BUDGET-1 says the tracker accepts an initial budget "in exact USD." The Appendix C
resolution says "internal unit = tokens, display unit = $." Those two are only
compatible if there's one token price. With two models (cheap/expensive) at different
per-token rates, a single token counter can't represent a USD budget — 1,000 "budget
tokens" means something different depending on which model spends them. **Decision: the
ledger's source of truth is USD**, decremented by the *actual* cost of each call (from
provider usage metadata, converted via a per-model rate card). Token counting is used
only as the REQ-BUDGET-6 fallback estimator when a provider doesn't return usage
metadata — never as the primary unit. This satisfies "provider-agnostic and
deterministic" (the actual intent behind the Appendix C note) without the double-counting
bug a shared token ledger would introduce.

**3. "Malformed output" is never defined (REQ-ROUTER-6).**
The retry-escalate requirement needs a validator to know what counts as malformed. The
SRS doesn't specify one. **Decision: `RouterConfig.validator` is a pluggable
`Callable[[ModelResponse], bool]`, defaulting to "non-empty text and no provider error
flag."** Anyone benchmarking against GSM8K/TriviaQA can swap in a stricter validator
(e.g., "contains a parseable final-answer token") without touching router internals.

**4. Historical failure rate has a cold-start problem (REQ-DIFF-1).**
"Historical failure rate for this step type" is undefined for a step type seen zero
times. **Decision: `FailureRateStore` returns a configurable prior (default 0.5) until a
step type has `min_samples` (default 5) observations**, then switches to the observed
rate. Document this in the estimator's docstring so it doesn't look like a bug during
benchmarking (early-run difficulty scores will look artificially uniform).

One more implementation-level (not contradiction-level) note worth flagging up front:
**REQ-BUDGET-7's thread-safety only matters if the graph actually executes branches
concurrently.** LangGraph supports both sync and async execution; a plain
`threading.Lock` protects sync/threaded concurrency but does nothing for `asyncio`
concurrency on a single thread (where you don't need a lock, you need to avoid awaiting
inside the critical section). Pick one execution model for v1.0 and say so — don't build
both paths without being asked.

---

## 1. Architecture at a glance

```
Upstream Agent Step
        │
        ▼
   RouterNode  ──getDifficultyScore()──▶  DifficultyEstimator ──▶ FailureRateStore
        │  ──getRemainingBudget()────▶  BudgetTracker
        │
        ├──(cheap)──▶ Cheap Model  ──┐
        └──(expensive)─▶ Expensive Model ─┤
                                          ▼
                              Downstream Agent Step
                                          │
                                          ▼
                              BudgetTracker.record_actual_cost()
                                          │
                                          ▼
                              DecisionLogger ──▶ SQLite (routing_decisions, budget_events)
```

Five components, each independently unit-testable (SRS §5.4):

| Component | Responsibility | REQ IDs |
|---|---|---|
| `DifficultyEstimator` | score an upcoming step 0–1 | DIFF-1..4 |
| `BudgetTracker` | own the USD ledger, thresholds, hard/soft stop | BUDGET-1..7 |
| `RouterNode` | LangGraph node; wires the above into a routing decision | ROUTER-1..6 |
| `DecisionLogger` / SQLite store | persist every decision for audit + benchmarking | ROUTER-4, §6 |
| Benchmark harness + Streamlit dashboard | run always-cheap/always-expensive/router conditions, plot Pareto curve | BENCH-1..6 |

---

## 2. Repository layout

```
costrouter/
├── pyproject.toml
├── README.md
├── .env.example
├── costrouter/
│   ├── __init__.py
│   ├── types.py            # dataclasses, enums, protocols
│   ├── budget.py            # BudgetTracker
│   ├── difficulty.py        # DifficultyEstimator, FailureRateStore
│   ├── router.py            # RouterNode, built-in policies
│   ├── store.py              # SQLite schema + DecisionLogger
│   └── exceptions.py
├── benchmark/
│   ├── __init__.py
│   ├── datasets.py           # TriviaQA / GSM8K loaders
│   ├── graders.py            # exact-match graders per dataset
│   └── harness.py            # 3-condition runner + policy sweep
├── dashboard/
│   └── app.py                 # Streamlit + plotly Pareto view
├── examples/
│   └── minimal_graph.py       # smallest possible LangGraph + RouterNode
├── tests/
│   ├── conftest.py             # MockChatModel, fixtures
│   ├── test_budget.py
│   ├── test_difficulty.py
│   ├── test_router.py
│   └── test_harness_smoke.py
└── notebooks/
    └── pareto_report.ipynb     # SRS §2.6 "short report" deliverable
```

---

## 3. Core types (`costrouter/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol


class ModelTier(str, Enum):
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


class BudgetMode(str, Enum):
    HARD_STOP = "hard_stop"   # halt graph execution when budget hits 0
    SOFT = "soft"              # force cheap-only routing when budget hits 0


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None   # None -> BudgetTracker estimates via rate card
    error: Optional[str] = None


class ChatModelLike(Protocol):
    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse: ...


@dataclass
class DifficultyEstimate:
    score: float                       # normalized 0..1, REQ-DIFF-4
    confidence: Optional[float] = None
    latency_ms: float = 0.0


@dataclass
class RoutingDecision:
    step_id: str
    task_id: Optional[str]
    difficulty_score: float
    budget_remaining_before: float
    budget_remaining_after: float
    model_chosen: ModelTier
    budget_forced: bool
    retried: bool
    actual_cost_usd: float
    latency_ms: float
    estimator_latency_ms: float
    error: Optional[str] = None


PolicyFn = Callable[[float, float, "RouterConfig"], ModelTier]
Validator = Callable[[ModelResponse], bool]


@dataclass
class RouterConfig:
    threshold: float = 0.5
    budget_mode: BudgetMode = BudgetMode.SOFT
    warn_thresholds_pct: tuple = (0.5, 0.1)     # REQ-BUDGET-4
    retry_on_error: bool = True                  # REQ-ROUTER-6
    validator: Optional[Validator] = None
```

---

## 4. Module specs

### 4.1 `BudgetTracker` (REQ-BUDGET-1..7)

Ledger is USD (see Decision #2 above). Rate card is `dict[ModelTier, dict[str, float]]`
with `"prompt"` / `"completion"` per-token USD rates, used only as the fallback
estimator.

```python
import threading

class BudgetTracker:
    def __init__(self, initial_budget_usd: float,
                 rate_card: dict[ModelTier, dict[str, float]],
                 mode: BudgetMode = BudgetMode.SOFT,
                 warn_thresholds_pct=(0.5, 0.1),
                 on_threshold: Optional[Callable[[float, float], None]] = None):
        self._initial = initial_budget_usd
        self._remaining = initial_budget_usd
        self._rate_card = rate_card
        self._mode = mode
        self._warn_thresholds = sorted(warn_thresholds_pct, reverse=True)
        self._fired: set[float] = set()
        self._on_threshold = on_threshold
        self._lock = threading.Lock()

    def remaining_budget(self) -> float:
        with self._lock:
            return self._remaining

    def can_afford(self, tier: ModelTier, est_prompt_tokens: int, est_completion_tokens: int) -> bool:
        rates = self._rate_card[tier]
        est_cost = est_prompt_tokens * rates["prompt"] + est_completion_tokens * rates["completion"]
        return self.remaining_budget() >= est_cost

    def record_actual_cost(self, tier: ModelTier, response: ModelResponse) -> float:
        with self._lock:
            cost = response.cost_usd
            if cost is None:                       # REQ-BUDGET-6 fallback
                rates = self._rate_card[tier]
                cost = (response.prompt_tokens or 0) * rates["prompt"] \
                     + (response.completion_tokens or 0) * rates["completion"]
            self._remaining -= cost
            self._maybe_fire_thresholds()
            return cost

    def exhausted(self) -> bool:
        return self.remaining_budget() <= 0

    def _maybe_fire_thresholds(self) -> None:
        pct = self._remaining / self._initial if self._initial else 0
        for t in self._warn_thresholds:
            if pct <= t and t not in self._fired:
                self._fired.add(t)
                if self._on_threshold:
                    self._on_threshold(t, self._remaining)
```

`can_afford` needs *some* estimate of expensive-model token usage before the call
happens — use a cheap heuristic (e.g. prompt length × a fixed expansion factor) rather
than calling the estimator again; keep this estimate crude on purpose, since REQ-DIFF-3's
cost/latency budget applies to the difficulty estimator, not to this pre-flight check.

### 4.2 `DifficultyEstimator` (REQ-DIFF-1..4)

Pure local computation → effectively free and well under the 200ms bound by
construction, since it never makes a network call.

```python
import re, time
from abc import ABC, abstractmethod

REASONING_CUES = re.compile(
    r"\b(step by step|calculate|prove|derive|explain why|multi-step|compare)\b", re.I
)

class DifficultyEstimator(ABC):
    @abstractmethod
    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate: ...


class FailureRateStore:
    """Cold-start handled with a prior until min_samples observed (Decision #4)."""
    def __init__(self, prior: float = 0.5, min_samples: int = 5):
        self._prior, self._min_samples = prior, min_samples
        self._counts: dict[str, tuple[int, int]] = {}   # step_type -> (successes, failures)

    def failure_rate(self, step_type: str) -> float:
        s, f = self._counts.get(step_type, (0, 0))
        return self._prior if (s + f) < self._min_samples else f / (s + f)

    def record(self, step_type: str, failed: bool) -> None:
        s, f = self._counts.get(step_type, (0, 0))
        self._counts[step_type] = (s + (0 if failed else 1), f + (1 if failed else 0))


class HeuristicDifficultyEstimator(DifficultyEstimator):
    def __init__(self, failure_store: FailureRateStore,
                 weights=(0.3, 0.3, 0.4), length_norm_chars: int = 2000):
        self._store = failure_store
        self._w_len, self._w_cue, self._w_hist = weights
        self._length_norm_chars = length_norm_chars

    def estimate(self, prompt: str, step_type: str) -> DifficultyEstimate:
        t0 = time.perf_counter()
        len_score = min(len(prompt) / self._length_norm_chars, 1.0)
        cue_score = 1.0 if REASONING_CUES.search(prompt) else 0.0
        hist_score = self._store.failure_rate(step_type)
        raw = self._w_len * len_score + self._w_cue * cue_score + self._w_hist * hist_score
        return DifficultyEstimate(score=max(0.0, min(raw, 1.0)),
                                   latency_ms=(time.perf_counter() - t0) * 1000)
```

Leave `DifficultyEstimator` as an ABC on purpose — REQ-DIFF-2 restricts v1.0 to
heuristics but explicitly wants the interface open for a future learned model.

### 4.3 `RouterNode` (REQ-ROUTER-1..6)

LangGraph nodes are callables of shape `(state, config) -> partial_state_update` (a
`StateGraph` node can be a plain function, a `Runnable`, or any callable — the return
value is merged into state via each field's reducer, default last-write-wins). Implement
`RouterNode` as a class with `__call__` so dependencies (estimator, budget, models) are
injected once at construction rather than threaded through state.

```python
import time
from langchain_core.runnables import RunnableConfig

def threshold_policy(difficulty: float, remaining_budget: float, config: RouterConfig) -> ModelTier:
    """REQ-ROUTER-2 built-in policy."""
    return ModelTier.EXPENSIVE if difficulty > config.threshold else ModelTier.CHEAP


class RouterNode:
    def __init__(self, cheap_model: ChatModelLike, expensive_model: ChatModelLike,
                 estimator: DifficultyEstimator, budget: BudgetTracker,
                 config: RouterConfig, policy: PolicyFn = threshold_policy,
                 logger: Optional["DecisionLogger"] = None):
        self._cheap, self._expensive = cheap_model, expensive_model
        self._estimator, self._budget, self._config = estimator, budget, config
        self._policy, self._logger = policy, logger

    def __call__(self, state: dict, config: Optional[RunnableConfig] = None) -> dict:
        step_id, prompt = state["step_id"], state["prompt"]
        step_type = state.get("step_type", "default")

        # REQ-ROUTER-5: fail open to cheap if the estimator throws
        try:
            est = self._estimator.estimate(prompt, step_type)
        except Exception:
            est = DifficultyEstimate(score=0.0, latency_ms=0.0)

        remaining_before = self._budget.remaining_budget()
        tier = self._policy(est.score, remaining_before, self._config)

        # REQ-ROUTER-3: budget-forced override
        budget_forced = False
        approx_tokens = max(len(prompt) // 4, 1)
        if tier == ModelTier.EXPENSIVE and not self._budget.can_afford(
                ModelTier.EXPENSIVE, approx_tokens, approx_tokens):
            tier, budget_forced = ModelTier.CHEAP, True
        if self._config.budget_mode == BudgetMode.HARD_STOP and self._budget.exhausted():
            raise BudgetExhaustedError(step_id)

        # model call + REQ-ROUTER-6 retry-escalate
        model, retried, t0 = (self._cheap if tier == ModelTier.CHEAP else self._expensive), False, time.perf_counter()
        validator = self._config.validator or (lambda r: bool(r.text) and not r.error)
        try:
            response = model.invoke(prompt)
            if not validator(response):
                raise MalformedOutputError(step_id)
        except Exception:
            if self._config.retry_on_error and tier == ModelTier.CHEAP:
                tier, retried = ModelTier.EXPENSIVE, True
                response = self._expensive.invoke(prompt)
            else:
                raise
        latency_ms = (time.perf_counter() - t0) * 1000

        cost = self._budget.record_actual_cost(tier, response)         # REQ-BUDGET-2
        decision = RoutingDecision(
            step_id=step_id, task_id=state.get("task_id"), difficulty_score=est.score,
            budget_remaining_before=remaining_before, budget_remaining_after=self._budget.remaining_budget(),
            model_chosen=tier, budget_forced=budget_forced, retried=retried,
            actual_cost_usd=cost, latency_ms=latency_ms, estimator_latency_ms=est.latency_ms,
        )
        if self._logger:
            self._logger.log(decision)                                  # REQ-ROUTER-4

        return {"response": response.text, "last_decision": decision}   # partial update
```

Register with `builder.add_node("router", router_instance)` — don't wrap it in an extra
lambda; a callable instance is a first-class node.

### 4.4 Persistence (`costrouter/store.py`, SRS §6)

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    condition TEXT NOT NULL CHECK(condition IN ('always_cheap','always_expensive','router')),
    task_set TEXT NOT NULL,
    policy_config_json TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    step_id TEXT NOT NULL,
    task_id TEXT,
    difficulty_score REAL NOT NULL,
    budget_remaining_before REAL NOT NULL,
    budget_remaining_after REAL NOT NULL,
    model_chosen TEXT NOT NULL CHECK(model_chosen IN ('cheap','expensive')),
    budget_forced INTEGER NOT NULL DEFAULT 0,
    retried INTEGER NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    estimator_latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE budget_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    threshold_pct REAL NOT NULL,
    remaining_budget REAL NOT NULL,
    triggered_at TEXT NOT NULL
);

CREATE TABLE task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    correct INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL
);
```

Plain `sqlite3` + a thin `DecisionLogger` wrapper is enough — no ORM needed, and
`pandas.read_sql` reads straight from it for the dashboard. Keep every write in a single
transaction per step to satisfy "every routing decision must be traceable" (SRS §2.5)
even if the process crashes mid-run.

### 4.5 Benchmark harness (`benchmark/harness.py`, REQ-BENCH-1..6)

- **Datasets**: TriviaQA (easy) + GSM8K (hard), per Decision #1. Loaders return
  `(task_id, prompt, gold_answer)` tuples; keep loaders separate from grading logic.
- **Graders**: GSM8K grader extracts the final numeric answer (regex on the last number
  in the completion, or the standard `#### <answer>` GSM8K format) and compares
  numerically. TriviaQA grader normalizes case/whitespace/punctuation and checks against
  the gold-answer alias list.
- **Conditions**: `always_cheap`, `always_expensive`, `router` — same task set, same
  graph shape, only the node wired into the "model call" slot changes.
- **Policy sweep (REQ-BENCH-4)**: run `router` condition across a list of thresholds
  (e.g. `[0.2, 0.35, 0.5, 0.65, 0.8]`) or budget caps to get multiple points — a single
  router run is one dot, not a frontier.
- **Output**: write every task result + routing decision to SQLite via `run_id`; the
  dashboard aggregates cost/accuracy per `run_id` to draw the curve. Seed-control
  (REQ-BENCH-6) means passing `temperature=0` / provider `seed` param where supported and
  recording the run's config JSON alongside results.

### 4.6 Streamlit dashboard (`dashboard/app.py`)

`pandas.read_sql("SELECT ...", conn)` → group by `run_id`/condition → `plotly.express`
scatter+line for the Pareto curve (matches the sample chart in the SRS: cost on x,
accuracy on y, three series). Add a secondary view for budget-forced decision counts and
threshold-crossing events over time — that data already exists in `budget_events` and is
otherwise invisible, and it's the concrete evidence for the §5.2 "safety" requirement
(degradation should be traceable, not hidden).

---

## 5. Testing strategy (SRS §5.4)

| Test | What it proves | Fixture needed |
|---|---|---|
| `test_budget.py` | thresholds fire once each, hard-stop halts, soft-mode forces cheap, fallback estimator triggers when `cost_usd is None` | rate card fixture, no network |
| `test_budget.py::test_concurrent_decrements` | REQ-BUDGET-7 — N threads calling `record_actual_cost` concurrently sum to the correct remaining budget | `threading` |
| `test_difficulty.py` | length/cue/history each move the score in the right direction; cold-start prior returned below `min_samples` | none |
| `test_router.py` | REQ-ROUTER-3 (budget-forced), -5 (estimator throws → cheap fallback), -6 (malformed → retry-escalate, and no infinite retry loop) | `MockChatModel` returning canned `ModelResponse`s, incl. one that always errors |
| `test_harness_smoke.py` | 3-condition run over a 2–3-item synthetic task set completes and writes expected rows to a temp SQLite file | in-memory/temp DB |

`MockChatModel` (in `conftest.py`) implements `ChatModelLike` and takes a queue of
canned responses — this is what makes routing logic testable "without live API calls"
per SRS §5.4.

---

## 6. Non-functional requirements — how they get enforced, not just stated

- **Latency (§5.1)**: `DifficultyEstimate.latency_ms` and `RoutingDecision.latency_ms`
  are already captured on every call — write a test that asserts
  `estimator_latency_ms < 200` over a batch of realistic prompts, and a separate
  micro-benchmark script that asserts router overhead (everything in `__call__` minus
  the model `.invoke()` time) stays under 50ms. Don't try to enforce this at runtime with
  a hard timeout — that adds its own latency and complexity; assert it in CI instead.
- **Security (§5.3)**: API keys via `python-dotenv` + `os.environ`, never passed as
  literals or logged. `DecisionLogger` should log prompt *length* and a hash, not full
  prompt text, by default — full-text logging should be an explicit opt-in flag with a
  documented retention/deletion policy (SRS asks for one; put it in the README, not just
  in code comments).
- **Reproducibility (§5.4)**: pin `temperature=0` and pass a `seed` where the provider
  supports it; store the resolved config (threshold, budget cap, model names, seed) as
  JSON on the `runs` row so a Pareto point can be regenerated later.

---

## 7. Dependencies

```toml
[project]
name = "costrouter"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=0.6",
    "langchain-core",
    "pydantic>=2",
    "streamlit",
    "plotly",
    "pandas",
    "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock"]
```

`tiktoken` only if you need a real tokenizer for the REQ-BUDGET-6 fallback estimator
against a provider that gives zero usage metadata (OpenAI/Anthropic both return usage by
default, so this is a true fallback path, not the common case — don't over-invest here).

Note on LangGraph itself: it's a fast-moving library (`config_schema` was deprecated in
favor of `context_schema` as of v0.6.0). Pin a version in `pyproject.toml` and re-check
`docs.langchain.com` against whatever version you actually install before wiring
`RouterNode` in — don't trust the snippet in §4.3 verbatim if you're on a materially
different version.

---

## 8. Phased build plan

| Phase | Deliverable | Definition of Done |
|---|---|---|
| 0 | Repo scaffold, `pyproject.toml`, `.env.example` | `pip install -e .` succeeds; empty test suite runs |
| 1 | `BudgetTracker` | All `test_budget.py` cases pass, incl. concurrency (BUDGET-1..7) |
| 2 | `DifficultyEstimator` + `FailureRateStore` | `test_difficulty.py` passes; estimator returns in <5ms in practice (DIFF-1..4) |
| 3 | `RouterNode` + built-in policy + exceptions | `test_router.py` passes, incl. budget-forced, fail-open, retry-escalate (ROUTER-1..6) |
| 4 | SQLite schema + `DecisionLogger` | Router writes one row per decision; manual `sqlite3` inspection matches SRS §2.5 fields (ROUTER-4, §6) |
| 5 | `examples/minimal_graph.py` | A 2-node LangGraph (router → downstream) runs end-to-end against `MockChatModel`; becomes the README quickstart (§2.6) |
| 6 | Benchmark harness + TriviaQA/GSM8K loaders + graders | 3-condition run over a small real sample produces correct/incorrect labels matching manual spot-check (BENCH-1..3) |
| 7 | Policy sweep + Pareto aggregation | Sweeping 5 thresholds produces 5 distinct router points, not 1 (BENCH-4) |
| 8 | Streamlit dashboard | Reproduces a chart shaped like the SRS's sample Pareto figure from real run data (BENCH-5) |
| 9 | Docs + hardening | README quickstart, API docstrings, `pareto_report.ipynb`, latency/security checks from §6 all pass |

Build strictly in this order — phases 3–9 all depend on 1–2 being correct, and a bug in
the budget ledger is much cheaper to catch in `test_budget.py` than after a benchmark run
has burned real API cost.

---

## 9. Running this with Claude Opus 4.6 / Claude Code

Feed it both documents (the SRS + this plan) and work one phase at a time:

1. Ask for Phase N's implementation only, referencing the DoD row from §8 above.
2. Ask it to write the phase's test file *first* against the interfaces in §3–4, then the
   implementation.
3. Run the tests yourself (or have it run them) before moving to Phase N+1 — don't let it
   batch multiple phases in one shot; the whole point of the phased DoD table is to catch
   a bad `BudgetTracker` before it's load-bearing for four other modules.
4. When it hits one of the four decisions in §0, point it at that section instead of
   letting it invent its own resolution — that's exactly the kind of silent assumption
   that causes rework later.

---

## Appendix: Requirements traceability

| REQ ID | Module | Phase |
|---|---|---|
| ROUTER-1..6 | `costrouter/router.py` | 3 |
| DIFF-1..4 | `costrouter/difficulty.py` | 2 |
| BUDGET-1..7 | `costrouter/budget.py` | 1 |
| BENCH-1..6 | `benchmark/harness.py`, `datasets.py`, `graders.py` | 6–7 |
| §5.1 Performance | instrumentation in `types.py`/`router.py` + CI assertion | 9 |
| §5.2 Safety | `budget_forced` flag + dashboard view | 4, 8 |
| §5.3 Security | `.env` usage, log redaction | 0, 9 |
| §5.4 Quality attributes | module boundaries in §2, `MockChatModel` in tests | all |
| §6 SQLite persistence | `costrouter/store.py` schema | 4 |
