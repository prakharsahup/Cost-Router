"""Data + simulation layer for the CostRouter dashboard (no Streamlit imports).

- ``load_*``: read the benchmark database written by ``python -m benchmark.run``.
- ``replay``: re-run the real ``RouterNode`` over the 100 cached benchmark
  tasks with any threshold / budget / mode, in well under a second and with
  no API calls (every answer comes from the committed response cache).
- ``LiveSession``: a persistent router for the "Live Router" tab.  Answers
  come from the benchmark cache, then the committed showcase cache, then the
  live provider (if an API key is set) — so preset questions work offline.

Pre-fill the showcase cache (needs ``GROQ_API_KEY``)::

    python -m dashboard.engine --prewarm
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from benchmark.graders import final_answer_segment, grade_gsm8k, grade_triviaqa  # noqa: E402
from benchmark.harness import item_dataset, run_condition  # noqa: E402
from benchmark.prompts import CachedChatModel, has_final_answer, load_mixed  # noqa: E402
from benchmark.run import PROVIDERS, api_key_for  # noqa: E402
from costrouter.budget import BudgetTracker  # noqa: E402
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator  # noqa: E402
from costrouter.exceptions import BudgetExhaustedError  # noqa: E402
from costrouter.router import RouterNode  # noqa: E402
from costrouter.store import init_db  # noqa: E402
from costrouter.types import BudgetMode, ModelResponse, ModelTier, RouterConfig  # noqa: E402

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "response_cache"
SHOWCASE_CACHE = CACHE_DIR / "groq_showcase.jsonl"
LIVE_CACHE = ROOT / "results" / "live_cache.jsonl"  # gitignored
GRADERS = {"triviaqa": grade_triviaqa, "gsm8k": grade_gsm8k}

# Completion cap for live / showcase calls.  Higher than the benchmark's 1024
# because the expensive tier is never retried, and gpt-oss reasoning can run long.
LIVE_MAX_TOKENS = 2048

CUSTOM_TEMPLATE = "{question}\n\nEnd your reply with a line of the form 'Final answer: <answer>'."

# Preset questions for the live demo.  ``step_type`` keys the failure history.
SHOWCASE = [
    {"label": "Capital of Australia", "kind": "easy", "step_type": "trivia",
     "question": "What is the capital of Australia?"},
    {"label": "Who painted The Starry Night?", "kind": "easy", "step_type": "trivia",
     "question": "Who painted The Starry Night?"},
    {"label": "Cheapest way to buy 22 pens", "kind": "hard", "step_type": "math",
     "question": "A shop sells pens at $3 each or 4 for $10. Priya needs 22 pens and wants to "
                 "pay as little as possible. Analyze the options step by step, compare the "
                 "costs and justify the cheapest way to buy them."},
    {"label": "Train: derive the average speed", "kind": "hard", "step_type": "math",
     "question": "Prove step by step and justify each move: a train covers 180 km in 2.5 hours "
                 "and then 120 km in 1.5 hours. Compare its two average speeds and derive the "
                 "overall average speed."},
    {"label": "UK quiz-show trivia", "kind": "tricky", "step_type": "trivia",
     "task_id": "triviaqa-16",
     "question": "Judith Keppel was the first winner of the top prize on which UK programme?",
     "note": "From the benchmark: the small model spends its whole token budget thinking "
             "and returns nothing, so the validator triggers retry-escalate."},
    {"label": "Plain math word problem", "kind": "plain", "step_type": "math",
     "task_id": "gsm8k-41",
     "question": "Last year there were 50 students enrolled in a calligraphy class. This year, "
                 "there was a 20% increase in enrollment. How many students are enrolled this "
                 "year?",
     "note": "From the benchmark: no reasoning keywords and short, so the heuristic calls it "
             "easy — and the cheap model gets it right."},
    {"label": "Compare two job offers", "kind": "hard", "step_type": "reasoning",
     "question": "Offer A pays $85,000 with a $5,000 signing bonus; offer B pays $80,000 plus a "
                 "10% annual bonus. Compare the first-year totals, analyze the trade-offs and "
                 "justify which pays more in year one."},
]


def showcase_prompt(item: dict) -> str:
    """The exact prompt sent for a preset (benchmark presets reuse the benchmark prompt)."""
    if "task_id" in item:
        return next(p for tid, p, _ in benchmark_items() if tid == item["task_id"])
    return CUSTOM_TEMPLATE.format(question=item["question"])


# ---------------------------------------------------------------------------
# Benchmark database
# ---------------------------------------------------------------------------

def default_db_path() -> str:
    return os.environ.get("COSTROUTER_DB_PATH", str(ROOT / "costrouter.db"))


def open_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def load_runs(db_path: str) -> pd.DataFrame:
    """One row per finished run, with accuracy, cost and routing counts."""
    with open_readonly(db_path) as conn:
        runs = pd.read_sql(
            "SELECT run_id, condition, task_set, policy_config_json, started_at FROM runs "
            "WHERE finished_at IS NOT NULL ORDER BY started_at", conn)
        tasks = pd.read_sql("SELECT run_id, task_id, dataset, correct, total_cost_usd "
                            "FROM task_results", conn)
        decisions = pd.read_sql("SELECT run_id, model_chosen, retried, budget_forced "
                                "FROM routing_decisions", conn)
    if runs.empty:
        return runs
    cfg = runs.policy_config_json.map(lambda s: json.loads(s) if s else {})
    runs["threshold"] = cfg.map(lambda c: c.get("threshold"))
    runs["budget_usd"] = cfg.map(lambda c: c.get("budget_usd"))
    runs["budget_mode"] = cfg.map(lambda c: c.get("budget_mode", "soft"))
    agg = tasks.groupby("run_id").agg(accuracy=("correct", "mean"),
                                      total_cost=("total_cost_usd", "sum"),
                                      task_count=("task_id", "count"))
    by_ds = tasks.pivot_table(index="run_id", columns="dataset", values="correct", aggfunc="mean")
    by_ds.columns = [f"acc_{c}" for c in by_ds.columns]
    dec = decisions.assign(expensive=decisions.model_chosen.eq("expensive")).groupby("run_id").agg(
        expensive_share=("expensive", "mean"), retries=("retried", "sum"),
        budget_forced=("budget_forced", "sum"), decisions=("model_chosen", "count"))
    out = runs.join(agg, on="run_id").join(by_ds, on="run_id").join(dec, on="run_id")
    out["unanswered"] = out.task_count - out.decisions.fillna(0)
    out["label"] = out.apply(_run_label, axis=1)
    out["kind"] = out.apply(_run_kind, axis=1)
    return out


def _run_kind(r) -> str:
    if r.condition != "router":
        return r.condition
    if "budgetcap" in r.run_id:
        return "capped_hard" if r.budget_mode == "hard_stop" else "capped_soft"
    return "router"


def _run_label(r) -> str:
    if r.condition == "always_cheap":
        return "Always cheap"
    if r.condition == "always_expensive":
        return "Always expensive"
    base = f"Router t={r.threshold:g}"
    if "budgetcap" in r.run_id:
        mode = "HARD_STOP" if r.budget_mode == "hard_stop" else "SOFT"
        return f"{base} · cap ${r.budget_usd:.4f} · {mode}"
    return base


def load_meta(db_path: str) -> dict:
    """Run configuration (provider, models, prices) stored on the baseline run."""
    with open_readonly(db_path) as conn:
        row = conn.execute("SELECT policy_config_json FROM runs WHERE condition = 'always_cheap' "
                           "AND finished_at IS NOT NULL ORDER BY started_at LIMIT 1").fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def load_decisions(db_path: str) -> pd.DataFrame:
    with open_readonly(db_path) as conn:
        df = pd.read_sql("SELECT d.*, t.correct FROM routing_decisions d LEFT JOIN task_results t "
                         "ON t.run_id = d.run_id AND t.task_id = d.task_id ORDER BY d.id", conn)
    if not df.empty:
        df["dataset"] = df.task_id.map(item_dataset)
    return df


def load_budget_events(db_path: str) -> pd.DataFrame:
    with open_readonly(db_path) as conn:
        return pd.read_sql("SELECT * FROM budget_events ORDER BY id", conn)


def load_task_results(db_path: str) -> pd.DataFrame:
    with open_readonly(db_path) as conn:
        return pd.read_sql("SELECT * FROM task_results", conn)


def oracle_point(db_path: str) -> Optional[dict]:
    """Hindsight reference: per dataset, the baseline that was more accurate per dollar.

    Picks, for each dataset, whichever baseline has higher accuracy (the cheaper
    one on ties) — i.e. the best a router keyed only on task type could do.
    """
    runs = load_runs(db_path)
    tr = load_task_results(db_path)
    ids = {c: runs.loc[runs.condition == c, "run_id"] for c in ("always_cheap", "always_expensive")}
    if any(v.empty for v in ids.values()):
        return None
    cheap = tr[tr.run_id == ids["always_cheap"].iloc[0]]
    exp = tr[tr.run_id == ids["always_expensive"].iloc[0]]
    picks, parts = {}, []
    for ds in sorted(tr.dataset.unique()):
        c, e = cheap[cheap.dataset == ds], exp[exp.dataset == ds]
        use_exp = e.correct.mean() > c.correct.mean()
        picks[ds] = "expensive" if use_exp else "cheap"
        parts.append(e if use_exp else c)
    pick = pd.concat(parts)
    return {"accuracy": pick.correct.mean(), "total_cost": pick.total_cost_usd.sum(), "picks": picks}


# ---------------------------------------------------------------------------
# Replay simulator
# ---------------------------------------------------------------------------

def benchmark_items():
    return load_mixed(str(DATA_DIR / "triviaqa_sample.json"),
                      str(DATA_DIR / "gsm8k_sample.jsonl"), limit=50, seed=42)


def rate_card_from_meta(meta: dict) -> dict:
    rates = meta.get("rates_usd_per_1m") or {}
    spec = PROVIDERS[meta.get("provider", "groq")]
    out = {}
    for tier in (ModelTier.CHEAP, ModelTier.EXPENSIVE):
        name = meta.get(f"{tier.value}_model", spec[tier.value])
        p, c = rates.get(tier.value) or spec["rates"][name]
        out[tier] = {"prompt": p / 1e6, "completion": c / 1e6}
    return out


def replay(meta: dict, threshold: float, budget_usd: float, hard_stop: bool) -> dict:
    """Run the real RouterNode over the cached benchmark tasks.  No network."""
    provider = meta.get("provider", "groq")
    cache = CACHE_DIR / f"{provider}.jsonl"
    names = {t: meta.get(f"{t}_model", PROVIDERS[provider][t]) for t in ("cheap", "expensive")}
    models = {t: CachedChatModel(None, n, cache, offline=True) for t, n in names.items()}
    t0 = time.perf_counter()
    result = run_condition(
        items=benchmark_items(), condition="router", dataset_name="mixed", grader=GRADERS,
        cheap_model=models["cheap"], expensive_model=models["expensive"],
        rate_card=rate_card_from_meta(meta), budget_usd=budget_usd, threshold=threshold,
        conn=init_db(":memory:"), validator=has_final_answer,
        max_completion_tokens=meta.get("max_completion_tokens"),
        budget_mode=BudgetMode.HARD_STOP if hard_stop else BudgetMode.SOFT,
    )
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
    per = pd.DataFrame(result["per_task"])
    per["step"] = range(1, len(per) + 1)
    per["spent"] = per.cost_usd.cumsum()
    per["remaining"] = budget_usd - per.spent
    result["per_task_df"] = per
    answered = per.model_chosen.notna()
    result["expensive_share"] = per.model_chosen.eq("expensive").sum() / max(answered.sum(), 1)
    result["retries"] = int(per.retried.sum())
    result["budget_forced"] = int(per.budget_forced.sum())
    result["accuracy_by_dataset"] = per.groupby("dataset").correct.mean().to_dict()
    return result


# ---------------------------------------------------------------------------
# Live router session
# ---------------------------------------------------------------------------

class LiveModel:
    """Cached answers first (benchmark + showcase), then the live provider."""

    def __init__(self, name: str, raw: Any, read_caches: list[Path], write_cache: Optional[Path]):
        self.name = name
        self._ro = [CachedChatModel(None, name, p, offline=True) for p in read_caches]
        self._rw = CachedChatModel(raw, name, write_cache) if raw is not None and write_cache else None
        self.last_source = None

    def invoke(self, prompt: str, **kwargs: Any) -> ModelResponse:
        for cache in self._ro:
            resp = cache.invoke(prompt)
            if not resp.error:
                self.last_source = "cache"
                return resp
        if self._rw is None:
            self.last_source = "offline"
            return ModelResponse(text="", error="offline: no API key and no cached answer")
        hits = self._rw.hits
        resp = self._rw.invoke(prompt, **kwargs)
        self.last_source = "cache" if self._rw.hits > hits else "live"
        return resp


def build_models(meta: dict, *, write_cache: Optional[Path] = LIVE_CACHE,
                 extra_read: tuple[Path, ...] = (SHOWCASE_CACHE,)) -> tuple[dict, bool]:
    """``({"cheap": LiveModel, "expensive": LiveModel}, live_available)``."""
    provider = meta.get("provider", "groq")
    spec = PROVIDERS[provider]
    names = {t: meta.get(f"{t}_model", spec[t]) for t in ("cheap", "expensive")}
    raw = {}
    offline = os.environ.get("COSTROUTER_OFFLINE") == "1"
    key = None if offline else api_key_for(spec)
    if key:
        from costrouter.adapters import OpenAICompatibleChatModel
        for tier, name in names.items():
            raw[tier] = OpenAICompatibleChatModel(
                name, base_url=spec["base_url"], api_key=key,
                max_tokens=LIVE_MAX_TOKENS, max_retries=3, timeout=90)
    reads = [CACHE_DIR / f"{provider}.jsonl", *extra_read]
    models = {t: LiveModel(n, raw.get(t), reads, write_cache) for t, n in names.items()}
    return models, bool(key)


@dataclass
class LiveSession:
    """A persistent router whose budget and failure history carry across questions."""

    meta: dict
    budget_usd: float = 0.01
    threshold: float = 0.3
    hard_stop: bool = False
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.models, self.live_available = build_models(self.meta)
        self.store = FailureRateStore(prior=0.5, min_samples=5)
        self.estimator = HeuristicDifficultyEstimator(self.store)
        self.budget = BudgetTracker(
            self.budget_usd, rate_card_from_meta(self.meta),
            mode=BudgetMode.HARD_STOP if self.hard_stop else BudgetMode.SOFT,
            on_threshold=self._on_threshold)
        self.router = RouterNode(
            self.models["cheap"], self.models["expensive"], self.estimator, self.budget,
            RouterConfig(threshold=self.threshold, validator=has_final_answer,
                         max_completion_tokens=LIVE_MAX_TOKENS),
            failure_store=self.store)
        self._events: list = []

    def _on_threshold(self, pct: float, remaining: float) -> None:
        self._events.append({"pct": pct, "remaining": remaining})

    def route(self, prompt: str, step_type: str, label: str) -> dict:
        explain = self.estimator.explain(prompt, step_type)
        history_n = self.store.observations(step_type)
        before = self.budget.remaining_budget()
        self._events = []
        for m in self.models.values():
            m.last_source = None
        entry = {"n": len(self.history) + 1, "label": label, "step_type": step_type,
                 "prompt": prompt, "explain": explain, "history_n": history_n,
                 "threshold": self.threshold, "budget_before": before}
        t0 = time.perf_counter()
        try:
            out = self.router({"step_id": f"live-{entry['n']}", "step_type": step_type,
                               "prompt": prompt})
        except BudgetExhaustedError:
            entry.update(status="refused", tier=None, cost=0.0, text="",
                         error="HARD_STOP: budget exhausted — call refused")
        except Exception as exc:  # malformed output that couldn't be escalated, etc.
            entry.update(status="error", tier=None,
                         cost=before - self.budget.remaining_budget(), text="",
                         error=f"{type(exc).__name__}: {exc}")
        else:
            d = out["last_decision"]
            entry.update(status="ok", tier=d.model_chosen.value, cost=d.actual_cost_usd,
                         text=out["response"], budget_forced=d.budget_forced,
                         retried=d.retried, error=d.error,
                         model=self.models[d.model_chosen.value].name,
                         source=self.models[d.model_chosen.value].last_source)
        entry["wall_ms"] = (time.perf_counter() - t0) * 1000
        entry["budget_after"] = self.budget.remaining_budget()
        entry["events"] = list(self._events)
        entry["final_answer"] = final_answer_segment(entry["text"]).strip("* ") if entry["text"] else ""
        self.history.append(entry)
        return entry


# ---------------------------------------------------------------------------
# CLI: pre-fill the showcase cache so presets work offline
# ---------------------------------------------------------------------------

def prewarm_showcase(db_path: str) -> None:
    meta = load_meta(db_path)
    models, live = build_models(meta, write_cache=SHOWCASE_CACHE, extra_read=())
    if not live:
        sys.exit("Set GROQ_API_KEY in .env to pre-fill the showcase cache.")
    for item in SHOWCASE:
        for tier, m in models.items():
            r = m.invoke(showcase_prompt(item))
            status = r.error or f"{m.last_source}: ...{r.text[-60:]!r}"
            print(f"{item['label']:<34} {tier:<9} {status}")


if __name__ == "__main__":
    if "--prewarm" in sys.argv:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        prewarm_showcase(default_db_path())
