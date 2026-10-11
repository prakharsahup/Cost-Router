"""Run the real CostRouter benchmark end to end and write ``costrouter.db``.

Steps:

1. Load the TriviaQA + GSM8K samples from ``data/`` (``download_data.py``),
   template them and shuffle them into one mixed task set.
2. Pre-warm a response cache: call the cheap and the expensive model once on
   every prompt (the only calls any condition can make).
3. Run every condition from that cache into SQLite:
   ``always_cheap``, ``always_expensive``, a 5-point router threshold sweep
   (REQ-BENCH-4), and one budget-capped router run that exercises
   budget-forced routing and threshold events.
4. Print a summary and write ``reports/summary.json`` and
   ``reports/spot_check.csv`` (every graded answer, for manual checking).

Providers are OpenAI-compatible endpoints; the API key is read from the
environment (or ``.env``) and never logged::

    python -m benchmark.run --provider groq --fresh
    python -m benchmark.run --offline --fresh      # rebuild the DB from the cache
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from costrouter.types import BudgetMode, ModelTier
from costrouter.store import init_db

from benchmark.graders import final_answer_segment, grade_gsm8k, grade_triviaqa
from benchmark.harness import prompt_score_thresholds, run_condition
from benchmark.prompts import CachedChatModel, has_final_answer, load_mixed

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# Prices are USD per 1M tokens (prompt, completion).  Groq/OpenAI prices are
# the published list prices at the time of writing — override with
# --cheap-rates / --expensive-rates if they change.  OpenRouter prices are
# fetched live from its public model list.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "cheap": "openai/gpt-oss-20b",
        "expensive": "openai/gpt-oss-120b",
        # console.groq.com/docs/models, checked 2026-10-12
        "rates": {"openai/gpt-oss-20b": (0.075, 0.30),
                  "openai/gpt-oss-120b": (0.15, 0.60)},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "cheap": "meta-llama/llama-3.1-8b-instruct",
        "expensive": "meta-llama/llama-3.3-70b-instruct",
        "rates": {},
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "cheap": "gpt-4o-mini",
        "expensive": "gpt-4o",
        "rates": {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)},
    },
}

GRADERS = {"triviaqa": grade_triviaqa, "gsm8k": grade_gsm8k}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def api_key_for(spec: dict) -> str | None:
    """The provider key from the environment, ignoring ``.env.example`` placeholders."""
    key = os.environ.get(spec["key_env"], "").strip()
    return key if key and not key.endswith("...") else None


def detect_provider(offline: bool) -> str:
    for name, spec in PROVIDERS.items():
        if api_key_for(spec):
            return name
    if offline:
        for name in PROVIDERS:
            if (DATA_DIR / "response_cache" / f"{name}.jsonl").is_file():
                return name
    sys.exit("No provider key found. Put GROQ_API_KEY or OPENROUTER_API_KEY in .env "
             "(or pass --offline to rebuild from an existing cache).")


def openrouter_rates(model: str) -> tuple[float, float]:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        for m in json.load(r)["data"]:
            if m["id"] == model:
                p = m["pricing"]
                return float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
    sys.exit(f"OpenRouter model {model!r} not found in its model list.")


def parse_rates(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    prompt, completion = (float(x) for x in text.split(","))
    return prompt, completion


def prewarm(models: dict[str, CachedChatModel], items, workers: int, attempts: int) -> int:
    """Fill the cache for every (model, prompt).  Returns the number of misses left."""
    jobs = [(tier, prompt) for tier in models for _, prompt, _ in items]
    done = failed = 0

    def call(tier: str, prompt: str) -> bool:
        for attempt in range(attempts):
            resp = models[tier].invoke(prompt)
            if not resp.error:
                return True
            log(f"  [{tier}] attempt {attempt + 1}/{attempts} failed: {resp.error[:160]}")
            time.sleep(min(60, 5 * 2 ** attempt))
        return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, t, p) for t, p in jobs]
        for fut in as_completed(futures):
            done += 1
            failed += not fut.result()
            if done % 20 == 0 or done == len(jobs):
                log(f"  pre-warm {done}/{len(jobs)} ({failed} failed)")
    return failed


def write_reports(results: list[dict], items, meta: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    gold = {tid: g for tid, _, g in items}
    summary = {
        "meta": meta,
        "runs": [
            {k: r[k] for k in ("run_id", "condition", "label", "threshold", "budget_usd",
                               "accuracy", "total_cost", "correct_count", "total_count",
                               "errors", "expensive_share", "retries", "budget_forced",
                               "accuracy_by_dataset")}
            for r in results
        ],
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (REPORTS_DIR / "spot_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "task_id", "dataset", "model", "correct", "graded_span", "gold"])
        for r in results:
            if r["condition"] == "router":
                continue  # router answers are copies of these two
            for t in r["per_task"]:
                g = gold[t["task_id"]]
                w.writerow([r["condition"], t["task_id"], t["dataset"], t["model_chosen"],
                            int(t["correct"]), final_answer_segment(t["answer"])[:200],
                            " | ".join(g[:5]) if isinstance(g, list) else g])


def summarize(r: dict, conn) -> None:
    rows = conn.execute(
        "SELECT model_chosen, retried, budget_forced FROM routing_decisions WHERE run_id = ?",
        (r["run_id"],),
    ).fetchall()
    n = max(len(rows), 1)
    r["expensive_share"] = sum(m == "expensive" for m, _, _ in rows) / n
    r["retries"] = sum(x for _, x, _ in rows)
    r["budget_forced"] = sum(x for _, _, x in rows)
    by_ds: dict[str, list[int]] = {}
    for t in r["per_task"]:
        by_ds.setdefault(t["dataset"], []).append(int(t["correct"]))
    r["accuracy_by_dataset"] = {k: sum(v) / len(v) for k, v in sorted(by_ds.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CostRouter benchmark.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--cheap-model")
    parser.add_argument("--expensive-model")
    parser.add_argument("--cheap-rates", help="USD per 1M tokens: 'prompt,completion'")
    parser.add_argument("--expensive-rates", help="USD per 1M tokens: 'prompt,completion'")
    parser.add_argument("--n", type=int, default=50, help="items per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="completion cap (reasoning models count hidden reasoning tokens)")
    parser.add_argument("--budget", type=float, default=10.0, help="USD budget per run")
    parser.add_argument("--db", default=str(ROOT / "costrouter.db"))
    parser.add_argument("--fresh", action="store_true", help="delete --db first")
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=6, help="pre-warm attempts per call")
    args = parser.parse_args()

    load_dotenv_if_available()
    provider = args.provider or detect_provider(args.offline)
    spec = PROVIDERS[provider]
    cheap_name = args.cheap_model or spec["cheap"]
    expensive_name = args.expensive_model or spec["expensive"]

    rates = {}
    for tier, name, override in ((ModelTier.CHEAP, cheap_name, args.cheap_rates),
                                 (ModelTier.EXPENSIVE, expensive_name, args.expensive_rates)):
        per_m = parse_rates(override) or spec["rates"].get(name)
        if per_m is None and provider == "openrouter":
            per_m = openrouter_rates(name)
        if per_m is None:
            sys.exit(f"No price known for {name!r}; pass --{tier.value}-rates.")
        rates[tier] = per_m
    rate_card = {t: {"prompt": p / 1e6, "completion": c / 1e6} for t, (p, c) in rates.items()}

    tq_path, gsm_path = DATA_DIR / "triviaqa_sample.json", DATA_DIR / "gsm8k_sample.jsonl"
    if not (tq_path.is_file() and gsm_path.is_file()):
        # The loaders would silently fall back to their built-in mini sets.
        sys.exit("data/ samples missing — run: python -m benchmark.download_data")
    items = load_mixed(str(tq_path), str(gsm_path), limit=args.n, seed=args.seed)
    log(f"Provider: {provider} | cheap={cheap_name} {rates[ModelTier.CHEAP]} | "
        f"expensive={expensive_name} {rates[ModelTier.EXPENSIVE]} (USD/1M tokens)")
    log(f"Tasks: {len(items)} (mixed TriviaQA + GSM8K)")

    raw = {}
    if not args.offline:
        from costrouter.adapters import OpenAICompatibleChatModel

        api_key = api_key_for(spec)
        if not api_key:
            sys.exit(f"{spec['key_env']} is not set (put it in .env).")
        for tier, name in (("cheap", cheap_name), ("expensive", expensive_name)):
            raw[tier] = OpenAICompatibleChatModel(
                name, base_url=spec["base_url"], api_key=api_key,
                seed=args.seed, max_tokens=args.max_tokens,
            )
    cache_path = DATA_DIR / "response_cache" / f"{provider}.jsonl"
    models = {
        tier: CachedChatModel(raw.get(tier), name, cache_path, offline=args.offline)
        for tier, name in (("cheap", cheap_name), ("expensive", expensive_name))
    }

    if not args.offline:
        log("Pre-warming response cache ...")
        left = prewarm(models, items, args.workers, args.attempts)
        if left:
            log(f"WARNING: {left} calls still failing; those tasks will be graded wrong. "
                "Re-run to retry them (successful calls are cached).")
    for m in models.values():
        m.hits = m.misses = 0
        m._offline = True  # conditions run purely from the cache

    if args.fresh and Path(args.db).exists():
        for suffix in ("", "-wal", "-shm"):
            Path(args.db + suffix).unlink(missing_ok=True)
    conn = init_db(args.db)

    thresholds = prompt_score_thresholds(items)
    run_config = {"provider": provider, "cheap_model": cheap_name,
                  "expensive_model": expensive_name,
                  "rates_usd_per_1m": {t.value: list(v) for t, v in rates.items()},
                  "temperature": 0.0, "n_per_dataset": args.n,
                  "sweep_thresholds": thresholds, "validator": "has_final_answer"}
    common = dict(items=items, dataset_name="mixed", grader=GRADERS,
                  cheap_model=models["cheap"], expensive_model=models["expensive"],
                  rate_card=rate_card, conn=conn, seed=args.seed,
                  max_completion_tokens=args.max_tokens, run_config=run_config)

    plan = [("always_cheap", "always_cheap", None, args.budget),
            ("always_expensive", "always_expensive", None, args.budget)]
    plan += [("router", f"router t={t:g}", t, args.budget) for t in thresholds]

    results = []
    for condition, label, threshold, budget in plan:
        r = run_condition(condition=condition, threshold=threshold if threshold is not None else 0.5,
                          budget_usd=budget, validator=has_final_answer,
                          run_id=f"{label.replace(' ', '-').replace('=', '')}-mixed", **common)
        r.update(label=label, threshold=threshold, budget_usd=budget)
        summarize(r, conn)
        results.append(r)
        log(f"  {label:<34} acc={r['accuracy']:.1%}  cost=${r['total_cost']:.5f}  "
            f"expensive={r['expensive_share']:.0%}  retries={r['retries']}  errors={r['errors']}")

    # Extra router runs, all replayed from the cache:
    # - budget-capped (SOFT and HARD_STOP): the most expensive-leaning
    #   threshold with half the always_expensive spend — shows budget-forced
    #   downgrades, threshold events, and how each mode treats the cap.
    cap = round(results[1]["total_cost"] * 0.5, 6)
    t_lo = thresholds[0]
    extras = [
        (f"router t={t_lo:g} capped ${cap:g} soft", f"router-t{t_lo:g}-budgetcap-mixed",
         dict(threshold=t_lo, budget_usd=cap)),
        (f"router t={t_lo:g} capped ${cap:g} hard", f"router-t{t_lo:g}-budgetcap-hardstop-mixed",
         dict(threshold=t_lo, budget_usd=cap, budget_mode=BudgetMode.HARD_STOP)),
    ]
    for label, run_id, kw in extras:
        r = run_condition(condition="router", validator=has_final_answer, run_id=run_id,
                          **kw, **common)
        r.update(label=label, threshold=kw["threshold"], budget_usd=kw["budget_usd"])
        summarize(r, conn)
        results.append(r)
        log(f"  {label:<34} acc={r['accuracy']:.1%}  cost=${r['total_cost']:.5f}  "
            f"expensive={r['expensive_share']:.0%}  forced={r['budget_forced']}  "
            f"errors={r['errors']}")

    meta = {**run_config, "max_tokens": args.max_tokens, "seed": args.seed,
            "tasks": len(items), "budget_cap_usd": cap, "db": Path(args.db).name}
    write_reports(results, items, meta)
    conn.close()
    log(f"Done. DB: {args.db} | reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
