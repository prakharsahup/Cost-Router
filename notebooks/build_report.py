"""Build ``pareto_report.ipynb`` (SRS §2.6 short report) and execute it.

The analysis cells read ``costrouter.db`` and ``reports/summary.json`` written
by ``python -m benchmark.run``; the narrative cells live in ``FINDINGS`` below.

    python notebooks/build_report.py            # build + execute in place
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "pareto_report.ipynb"
FINDINGS_FILE = HERE / "findings.md"

INTRO = """\
# CostRouter — Pareto Report

Short report for SRS §2.6: does difficulty- and budget-aware routing between a
cheap and an expensive model buy a better cost/accuracy trade-off than using
either model alone?

**Setup.** One mixed task set of real benchmark items — TriviaQA (`rc.nocontext`
validation, the "easy" set) and GSM8K (test, the "hard" set), sampled with a
fixed seed and shuffled together, as an agent would see easy and hard steps
interleaved. Every condition runs the same tasks through the same `RouterNode`:

* `always_cheap` / `always_expensive` — baselines (no retries).
* `router t=…` — the default threshold policy at 5 thresholds (REQ-BENCH-4).
  Thresholds sit at the 10/30/50/70/90th percentiles of the estimator's
  prompt-only score (length + reasoning cues), computed from the prompts alone
  — no model calls, no gold labels.
* `router … capped` — the lowest threshold with half the `always_expensive`
  spend as its budget, to exercise budget-forced routing (SRS §5.2).

The router's validator requires a `Final answer:` line; a cheap reply without one
is retry-escalated to the expensive model and counts as a failure in the
per-dataset history. Grading is exact match on the `Final answer:` line
(normalised alias match for TriviaQA, numeric match for GSM8K).

Responses are cached per (model, prompt) at temperature 0, so every condition
sees identical answers and differences between runs come from routing alone.
Costs are each call's real token usage × the provider's list price.
"""

SETUP = """\
import json, sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
conn = sqlite3.connect(f"{(ROOT / 'costrouter.db').as_uri()}?mode=ro", uri=True)
summary = json.loads((ROOT / "reports" / "summary.json").read_text(encoding="utf-8"))
meta = summary["meta"]
runs = pd.DataFrame(summary["runs"])
pd.set_option("display.width", 160)
print(f"Provider: {meta['provider']}")
print(f"Cheap:     {meta['cheap_model']}  (USD/1M tokens in, out: {meta['rates_usd_per_1m']['cheap']})")
print(f"Expensive: {meta['expensive_model']}  (USD/1M tokens in, out: {meta['rates_usd_per_1m']['expensive']})")
print(f"Tasks: {meta['tasks']}  ({meta['n_per_dataset']} TriviaQA + {meta['n_per_dataset']} GSM8K), "
      f"temperature {meta['temperature']}, max_tokens {meta['max_tokens']}, seed {meta['seed']}")
"""

TABLE = """\
table = runs.assign(
    triviaqa=runs.accuracy_by_dataset.map(lambda d: d.get("triviaqa")),
    gsm8k=runs.accuracy_by_dataset.map(lambda d: d.get("gsm8k")),
)[["label", "accuracy", "triviaqa", "gsm8k", "total_cost", "expensive_share",
   "retries", "budget_forced", "errors"]]
base_cost = runs.loc[runs.condition == "always_expensive", "total_cost"].iloc[0]
table["cost_vs_expensive"] = table.total_cost / base_cost
table.style.format({"accuracy": "{:.1%}", "triviaqa": "{:.1%}", "gsm8k": "{:.1%}",
                    "total_cost": "${:.5f}", "expensive_share": "{:.0%}",
                    "cost_vs_expensive": "{:.0%}"}).hide(axis="index")
"""

ORACLE = """\
# Reference points built from the baselines' graded answers (no extra calls):
# - per-dataset oracle: TriviaQA -> expensive, GSM8K -> cheap (picked with hindsight);
# - the dotted "mix" line in the chart: randomly sending a fraction p of tasks to the
#   expensive model.
tr = pd.read_sql("SELECT run_id, task_id, dataset, correct, total_cost_usd FROM task_results", conn)
cheap_tr = tr[tr.run_id == runs.loc[runs.condition == "always_cheap", "run_id"].iloc[0]].set_index("task_id")
exp_tr = tr[tr.run_id == runs.loc[runs.condition == "always_expensive", "run_id"].iloc[0]].set_index("task_id")
pick = pd.concat([exp_tr[exp_tr.dataset == "triviaqa"], cheap_tr[cheap_tr.dataset == "gsm8k"]])
oracle = {"accuracy": pick.correct.mean(), "total_cost": pick.total_cost_usd.sum()}
print(f"Per-dataset oracle: accuracy {oracle['accuracy']:.1%} at ${oracle['total_cost']:.5f} "
      f"({oracle['total_cost'] / base_cost:.0%} of always_expensive)")
print("Per-dataset accuracy and spend, cheap vs expensive:")
pd.DataFrame({
    "cheap_acc": cheap_tr.groupby("dataset").correct.mean(),
    "expensive_acc": exp_tr.groupby("dataset").correct.mean(),
    "cheap_cost": cheap_tr.groupby("dataset").total_cost_usd.sum(),
    "expensive_cost": exp_tr.groupby("dataset").total_cost_usd.sum(),
}).style.format({"cheap_acc": "{:.0%}", "expensive_acc": "{:.0%}",
                 "cheap_cost": "${:.5f}", "expensive_cost": "${:.5f}"})
"""

PARETO = """\
colors = {"always_cheap": "#2ca02c", "always_expensive": "#d62728", "router": "#1f77b4"}
is_cap = runs.run_id.str.contains("budgetcap")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
base = runs[runs.condition != "router"].sort_values("total_cost")
ax.plot(base.total_cost, base.accuracy, ":", color="grey", lw=1, label="random mix of baselines")
sweep = runs[(runs.condition == "router") & ~is_cap].sort_values("total_cost")
ax.plot(sweep.total_cost, sweep.accuracy, "--", color=colors["router"], lw=1, zorder=1)
for cond, group in runs[~is_cap].groupby("condition"):
    ax.scatter(group.total_cost, group.accuracy, s=70, color=colors[cond],
               label=cond if cond != "router" else "router sweep", zorder=2)
# The HARD_STOP run refuses steps once the cap is hit (see the budget section),
# so its accuracy isn't comparable on this chart.
soft = runs[is_cap & ~runs.run_id.str.contains("hardstop")]
ax.scatter(soft.total_cost, soft.accuracy, s=110, facecolors="none", edgecolors="#ff7f0e",
           linewidths=2, zorder=3, label="router, budget-capped (SOFT)")
ax.scatter(oracle["total_cost"], oracle["accuracy"], marker="*", s=220, color="#9467bd", zorder=3,
           label="per-dataset oracle (hindsight reference)")
for k, (_, r) in enumerate(sweep.iterrows()):
    ax.annotate(f"t={r.threshold:g}", (r.total_cost, r.accuracy), textcoords="offset points",
                xytext=(-12, 9 if k % 2 == 0 else -15), fontsize=8)
ax.axvline(runs.loc[is_cap, "budget_usd"].iloc[0], color="#ff7f0e", ls=":", lw=1)
ax.set_xlabel("Total cost for the 100-task set (USD)")
ax.set_ylabel("Accuracy (exact match)")
ax.set_title(f"Cost vs accuracy: {meta['cheap_model']} vs {meta['expensive_model']}")
ax.grid(alpha=0.3)
lo, hi = runs.loc[~runs.run_id.str.contains("hardstop"), "accuracy"].agg(["min", "max"])
ax.set_ylim(lo - 0.015, max(hi, oracle["accuracy"]) + 0.01)
ax.legend(loc="center left", fontsize=8)
fig.tight_layout()
(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs" / "pareto.png", dpi=150)
plt.show()
"""

ROUTING = """\
dec = pd.read_sql(
    "SELECT d.run_id, d.task_id, d.model_chosen, d.retried, d.budget_forced, d.difficulty_score "
    "FROM routing_decisions d JOIN runs r USING (run_id) WHERE r.condition = 'router'", conn)
dec["dataset"] = dec.task_id.str.rsplit("-", n=1).str[0]
dec["expensive"] = dec.model_chosen.eq("expensive")
labels = runs.set_index("run_id").label
share = (dec.groupby(["run_id", "dataset"]).expensive.mean().unstack()
            .rename(index=labels).sort_values("gsm8k"))
retry = dec.groupby(["run_id", "dataset"]).retried.sum().unstack().rename(index=labels)
print("Share of steps answered by the expensive model, per dataset:")
display(share.style.format("{:.0%}"))
print("Retry-escalations (cheap reply had no 'Final answer:' line):")
display(retry.loc[share.index])
"""

BUDGET = """\
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for _, r in runs[is_cap].iterrows():
    hard = "hardstop" in r.run_id
    traj = pd.read_sql("SELECT budget_remaining_after, budget_forced FROM routing_decisions "
                       "WHERE run_id = ? ORDER BY id", conn, params=(r.run_id,))
    ax.plot(range(1, len(traj) + 1), traj.budget_remaining_after,
            color="#d62728" if hard else "#1f77b4",
            label=f"{'HARD_STOP' if hard else 'SOFT'}: {len(traj)} steps answered, "
                  f"{int(traj.budget_forced.sum())} budget-forced")
ax.axhline(0, color="black", lw=0.8)
for pct in (0.5, 0.1):
    ax.axhline(pct * r.budget_usd, color="grey", ls=":", lw=1)
    ax.text(1, pct * r.budget_usd, f" {pct:.0%} warning", va="bottom", fontsize=8)
ax.set_xlabel("Step (decisions logged)")
ax.set_ylabel("Remaining budget (USD)")
ax.set_title(f"Budget-capped router runs: cap ${r.budget_usd:g} (half of always_expensive)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()
for _, r in runs[is_cap].iterrows():
    mode = "HARD_STOP" if "hardstop" in r.run_id else "SOFT"
    print(f"{mode:<9} spent ${r.total_cost:.5f} of ${r.budget_usd:g}, accuracy {r.accuracy:.0%}, "
          f"{r.budget_forced} budget-forced decisions, {r.errors} steps not answered")
events = pd.read_sql("SELECT r.run_id, e.threshold_pct, e.remaining_budget FROM budget_events e "
                     "JOIN runs r USING (run_id) ORDER BY e.id", conn)
events
"""

SPOT = """\
spot = pd.read_csv(ROOT / "reports" / "spot_check.csv")
print(spot.groupby(["run", "dataset"]).correct.agg(["mean", "count"]))
spot.sample(12, random_state=0)
"""


def build(execute: bool = True) -> None:
    findings = FINDINGS_FILE.read_text(encoding="utf-8") if FINDINGS_FILE.is_file() else (
        "## Findings\n\n_Run the benchmark, then fill in `notebooks/findings.md`._"
    )
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        md(INTRO),
        code(SETUP),
        md("## Results by condition\n\n`cost_vs_expensive` is each run's spend relative to "
           "`always_expensive`."),
        code(TABLE),
        md("## Where the two models differ\n\nThe headroom available to *any* router is set "
           "by where the cheap model loses accuracy."),
        code(ORACLE),
        md("## Pareto view (REQ-BENCH-5)\n\nCost on x, accuracy on y. The dashed line joins "
           "the router sweep, the dotted line is a random mix of the two baselines, and the "
           "vertical line marks the budget cap of the two capped runs."),
        code(PARETO),
        md("## How the router spent its budget\n\nWhere the expensive calls went, by dataset, "
           "and how many came from retry-escalation."),
        code(ROUTING),
        md("## Budget-capped runs (SRS §5.2: degradation is visible, not hidden)\n\n"
           "Same router, budget capped at half the `always_expensive` spend, in each "
           "`BudgetMode`."),
        code(BUDGET),
        md("## Grading spot-check (Phase 6)\n\nEvery baseline answer is in "
           "`reports/spot_check.csv` (graded span vs gold aliases). A random sample:"),
        code(SPOT),
        md(findings),
    ]
    if execute:
        ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(HERE)}})
    nbformat.write(nb, NOTEBOOK)
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    build(execute="--no-execute" not in sys.argv)
