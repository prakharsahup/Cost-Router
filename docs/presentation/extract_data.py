"""Export the numbers the slide deck uses (from costrouter.db) to deck_data.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard import engine  # noqa: E402
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator  # noqa: E402

db = str(ROOT / "costrouter.db")
runs = engine.load_runs(db)
meta = engine.load_meta(db)
dec = engine.load_decisions(db)
oracle = engine.oracle_point(db)

cols = ["run_id", "kind", "label", "threshold", "budget_usd", "accuracy", "acc_triviaqa",
        "acc_gsm8k", "total_cost", "expensive_share", "retries", "budget_forced", "unanswered"]
out_runs = [{k: (None if v != v else v) for k, v in r.items()} for r in runs[cols].to_dict("records")]

capped = {}
for _, r in runs[runs.kind.isin(["capped_soft", "capped_hard"])].iterrows():
    d = dec[dec.run_id == r.run_id]
    capped[r.kind] = {"remaining": [round(x, 7) for x in d.budget_remaining_after], "cap": r.budget_usd}

share = {}
for _, r in runs[runs.kind == "router"].iterrows():
    d = dec[dec.run_id == r.run_id]
    share[r.label] = d.assign(e=d.model_chosen.eq("expensive")).groupby("dataset").e.mean().to_dict()

# Difficulty breakdown for demo prompts, as a fresh live session sees them (cold start).
est = HeuristicDifficultyEstimator(FailureRateStore(prior=0.5, min_samples=5))
examples = []
for label in ["Capital of Australia", "Plain math word problem", "Cheapest way to buy 22 pens",
              "Train: derive the average speed"]:
    item = next(i for i in engine.SHOWCASE if i["label"] == label)
    ex = est.explain(engine.showcase_prompt(item), item["step_type"])
    examples.append({"label": label, "kind": item["kind"], **ex["contributions"],
                     "score": ex["score"], "cue_words": ex["matched_cues"]})

data = {"meta": meta, "runs": out_runs, "oracle": oracle, "capped": capped,
        "expensive_share": share, "difficulty_examples": examples,
        "tasks": int(runs.task_count.max())}
(Path(__file__).parent / "deck_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print(json.dumps({k: data[k] for k in ("oracle", "difficulty_examples", "expensive_share")}, indent=1))
