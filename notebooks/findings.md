## Findings

**1. The two models differ only on TriviaQA.** `gpt-oss-20b` and `gpt-oss-120b` both score
96% on GSM8K; the whole 5-point gap (84% vs 89% overall) is on TriviaQA, 72% vs 82%. The
expensive model costs 2× per token, but because the small model spends more hidden reasoning
tokens, `always_expensive` costs only 1.42× `always_cheap` ($0.0134 vs $0.0094 for 100 tasks).
On TriviaQA the cheap model saves almost nothing ($0.0045 vs $0.0049).

**2. The v1 heuristic router lands on, not above, the random-mix line.** The 5-threshold sweep
gives 5 distinct points (REQ-BENCH-4): 87–88% accuracy at 93–102% of the `always_expensive`
cost, with 53–98% of steps on the expensive model. That is about what randomly mixing the two
models would give. The reason is visible in the routing table: the length + reasoning-cue score
says the long GSM8K problems are hard, so they go to the expensive model, where nothing is
gained. The failure-history signal pulls the other way — truncated cheap TriviaQA replies fail
validation, which raises TriviaQA's history score (50% of TriviaQA went expensive at the
highest threshold, against ~0% predicted from the prompts alone) — but not enough to fix it.

**3. The headroom is real.** Routing by dataset with hindsight (TriviaQA → expensive,
GSM8K → cheap) would match `always_expensive`'s 89% at **73% of its cost**. A step-type-aware
policy, or the learned estimator that REQ-DIFF-2 leaves room for, is the next step. The lesson
for v1: prompt length is a poor difficulty proxy when the cheap model is itself a reasoning model.

**4. Retry-escalate earns its keep.** 7 of the 100 cheap replies were empty: `gpt-oss-20b`
spent the entire 1,024-token cap on hidden reasoning (6 of the 7 were TriviaQA). The router's
`Final answer:` validator caught every one that reached the cheap model and re-ran it on the
expensive model (1–4 escalations per run), which is why every router point beats
`always_cheap` on accuracy. Both attempts are charged to the ledger, which is why the two
lowest-threshold points cost slightly *more* than `always_expensive`.

**5. Budget modes behave as specified, and the difference matters.** With the budget capped at
half the `always_expensive` spend ($0.0067):

* **SOFT** downgraded 50 decisions to the cheap model and fired both warnings (50%, 10%), but
  kept answering after the cap — final spend $0.0113, 168% of the cap, at 84% accuracy. With a
  cheap model only ~1.4× cheaper, "cheap-only after exhaustion" is *not* a spending cap.
* **HARD_STOP** stayed under the cap ($0.0064) by refusing the remaining 53 steps.

Use HARD_STOP when the budget is a hard limit; SOFT when finishing the task matters more.

**6. Grading spot-check (Phase 6).** All 27 answers graded wrong and all 12 graded right
without an exact alias match were checked by hand; the graders made no mistakes. Two
dataset-label issues were found and left as-is (official labels are used):
GSM8K item 49's gold answer is 2280, but the correct total is 750 + 430 + 700 + 300 = 2180 —
both models answered 2180 and are marked wrong; and TriviaQA item 38's gold answer is Lady
Bracknell, but its alias list (drawn from a page about the play's characters) also contains
"Jack Worthing" and "Algernon Moncrieff", so the expensive model's "Jack Worthing" counts as right.

**Caveats.** 100 tasks gives a standard error of about ±3 points on accuracy, so 1-point
differences between router thresholds are noise; the cost differences are exact. One model
pair, one provider, list prices. Every condition replays the same cached response per
(model, prompt), so routing is the only variable — and the per-call latencies in the database
are not meaningful. `python -m benchmark.run --offline --fresh` rebuilds these exact numbers
from the committed cache.
