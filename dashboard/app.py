"""CostRouter dashboard — presentation edition (REQ-BENCH-5).

Tabs:
1. **Overview** — what CostRouter does, how a request flows, headline results.
2. **Live Router** — route a question through the real ``RouterNode`` and watch
   each step (difficulty → budget → model → validation → ledger).  Presets are
   answered from the committed caches, so they work without network.
3. **Playground** — drag the threshold / budget and replay all 100 benchmark
   tasks through the router instantly (no API calls).
4. **Results**, 5. **Budget Safety**, 6. **Audit Log** — the benchmark evidence.

Usage::

    streamlit run dashboard/app.py                 # uses ./costrouter.db
    streamlit run dashboard/app.py -- --db other.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import engine, ui  # noqa: E402

try:  # live calls in the Live Router tab read the provider key from .env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


TAB_LABELS = ["🏠 Overview", "⚡ Live Router", "🎛️ Playground", "📊 Results",
              "🛡️ Budget Safety", "🧾 Audit Log"]


def default_db_path() -> str:
    """``--db`` CLI arg (after streamlit's ``--``), else env var, else default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db")
    args, _ = parser.parse_known_args()
    return args.db or engine.default_db_path()


# ---------------------------------------------------------------------------
# Cached data access (keyed on the DB file's mtime so a re-run refreshes it)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _runs(db: str, mtime: float) -> pd.DataFrame:
    return engine.load_runs(db)


@st.cache_data(show_spinner=False)
def _meta(db: str, mtime: float) -> dict:
    return engine.load_meta(db)


@st.cache_data(show_spinner=False)
def _decisions(db: str, mtime: float) -> pd.DataFrame:
    return engine.load_decisions(db)


@st.cache_data(show_spinner=False)
def _events(db: str, mtime: float) -> pd.DataFrame:
    return engine.load_budget_events(db)


@st.cache_data(show_spinner=False)
def _oracle(db: str, mtime: float):
    return engine.oracle_point(db)


@st.cache_data(show_spinner=False, max_entries=256)
def _replay(meta_json: str, threshold: float, budget: float, hard: bool) -> dict:
    return engine.replay(json.loads(meta_json), threshold, budget, hard)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def pct(x: float) -> str:
    return f"{x:.0%}"


def usd(x: float, digits: int = 4) -> str:
    return f"${x:.{digits}f}"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def hero(meta: dict) -> None:
    cheap, exp = meta.get("cheap_model", "cheap model"), meta.get("expensive_model", "expensive model")
    html('<div class="cr-hero"><h1>CostRouter</h1>'
         '<p class="tag">Send every LLM call to the cheapest model that can handle it — '
         'without ever losing track of the budget.</p>'
         '<p class="sub">A drop-in LangGraph node that scores how hard each agent step is, '
         'checks the remaining USD budget, routes to a cheap or an expensive model, validates '
         'the answer, escalates when the cheap model fails, and logs every decision.</p>'
         f'<span class="cr-pill cheap">Cheap · {ui.esc(cheap)}</span>'
         f'<span class="cr-pill expensive">Expensive · {ui.esc(exp)}</span>'
         f'<span class="cr-pill">Provider · {ui.esc(meta.get("provider", "?")).title()}</span>'
         '<span class="cr-pill">100 real benchmark tasks</span></div>')


def tab_overview(runs: pd.DataFrame, meta: dict, oracle) -> None:
    hero(meta)
    cheap = runs[runs.kind == "always_cheap"].iloc[0]
    exp = runs[runs.kind == "always_expensive"].iloc[0]
    sweep = runs[runs.kind == "router"]
    best = sweep.sort_values(["accuracy", "total_cost"], ascending=[False, True]).iloc[0]
    cards = [
        ui.card("Always cheap", pct(cheap.accuracy), f"accuracy · {usd(cheap.total_cost)} for 100 tasks",
                "cheap"),
        ui.card("Always expensive", pct(exp.accuracy), f"accuracy · {usd(exp.total_cost)} for 100 tasks",
                "expensive"),
        ui.card("CostRouter v1", pct(best.accuracy),
                f"accuracy · {pct(best.total_cost / exp.total_cost)} of the expensive cost", "router"),
    ]
    if oracle:
        cards.append(ui.card("Headroom: route by task type", pct(oracle["accuracy"]),
                             f"accuracy · {pct(oracle['total_cost'] / exp.total_cost)} of the "
                             "expensive cost", "oracle"))
    html(ui.grid(cards, 4))

    st.markdown("### How one agent step flows through CostRouter")
    html(ui.flow([
        ui.step(1, "Score difficulty", "0 – 1", "prompt length · reasoning cues · failure history"),
        ui.step(2, "Check the budget", "USD ledger", "can we afford the expensive model?"),
        ui.step(3, "Route", f'{ui.badge("CHEAP", "cheap")} / {ui.badge("EXPENSIVE", "expensive")}',
                "difficulty above the threshold → expensive"),
        ui.step(4, "Validate", "retry-escalate", "bad cheap answer → re-run on the expensive model"),
        ui.step(5, "Charge & log", "audit trail", "real token cost → ledger → SQLite"),
    ]))

    st.markdown("### What's inside")
    html(ui.grid([
        ui.feature("🎯", "Difficulty estimator", "Pure-Python heuristic, &lt;1 ms per step. "
                   "Pluggable interface ready for a learned model."),
        ui.feature("💵", "USD budget ledger", "Thread-safe, priced from real token usage. "
                   "50% / 10% warnings, SOFT or HARD_STOP modes."),
        ui.feature("🛡️", "Budget-forced routing", "When the expensive call can't be afforded, "
                   "the step is downgraded — and the downgrade is recorded."),
        ui.feature("🔁", "Retry-escalate", "A pluggable validator catches empty or malformed "
                   "cheap answers and re-runs them on the expensive model."),
        ui.feature("🧾", "Full audit trail", "Every decision in SQLite: score, model, cost, "
                   "budget before/after. Prompts stored as hashes only."),
        ui.feature("📈", "Benchmark harness", "Baselines, threshold sweeps and budget caps on "
                   "real TriviaQA + GSM8K, replayable offline."),
    ], 3))

    st.markdown("### The honest result")
    html('<div class="cr-callout">On this model pair the two models score the <b>same on math '
         f'({pct(cheap.acc_gsm8k)})</b> and differ only on <b>trivia ({pct(cheap.acc_triviaqa)} vs '
         f'{pct(exp.acc_triviaqa)})</b>. The v1 heuristic treats long math problems as "hard", so it '
         'spends where nothing is gained and lands near a random mix of the two models. Routing by '
         'task type would keep full accuracy at a fraction of the cost — that headroom is what the '
         'next estimator targets. The plumbing — budgets, escalation, audit — works as specified.</div>')


def _session(meta: dict, threshold: float, budget: float, hard: bool) -> engine.LiveSession:
    key = (threshold, budget, hard)
    sess = st.session_state.get("live")
    if sess is None or st.session_state.get("live_key") != key:
        sess = engine.LiveSession(meta, budget_usd=budget, threshold=threshold, hard_stop=hard)
        st.session_state["live"], st.session_state["live_key"] = sess, key
    return sess


def _render_flow(e: dict) -> None:
    """Full-width row of the five routing steps for one decision."""
    ex, t = e["explain"], e["threshold"]
    wants = "expensive" if ex["score"] > t else "cheap"
    tier, status = e.get("tier"), e["status"]
    forced, retried = e.get("budget_forced", False), e.get("retried", False)

    s1 = ui.step(1, "Difficulty score", f"{ex['score']:.2f}",
                 f"threshold {t:g} → wants {wants.upper()}", "on")
    if status == "refused":
        s2 = ui.step(2, "Budget check", usd(e["budget_before"], 5), "HARD_STOP: budget gone → refused", "bad")
    elif forced:
        s2 = ui.step(2, "Budget check", usd(e["budget_before"], 5),
                     "can't afford expensive → forced to cheap", "warn")
    else:
        s2 = ui.step(2, "Budget check", usd(e["budget_before"], 5), "affordable ✓", "on")
    if tier and retried:
        s3 = ui.step(3, "Model", f'{ui.badge("CHEAP", "cheap")} → {ui.badge("EXPENSIVE", "expensive")}',
                     f"escalated to {ui.esc(e.get('model', ''))}", "warn")
    elif tier:
        s3 = ui.step(3, "Model", ui.badge(tier.upper(), tier), ui.esc(e.get("model", "")), tier)
    else:
        s3 = ui.step(3, "Model", ui.badge("NONE", "muted"), "no call made", "bad")
    if status == "ok" and retried:
        s4 = ui.step(4, "Validate", "✗ → retried", "cheap answer was empty/malformed; "
                     "re-ran on the expensive model", "warn")
    elif status == "ok":
        s4 = ui.step(4, "Validate", "✓ valid", "has a 'Final answer:' line", "on")
    else:
        s4 = ui.step(4, "Validate", "✗ failed", ui.esc(e.get("error") or "")[:120], "bad")
    src = {"cache": "cached answer", "live": "live API call", "offline": "offline"}.get(e.get("source"), "")
    timing = "" if e.get("source") == "cache" else f" · {e['wall_ms']:.0f} ms"
    s5 = ui.step(5, "Charge & log", usd(e["cost"], 5),
                 f"{src}{timing} · budget left {usd(e['budget_after'], 5)}", "on")
    html(ui.flow([s1, s2, s3, s4, s5]))


def _render_details(e: dict) -> None:
    """Answer, score breakdown and raw record for one decision."""
    ex, t, status = e["explain"], e["threshold"], e["status"]
    hist_note = (f"cold-start prior ({e['history_n']}/5 seen)" if e["history_n"] < 5
                 else f"{ex['signals']['history']:.0%} recent failures")
    answer = e["final_answer"] or ("(refused — budget exhausted)" if status == "refused"
                                   else "(no valid answer)")
    html(f'<div class="cr-answer"><div class="q">Q{e["n"]} · {ui.esc(e["label"])}</div>'
         f'<div class="a">{ui.esc(answer)}</div></div>')
    for ev in e["events"]:
        st.warning(f"Budget warning: crossed the {ev['pct']:.0%} threshold "
                   f"({usd(ev['remaining'], 5)} left)", icon="⚠️")
    st.markdown("**Why this score?**")
    st.plotly_chart(ui.difficulty_bar(ex, t), width="stretch", key=f"diff-{e['n']}")
    cues = ", ".join(ex["matched_cues"]) or "none"
    st.caption(f"Reasoning cues found: {cues} · history: {hist_note}")
    with st.expander("Full model response and decision record"):
        st.markdown(e["text"] or "_(empty response)_")
        st.json({k: e.get(k) for k in ("tier", "model", "status", "budget_forced", "retried",
                                       "cost", "budget_before", "budget_after", "error", "source")}
                | {"difficulty": ex})


def tab_live(meta: dict) -> None:
    st.markdown("## ⚡ Live Router")
    st.caption("Every question goes through the real RouterNode. Budget and failure history "
               "persist across questions in this session.")
    cfg1, cfg2, cfg3, cfg4 = st.columns([1.3, 1.3, 1, 0.9])
    threshold = cfg1.slider("Difficulty threshold", 0.05, 0.60, 0.30, 0.01, key="live_threshold",
                            help="Scores above this go to the expensive model.")
    budget = cfg2.select_slider(
        "Session budget (USD)", options=[0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05], value=0.01,
        format_func=lambda v: f"${v:g}", key="live_budget")
    hard = cfg3.toggle("HARD_STOP mode", value=False, key="live_hard",
                       help="Off = SOFT: keep answering with the cheap model once the budget is gone.")
    sess = _session(meta, threshold, budget, hard)
    if cfg4.button("↺ New session", width="stretch"):
        st.session_state.pop("live", None)
        sess = _session(meta, threshold, budget, hard)

    flow_slot = st.container()
    pick, show = st.columns([1, 1.9], gap="large")
    entry = None
    with pick:
        st.markdown("#### Pick a question")
        icons = {"easy": "🟢", "hard": "🔴", "tricky": "🟡", "plain": "🔵"}
        for i, item in enumerate(engine.SHOWCASE):
            if st.button(f"{icons[item['kind']]}  {item['label']}", key=f"preset-{i}",
                         width="stretch", help=item.get("note")):
                with st.spinner("Routing…"):
                    entry = sess.route(engine.showcase_prompt(item), item["step_type"],
                                       item["question"])
        st.markdown("#### …or ask your own")
        question = st.text_area("Your question", key="custom_q", height=110,
                                placeholder="e.g. Explain why the sky is blue, step by step.",
                                label_visibility="collapsed")
        kind = st.selectbox("Task type (keys the failure history)",
                            ["general", "trivia", "math", "reasoning"], key="custom_type")
        if st.button("Route my question →", type="primary", width="stretch",
                     disabled=not (question.strip() and sess.live_available)):
            with st.spinner("Calling the model…"):
                entry = sess.route(engine.CUSTOM_TEMPLATE.format(question=question.strip()),
                                   kind, question.strip())
        if not sess.live_available:
            st.caption("Live calls off (no API key) — presets still work from the cache.")
        else:
            st.caption(f"Live calls on via {meta.get('provider', '?').title()}.")

    if entry is None and sess.history:
        entry = sess.history[-1]
    with flow_slot:
        if entry is not None:
            _render_flow(entry)
    with show:
        if entry is None:
            html('<div class="cr-callout">Pick a preset on the left. Try an <b>easy</b> question, '
                 'then a <b>hard</b> one, then the <b>tricky</b> trivia to see retry-escalate. '
                 'Drop the budget to <b>$0.0005</b> to watch budget-forced routing, or pick <b>$0.002</b> '
                 'with <b>HARD_STOP</b> on to see calls refused once the worst-case cost no longer fits.</div>')
        else:
            _render_details(entry)

    if sess.history:
        st.markdown("#### Session ledger")
        spent = sess.budget_usd - sess.budget.remaining_budget()
        h = pd.DataFrame(sess.history)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Questions", len(h))
        m2.metric("Cheap / expensive", f"{(h.tier == 'cheap').sum()} / {(h.tier == 'expensive').sum()}")
        m3.metric(f"Spent (of ${sess.budget_usd:g})", usd(spent, 5))
        m4.metric("Budget left", usd(sess.budget.remaining_budget(), 5))
        st.progress(min(max(spent / sess.budget_usd, 0.0), 1.0),
                    text=f"{min(spent / sess.budget_usd, 9.99):.0%} of the budget used")
        table = h.assign(
            score=h.explain.map(lambda x: round(x["score"], 3)),
            model=h.tier.fillna("—"),
            flags=h.apply(lambda r: " ".join(f for f, on in (("forced", r.get("budget_forced")),
                                                             ("retried", r.get("retried")),
                                                             ("refused", r.status == "refused"))
                                             if on is True), axis=1),
            answer=h.final_answer.str.slice(0, 60),
        )[["n", "label", "step_type", "score", "model", "flags", "cost", "budget_after", "answer"]]
        st.dataframe(table.iloc[::-1], width="stretch", hide_index=True,
                     column_config={"cost": st.column_config.NumberColumn("cost", format="$%.6f"),
                                    "budget_after": st.column_config.NumberColumn(
                                        "budget left", format="$%.5f")})


def tab_playground(runs: pd.DataFrame, meta: dict, oracle) -> None:
    st.markdown("## 🎛️ Playground — replay the benchmark with your settings")
    exp = runs[runs.kind == "always_expensive"].iloc[0]
    c1, c2, c3 = st.columns([1.6, 1.6, 1])
    sweep = sorted(meta.get("sweep_thresholds") or [0.039])
    threshold = c1.slider("Difficulty threshold", 0.0, 0.25, float(sweep[len(sweep) // 2]), 0.001,
                          format="%.3f", key="pg_threshold",
                          help=f"The benchmark sweep used {', '.join(f'{t:g}' for t in sweep)}.")
    capped = c2.toggle("Cap the budget", value=False, key="pg_capped")
    budget = c2.slider("Budget cap (USD)", 0.002, 0.014, 0.0067, 0.0001, format="$%.4f",
                       key="pg_budget", disabled=not capped)
    hard = c3.radio("When the budget runs out", ["SOFT", "HARD_STOP"], key="pg_mode",
                    disabled=not capped, help="SOFT keeps answering with the cheap model; "
                                              "HARD_STOP refuses further calls.") == "HARD_STOP"
    r = _replay(json.dumps(meta, sort_keys=True), round(threshold, 4),
                round(budget, 6) if capped else 10.0, hard and capped)
    per = r["per_task_df"]

    m = st.columns(6)
    m[0].metric("Accuracy", pct(r["accuracy"]),
                f"{(r['accuracy'] - exp.accuracy) * 100:+.0f} pts vs expensive")
    m[1].metric("Cost", usd(r["total_cost"]),
                f"{r['total_cost'] / exp.total_cost - 1:+.0%} vs expensive", delta_color="inverse")
    m[2].metric("Sent to expensive", pct(r["expensive_share"]))
    m[3].metric("Retry-escalations", r["retries"])
    m[4].metric("Budget-forced", r["budget_forced"])
    m[5].metric("Refused / failed", r["errors"])
    st.caption(f"Replayed {r['total_count']} tasks through the real RouterNode in "
               f"{r['elapsed_ms']:.0f} ms — answers come from the recorded benchmark responses, "
               "so there are no API calls. Settings found by exploring here are tuned on these "
               "same tasks — treat them as exploration, not a benchmark result.")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(ui.pareto(runs, oracle, you={"total_cost": r["total_cost"],
                                                     "accuracy": r["accuracy"]}),
                        width="stretch", key="pg-pareto")
    with right:
        st.markdown("#### Every task, one square")
        html(ui.waffle(per))
        by = r["accuracy_by_dataset"]
        st.markdown(f"TriviaQA **{pct(by.get('triviaqa', 0))}** · GSM8K **{pct(by.get('gsm8k', 0))}**")
    if capped:
        st.plotly_chart(ui.budget_trajectories(
            [{"name": "Budget left", "y": list(per.remaining), "color": ui.ROUTER,
              "forced": list(per.budget_forced)}],
            height=330, title=f"Budget over the run (cap ${budget:.4f}, {'HARD_STOP' if hard else 'SOFT'})"),
            width="stretch", key="pg-budget")


def tab_results(runs: pd.DataFrame, decisions: pd.DataFrame, oracle) -> None:
    st.markdown("## 📊 Benchmark results")
    st.plotly_chart(ui.pareto(runs, oracle, height=580), width="stretch", key="res-pareto")
    exp_cost = runs.loc[runs.kind == "always_expensive", "total_cost"].iloc[0]
    table = runs.assign(cost_vs_expensive=runs.total_cost / exp_cost)[
        ["label", "accuracy", "acc_triviaqa", "acc_gsm8k", "total_cost", "cost_vs_expensive",
         "expensive_share", "retries", "budget_forced", "unanswered"]]
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "label": "Run",
        "accuracy": st.column_config.ProgressColumn("Accuracy", format="percent", min_value=0, max_value=1),
        "acc_triviaqa": st.column_config.NumberColumn("TriviaQA", format="percent"),
        "acc_gsm8k": st.column_config.NumberColumn("GSM8K", format="percent"),
        "total_cost": st.column_config.NumberColumn("Cost", format="$%.5f"),
        "cost_vs_expensive": st.column_config.NumberColumn("vs expensive", format="percent"),
        "expensive_share": st.column_config.NumberColumn("Expensive share", format="percent"),
        "retries": "Retries", "budget_forced": "Budget-forced", "unanswered": "Unanswered"})
    a, b = st.columns(2, gap="large")
    a.plotly_chart(ui.dataset_bars(runs), width="stretch", key="res-ds")
    b.plotly_chart(ui.share_heatmap(decisions, runs), width="stretch", key="res-heat")
    st.caption("Grading was checked by hand (every wrong answer and every non-exact match). "
               "One GSM8K gold label is wrong (item 49: 2280; the correct total is 2180) and is kept "
               "as published. Full write-up: notebooks/pareto_report.ipynb.")


def tab_budget(runs: pd.DataFrame, decisions: pd.DataFrame, events: pd.DataFrame) -> None:
    st.markdown("## 🛡️ Budget safety — degradation you can see")
    capped = runs[runs.kind.isin(["capped_soft", "capped_hard"])]
    if capped.empty:
        st.info("No budget-capped runs in this database.")
        return
    cap = capped.budget_usd.iloc[0]
    soft = capped[capped.kind == "capped_soft"]
    hard = capped[capped.kind == "capped_hard"]
    cards = []
    if not soft.empty:
        s = soft.iloc[0]
        cards.append(ui.card("SOFT mode", usd(s.total_cost),
                             f"spent vs a {usd(cap)} cap ({pct(s.total_cost / cap)}) · "
                             f"{int(s.budget_forced)} steps downgraded · accuracy {pct(s.accuracy)}",
                             "budget"))
    if not hard.empty:
        h = hard.iloc[0]
        cards.append(ui.card("HARD_STOP mode", usd(h.total_cost),
                             f"never exceeds the {usd(cap)} cap · {int(h.unanswered)} steps refused · "
                             f"accuracy {pct(h.accuracy)}", "expensive"))
    html(ui.grid(cards, 2))
    series = []
    for _, r in capped.iterrows():
        d = decisions[decisions.run_id == r.run_id]
        hard_run = r.kind == "capped_hard"
        series.append({"name": "HARD_STOP" if hard_run else "SOFT",
                       "y": list(d.budget_remaining_after),
                       "color": ui.BAD if hard_run else ui.ROUTER,
                       "dash": "dash" if hard_run else "solid", "width": 3 if hard_run else 5,
                       "end_note": (f"HARD_STOP stops: {int(r.unanswered)} steps refused"
                                    if hard_run else f"SOFT finishes {usd(-d.budget_remaining_after.iloc[-1])} over"),
                       "forced": None if hard_run else list(d.budget_forced.astype(bool))})
    series.sort(key=lambda s: s["name"] == "HARD_STOP")  # draw the dashed line on top
    st.plotly_chart(ui.budget_trajectories(series, height=440,
                                           title=f"Budget left per step — cap {usd(cap)} "
                                                 "(half of the always-expensive spend)"),
                    width="stretch", key="budget-traj")
    html('<div class="cr-callout"><b>SOFT</b> downgrades to the cheap model and keeps going, so '
         'the task finishes but spend can pass the cap. <b>HARD_STOP</b> refuses any call it can\'t '
         'afford, so the cap holds exactly. Both fire 50% and 10% warnings, and every downgrade is '
         'flagged <code>budget_forced</code> in the audit log.</div>')
    ev = events[events.run_id.isin(capped.run_id)].merge(runs[["run_id", "label"]], on="run_id")
    if not ev.empty:
        st.dataframe(ev[["label", "threshold_pct", "remaining_budget", "triggered_at"]],
                     hide_index=True, width="stretch", column_config={
                         "label": "Run", "threshold_pct": st.column_config.NumberColumn(
                             "Warning at", format="percent"),
                         "remaining_budget": st.column_config.NumberColumn("Left", format="$%.5f")})


def tab_audit(runs: pd.DataFrame, decisions: pd.DataFrame, db_path: str) -> None:
    st.markdown("## 🧾 Audit log — every routing decision")
    labels = dict(zip(runs.run_id, runs.label))
    choice = st.selectbox("Run", list(labels), format_func=labels.get, key="audit_run")
    d = decisions[decisions.run_id == choice]
    cols = ["step_id", "dataset", "difficulty_score", "model_chosen", "budget_forced", "retried",
            "correct", "actual_cost_usd", "budget_remaining_before", "budget_remaining_after",
            "prompt_length", "prompt_hash", "error"]
    st.dataframe(d[cols], width="stretch", hide_index=True, height=520, column_config={
        "difficulty_score": st.column_config.NumberColumn("difficulty", format="%.3f"),
        "actual_cost_usd": st.column_config.NumberColumn("cost", format="$%.6f"),
        "budget_remaining_before": st.column_config.NumberColumn("budget before", format="$%.5f"),
        "budget_remaining_after": st.column_config.NumberColumn("budget after", format="$%.5f")})
    st.download_button("Download this run as CSV", d[cols].to_csv(index=False),
                       file_name=f"{choice}.csv", mime="text/csv")
    st.caption(f"Source: {db_path} (opened read-only). Prompts are stored as length + SHA-256 "
               "prefix, never as text.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="CostRouter", page_icon="🔀", layout="wide",
                       initial_sidebar_state="collapsed")
    html(ui.CSS)
    db_path = default_db_path()
    if not os.path.isfile(db_path):
        hero({})
        st.warning(f"Database not found at `{db_path}`.")
        st.info("Generate it with:\n```\npython -m benchmark.run --fresh\n```\n"
                "or rebuild it from the committed response cache, no API key "
                "needed:\n```\npython -m benchmark.run --offline --fresh\n```")
        return
    mtime = os.path.getmtime(db_path)
    runs = _runs(db_path, mtime)
    if runs.empty or not {"always_cheap", "always_expensive"} <= set(runs.kind):
        hero({})
        st.info("No completed baseline runs found. Run `python -m benchmark.run --fresh` first.")
        return
    meta, oracle = _meta(db_path, mtime), _oracle(db_path, mtime)
    decisions, events = _decisions(db_path, mtime), _events(db_path, mtime)

    # Navigation is a keyed widget, so the selected page survives reruns (button
    # clicks, sliders), and only the selected page is rendered.
    pages = {
        TAB_LABELS[0]: lambda: tab_overview(runs, meta, oracle),
        TAB_LABELS[1]: lambda: tab_live(meta),
        TAB_LABELS[2]: lambda: tab_playground(runs, meta, oracle),
        TAB_LABELS[3]: lambda: tab_results(runs, decisions, oracle),
        TAB_LABELS[4]: lambda: tab_budget(runs, decisions, events),
        TAB_LABELS[5]: lambda: tab_audit(runs, decisions, db_path),
    }
    choice = st.segmented_control("Section", TAB_LABELS, default=TAB_LABELS[0], key="nav",
                                  required=True, label_visibility="collapsed", width="stretch")
    pages[choice or TAB_LABELS[0]]()


if __name__ == "__main__":
    main()
