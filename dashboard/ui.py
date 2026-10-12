"""Presentation layer for the dashboard: theme CSS, HTML components, charts."""

from __future__ import annotations

import html
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

CHEAP = "#22c55e"
EXPENSIVE = "#f97316"
ROUTER = "#4f8cff"
ORACLE = "#a78bfa"
BUDGET = "#f59e0b"
BAD = "#ef4444"
MUTED = "#8b95b5"
GRID = "rgba(139,149,181,0.18)"

CSS = """
<style>
html { font-size: 18px; }
.block-container { padding-top: 1.2rem; max-width: 1560px; }
#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 700; }
[data-testid="stMetricLabel"] p { font-size: 0.95rem; color: #aab3cf; }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid rgba(139,149,181,.25); }
.stTabs [data-baseweb="tab"] { font-size: 1.08rem; font-weight: 600; padding: 0.7rem 1.1rem;
  border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { background: rgba(79,140,255,.14); }
.stTabs [data-baseweb="tab"] p { font-size: 1.12rem; font-weight: 650; }
[data-testid="stButtonGroup"] { margin-bottom: 1rem; }
[data-testid="stButtonGroup"] button { min-height: 3.1rem; border-radius: 12px !important; }
[data-testid="stButtonGroup"] button p { font-size: 1.12rem; font-weight: 700; }
.cr-grid + div, .cr-grid { margin-bottom: .6rem; }
h3 { margin-top: 1.1rem !important; }
.stButton > button { border-radius: 12px; font-weight: 600; min-height: 2.9rem; }

.cr-hero { padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
  background: radial-gradient(1200px 400px at 0% 0%, rgba(79,140,255,.35), transparent 60%),
              radial-gradient(900px 400px at 100% 100%, rgba(249,115,22,.22), transparent 60%),
              #121a33; border: 1px solid rgba(79,140,255,.35); }
.cr-hero h1 { font-size: 3.3rem; margin: 0 0 .4rem 0; line-height: 1.05; }
.cr-hero .tag { font-size: 1.45rem; color: #dbe3ff; margin: 0 0 .8rem 0; font-weight: 500; }
.cr-hero .sub { font-size: 1.05rem; color: #aab3cf; margin: 0; max-width: 70rem; }
.cr-pill { display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-size: .9rem;
  font-weight: 600; margin: .9rem .45rem 0 0; border: 1px solid rgba(255,255,255,.18);
  background: rgba(255,255,255,.06); color: #e8ecf6; }
.cr-pill.cheap { border-color: rgba(34,197,94,.6); color: #86efac; }
.cr-pill.expensive { border-color: rgba(249,115,22,.6); color: #fdba74; }

.cr-grid { display: grid; gap: 1rem; }
.cr-grid.g4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.cr-grid.g3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cr-grid.g2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
@media (max-width: 1000px) { .cr-grid.g4, .cr-grid.g3 { grid-template-columns: repeat(2, 1fr); } }
.cr-card { background: #141b2f; border: 1px solid rgba(139,149,181,.22); border-radius: 18px;
  padding: 1.15rem 1.3rem; height: 100%; }
.cr-card .k { font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; color: #8b95b5;
  font-weight: 700; }
.cr-card .v { font-size: 2.3rem; font-weight: 800; margin: .25rem 0 .1rem 0; line-height: 1.1; }
.cr-card .s { font-size: .98rem; color: #aab3cf; }
.cr-card .icon { font-size: 1.7rem; }
.cr-card h4 { margin: .35rem 0 .3rem 0; font-size: 1.15rem; }
.cr-card p { margin: 0; color: #aab3cf; font-size: .98rem; }
.cr-accent-cheap { border-top: 4px solid #22c55e; }
.cr-accent-expensive { border-top: 4px solid #f97316; }
.cr-accent-router { border-top: 4px solid #4f8cff; }
.cr-accent-oracle { border-top: 4px solid #a78bfa; }
.cr-accent-budget { border-top: 4px solid #f59e0b; }

.cr-flow { display: flex; align-items: stretch; gap: .5rem; margin: .4rem 0 1rem 0; }
.cr-step { flex: 1 1 0; background: #141b2f; border: 1px solid rgba(139,149,181,.25);
  border-radius: 16px; padding: .9rem 1rem; position: relative; min-width: 0; }
.cr-step .n { font-size: .78rem; font-weight: 800; color: #8b95b5; letter-spacing: .1em; }
.cr-step .t { font-size: 1.05rem; font-weight: 700; margin: .15rem 0 .35rem 0; }
.cr-step .val { font-size: 1.5rem; font-weight: 800; line-height: 1.15; word-break: break-word; }
.cr-step .d { font-size: .88rem; color: #aab3cf; margin-top: .3rem; }
.cr-step.on { border-color: #4f8cff; box-shadow: 0 0 0 1px #4f8cff inset, 0 0 24px rgba(79,140,255,.25); }
.cr-step.cheap { border-color: #22c55e; box-shadow: 0 0 0 1px #22c55e inset, 0 0 24px rgba(34,197,94,.25); }
.cr-step.expensive { border-color: #f97316; box-shadow: 0 0 0 1px #f97316 inset, 0 0 24px rgba(249,115,22,.25); }
.cr-step.warn { border-color: #f59e0b; box-shadow: 0 0 0 1px #f59e0b inset, 0 0 24px rgba(245,158,11,.25); }
.cr-step.bad { border-color: #ef4444; box-shadow: 0 0 0 1px #ef4444 inset; }
.cr-arrow { align-self: center; color: #4f8cff; font-size: 1.6rem; font-weight: 800; }

.cr-badge { display: inline-block; padding: .22rem .7rem; border-radius: 8px; font-weight: 800;
  font-size: .85rem; letter-spacing: .05em; }
.cr-badge.cheap { background: rgba(34,197,94,.18); color: #86efac; }
.cr-badge.expensive { background: rgba(249,115,22,.18); color: #fdba74; }
.cr-badge.warn { background: rgba(245,158,11,.18); color: #fcd34d; }
.cr-badge.bad { background: rgba(239,68,68,.18); color: #fca5a5; }
.cr-badge.muted { background: rgba(139,149,181,.18); color: #cbd2e6; }

.cr-answer { background: #0f172a; border: 1px solid rgba(139,149,181,.25); border-radius: 18px;
  padding: 1.1rem 1.4rem; }
.cr-answer .q { color: #aab3cf; font-size: 1rem; margin-bottom: .5rem; }
.cr-answer .a { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
.cr-callout { border-left: 5px solid #a78bfa; background: rgba(167,139,250,.08); border-radius: 12px;
  padding: 1rem 1.3rem; margin: .6rem 0 1rem 0; font-size: 1.02rem; }
.cr-callout b { color: #ddd6fe; }

.cr-waffle { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 5px;
  margin: .3rem 0 .6rem 0; width: 100%; box-sizing: border-box; }
.cr-cell { aspect-ratio: 1 / 1; border-radius: 6px; display: flex; align-items: center;
  justify-content: center; font-size: .85rem; font-weight: 900; color: rgba(10,15,30,.85); }
.cr-cell.cheap { background: #22c55e; } .cr-cell.expensive { background: #f97316; }
.cr-cell.none { background: #2a3350; color: #8b95b5; }
.cr-cell.x { opacity: .55; outline: 2px solid #ef4444; outline-offset: -2px; }
.cr-legend > span { margin-right: 1.1rem; color: #aab3cf; font-size: .95rem; white-space: nowrap;
  display: inline-block; }
.cr-dot { display: inline-block; width: .85rem; height: .85rem; border-radius: 3px; margin-right: .35rem;
  vertical-align: -1px; }
.cr-small { color: #8b95b5; font-size: .9rem; }
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def card(key: str, value: str, sub: str = "", accent: str = "router") -> str:
    return (f'<div class="cr-card cr-accent-{accent}"><div class="k">{esc(key)}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def feature(icon: str, title: str, text: str) -> str:
    return (f'<div class="cr-card"><div class="icon">{icon}</div><h4>{esc(title)}</h4>'
            f'<p>{text}</p></div>')


def grid(items: list[str], cols: int = 4) -> str:
    return f'<div class="cr-grid g{cols}">' + "".join(items) + "</div>"


def step(n: int, title: str, value: str, detail: str = "", state: str = "") -> str:
    return (f'<div class="cr-step {state}"><div class="n">STEP {n}</div><div class="t">{esc(title)}</div>'
            f'<div class="val">{value}</div><div class="d">{detail}</div></div>')


def flow(steps: list[str]) -> str:
    return '<div class="cr-flow">' + '<div class="cr-arrow">→</div>'.join(steps) + "</div>"


def badge(text: str, kind: str) -> str:
    return f'<span class="cr-badge {kind}">{esc(text)}</span>'


def waffle(per: pd.DataFrame) -> str:
    cells = []
    for _, r in per.iterrows():
        tier = r.model_chosen if isinstance(r.model_chosen, str) else "none"
        ok = bool(r.correct)
        mark = "✓" if ok else ("–" if tier == "none" else "✗")
        title = f"{r.task_id} · {r.dataset} · {tier} · {'correct' if ok else 'wrong'}"
        cells.append(f'<div class="cr-cell {tier}{"" if ok or tier == "none" else " x"}" '
                     f'title="{esc(title)}">{mark}</div>')
    legend = ('<div class="cr-legend">'
              f'<span><span class="cr-dot" style="background:{CHEAP}"></span>cheap model</span>'
              f'<span><span class="cr-dot" style="background:{EXPENSIVE}"></span>expensive model</span>'
              '<span><span class="cr-dot" style="background:#2a3350"></span>refused (HARD_STOP)</span>'
              '<span>✓ correct &nbsp; ✗ wrong</span></div>')
    return '<div class="cr-waffle">' + "".join(cells) + "</div>" + legend


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _layout(fig: go.Figure, height: int = 460, **kw) -> go.Figure:
    layout = dict(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(size=16, color="#e8ecf6"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(20,27,47,.85)", bordercolor="rgba(139,149,181,.3)", borderwidth=1))
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def pareto(runs: pd.DataFrame, oracle: Optional[dict] = None, you: Optional[dict] = None,
           height: int = 520) -> go.Figure:
    fig = go.Figure()
    base = runs[runs.kind.isin(["always_cheap", "always_expensive"])].sort_values("total_cost")
    if len(base) == 2:
        fig.add_trace(go.Scatter(x=base.total_cost, y=base.accuracy, mode="lines",
                                 line=dict(color=MUTED, dash="dot", width=2),
                                 name="Random mix of the two models", hoverinfo="skip"))
    sweep = runs[runs.kind == "router"].sort_values("total_cost")
    fig.add_trace(go.Scatter(
        x=sweep.total_cost, y=sweep.accuracy, mode="lines+markers", name="Router (threshold sweep)",
        line=dict(color=ROUTER, dash="dash", width=2), marker=dict(size=15, color=ROUTER),
        text=sweep.label, hovertemplate="%{text}<br>$%{x:.5f} · %{y:.0%}<extra></extra>"))
    for kind, color, name, symbol in (("always_cheap", CHEAP, "Always cheap", "circle"),
                                      ("always_expensive", EXPENSIVE, "Always expensive", "circle"),
                                      ("capped_soft", BUDGET, "Router, budget-capped (SOFT)", "circle-open")):
        d = runs[runs.kind == kind]
        if not d.empty:
            fig.add_trace(go.Scatter(
                x=d.total_cost, y=d.accuracy, mode="markers", name=name, text=d.label,
                marker=dict(size=20, color=color, symbol=symbol, line=dict(width=3, color=color)),
                hovertemplate="%{text}<br>$%{x:.5f} · %{y:.0%}<extra></extra>"))
    if oracle:
        fig.add_trace(go.Scatter(
            x=[oracle["total_cost"]], y=[oracle["accuracy"]], mode="markers",
            name="Task-type routing (hindsight)", marker=dict(size=26, color=ORACLE, symbol="star"),
            hovertemplate="Route by task type<br>$%{x:.5f} · %{y:.0%}<extra></extra>"))
    if you:
        fig.add_trace(go.Scatter(
            x=[you["total_cost"]], y=[you["accuracy"]], mode="markers+text", name="Your setting",
            text=["  YOU"], textposition="middle right", textfont=dict(size=18, color="#ffffff"),
            marker=dict(size=30, color="#ffffff", symbol="diamond", line=dict(width=3, color=ROUTER)),
            hovertemplate="Your setting<br>$%{x:.5f} · %{y:.0%}<extra></extra>"))
    ys = list(runs[runs.kind != "capped_hard"].accuracy) + ([oracle["accuracy"]] if oracle else []) \
        + ([you["accuracy"]] if you else [])
    lo, hi = min(ys), max(ys)
    _layout(fig, height, title="Cost vs accuracy on 100 real tasks",
            legend_orientation="h", legend_y=-0.18)
    fig.update_xaxes(title="Total cost for 100 tasks (USD)", tickprefix="$", tickformat=".4f")
    fig.update_yaxes(title="Accuracy", tickformat=".0%", range=[lo - 0.03, hi + 0.02])
    return fig


def difficulty_bar(explain: dict, threshold: float, height: int = 230) -> go.Figure:
    c = explain["contributions"]
    labels = {"length": "Prompt length", "cues": "Reasoning cues", "history": "Failure history"}
    colors = {"length": "#60a5fa", "cues": "#c084fc", "history": "#f472b6"}
    fig = go.Figure()
    for k in ("length", "cues", "history"):
        fig.add_trace(go.Bar(y=["score"], x=[c[k]], orientation="h", name=labels[k],
                             marker_color=colors[k], text=[f"{c[k]:.2f}"], textposition="inside",
                             hovertemplate=f"{labels[k]}: %{{x:.3f}}<extra></extra>"))
    top = max(0.6, explain["score"] * 1.25, threshold * 1.25)
    fig.add_vline(x=threshold, line=dict(color="#ffffff", width=3, dash="dash"),
                  annotation_text=f"threshold {threshold:g}", annotation_position="top",
                  annotation_font_size=15)
    _layout(fig, height, barmode="stack", showlegend=True, legend_orientation="h",
            legend_y=-0.35, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(range=[0, top], title=None)
    return fig


def budget_trajectories(series: list[dict], height: int = 400, title: str = "") -> go.Figure:
    """``series``: [{"name", "y" (remaining after each step), "color", "forced" (bool list)}]."""
    fig = go.Figure()
    for s in series:
        x = list(range(1, len(s["y"]) + 1))
        fig.add_trace(go.Scatter(x=x, y=s["y"], mode="lines", name=s["name"],
                                 line=dict(color=s["color"], width=s.get("width", 4),
                                           dash=s.get("dash", "solid"))))
        if s.get("end_note") and x:
            fig.add_annotation(x=x[-1], y=s["y"][-1], text=s["end_note"], showarrow=True,
                               arrowhead=2, arrowcolor=s["color"], ax=40, ay=-50,
                               font=dict(size=15, color=s["color"]))
        if s.get("forced") is not None:
            fx = [i for i, f in zip(x, s["forced"]) if f]
            fy = [y for y, f in zip(s["y"], s["forced"]) if f]
            if fx:
                fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers", showlegend=False,
                                         marker=dict(size=8, color=BUDGET),
                                         hovertemplate="budget-forced to cheap<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#ffffff", width=2))
    _layout(fig, height, title=title, legend_orientation="h", legend_y=-0.2)
    fig.update_xaxes(title="Step")
    fig.update_yaxes(title="Budget left (USD)", tickprefix="$", tickformat=".4f")
    return fig


def dataset_bars(runs: pd.DataFrame, height: int = 420) -> go.Figure:
    d = runs[runs.kind.isin(["always_cheap", "always_expensive", "router"])]
    fig = go.Figure()
    for col, name, color in (("acc_triviaqa", "TriviaQA (knowledge)", "#38bdf8"),
                             ("acc_gsm8k", "GSM8K (math)", "#e879f9")):
        if col in d:
            fig.add_trace(go.Bar(x=d.label, y=d[col], name=name, marker_color=color,
                                 text=[f"{v:.0%}" for v in d[col]], textposition="outside"))
    _layout(fig, height, barmode="group", title="Accuracy by task type",
            legend_orientation="h", legend_y=-0.32, legend_x=0, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    return fig


def share_heatmap(decisions: pd.DataFrame, runs: pd.DataFrame, height: int = 380) -> go.Figure:
    d = decisions[decisions.run_id.isin(runs.loc[runs.kind == "router", "run_id"])]
    share = (d.assign(exp=d.model_chosen.eq("expensive"))
              .groupby(["run_id", "dataset"]).exp.mean().unstack())
    share.index = share.index.map(runs.set_index("run_id").label)
    share = share.sort_values(share.columns[0])
    fig = go.Figure(go.Heatmap(z=share.values, x=list(share.columns), y=list(share.index),
                               colorscale=[[0, "#13203d"], [0.5, "#7c3f1d"], [1, EXPENSIVE]],
                               zmin=0, zmax=1,
                               text=[[f"{v:.0%}" for v in row] for row in share.values],
                               texttemplate="%{text}", textfont=dict(size=18, color="#ffffff"),
                               colorbar=dict(title="expensive", tickformat=".0%")))
    _layout(fig, height, title="Share of steps sent to the expensive model")
    return fig
