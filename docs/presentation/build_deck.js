// Builds ../CostRouter_Presentation.pptx from deck_data.json (python extract_data.py).
// Run: npm install && node build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "..", "CostRouter_Presentation.pptx");

// ---------------------------------------------------------------------------
// Theme — matches the dashboard: night navy, green = cheap, orange = expensive
// ---------------------------------------------------------------------------
const THEME = {
  name: "CostRouter Night",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1020",   // slide background
    lt1: "FFFFFF",   // primary text
    dk2: "172038",   // card fill
    lt2: "AAB3CF",   // muted text
    accent1: "4F8CFF", // router blue
    accent2: "22C55E", // cheap green
    accent3: "F97316", // expensive orange
    accent4: "A78BFA", // insight violet
    accent5: "F59E0B", // budget amber
    accent6: "EF4444", // refusal red
    hlink: "4F8CFF",
    folHlink: "A78BFA",
  },
};
const H = THEME.colors; // hex, for icons and hex-only options
const HEX = { len: "60A5FA", cue: "C084FC", hist: "F472B6", grid: "26304D", axis: "AAB3CF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "CostRouter — cost-aware model routing for LangGraph agents";
pres.subject = "Project presentation";
pres.author = "CostRouter";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const TINT = { blue: C.accent1, green: C.accent2, orange: C.accent3, violet: C.accent4, amber: C.accent5, red: C.accent6 };
const TINT_HEX = { blue: H.accent1, green: H.accent2, orange: H.accent3, violet: H.accent4, amber: H.accent5, red: H.accent6 };

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.05, w: 6.6, h: 1.2,
      fontSize: 60, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.4, w: 6.4, h: 1.3,
      fontSize: 20, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.dk1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontSize: 28, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "CostRouter", options: { x: 0.6, y: 7.02, w: 3, h: 0.28, fontSize: 10,
      color: C.background2, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.28, fontSize: 10, color: H.lt2, align: "right" },
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const iconCache = new Map();
async function icon(Comp, hex) {
  const key = `${Comp.name}-${hex}`;
  if (iconCache.has(key)) return iconCache.get(key);
  let svg = renderToStaticMarkup(React.createElement(Comp, { color: `#${hex}`, size: "256" }));
  if (!svg.includes("xmlns=")) svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  const png = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  const data = "image/png;base64," + png.toString("base64");
  iconCache.set(key, data);
  return data;
}

let objN = 0;
const nm = (s) => `${s}-${++objN}`;

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: opts.r ?? 0.12, objectName: nm(opts.name || "card"),
    fill: { color: opts.fill || C.text2, transparency: opts.transparency || 0 },
    line: opts.border ? { color: opts.border, width: opts.borderW || 1.5 } : { type: "none" },
  });
}

function text(slide, t, x, y, w, h, opts = {}) {
  slide.addText(t, { x, y, w, h, isTextBox: true, margin: 0, color: C.background1, fontSize: 14,
    valign: "top", objectName: nm(opts.name || "text"), ...opts });
}

async function iconCircle(slide, Comp, tint, x, y, d) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, objectName: nm("icon-bg"),
    fill: { color: TINT[tint], transparency: 78 }, line: { type: "none" } });
  const s = d * 0.5;
  slide.addImage({ data: await icon(Comp, TINT_HEX[tint]), x: x + (d - s) / 2, y: y + (d - s) / 2,
    w: s, h: s, objectName: nm("icon") });
}

// Straight arrow from (x1,y1) to (x2,y2).
function arrow(slide, x1, y1, x2, y2, opts = {}) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2);
  slide.addShape(pres.shapes.LINE, {
    x, y, w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
    flipH: x2 < x1, flipV: y2 < y1, objectName: nm("arrow"),
    line: { color: opts.color || C.accent1, width: opts.width || 2, dashType: opts.dash || "solid",
      endArrowType: opts.noHead ? undefined : "triangle" },
  });
}

function chip(slide, t, x, y, w, h, tint, opts = {}) {
  card(slide, x, y, w, h, { fill: TINT[tint], transparency: opts.transparency ?? 80, r: h / 2, name: "chip" });
  text(slide, t, x, y, w, h, { align: "center", valign: "middle", fontSize: opts.fontSize || 13,
    bold: opts.bold ?? true, color: opts.color || C.background1, name: "chip-text" });
}

const pct = (v) => `${Math.round(v * 100)}%`;
const usd = (v, d = 4) => `$${v.toFixed(d)}`;

// Chart text styling shared by every chart (identical labels across the deck).
const CHART_TEXT = {
  catAxisLabelColor: HEX.axis, valAxisLabelColor: HEX.axis, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelColor: H.lt1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
  legendColor: H.lt2, legendFontSize: 12, legendFontFace: "+mn-lt",
  titleColor: H.lt1, titleFontSize: 14, titleFontFace: "+mn-lt",
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
};

const runs = DATA.runs;
const byKind = (k) => runs.filter((r) => r.kind === k);
const cheapRun = byKind("always_cheap")[0];
const expRun = byKind("always_expensive")[0];
const sweep = byKind("router").sort((a, b) => a.threshold - b.threshold);
const best = [...sweep].sort((a, b) => b.accuracy - a.accuracy || a.total_cost - b.total_cost)[0];
const soft = byKind("capped_soft")[0];
const hard = byKind("capped_hard")[0];
const oracle = DATA.oracle;
const meta = DATA.meta;

async function build() {
  // ===================================================================
  // 1. Title
  // ===================================================================
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  text(s, "PROJECT PRESENTATION", 0.7, 1.55, 6, 0.35, { fontSize: 14, bold: true, color: C.accent1, charSpacing: 2 });
  s.addText("CostRouter", { placeholder: "title" });
  s.addText("Send every LLM call to the cheapest model that can handle it — without losing track of the budget",
    { placeholder: "body" });
  text(s, "Cost-aware model routing for LangGraph agents", 0.7, 4.85, 6.4, 0.4, { fontSize: 16, color: C.background2 });
  s.addImage({ data: await icon(fa.FaGithub, H.lt2), x: 0.7, y: 6.32, w: 0.32, h: 0.32, objectName: nm("gh") });
  text(s, "github.com/prakharsahup/Cost-Router", 1.12, 6.3, 5, 0.36, { fontSize: 14, color: C.background2, valign: "middle" });

  // Hero diagram: agent → router → cheap / expensive
  const hx = 7.55;
  await iconCircle(s, fa.FaRobot, "violet", hx, 3.05, 1.1);
  text(s, "Agent step", hx - 0.25, 4.25, 1.6, 0.3, { fontSize: 13, align: "center", color: C.background2 });
  arrow(s, hx + 1.2, 3.6, hx + 1.85, 3.6);
  s.addShape(pres.shapes.OVAL, { x: hx + 1.95, y: 2.75, w: 1.7, h: 1.7, objectName: nm("router-glow"),
    fill: { color: C.accent1, transparency: 70 }, line: { color: C.accent1, width: 2.5 } });
  s.addImage({ data: await icon(fa.FaRoute, H.lt1), x: hx + 2.4, y: 3.2, w: 0.8, h: 0.8, objectName: nm("router-icon") });
  text(s, "CostRouter", hx + 1.65, 4.55, 2.3, 0.35, { fontSize: 15, bold: true, align: "center" });
  arrow(s, hx + 3.7, 3.25, hx + 4.35, 2.35, { color: C.accent2 });
  arrow(s, hx + 3.7, 3.95, hx + 4.35, 4.85, { color: C.accent3 });
  await iconCircle(s, fa.FaFeather, "green", hx + 4.35, 1.55, 1.0);
  text(s, "Cheap model", hx + 3.85, 2.6, 2.0, 0.3, { fontSize: 13, bold: true, align: "center", color: C.accent2 });
  await iconCircle(s, fa.FaGem, "orange", hx + 4.35, 4.55, 1.0);
  text(s, "Expensive model", hx + 3.85, 5.6, 2.0, 0.3, { fontSize: 13, bold: true, align: "center", color: C.accent3 });
  s.addNotes(
    "Hi everyone. This is CostRouter, a project about spending less on AI models without getting worse answers. " +
    "Modern AI agents call a language model at every step. CostRouter sits in front of those calls and decides, " +
    "step by step, whether a cheap model is good enough or the expensive one is needed — while keeping track of a budget in dollars. " +
    "I'll explain the idea, how it works, how we tested it on real models, what we found, and then show it live.");

  // ===================================================================
  // 2. The problem
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Agents pay premium prices for every step — even easy ones", { placeholder: "title" });
  const steps = [
    ["Look up a capital city", "green"], ["Format a date", "green"], ["Summarize an email", "amber"],
    ["Compare two job offers", "orange"], ["Answer a quick fact", "green"], ["Prove a math result", "orange"],
  ];
  const sy0 = 1.55, sh = 0.62, sg = 0.2, yc = sy0 + (steps.length * (sh + sg) - sg) / 2;
  for (const [i, [label, tint]] of steps.entries()) {
    const y = sy0 + i * (sh + sg);
    card(s, 0.6, y, 3.7, sh, { name: "step" });
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + sh / 2 - 0.1, w: 0.2, h: 0.2, fill: { color: TINT[tint] },
      line: { type: "none" }, objectName: nm("dot") });
    text(s, label, 1.15, y, 3.0, sh, { fontSize: 15, valign: "middle" });
    arrow(s, 4.35, y + sh / 2, 5.45, yc, { color: C.accent3, width: 1.5 });
  }
  card(s, 5.5, yc - 1.0, 2.3, 2.0, { fill: C.accent3, transparency: 82, border: C.accent3, name: "big-model" });
  s.addImage({ data: await icon(fa.FaGem, H.accent3), x: 6.35, y: yc - 0.8, w: 0.6, h: 0.6, objectName: nm("gem") });
  text(s, "One big model for everything", 5.6, yc - 0.1, 2.1, 0.9, { fontSize: 15, bold: true, align: "center", valign: "middle" });
  text(s, [{ text: "● ", options: { color: C.accent2 } }, { text: "easy    ", options: {} },
    { text: "● ", options: { color: C.accent5 } }, { text: "medium    ", options: {} },
    { text: "● ", options: { color: C.accent3 } }, { text: "hard", options: {} }],
    0.6, 6.55, 3.7, 0.3, { fontSize: 12, color: C.background2 });

  card(s, 8.3, 1.55, 4.4, 1.55, { name: "stat" });
  text(s, "2×", 8.6, 1.7, 1.6, 1.25, { fontSize: 54, bold: true, color: C.accent3, valign: "middle", fontFace: "+mj-lt" });
  text(s, "price per token of the bigger model in our test (gpt-oss-120b vs gpt-oss-20b)", 10.15, 1.75, 2.35, 1.2,
    { fontSize: 14, color: C.background2, valign: "middle" });
  card(s, 8.3, 3.3, 4.4, 1.55, { name: "stat" });
  text(s, pct(cheapRun.accuracy), 8.6, 3.45, 1.6, 1.25, { fontSize: 54, bold: true, color: C.accent2, valign: "middle", fontFace: "+mj-lt" });
  text(s, `of our test questions the small model already got right (big model: ${pct(expRun.accuracy)})`, 10.15, 3.5, 2.35, 1.2,
    { fontSize: 14, color: C.background2, valign: "middle" });
  card(s, 8.3, 5.05, 4.4, 1.6, { fill: C.accent4, transparency: 82, border: C.accent4, name: "question" });
  text(s, "Can each step go to the cheapest model that can handle it — without ever overspending?", 8.55, 5.15, 3.95, 1.4,
    { fontSize: 16, bold: true, valign: "middle" });
  s.addNotes(
    "An agent breaks a task into many small steps, and most teams send every step to their best, most expensive model. " +
    "But many steps are easy: looking something up, formatting text. In our test the bigger model costs twice as much per token, " +
    "and the small model already answered 84% of questions correctly versus 89% for the big one. " +
    "So the question for this project is: can we pick the right model per step automatically, and guarantee we never blow the budget?");

  // ===================================================================
  // 3. The solution — architecture
  // ===================================================================
  pres.addSection({ title: "How it works" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("One LangGraph node picks the model for every step", { placeholder: "title" });
  const ay = 2.9;
  card(s, 0.6, ay - 0.5, 2.0, 1.0, { name: "agent" });
  text(s, "Agent step", 0.6, ay - 0.5, 2.0, 1.0, { fontSize: 16, bold: true, align: "center", valign: "middle" });
  arrow(s, 2.65, ay, 3.15, ay);
  card(s, 3.2, ay - 0.85, 3.1, 1.7, { fill: C.accent1, transparency: 80, border: C.accent1, borderW: 2, name: "router" });
  s.addImage({ data: await icon(fa.FaRoute, H.accent1), x: 3.45, y: ay - 0.6, w: 0.5, h: 0.5, objectName: nm("ri") });
  text(s, "RouterNode", 4.05, ay - 0.62, 2.1, 0.5, { fontSize: 20, bold: true, valign: "middle" });
  text(s, "score → budget check → route → validate → log", 3.45, ay + 0.05, 2.7, 0.65, { fontSize: 13, color: C.background2 });
  arrow(s, 6.35, ay - 0.2, 7.25, ay - 1.0, { color: C.accent2 });
  arrow(s, 6.35, ay + 0.2, 7.25, ay + 1.0, { color: C.accent3 });
  card(s, 7.3, ay - 1.5, 2.6, 1.0, { fill: C.accent2, transparency: 82, border: C.accent2, name: "cheap" });
  s.addImage({ data: await icon(fa.FaFeather, H.accent2), x: 7.5, y: ay - 1.22, w: 0.45, h: 0.45, objectName: nm("ci") });
  text(s, [{ text: "Cheap model", options: { bold: true, fontSize: 15, breakLine: true } },
    { text: meta.cheap_model, options: { fontSize: 12, color: C.background2 } }], 8.1, ay - 1.42, 1.75, 0.85, { valign: "middle" });
  card(s, 7.3, ay + 0.5, 2.6, 1.0, { fill: C.accent3, transparency: 82, border: C.accent3, name: "expensive" });
  s.addImage({ data: await icon(fa.FaGem, H.accent3), x: 7.5, y: ay + 0.78, w: 0.45, h: 0.45, objectName: nm("ei") });
  text(s, [{ text: "Expensive model", options: { bold: true, fontSize: 15, breakLine: true } },
    { text: meta.expensive_model, options: { fontSize: 12, color: C.background2 } }], 8.1, ay + 0.58, 1.75, 0.85, { valign: "middle" });
  arrow(s, 9.95, ay - 1.0, 10.65, ay - 0.2, { color: C.accent2 });
  arrow(s, 9.95, ay + 1.0, 10.65, ay + 0.2, { color: C.accent3 });
  card(s, 10.7, ay - 0.5, 2.0, 1.0, { name: "next" });
  text(s, "Next step", 10.7, ay - 0.5, 2.0, 1.0, { fontSize: 16, bold: true, align: "center", valign: "middle" });

  // helper bus
  const busY = 4.55;
  arrow(s, 4.75, ay + 0.85, 4.75, busY, { noHead: true, color: C.background2, width: 1.5 });
  arrow(s, 2.5, busY, 10.85, busY, { noHead: true, color: C.background2, width: 1.5 });
  const helpers = [
    [fa.FaGaugeHigh, "blue", "Difficulty estimator", "Scores each step 0–1 from prompt length, reasoning words and past failures. Under 1 ms, no API call."],
    [fa.FaWallet, "amber", "Budget tracker", "A USD ledger priced from real token usage, with 50% / 10% warnings and SOFT or HARD_STOP limits."],
    [fa.FaDatabase, "violet", "Decision logger", "Every decision goes to SQLite: score, model, cost, budget before and after. Prompts kept as hashes only."],
  ];
  for (const [i, [Ic, tint, title, desc]] of helpers.entries()) {
    const x = 0.6 + i * 4.15;
    arrow(s, x + 1.9, busY, x + 1.9, 4.9, { color: C.background2, width: 1.5 });
    card(s, x, 4.95, 3.8, 1.75, { name: "helper" });
    await iconCircle(s, Ic, tint, x + 0.25, 5.15, 0.7);
    text(s, title, x + 1.1, 5.2, 2.6, 0.45, { fontSize: 16, bold: true, valign: "middle" });
    text(s, desc, x + 0.25, 5.9, 3.35, 0.75, { fontSize: 12, color: C.background2 });
  }
  s.addNotes(
    "Here's the architecture. LangGraph is a popular framework for building agents as a graph of steps. " +
    "CostRouter is a single node you drop into that graph. Each step comes in, the RouterNode decides, and the step goes to either " +
    "the cheap model or the expensive model, then on to the next step. " +
    "Underneath are three parts: a difficulty estimator that scores the step, a budget tracker that keeps a ledger in real dollars, " +
    "and a decision logger that records every choice in a SQLite database so you can audit it later.");

  // ===================================================================
  // 4. Five checks per step
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Every step goes through the same five checks", { placeholder: "title" });
  const flow = [
    [fa.FaGaugeHigh, "blue", "Score difficulty", "Length, reasoning words and failure history become one score from 0 to 1"],
    [fa.FaWallet, "amber", "Check the budget", "Can the ledger cover the expensive model, even in the worst case?"],
    [fa.FaCodeBranch, "green", "Route", "Score above the threshold and affordable → expensive; otherwise cheap"],
    [fa.FaRotate, "orange", "Validate", "Empty or malformed cheap answer → retry once on the expensive model"],
    [fa.FaClipboardList, "violet", "Charge & log", "Real token cost goes to the ledger; the decision goes to SQLite"],
  ];
  const colW = 12.1 / 5;
  for (const [i, [Ic, tint, title, desc]] of flow.entries()) {
    const cx = 0.6 + i * colW + colW / 2;
    await iconCircle(s, Ic, tint, cx - 0.6, 1.6, 1.2);
    s.addShape(pres.shapes.OVAL, { x: cx + 0.25, y: 1.5, w: 0.42, h: 0.42, fill: { color: TINT[tint] },
      line: { type: "none" }, objectName: nm("num") });
    text(s, String(i + 1), cx + 0.25, 1.5, 0.42, 0.42, { fontSize: 14, bold: true, align: "center", valign: "middle", color: C.text1 });
    text(s, title, cx - colW / 2 + 0.1, 2.95, colW - 0.2, 0.45, { fontSize: 18, bold: true, align: "center" });
    text(s, desc, cx - colW / 2 + 0.2, 3.45, colW - 0.4, 1.1, { fontSize: 14, color: C.background2, align: "center" });
    if (i < flow.length - 1) arrow(s, cx + 0.75, 2.2, cx + colW - 0.75, 2.2);
  }
  card(s, 0.6, 4.95, 12.1, 1.75, { name: "example" });
  text(s, "Example from the live demo — “What's the cheapest way to buy 22 pens?”", 0.9, 5.1, 11.5, 0.4,
    { fontSize: 15, bold: true });
  const ex = [["score 0.54", "blue"], ["> threshold 0.30", "blue"], ["budget OK", "amber"],
    ["EXPENSIVE · gpt-oss-120b", "orange"], ["valid: $56", "green"], ["cost $0.00066", "violet"]];
  let ex0 = 0.9;
  const exW = [1.35, 1.8, 1.3, 2.7, 1.35, 1.65];
  for (const [i, [t, tint]] of ex.entries()) {
    chip(s, t, ex0, 5.75, exW[i], 0.55, tint, { fontSize: 13 });
    if (i < ex.length - 1) arrow(s, ex0 + exW[i] + 0.03, 6.025, ex0 + exW[i] + 0.19, 6.025, { width: 1.5 });
    ex0 += exW[i] + 0.22;
  }
  s.addNotes(
    "Every single step goes through the same five checks. One: score how difficult it looks. Two: check the budget can cover the " +
    "expensive model in the worst case. Three: route — if the score is above a threshold and we can afford it, use the expensive model, " +
    "otherwise the cheap one. Four: validate the answer — if the cheap model returns something empty or broken, retry once on the expensive model. " +
    "Five: charge the real cost to the ledger and log the decision. At the bottom is a real example from the live demo: the pens question scored 0.54, " +
    "above the 0.30 threshold, so it went to the expensive model, which answered $56 for well under a tenth of a cent.");

  // ===================================================================
  // 5. Difficulty score
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Difficulty: three cheap signals, scored in under 1 ms", { placeholder: "title" });
  const signals = [
    [fa.FaRulerHorizontal, "blue", "Prompt length", "× 0.3", "Longer prompts are assumed harder (full marks at 2,000 characters)"],
    [fa.FaLightbulb, "violet", "Reasoning words", "× 0.3", "“step by step”, “prove”, “compare”, “justify”… (full marks at 2 different words)"],
    [fa.FaClockRotateLeft, "red", "Failure history", "× 0.4", "How often this kind of step failed validation before (starts at a 0.5 prior)"],
  ];
  for (const [i, [Ic, tint, title, w, desc]] of signals.entries()) {
    const y = 1.55 + i * 1.32;
    card(s, 0.6, y, 5.4, 1.15, { name: "signal" });
    await iconCircle(s, Ic, tint, 0.8, y + 0.2, 0.75);
    text(s, title, 1.75, y + 0.15, 2.8, 0.4, { fontSize: 17, bold: true });
    chip(s, w, 4.75, y + 0.17, 1.0, 0.4, tint, { fontSize: 14 });
    text(s, desc, 1.75, y + 0.55, 4.05, 0.55, { fontSize: 13, color: C.background2 });
  }
  card(s, 0.6, 5.6, 5.4, 1.1, { fill: C.accent1, transparency: 82, border: C.accent1, name: "formula" });
  text(s, [{ text: "score = 0.3·length + 0.3·words + 0.4·history", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "score above the threshold → expensive model", options: { fontSize: 14, color: C.background2 } }],
    0.85, 5.68, 5.0, 0.95, { valign: "middle" });

  const exs = [...DATA.difficulty_examples].reverse();
  const cats = exs.map((e) => `${e.label}  (${e.score.toFixed(2)})`);
  s.addChart(pres.charts.BAR, [
    { name: "Prompt length", labels: cats, values: exs.map((e) => +e.length.toFixed(3)) },
    { name: "Reasoning words", labels: cats, values: exs.map((e) => +e.cues.toFixed(3)) },
    { name: "Failure history", labels: cats, values: exs.map((e) => +e.history.toFixed(3)) },
  ], {
    x: 6.4, y: 1.9, w: 6.3, h: 4.55, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 60,
    chartColors: [HEX.len, HEX.cue, HEX.hist], showLegend: true, legendPos: "b", showTitle: false,
    valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
    layout: { x: 0.42, y: 0.12, w: 0.54, h: 0.7 },
    ...CHART_TEXT, objectName: nm("difficulty-chart"),
  });
  // threshold marker over the plot area (plot spans x 0.42–0.96 of the chart, axis 0–0.6)
  text(s, "Score breakdown for four demo questions", 6.4, 1.5, 6.3, 0.35, { fontSize: 15, bold: true });
  const plotX = 6.4 + 0.42 * 6.3, plotW = 0.54 * 6.3, plotY = 1.9 + 0.12 * 4.55, plotH = 0.7 * 4.55;
  const tx = plotX + (0.3 / 0.6) * plotW;
  arrow(s, tx, plotY - 0.05, tx, plotY + plotH, { noHead: true, color: C.background1, width: 2, dash: "dash" });
  text(s, "threshold 0.30", tx - 0.8, plotY - 0.36, 1.6, 0.28, { fontSize: 12, bold: true, align: "center" });
  text(s, "A fresh session: every kind of step starts at the 0.5 history prior, which adds 0.20.", 6.4, 6.5, 6.3, 0.3,
    { fontSize: 11, color: C.background2 });
  s.addNotes(
    "How do we judge difficulty without asking an AI? Three cheap signals. Prompt length, because longer questions tend to be harder. " +
    "Reasoning words like 'step by step', 'prove' or 'compare'. And history: how often this kind of step has failed before. " +
    "They're combined with fixed weights into one score. The chart shows four real demo questions: the capital-of-Australia question scores 0.21 " +
    "and the plain math problem 0.25, so both go to the cheap model, while the pens and train questions are full of reasoning words and score 0.54, " +
    "above the 0.30 threshold, so they go to the expensive model. All of this takes under a millisecond and costs nothing.");

  // ===================================================================
  // 6. Budget safety
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("A dollar budget that degrades visibly — or holds a hard cap", { placeholder: "title" });
  const feats = [
    [fa.FaCoins, "amber", "Charged in real dollars from each call's token usage"],
    [fa.FaTriangleExclamation, "amber", "Warnings when 50% and 10% of the budget is left"],
    [fa.FaArrowDownWideShort, "blue", "Can't afford the expensive model? The step is downgraded and flagged"],
  ];
  for (const [i, [Ic, tint, t]] of feats.entries()) {
    const y = 1.5 + i * 0.72;
    await iconCircle(s, Ic, tint, 0.6, y, 0.55);
    text(s, t, 1.35, y, 4.35, 0.55, { fontSize: 14, valign: "middle" });
  }
  const modes = [
    ["SOFT mode", "blue", usd(soft.total_cost), `spent vs a ${usd(soft.budget_usd)} cap (${pct(soft.total_cost / soft.budget_usd)})`,
      `${soft.budget_forced} steps downgraded · finishes the job · ${pct(soft.accuracy)} accuracy`],
    ["HARD_STOP mode", "red", usd(hard.total_cost), `never over the ${usd(hard.budget_usd)} cap`,
      `${hard.unanswered} steps refused · stops early · ${pct(hard.accuracy)} accuracy`],
  ];
  for (const [i, [title, tint, big, sub, foot]] of modes.entries()) {
    const x = 0.6 + i * 2.65;
    card(s, x, 3.85, 2.5, 2.85, { fill: TINT[tint], transparency: 84, border: TINT[tint], name: "mode" });
    text(s, title, x + 0.2, 4.0, 2.1, 0.35, { fontSize: 15, bold: true, color: TINT[tint] });
    text(s, big, x + 0.2, 4.4, 2.1, 0.7, { fontSize: 32, bold: true, fontFace: "+mj-lt", valign: "middle" });
    text(s, sub, x + 0.2, 5.12, 2.1, 0.55, { fontSize: 12, color: C.background2 });
    text(s, foot, x + 0.2, 5.75, 2.1, 0.85, { fontSize: 12 });
  }
  const softY = DATA.capped.capped_soft.remaining, hardY = DATA.capped.capped_hard.remaining;
  const stepLabels = softY.map((_, i) => String(i + 1));
  s.addChart(pres.charts.LINE, [
    { name: "SOFT", labels: stepLabels, values: softY },
    { name: "HARD_STOP", labels: stepLabels, values: stepLabels.map((_, i) => (i < hardY.length ? hardY[i] : null)) },
  ], {
    x: 5.95, y: 1.45, w: 6.75, h: 5.25, chartColors: [H.accent1, H.accent6], lineSize: 3, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", showTitle: true,
    title: `Budget left per step (cap ${usd(soft.budget_usd)})`,
    valAxisLabelFormatCode: "$0.000", catAxisLabelFrequency: 10, displayBlanksAs: "gap",
    ...CHART_TEXT, objectName: nm("budget-chart"),
  });
  s.addNotes(
    "The budget tracker is a ledger in real dollars: every call is charged from its actual token usage. " +
    "It warns at 50% and 10% left, and when the expensive model is no longer affordable, the step is downgraded to the cheap model and flagged. " +
    "There are two modes. In this test the budget was capped at half of what the expensive model alone would spend. " +
    "SOFT mode keeps answering with the cheap model after the money runs out, so it finished every question but ended at 168% of the cap — the blue line goes below zero. " +
    "HARD_STOP refuses any call it can't afford: it stayed under the cap, but stopped answering after 47 questions. So: SOFT when finishing matters, HARD_STOP when the budget is a hard limit.");

  // ===================================================================
  // 7. Retry-escalate + audit
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Cheap model failed? Escalate — and log every decision", { placeholder: "title" });
  card(s, 0.6, 1.5, 6.1, 0.95, { name: "q" });
  s.addImage({ data: await icon(fa.FaCircleQuestion, H.lt2), x: 0.85, y: 1.75, w: 0.45, h: 0.45, objectName: nm("qi") });
  text(s, "“Judith Keppel was the first winner of the top prize on which UK programme?”", 1.5, 1.55, 5.0, 0.85,
    { fontSize: 15, italic: true, valign: "middle" });
  arrow(s, 1.6, 2.5, 1.6, 2.95);
  const row = async (y, Ic, tint, model, resIc, resTint, result) => {
    card(s, 0.6, y, 2.3, 0.85, { fill: TINT[tint], transparency: 82, border: TINT[tint], name: "model" });
    s.addImage({ data: await icon(Ic, TINT_HEX[tint]), x: 0.8, y: y + 0.22, w: 0.42, h: 0.42, objectName: nm("mi") });
    text(s, model, 1.3, y, 1.55, 0.85, { fontSize: 14, bold: true, valign: "middle" });
    arrow(s, 2.95, y + 0.425, 3.45, y + 0.425, { color: TINT[resTint] });
    card(s, 3.5, y, 3.2, 0.85, { name: "result" });
    s.addImage({ data: await icon(resIc, TINT_HEX[resTint]), x: 3.7, y: y + 0.22, w: 0.42, h: 0.42, objectName: nm("res") });
    text(s, result, 4.25, y, 2.35, 0.85, { fontSize: 13, valign: "middle" });
  };
  await row(3.0, fa.FaFeather, "green", "Cheap model", fa.FaCircleXmark, "red", "Empty reply: spent all 1,024 tokens thinking");
  arrow(s, 1.6, 3.9, 1.6, 4.35, { color: C.accent3 });
  text(s, "validator: no “Final answer:” line → retry on the expensive model", 1.85, 3.92, 4.8, 0.4,
    { fontSize: 12, color: C.accent3, valign: "middle" });
  await row(4.4, fa.FaGem, "orange", "Expensive model", fa.FaCircleCheck, "green", "“Who Wants to Be a Millionaire?” — correct");
  card(s, 0.6, 5.55, 6.1, 1.15, { fill: C.accent4, transparency: 84, border: C.accent4, name: "fact" });
  text(s, [{ text: "7 of 100", options: { bold: true, fontSize: 22, color: C.accent4 } },
    { text: "  cheap replies in the benchmark were empty. Whenever the router sent one of those to the cheap model, the validator caught it and — budget permitting — escalated.",
      options: { fontSize: 13 } }], 0.85, 5.62, 5.65, 1.0, { valign: "middle" });

  card(s, 7.2, 1.5, 5.5, 5.2, { name: "audit" });
  await iconCircle(s, fa.FaDatabase, "violet", 7.45, 1.7, 0.6);
  text(s, "The audit record (SQLite)", 8.2, 1.7, 4.3, 0.6, { fontSize: 17, bold: true, valign: "middle" });
  const rec = [
    ["step_id", "triviaqa-16"], ["difficulty_score", "0.058"], ["model_chosen", "expensive"], ["retried", "true"],
    ["budget_forced", "false"], ["actual_cost_usd", "0.000427"], ["budget_before", "9.992341"], ["budget_after", "9.991914"],
    ["prompt_hash", "b764d66dcf871fbc"], ["error", "MalformedOutputError"],
  ];
  text(s, rec.map(([k, v], i) => ({ text: `${k.padEnd(17, " ")}${v}`, options: { breakLine: i < rec.length - 1,
    color: k === "retried" || k === "model_chosen" ? C.accent3 : C.background1 } })),
    7.5, 2.55, 5.0, 3.5, { fontSize: 15, fontFace: "Courier New", paraSpaceAfter: 4 });
  text(s, "A real row from the t = 0.065 benchmark run. Prompts are stored as length + hash, never as text.", 7.5, 6.05, 5.0, 0.55,
    { fontSize: 12, color: C.background2 });
  s.addNotes(
    "Cheap models sometimes fail in obvious ways. Here's a real case from the benchmark: on this trivia question, the small model spent its entire " +
    "1,024-token allowance thinking and returned nothing. CostRouter's validator noticed there was no final answer and retried once on the expensive model, " +
    "which got it right. That happened to 7 of the 100 cheap answers; the validator caught every one the router sent to the cheap model, and escalated it whenever the budget allowed. " +
    "On the right is the actual database row for that decision: the score, the model, that it was retried, the exact cost and the budget before and after. " +
    "We store a hash of the prompt instead of the text, for privacy.");

  // ===================================================================
  // 8. The experiment
  // ===================================================================
  pres.addSection({ title: "Evaluation" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
  s.addText("We tested it on 100 real questions with two real models", { placeholder: "title" });
  const apiCost = cheapRun.total_cost + expRun.total_cost;
  const facts = [
    [fa.FaListCheck, "blue", "100", "real questions", "50 TriviaQA (knowledge) + 50 GSM8K (math word problems), seeded sample, shuffled together"],
    [fa.FaMicrochip, "orange", "2", "real models on Groq", `${meta.cheap_model} (cheap) vs ${meta.expensive_model} (expensive)`],
    [fa.FaTags, "amber", "2×", "price gap", "$0.075 / $0.30 vs $0.15 / $0.60 per million input / output tokens"],
    [fa.FaFlask, "violet", "9", "runs", "2 baselines, a 5-threshold sweep, and 2 budget caps (SOFT and HARD_STOP)"],
    [fa.FaScaleBalanced, "green", usd(apiCost, 3), "total API cost", "every answer cached at temperature 0, so all runs see identical responses"],
    [fa.FaVial, "blue", "139", "automated tests", "plus hand-checked grading of every answer marked wrong"],
  ];
  for (const [i, [Ic, tint, big, label, desc]] of facts.entries()) {
    const col = i % 3, rowI = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.55 + rowI * 2.6;
    card(s, x, y, 3.85, 2.35, { name: "fact" });
    await iconCircle(s, Ic, tint, x + 0.25, y + 0.25, 0.75);
    text(s, big, x + 1.2, y + 0.18, 2.5, 0.65, { fontSize: 36, bold: true, color: TINT[tint], fontFace: "+mj-lt", valign: "middle" });
    text(s, label, x + 1.2, y + 0.8, 2.5, 0.35, { fontSize: 14, bold: true });
    text(s, desc, x + 0.25, y + 1.3, 3.35, 0.9, { fontSize: 13, color: C.background2 });
  }
  s.addNotes(
    "To find out whether it actually works, we ran a real experiment. 100 questions from two well-known benchmarks: TriviaQA for general knowledge and " +
    "GSM8K for math word problems. Two real models from OpenAI's open-weight gpt-oss family, served by Groq; the big one costs twice as much per token. " +
    "We ran nine configurations: always cheap, always expensive, five router thresholds and two budget caps. " +
    "Every answer was recorded once, so all nine runs see exactly the same responses — the only difference is the routing. The whole experiment cost about two cents. " +
    "The code has 139 automated tests, and I checked the grading by hand.");

  // ===================================================================
  // 9. Results
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
  s.addText(`Results: v1 ≈ a random mix; task-type routing could save ${pct(1 - oracle.total_cost / expRun.total_cost)}`,
    { placeholder: "title" });
  const pts = [
    ["Always cheap", [cheapRun]], ["Always expensive", [expRun]], ["Router (threshold sweep)", sweep],
    ["Router, budget-capped (SOFT)", [soft]], ["Task-type routing (hindsight)", [{ total_cost: oracle.total_cost, accuracy: oracle.accuracy }]],
  ];
  const xs = pts.flatMap(([, rs]) => rs.map((r) => +r.total_cost.toFixed(5)));
  let off = 0;
  const ySeries = pts.map(([name, rs]) => {
    const values = xs.map((_, j) => (j >= off && j < off + rs.length ? +rs[j - off].accuracy.toFixed(3) : null));
    off += rs.length;
    return { name, values };
  });
  s.addChart(pres.charts.SCATTER, [{ name: "Cost", values: xs }, ...ySeries], {
    x: 0.6, y: 1.45, w: 7.9, h: 5.3, lineSize: 0, lineDataSymbolSize: 14, lineDataSymbol: "circle",
    chartColors: [H.accent2, H.accent3, H.accent1, H.accent5, H.accent4],
    showLegend: true, legendPos: "b", showTitle: true, title: "Cost vs accuracy, 100 tasks per run",
    valAxisMinVal: 0.82, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0%",
    catAxisMinVal: 0.009, catAxisMaxVal: 0.014, catAxisLabelFormatCode: "$0.000",
    showValAxisTitle: true, valAxisTitle: "Accuracy", showCatAxisTitle: true, catAxisTitle: "Total cost for 100 tasks (US dollars)",
    valAxisTitleColor: HEX.axis, catAxisTitleColor: HEX.axis, valAxisTitleFontSize: 12, catAxisTitleFontSize: 12,
    valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
    ...CHART_TEXT, catGridLine: { color: HEX.grid, size: 0.75 }, objectName: nm("pareto"),
  });
  const callouts = [
    ["Always cheap", "green", pct(cheapRun.accuracy), `${usd(cheapRun.total_cost)} · ${pct(cheapRun.total_cost / expRun.total_cost)} of the cost`],
    ["Always expensive", "orange", pct(expRun.accuracy), `${usd(expRun.total_cost)} · the baseline`],
    ["Best router setting", "blue", pct(best.accuracy), `${pct(best.total_cost / expRun.total_cost)} of the cost — about a random mix`],
    ["Route by task type", "violet", pct(oracle.accuracy), `${pct(oracle.total_cost / expRun.total_cost)} of the cost — same accuracy`],
  ];
  for (const [i, [label, tint, big, sub]] of callouts.entries()) {
    const y = 1.45 + i * 1.33;
    card(s, 8.9, y, 3.8, 1.18, { name: "callout", fill: i === 3 ? C.accent4 : C.text2, transparency: i === 3 ? 84 : 0,
      border: i === 3 ? C.accent4 : undefined });
    text(s, big, 9.1, y + 0.12, 1.35, 0.95, { fontSize: 34, bold: true, color: TINT[tint], fontFace: "+mj-lt", valign: "middle" });
    text(s, [{ text: label, options: { bold: true, fontSize: 14, breakLine: true } },
      { text: sub, options: { fontSize: 12, color: C.background2 } }], 10.5, y + 0.12, 2.1, 0.95, { valign: "middle" });
  }
  s.addNotes(
    "Here are the results. The x-axis is cost, the y-axis is accuracy, so up and to the left is better. " +
    "Always cheap got 84% for under a cent; always expensive got 89%. Our router settings, in blue, land between them — the best one gets 88% at 96% of the cost. " +
    "Honestly, that's about what you'd get by randomly mixing the two models. " +
    "But look at the purple point: if we had routed by task type — trivia to the big model, math to the small one — we'd keep the full 89% at 73% of the cost. " +
    "So the infrastructure works, and there's real money on the table; the next slide shows why v1 misses it.");

  // ===================================================================
  // 10. Why
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
  s.addText("Why: the models differ only on trivia, not on math", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Cheap model", labels: ["TriviaQA (knowledge)", "GSM8K (math)"], values: [cheapRun.acc_triviaqa, cheapRun.acc_gsm8k] },
    { name: "Expensive model", labels: ["TriviaQA (knowledge)", "GSM8K (math)"], values: [expRun.acc_triviaqa, expRun.acc_gsm8k] },
  ], {
    x: 0.6, y: 1.45, w: 5.9, h: 4.15, barDir: "col", barGapWidthPct: 70, chartColors: [H.accent2, H.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", showLegend: true, legendPos: "b",
    showTitle: true, title: "Accuracy by task type", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.2,
    valAxisLabelFormatCode: "0%", ...CHART_TEXT, objectName: nm("acc-by-type"),
  });
  const shareLabels = Object.keys(DATA.expensive_share).map((k) => k.replace("Router ", ""));
  const shareVals = Object.values(DATA.expensive_share);
  s.addChart(pres.charts.BAR, [
    { name: "GSM8K (math)", labels: shareLabels, values: shareVals.map((v) => v.gsm8k) },
    { name: "TriviaQA (knowledge)", labels: shareLabels, values: shareVals.map((v) => v.triviaqa) },
  ], {
    x: 6.85, y: 1.45, w: 5.85, h: 4.15, barDir: "col", barGapWidthPct: 60, chartColors: ["E879F9", "38BDF8"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", showLegend: true, legendPos: "b",
    showTitle: true, title: "Share sent to the expensive model, by router threshold", valAxisMinVal: 0, valAxisMaxVal: 1.15,
    valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0%", ...CHART_TEXT, objectName: nm("share-by-type"),
  });
  card(s, 0.6, 5.8, 12.1, 0.9, { fill: C.accent4, transparency: 84, border: C.accent4, name: "takeaway" });
  await iconCircle(s, fa.FaLightbulb, "violet", 0.8, 5.92, 0.66);
  text(s, [{ text: "Long math looks “hard” to the heuristic, but both models score 96% there. ", options: { bold: true } },
    { text: "The money is wasted on GSM8K, while trivia — the one place the big model helps — is where it's needed.", options: {} }],
    1.65, 5.85, 10.9, 0.8, { fontSize: 14, valign: "middle" });
  s.addNotes(
    "Why does v1 miss it? On the left: on math, both models score 96% — the small model is a reasoning model and handles GSM8K fine. " +
    "The whole accuracy gap is on trivia: 72% versus 82%. On the right: but our heuristic thinks long math problems are hard, " +
    "so at most thresholds it sends almost all of the math to the expensive model, where nothing is gained. " +
    "The lesson is that prompt length is a poor proxy for difficulty when the cheap model can reason. That's exactly what version 2 should learn.");

  // ===================================================================
  // 11. Live demo
  // ===================================================================
  pres.addSection({ title: "Demo and wrap-up" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and wrap-up" });
  s.addText("Live demo: watch every routing decision", { placeholder: "title" });
  const iw = 7.6, ih = iw / (1415 / 915);
  card(s, 0.5, 1.4, iw + 0.2, ih + 0.2, { fill: C.accent1, transparency: 70, r: 0.08, name: "frame" });
  s.addImage({ path: path.join(__dirname, "img", "live_crop.png"), x: 0.6, y: 1.5, w: iw, h: ih,
    altText: "CostRouter dashboard, Live Router page: the five routing steps for a retry-escalated question", objectName: nm("live-shot") });
  const demo = [
    [fa.FaFeather, "green", "Easy question → cheap model"],
    [fa.FaGem, "orange", "Hard question → expensive; tricky trivia → retry"],
    [fa.FaWallet, "amber", "Shrink the budget → budget-forced downgrades"],
    [fa.FaSliders, "blue", "Playground: drag the threshold, watch cost and accuracy move"],
  ];
  for (const [i, [Ic, tint, t]] of demo.entries()) {
    const y = 1.45 + i * 0.78;
    await iconCircle(s, Ic, tint, 8.75, y, 0.6);
    text(s, t, 9.5, y - 0.02, 3.2, 0.66, { fontSize: 14, valign: "middle" });
  }
  const pw = 3.95, ph = pw / (1415 / 540);
  card(s, 8.7, 4.7, pw + 0.1, ph + 0.1, { fill: C.accent1, transparency: 70, r: 0.06, name: "frame" });
  s.addImage({ path: path.join(__dirname, "img", "playground_crop.png"), x: 8.75, y: 4.75, w: pw, h: ph,
    altText: "Dashboard Playground: cost vs accuracy chart and a grid of all 100 tasks", objectName: nm("pg-shot") });
  chip(s, "streamlit run dashboard/app.py", 8.75, 6.3, 3.95, 0.42, "blue", { fontSize: 12, bold: false });
  s.addNotes(
    "Now let me show it live. [Switch to the dashboard.] On the Live Router page I'll ask an easy question — it goes to the cheap model. " +
    "Then a hard one with lots of reasoning words — expensive. Then the tricky trivia question: watch step 4 turn amber as the cheap answer comes back empty " +
    "and gets retried on the expensive model. Then I'll shrink the budget so hard questions get downgraded. " +
    "Finally the Playground replays all 100 benchmark questions instantly as I drag the threshold. " +
    "If anything goes wrong with the live demo, these screenshots show the same screens.");

  // ===================================================================
  // 12. Lessons + next steps
  // ===================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and wrap-up" });
  s.addText("What we learned and what comes next", { placeholder: "title" });
  text(s, "LESSONS", 0.6, 1.45, 5.8, 0.35, { fontSize: 13, bold: true, color: C.accent1, charSpacing: 3 });
  const lessons = [
    [fa.FaCircleCheck, "green", "The plumbing works", "Budgets, escalation and the audit trail behave as specified on real models"],
    [fa.FaRulerHorizontal, "amber", "Length is a weak signal", "When the cheap model can reason, “long” doesn't mean “hard”"],
    [fa.FaMagnifyingGlass, "violet", "Measure honestly", "Fixed seeds, cached answers and hand-checked grading made the result trustworthy"],
  ];
  for (const [i, [Ic, tint, t, d]] of lessons.entries()) {
    const y = 1.95 + i * 1.12;
    await iconCircle(s, Ic, tint, 0.6, y, 0.7);
    text(s, t, 1.5, y - 0.02, 4.9, 0.38, { fontSize: 16, bold: true });
    text(s, d, 1.5, y + 0.36, 4.9, 0.6, { fontSize: 13, color: C.background2 });
  }
  text(s, "NEXT VERSION", 7.0, 1.45, 5.7, 0.35, { fontSize: 13, bold: true, color: C.accent3, charSpacing: 3 });
  const next = [
    [fa.FaFeather, "green", "Cheap first, escalate on failure"],
    [fa.FaBrain, "violet", "Learn difficulty per kind of step"],
    [fa.FaLayerGroup, "blue", "More than two model tiers, async agents"],
    [fa.FaRocket, "orange", "CI, a PyPI package, live provider prices"],
  ];
  for (const [i, [Ic, tint, t]] of next.entries()) {
    const y = 1.95 + i * 0.84;
    card(s, 7.0, y, 5.7, 0.68, { name: "next" });
    s.addImage({ data: await icon(Ic, TINT_HEX[tint]), x: 7.2, y: y + 0.16, w: 0.36, h: 0.36, objectName: nm("ni") });
    text(s, t, 7.75, y, 4.8, 0.68, { fontSize: 15, valign: "middle" });
  }
  card(s, 0.6, 5.6, 12.1, 1.1, { fill: C.accent1, transparency: 80, border: C.accent1, name: "qa" });
  text(s, "Questions?", 0.9, 5.6, 4.0, 1.1, { fontSize: 32, bold: true, fontFace: "+mj-lt", valign: "middle" });
  s.addImage({ data: await icon(fa.FaGithub, H.lt1), x: 6.6, y: 5.92, w: 0.46, h: 0.46, objectName: nm("gh2") });
  text(s, "github.com/prakharsahup/Cost-Router", 7.2, 5.6, 5.3, 1.1, { fontSize: 18, valign: "middle" });
  s.addNotes(
    "To wrap up. Three lessons: the plumbing works — budgets, escalation and logging behave exactly as designed on real models. " +
    "Prompt length turned out to be a weak difficulty signal once the cheap model can reason. And measuring honestly — fixed seeds, recorded answers, " +
    "hand-checked grading — is what let us trust a result that wasn't the one we hoped for. " +
    "Next version: try the cheap model first and escalate only on failure, learn difficulty per kind of step, support more models and async agents, " +
    "and package it properly. The code is on GitHub. Thank you — happy to take questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("Wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
