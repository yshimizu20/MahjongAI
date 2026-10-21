/* Game viewer (T-052): renders one trajectory JSON (viewer/export.py) step by step. Vanilla JS, no build step.
   The reducer (`reduce`) is pure: (view, action) -> view; `render` draws a view. Every string from a game reaches
   the DOM as a text node (never innerHTML), and a game is validated (`validateGame`) before it is shown. */
"use strict";

const KIND_FILE = [].concat(
  [1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => `Man${n}`),
  [1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => `Pin${n}`),
  [1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => `Sou${n}`),
  ["Ton", "Nan", "Shaa", "Pei", "Haku", "Hatsu", "Chun"]);
const RED = { 16: "Man5-Dora", 52: "Pin5-Dora", 88: "Sou5-Dora" };
const WINDS = ["東", "南", "西", "北"];
const SUITS = "mps";
const DRAW_REASON = { exhaustive: "流局", yao9: "九種九牌", ron3: "三家和", reach4: "四家立直", kaze4: "四風連打", kan4: "四槓散了", nm: "流し満貫" };

function tileFile(id) { return RED[id] || KIND_FILE[id >> 2]; }
function tileName(id) {  // the 37-slot name used by the exporter's actions
  const k = id >> 2;
  if (RED[id]) return `5${SUITS[Math.floor(k / 9)]}r`;
  return k < 27 ? `${(k % 9) + 1}${SUITS[Math.floor(k / 9)]}` : WINDS.concat(["白", "發", "中"])[k - 27];
}
function slotOf(id) { return RED[id] ? 34 + Math.floor((id >> 2) / 9) : id >> 2; }
function el(tag, cls, text) {  // text is set as a text node: game strings never become markup
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function tileEl(id, extra) {
  const t = el("div", "tile" + (extra ? " " + extra : ""));
  if (id === null || id === undefined) { t.classList.add("back"); return t; }
  if ((id >> 2) === 31) t.classList.add("haku");  // 白: the set's face is plain white; a frame keeps it from reading as empty
  const img = document.createElement("img");
  img.src = `tiles/${tileFile(id)}.svg`;
  img.alt = tileName(id);
  img.draggable = false;
  t.appendChild(img);
  return t;
}
function roundName(index) { return `${WINDS[Math.floor(index / 4)]}${(index % 4) + 1}局`; }
function fmtPts(p) { return p.toLocaleString("en-US"); }

/* ---- validation of a trajectory JSON (API or file): shape and field types, the first violation names its path ---- */
function validateGame(game) {
  const bad = (path, what) => { throw new Error(`invalid trajectory: ${path} ${what}`); };
  const isInt = (x, lo, hi) => Number.isInteger(x) && x >= lo && x <= hi;
  const str = (x, p) => { if (typeof x !== "string") bad(p, "is not a string"); };
  const num = (x, p) => { if (typeof x !== "number" || !Number.isFinite(x)) bad(p, "is not a finite number"); };
  const bool = (x, p) => { if (typeof x !== "boolean") bad(p, "is not a boolean"); };
  const arr = (x, p, n) => { if (!Array.isArray(x) || (n !== undefined && x.length !== n)) bad(p, n === undefined ? "is not an array" : `is not an array of ${n}`); return x; };
  const obj = (x, p) => { if (!x || typeof x !== "object" || Array.isArray(x)) bad(p, "is not an object"); return x; };
  const tile = (x, p) => { if (!isInt(x, 0, 135)) bad(p, "is not a tile id"); };
  const tiles = (x, p) => arr(x, p).forEach((t, i) => tile(t, `${p}[${i}]`));
  const seat = (x, p) => { if (!isInt(x, 0, 3)) bad(p, "is not a seat"); };
  const seatOrNull = (x, p) => { if (x !== null) seat(x, p); };
  const act = (x, p) => { obj(x, p); if (!isInt(x.idx, 0, 153)) bad(`${p}.idx`, "is not an action index"); str(x.name, `${p}.name`); };
  const meld = (m, p) => {
    obj(m, p);
    if (!["chi", "pon", "minkan", "ankan", "kakan"].includes(m.kind)) bad(`${p}.kind`, "is not a meld kind");
    tiles(m.tiles, `${p}.tiles`);
    if (m.called !== null) tile(m.called, `${p}.called`);
    seatOrNull(m.from, `${p}.from`);
    if (m.added !== null) tile(m.added, `${p}.added`);
  };
  obj(game, "game");
  str(game.source, "source");
  arr(game.players, "players", 4).forEach((n, i) => str(n, `players[${i}]`));
  bool(game.complete, "complete");
  num(game.rounds_dropped, "rounds_dropped");
  arr(game.rounds, "rounds").forEach((r, ri) => {
    const rp = `rounds[${ri}]`;
    obj(r, rp);
    if (!isInt(r.index, 0, 15)) bad(`${rp}.index`, "is not a round index");
    if (!(typeof r.honba === "string" || typeof r.honba === "number")) bad(`${rp}.honba`, "is not a count");
    seat(r.oya, `${rp}.oya`);
    arr(r.steps, `${rp}.steps`).forEach((s, si) => {
      const sp = `${rp}.steps[${si}]`;
      obj(s, sp);
      if (!["during", "discard", "post"].includes(s.kind)) bad(`${sp}.kind`, "is not a point kind");
      seat(s.actor, `${sp}.actor`);
      arr(s.hands, `${sp}.hands`, 4).forEach((h, p) => tiles(h, `${sp}.hands[${p}]`));
      if (s.drawn !== null) { obj(s.drawn, `${sp}.drawn`); seat(s.drawn.seat, `${sp}.drawn.seat`); tile(s.drawn.tile, `${sp}.drawn.tile`); }
      arr(s.ponds, `${sp}.ponds`, 4).forEach((pond, p) => arr(pond, `${sp}.ponds[${p}]`).forEach((e, i) => {
        const ep = `${sp}.ponds[${p}][${i}]`;
        obj(e, ep); tile(e.tile, `${ep}.tile`); bool(e.tsumogiri, `${ep}.tsumogiri`); bool(e.riichi, `${ep}.riichi`);
        seatOrNull(e.called_by, `${ep}.called_by`);
      }));
      arr(s.melds, `${sp}.melds`, 4).forEach((ms, p) => arr(ms, `${sp}.melds[${p}]`).forEach((m, i) => meld(m, `${sp}.melds[${p}][${i}]`)));
      arr(s.scores, `${sp}.scores`, 4).forEach((x, p) => num(x, `${sp}.scores[${p}]`));
      arr(s.reaches, `${sp}.reaches`, 4).forEach((x, p) => num(x, `${sp}.reaches[${p}]`));
      num(s.honba, `${sp}.honba`); num(s.kyotaku, `${sp}.kyotaku`); num(s.draws_left, `${sp}.draws_left`);
      tiles(s.doras, `${sp}.doras`);
      if (!isInt(s.round, 0, 15)) bad(`${sp}.round`, "is not a round index");
      seat(s.oya, `${sp}.oya`);
      arr(s.decisions, `${sp}.decisions`).forEach((d, di) => {
        const dp = `${sp}.decisions[${di}]`;
        obj(d, dp); seat(d.seat, `${dp}.seat`);
        if (!["during", "discard", "post"].includes(d.head)) bad(`${dp}.head`, "is not a head");
        arr(d.legal, `${dp}.legal`).forEach((a, i) => act(a, `${dp}.legal[${i}]`));
        obj(d.logits, `${dp}.logits`); obj(d.probs, `${dp}.probs`);
        Object.entries(d.probs).forEach(([k, v]) => { if (!/^\d+$/.test(k)) bad(`${dp}.probs.${k}`, "is not an index"); num(v, `${dp}.probs.${k}`); });
        Object.entries(d.logits).forEach(([k, v]) => { if (!/^\d+$/.test(k)) bad(`${dp}.logits.${k}`, "is not an index"); num(v, `${dp}.logits.${k}`); });
        if (d.model_top !== null) act(d.model_top, `${dp}.model_top`);
        obj(d.actual, `${dp}.actual`);
        if (!["executed", "observed_pass", "unknown"].includes(d.actual.status)) bad(`${dp}.actual.status`, "is not a status");
        if (d.actual.action !== null) act(d.actual.action, `${dp}.actual.action`);
      });
    });
    arr(r.results, `${rp}.results`).forEach((x, xi) => {
      const xp = `${rp}.results[${xi}]`;
      obj(x, xp);
      arr(x.deltas, `${xp}.deltas`, 4).forEach((v, p) => num(v, `${xp}.deltas[${p}]`));
      arr(x.scores_after, `${xp}.scores_after`, 4).forEach((v, p) => num(v, `${xp}.scores_after[${p}]`));
      if (x.final_scores !== undefined) arr(x.final_scores, `${xp}.final_scores`, 4).forEach((v, p) => num(v, `${xp}.final_scores[${p}]`));
      if (x.type === "agari") {
        seat(x.who, `${xp}.who`); seat(x.from, `${xp}.from`); num(x.fu, `${xp}.fu`); num(x.ten, `${xp}.ten`); str(x.limit, `${xp}.limit`);
        arr(x.yaku, `${xp}.yaku`).forEach((y, i) => { arr(y, `${xp}.yaku[${i}]`, 2); str(y[0], `${xp}.yaku[${i}][0]`); num(y[1], `${xp}.yaku[${i}][1]`); });
        arr(x.yakuman, `${xp}.yakuman`).forEach((y, i) => str(y, `${xp}.yakuman[${i}]`));
        tiles(x.hand, `${xp}.hand`); arr(x.melds, `${xp}.melds`).forEach((m, i) => meld(m, `${xp}.melds[${i}]`));
        tile(x.machi, `${xp}.machi`); tiles(x.doras, `${xp}.doras`); tiles(x.ura, `${xp}.ura`);
      } else if (x.type === "ryuukyoku") {
        str(x.reason, `${xp}.reason`);
        if (x.tenpai !== null) arr(x.tenpai, `${xp}.tenpai`).forEach((p, i) => seat(p, `${xp}.tenpai[${i}]`));
        arr(x.revealed, `${xp}.revealed`).forEach((p, i) => seat(p, `${xp}.revealed[${i}]`));
      } else bad(`${xp}.type`, "is not agari / ryuukyoku");
    });
  });
  return game;
}

/* ---- state ---- */
const view = { game: null, round: 0, step: 0, pov: 0, open: true, playing: false };
let timer = null;

function stepsOf(v) { return v.game ? v.game.rounds[v.round].steps : []; }
function clampStep(v, s) { return Math.max(0, Math.min(stepsOf(v).length, s)); }  // steps.length = the result view

function reduce(v, a) {
  const n = Object.assign({}, v);
  switch (a.type) {
    case "load": n.game = a.game; n.round = 0; n.step = 0; n.playing = false; break;
    case "round": n.round = Math.max(0, Math.min(n.game.rounds.length - 1, a.round)); n.step = 0; break;
    case "step": n.step = clampStep(n, a.step); break;
    case "next": {
      if (n.step < stepsOf(n).length) n.step += 1;
      else if (n.round < n.game.rounds.length - 1) { n.round += 1; n.step = 0; }
      else n.playing = false;
      break;
    }
    case "prev": {
      if (n.step > 0) n.step -= 1;
      else if (n.round > 0) { n.round -= 1; n.step = stepsOf(n).length; }
      break;
    }
    case "nextDiscard": {
      const s = stepsOf(n);
      let i = n.step + 1;
      while (i < s.length && s[i].kind !== "discard") i += 1;
      n.step = Math.min(i, s.length);
      break;
    }
    case "prevDiscard": {
      const s = stepsOf(n);
      let i = Math.min(n.step, s.length) - 1;
      while (i > 0 && s[i].kind !== "discard") i -= 1;
      n.step = Math.max(i, 0);
      break;
    }
    case "pov": n.pov = a.pov & 3; break;
    case "open": n.open = !!a.open; break;
    case "play": n.playing = !!a.playing; break;
    default: throw new Error(`unknown action ${a.type}`);
  }
  return n;
}

function dispatch(a) {
  Object.assign(view, reduce(view, a));
  render(view);
}

/* ---- rendering ---- */
let pickerGame = null;  // the game whose rounds the round picker currently lists (rebuilt on any game change)

function render(v) {
  const g = v.game;
  const roundSel = document.getElementById("round-select");
  if (g !== pickerGame) {
    roundSel.replaceChildren();
    if (g) g.rounds.forEach((r, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `${i + 1}: ${roundName(r.index)} ${r.honba}本場`;
      roundSel.appendChild(o);
    });
    pickerGame = g;
  }
  if (!g) { document.getElementById("step-label").textContent = "–"; return; }
  roundSel.value = v.round;
  const steps = stepsOf(v);
  const atResult = v.step >= steps.length;
  const step = steps[Math.min(v.step, steps.length - 1)];
  document.getElementById("step-label").textContent = atResult ? `結果 / ${steps.length}` : `${v.step + 1} / ${steps.length}`;
  document.getElementById("play").classList.toggle("on", v.playing);
  document.getElementById("play").textContent = v.playing ? "⏸ pause" : "▶ play";
  document.getElementById("pov").value = v.pov;
  document.getElementById("open-hands").checked = v.open;
  renderTable(v, g, step, atResult ? g.rounds[v.round].results : null);
  renderPanel(v, g, step, atResult ? g.rounds[v.round].results : null);
}

function renderTable(v, g, step, results) {
  const table = document.getElementById("table");
  const scores = results && results.length ? results[results.length - 1].scores_after : step.scores;
  for (let pos = 0; pos < 4; pos++) {
    const seat = (v.pov + pos) & 3;
    const box = table.querySelector(`.seat.pos-${pos}`);
    box.classList.toggle("pov", pos === 0);
    box.replaceChildren(renderPond(step.ponds[seat]), renderHandRow(v, g, step, seat, pos === 0, results));
  }
  // centre: round, sticks, draws, scores
  const c = document.getElementById("center");
  const info = el("div", "info");
  info.appendChild(el("div", "round", roundName(step.round)));
  info.appendChild(el("div", "", `${step.honba}本場 · 供託 ${step.kyotaku}`));
  info.appendChild(el("div", "", `残り ${step.draws_left}`));
  c.replaceChildren(info);
  for (let pos = 0; pos < 4; pos++) {
    const seat = (v.pov + pos) & 3;
    const s = el("div", `score p${pos}` + (seat === step.oya ? " oya" : ""));
    const wind = WINDS[(seat - step.oya + 4) & 3];
    s.appendChild(el("span", "wind", `${wind}${seat === step.oya ? " 親" : ""}`));
    s.appendChild(el("span", "pts", fmtPts(scores[seat])));
    if (step.reaches[seat]) s.appendChild(el("span", "stick"));
    c.appendChild(s);
  }
  document.getElementById("doras").replaceChildren(...step.doras.map((id) => tileEl(id)));
}

function renderPond(pond) {
  const p = el("div", "pond");
  pond.forEach((e) => {
    // a riichi declaration lies sideways; a called tile stays in the pond, sideways and dimmed (the meld holds it)
    const cls = [e.riichi ? "riichi" : "", e.called_by !== null ? "called" : "", e.riichi || e.called_by !== null ? "side" : "",
      e.tsumogiri ? "tsumogiri" : ""].filter(Boolean).join(" ");
    p.appendChild(tileEl(e.tile, cls));
  });
  return p;
}

function discardDecision(step, seat) {
  return step.decisions.find((d) => d.seat === seat && d.head === "discard") || null;
}

function renderHandRow(v, g, step, seat, isPov, results) {
  const row = el("div", "hand-row");
  const name = el("span", "name" + (step.actor === seat && !results ? " turn" : ""), g.players[seat] || `seat ${seat}`);
  row.appendChild(name);
  const hand = el("div", "hand");
  const open = v.open || isPov || !!results;
  const drawn = step.drawn && step.drawn.seat === seat ? step.drawn.tile : null;
  const dec = open ? discardDecision(step, seat) : null;
  const top = dec && dec.model_top ? dec.model_top.idx : null;
  const actual = dec && dec.actual.action ? dec.actual.action.idx : null;
  const tiles = step.hands[seat].filter((t) => t !== drawn);
  const place = (id, parent) => {
    const h = el("div", "htile");
    h.appendChild(open ? tileEl(id) : tileEl(null));
    if (dec) {
      const slot = slotOf(id);
      const p = dec.probs[String(slot)] || 0;
      const bar = el("div", "bar");
      bar.appendChild(el("i")).style.width = `${Math.round(p * 100)}%`;
      h.appendChild(bar);
      h.title = `${tileName(id)}: ${(p * 100).toFixed(1)}%`;
      if (slot === top) h.classList.add("top");
      if (slot === actual) h.classList.add("actual");
    }
    parent.appendChild(h);
  };
  tiles.forEach((id) => place(id, hand));
  row.appendChild(hand);
  if (drawn !== null) {
    const d = el("div", "drawn");
    place(drawn, d);
    row.appendChild(d);
  }
  const melds = el("div", "melds");
  step.melds[seat].forEach((m) => melds.appendChild(renderMeld(m, seat)));
  row.appendChild(melds);
  return row;
}

function renderMeld(m, seat) {
  const box = el("div", "meld");
  if (m.kind === "ankan") {
    [null, m.tiles[1], m.tiles[2], null].forEach((id) => box.appendChild(tileEl(id)));
    return box;
  }
  // the called tile sideways at the position of the seat it came from: kamicha right, toimen middle, shimocha left
  const rel = (m.from - seat + 4) & 3;  // 3 kamicha, 2 toimen, 1 shimocha
  const others = m.tiles.filter((t) => t !== m.called && t !== m.added);  // a kakan's added tile goes last, sideways
  const order = [];
  if (rel === 3) order.push(...others, m.called);
  else if (rel === 2) order.push(others[0], m.called, ...others.slice(1));
  else order.push(m.called, ...others);
  order.forEach((id) => box.appendChild(tileEl(id, id === m.called ? "side" : "")));
  if (m.kind === "kakan") box.appendChild(tileEl(m.added, "side"));
  return box;
}

function renderPanel(v, g, step, results) {
  const panel = document.getElementById("panel");
  panel.replaceChildren();
  if (results) {
    results.forEach((r) => panel.appendChild(renderResult(r, g)));
    if (!results.length) panel.appendChild(el("div", "hint", "この局の結果はログにありません (BYE)"));
  }
  const title = el("h3", "", results ? "局終了" : `${step.kind} · ${g.players[step.actor]} (seat ${step.actor})`);
  panel.appendChild(title);
  if (!results) {
    if (!step.decisions.length) panel.appendChild(el("div", "hint", "no decision at this point"));
    step.decisions.forEach((d) => panel.appendChild(renderDecision(d, g, step)));
  }
  panel.appendChild(el("div", "kbd", "← → point · ↑ ↓ discard · space play · Home / End"));
}

function matchMark(d) {
  if (!d.model_top || d.actual.status === "unknown") return null;
  const actualIdx = d.actual.action ? d.actual.action.idx : 0;  // observed pass = index 0 on during / post
  return d.model_top.idx === actualIdx;
}

function renderDecision(d, g, step) {
  const box = el("div", "decision");
  const wind = WINDS[(d.seat - step.oya + 4) & 3];
  const who = el("div", "who", `${wind} ${g.players[d.seat]} (seat ${d.seat})`);
  who.appendChild(el("span", "head", d.head));
  box.appendChild(who);
  const rows = d.legal.map((a) => ({ a, p: d.probs[String(a.idx)] || 0, logit: d.logits[String(a.idx)] }))
    .sort((x, y) => y.p - x.p);
  const bars = el("div", "bars");
  const top = d.model_top ? d.model_top.idx : null;
  const actual = d.actual.action ? d.actual.action.idx : (d.actual.status === "observed_pass" ? 0 : null);
  const show = rows.slice(0, 5);
  const other = rows.slice(5).reduce((s, r) => s + r.p, 0);
  const hasModel = Object.keys(d.probs).length > 0;
  show.forEach((r) => {
    const lbl = el("div", "lbl" + (r.a.idx === top ? " model" : "") + (r.a.idx === actual ? " actual" : ""), r.a.name);
    if (r.logit !== undefined) lbl.title = `logit ${r.logit.toFixed(3)}`;
    bars.appendChild(lbl);
    const track = el("div", "track");
    track.appendChild(el("i")).style.width = `${Math.round(r.p * 100)}%`;
    bars.appendChild(track);
    bars.appendChild(el("div", "pct", hasModel ? `${(r.p * 100).toFixed(1)}%` : "–"));
  });
  if (rows.length > 5) {
    bars.appendChild(el("div", "lbl", `other (${rows.length - 5})`));
    const track = el("div", "track");
    track.appendChild(el("i", "other")).style.width = `${Math.round(other * 100)}%`;
    bars.appendChild(track);
    bars.appendChild(el("div", "pct", hasModel ? `${(other * 100).toFixed(1)}%` : "–"));
  }
  box.appendChild(bars);
  const verdict = el("div", "verdict");
  const forced = d.legal.length === 1;
  verdict.appendChild(el("span", "", forced ? "forced: " : "model: "));
  verdict.appendChild(el("span", "model", d.model_top ? d.model_top.name : "(no model)"));
  verdict.appendChild(el("span", "", "· log: "));
  const actualText = d.actual.status === "executed" ? d.actual.action.name
    : d.actual.status === "observed_pass" ? "pass" : "? (pre-empted)";
  verdict.appendChild(el("span", "actual", actualText));
  const m = matchMark(d);
  if (m !== null && !forced) verdict.appendChild(el("span", "mark " + (m ? "ok" : "ng"), m ? "✓" : "✗"));
  box.appendChild(verdict);
  return box;
}

function renderResult(r, g) {
  const box = el("div", "result");
  const names = (seats) => seats.map((p) => g.players[p]).join(", ") || "なし";
  if (r.type === "agari") {
    const how = r.who === r.from ? "ツモ" : `ロン ← ${g.players[r.from]}`;
    box.appendChild(el("div", "title", `${g.players[r.who]} ${how} · ${r.ten.toLocaleString("en-US")}点 ${r.limit || `${r.fu}符`}`));
    const tiles = el("div", "tiles");
    r.hand.forEach((id) => tiles.appendChild(tileEl(id)));
    r.melds.forEach((m) => { tiles.appendChild(el("span", "gap")); m.tiles.forEach((id) => tiles.appendChild(tileEl(id))); });
    box.appendChild(tiles);
    const yaku = r.yakuman.length ? r.yakuman.map((y) => `${y} 役満`) : r.yaku.map(([n, h]) => `${n} ${h}翻`);
    box.appendChild(el("div", "", yaku.join(" · ")));
    const dora = el("div", "tiles");
    dora.appendChild(el("span", "", "ドラ "));
    r.doras.forEach((id) => dora.appendChild(tileEl(id)));
    if (r.ura.length) { dora.appendChild(el("span", "", "　裏 ")); r.ura.forEach((id) => dora.appendChild(tileEl(id))); }
    box.appendChild(dora);
  } else {
    // tenpai is listed only where the XML establishes it (an exhaustive draw); a kyuushu hand is "公開"
    const reason = DRAW_REASON[r.reason] || r.reason;
    const detail = r.tenpai !== null ? ` · 聴牌: ${names(r.tenpai)}` : r.revealed.length ? ` · 公開: ${names(r.revealed)}` : "";
    box.appendChild(el("div", "title", `${reason}${detail}`));
  }
  const t = el("table");
  for (let p = 0; p < 4; p++) {
    const d = r.deltas[p];
    const tr = el("tr");
    tr.appendChild(el("td", "", g.players[p]));
    tr.appendChild(el("td", `delta ${d > 0 ? "pos" : d < 0 ? "neg" : ""}`, `${d > 0 ? "+" : ""}${fmtPts(d)}`));
    tr.appendChild(el("td", "", fmtPts(r.scores_after[p])));
    t.appendChild(tr);
  }
  box.appendChild(t);
  if (r.final_scores) box.appendChild(el("div", "", `終局: ${r.final_scores.map((s, p) => `${g.players[p]} ${fmtPts(s)}`).join(" · ")}`));
  return box;
}

/* ---- loading: one generation counter for API and file loads; a stale completion (or error) is dropped ---- */
function setStatus(msg, err) {
  const s = document.getElementById("status");
  s.textContent = msg || "";
  s.classList.toggle("err", !!err);
}

let loadGen = 0;
let inflight = null;  // the AbortController of the fetch in progress

function beginLoad() {
  if (inflight) { inflight.abort(); inflight = null; }
  return ++loadGen;
}

async function fetchJson(url, signal) {
  const r = await fetch(url, { signal });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || `${r.status}`);
  return body;
}

function showGame(game, note) {
  validateGame(game);
  dispatch({ type: "load", game });
  setStatus(`${game.source}: ${game.rounds.length} rounds${game.complete ? "" : ` (BYE: ${game.rounds_dropped} dropped)`}${note}`);
}

async function loadGamePath(path) {
  if (!path) return;
  const gen = beginLoad();
  const ctl = new AbortController();
  inflight = ctl;
  setStatus(`loading ${path.split("/").pop()} …`);
  try {
    const game = await fetchJson(`/api/game?path=${encodeURIComponent(path)}`, ctl.signal);
    if (gen !== loadGen) return;  // superseded by a newer selection or file
    inflight = null;
    showGame(game, "");
  } catch (e) {
    if (gen !== loadGen) return;
    inflight = null;
    setStatus(`error: ${e.message}`, true);
  }
}

function loadJsonFile(file) {
  const gen = beginLoad();
  const reader = new FileReader();
  reader.onload = () => {
    if (gen !== loadGen) return;
    try {
      const game = JSON.parse(reader.result);
      showGame(game, " (file)");
    } catch (e) {
      setStatus(`error: ${e.message}`, true);
    }
  };
  reader.readAsText(file);
}

function fillSelect(sel, games) {
  sel.replaceChildren(el("option", "", `(${games.length} games)`));
  sel.options[0].value = "";
  games.forEach((g) => {
    const o = document.createElement("option");
    o.value = g.path;
    o.textContent = g.name;
    sel.appendChild(o);
  });
}

async function listGames() {
  const date = document.getElementById("date").value.trim();
  try {
    const body = await fetchJson(`/api/games?date=${encodeURIComponent(date)}`);
    fillSelect(document.getElementById("game-select"), body.games);
    setStatus(`${body.games.length} games on ${date}`);
  } catch (e) {
    setStatus(`error: ${e.message}`, true);
  }
}

function setPlaying(on) {
  dispatch({ type: "play", playing: on });
  if (timer) { clearInterval(timer); timer = null; }
  if (on) {
    timer = setInterval(() => {
      if (!view.playing) { clearInterval(timer); timer = null; return; }
      dispatch({ type: "next" });
      if (!view.playing) { clearInterval(timer); timer = null; render(view); }
    }, Number(document.getElementById("speed").value));
  }
}

function bind() {
  const $ = (id) => document.getElementById(id);
  $("list-games").onclick = listGames;
  $("date").onkeydown = (e) => { if (e.key === "Enter") listGames(); };
  $("game-select").onchange = (e) => loadGamePath(e.target.value);
  $("fixture-select").onchange = (e) => loadGamePath(e.target.value);
  $("json-file").onchange = (e) => { if (e.target.files[0]) loadJsonFile(e.target.files[0]); };
  $("round-select").onchange = (e) => dispatch({ type: "round", round: Number(e.target.value) });
  $("first").onclick = () => dispatch({ type: "step", step: 0 });
  $("last").onclick = () => dispatch({ type: "step", step: Infinity });
  $("prev").onclick = () => dispatch({ type: "prev" });
  $("next").onclick = () => dispatch({ type: "next" });
  $("prev-discard").onclick = () => dispatch({ type: "prevDiscard" });
  $("next-discard").onclick = () => dispatch({ type: "nextDiscard" });
  $("play").onclick = () => setPlaying(!view.playing);
  $("speed").onchange = () => { if (view.playing) setPlaying(true); };
  $("pov").onchange = (e) => dispatch({ type: "pov", pov: Number(e.target.value) });
  $("open-hands").onchange = (e) => dispatch({ type: "open", open: e.target.checked });
  document.addEventListener("keydown", (e) => {
    if (!view.game || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const map = { ArrowRight: "next", ArrowLeft: "prev", ArrowDown: "nextDiscard", ArrowUp: "prevDiscard" };
    if (map[e.key]) { e.preventDefault(); dispatch({ type: map[e.key] }); }
    else if (e.key === " ") { e.preventDefault(); setPlaying(!view.playing); }
    else if (e.key === "Home") dispatch({ type: "step", step: 0 });
    else if (e.key === "End") dispatch({ type: "step", step: Infinity });
  });
  fetchJson("/api/fixtures").then((b) => fillSelect($("fixture-select"), b.games)).catch(() => {});
}

if (typeof window !== "undefined" && typeof document !== "undefined" && document.getElementById("table")) {
  bind();
  render(view);
  window.viewerState = () => ({ source: view.game ? view.game.source : null, round: view.round, step: view.step, loadGen });
}
if (typeof module !== "undefined") module.exports = { reduce, tileFile, tileName, slotOf, matchMark, validateGame };
