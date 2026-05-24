// ChessDorrie frontend (WebSocket / ponder version).
//
// Connects to /ws/session/{id} on load. The backend continuously
// re-thinks the current position and pushes:
//   * {type:"session", session_id}           — remember for reconnects
//   * {type:"position", fen, dests, ...}      — on every board change
//   * {type:"snapshot", result, metrics}      — ~2 Hz updates
//   * {type:"metrics", metrics}               — engine-only ticks
//   * {type:"ping"}                            — heartbeat
//
// Client sends:
//   * {type:"move", uci}    — when the user drags a piece
//   * {type:"side", color}  — when the user picks ⚪ / ⚫ / 👁
//   * {type:"style", style} — opponent psychology toggle
//   * {type:"elo", n}
//   * {type:"reset", fen}   — load a new position
//   * {type:"pong"}

import { Chessground } from "/static/vendor/chessground.min.js";

const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const state = {
  fen: STARTING_FEN,
  elo: 1500,
  style: "balanced",
  mode: "lite",            // "lite" | "hybrid" | "deep"
  playstyle: "direct",     // "direct" | "setup" | "setup_deep"
  autoplay: false,
  autostyle: false,
  side: null,              // "white" | "black" | null (observer)
  orientation: "white",
  sessionId: localStorage.getItem("chessdorrie_session") || "new",
  ws: null,
  reconnectDelay: 1000,    // exponential backoff cap
  pendingDests: new Map(),
  lastSnapshot: null,
};

// --- chessground ---------------------------------------------------- //

const boardEl = document.getElementById("board");
const cg = Chessground(boardEl, {
  fen: state.fen,
  orientation: state.orientation,
  turnColor: "white",
  draggable: { enabled: true, showGhost: true },
  movable: {
    free: false,
    color: "both",
    showDests: true,
    dests: new Map(),
    events: {
      after: (orig, dest, _metadata) => sendMove(orig + dest + maybePromotion(dest)),
    },
  },
  drawable: { enabled: true, defaultSnapToValidMove: true },
  highlight: { lastMove: true, check: true },
});

window.addEventListener("resize", () => cg.redrawAll());

// --- WebSocket lifecycle ------------------------------------------- //

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/session/${state.sessionId}`;
}

function connect() {
  const url = wsUrl();
  setStatus(`Connecting to ${url}…`);
  const ws = new WebSocket(url);
  state.ws = ws;

  ws.onopen = () => {
    setStatus("Connected. Engine pondering…");
    state.reconnectDelay = 1000;
    // After connect, push our current preferences so the session
    // syncs to them.
    sendJson({ type: "style", style: state.style });
    sendJson({ type: "elo", elo: state.elo });
    sendJson({ type: "side", color: state.side });
    sendJson({ type: "mode", mode: state.mode });
    sendJson({ type: "playstyle", playstyle: state.playstyle });
    sendJson({ type: "autoplay", on: state.autoplay });
    sendJson({ type: "autostyle", on: state.autostyle });
  };

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handleMessage(msg);
  };

  ws.onclose = () => {
    setStatus("Disconnected — reconnecting…");
    state.ws = null;
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
  };

  ws.onerror = (e) => {
    console.error("ws error", e);
  };
}

function sendJson(payload) {
  if (state.ws && state.ws.readyState === 1) {
    state.ws.send(JSON.stringify(payload));
  }
}

function sendMove(uci) {
  sendJson({ type: "move", uci });
}

// --- incoming message handlers ------------------------------------- //

function handleMessage(msg) {
  switch (msg.type) {
    case "session":
      state.sessionId = msg.session_id;
      localStorage.setItem("chessdorrie_session", msg.session_id);
      break;
    case "position":
      applyPosition(msg);
      break;
    case "snapshot":
      applySnapshot(msg);
      break;
    case "metrics":
      applyMetrics(msg.metrics);
      break;
    case "ping":
      sendJson({ type: "pong" });
      break;
    case "error":
      setStatus("error: " + msg.msg);
      break;
  }
}

function applyPosition(p) {
  state.fen = p.fen;
  document.getElementById("fen-input").value = p.fen;
  const dests = new Map();
  for (const [from, to] of Object.entries(p.dests || {})) dests.set(from, to);
  state.pendingDests = dests;
  const movableColor = sideForMovable(p.bot_side, p.turn);
  cg.set({
    fen: p.fen,
    turnColor: p.turn,
    check: p.in_check,
    movable: { dests, color: movableColor },
    lastMove: p.last_move_uci ? [p.last_move_uci.slice(0, 2), p.last_move_uci.slice(2, 4)] : undefined,
  });
  if (p.is_game_over) {
    setStatus(p.is_checkmate ? "Checkmate." : (p.is_stalemate ? "Stalemate." : "Game over."));
  }
  renderOppDetected(p.opp_detected, p.style);
  // Sync server-applied autostyle change
  if (p.style && p.style !== state.style) {
    state.style = p.style;
    const sel = document.getElementById("style-select");
    if (sel) sel.value = p.style;
  }
}

function renderOppDetected(opp, currentStyle) {
  const row = document.getElementById("opp-detected-row");
  if (!opp || opp.observed_moves < 1) {
    row.style.display = "none";
    return;
  }
  row.style.display = "";
  const emoji = opp.detected_style === "greedy" ? "😋"
              : opp.detected_style === "cautious" ? "🪖"
              : "⚖";
  const label = document.getElementById("opp-detected");
  label.innerHTML = `${emoji} ${opp.detected_style} ` +
    `<span class="hint">(${opp.observed_moves} moves, ` +
    `${Math.round(opp.confidence * 100)}% conf)</span>`;
  const applyBtn = document.getElementById("opp-apply-btn");
  applyBtn.style.display = (opp.detected_style !== currentStyle && opp.confidence > 0.3) ? "" : "none";
  applyBtn.onclick = () => {
    state.style = opp.detected_style;
    document.getElementById("style-select").value = opp.detected_style;
    sendJson({ type: "style", style: opp.detected_style });
  };
}

function sideForMovable(bot_side, turn) {
  // bot_side = "white" → bot plays for white → user is black → user can move only on black's turn
  if (!bot_side) return "both";
  const userColor = bot_side === "white" ? "black" : "white";
  // Allow dragging only if it's the user's turn. chessground accepts "white" | "black" | "both".
  // If it's not the user's turn, no pieces are draggable.
  return turn === userColor ? userColor : userColor;
}

function applySnapshot(msg) {
  state.lastSnapshot = msg;
  const r = msg.result;
  const m = msg.result.metrics || msg.metrics || {};

  applyMetrics(m);

  const top = r.candidates[0] || null;
  if (!top) {
    setStatus("No candidates — terminal position?");
    document.querySelector("#candidates tbody").innerHTML = "";
    return;
  }

  setMeter("obj", top.objective_eval, fmtCp(top.objective_eval));
  setMeter("troll", top.expected_eval, fmtCp(top.expected_eval));
  setAngerMeter("anger", top.anger_probability);
  setHumanMeter("human", top.human_factor);

  renderCandidates(r.candidates, r.objective_best_uci, r.troll_best_uci);
  drawArrows(r);
  renderReplies(top);

  const objSan = sanFromUci(r.candidates, r.objective_best_uci);
  const trollSan = top.move_san;
  setStatus(
    `${r.side_to_move} to move · SF: ${objSan} · Troll: ${trollSan}` +
    (msg.cache_hit ? " (cache hit)" : "")
  );
}

function applyMetrics(m) {
  if (!m) return;
  if (typeof m.depth === "number") setEngineField("depth", `${m.depth}/${m.seldepth || m.depth}`);
  if (typeof m.nps === "number") setEngineField("nps", fmtNumber(m.nps));
  if (typeof m.nodes === "number") setEngineField("nodes", fmtNumber(m.nodes));
  if (typeof m.elapsed_ms === "number") setEngineField("elapsed", `${(m.elapsed_ms/1000).toFixed(1)}s`);
  if (m.gpu_util !== null && m.gpu_util !== undefined) {
    setEngineField("gpu", `${m.gpu_util.toFixed(0)}%`);
    const bar = document.getElementById("gpu-bar");
    if (bar) bar.style.width = `${Math.min(100, m.gpu_util)}%`;
  } else {
    setEngineField("gpu", "—");
  }
  if (m.vram_mb !== null && m.vram_mb !== undefined) {
    setEngineField("vram", `${(m.vram_mb/1024).toFixed(1)}G`);
  }
  // Depth bar fill (target=24)
  const dbar = document.getElementById("depth-bar");
  if (dbar && typeof m.depth === "number") {
    dbar.style.width = `${Math.min(100, m.depth / 24 * 100)}%`;
  }
}

// --- rendering helpers --------------------------------------------- //

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function setEngineField(key, value) {
  const el = document.getElementById(`engine-${key}`);
  if (el) el.textContent = value;
}

function fmtCp(cp) {
  if (cp === null || cp === undefined) return "—";
  if (Math.abs(cp) > 50000) {
    const m = 100000 - Math.abs(cp);
    return (cp > 0 ? "#" : "-#") + m;
  }
  return (cp >= 0 ? "+" : "") + (cp / 100).toFixed(2);
}

function fmtPct(p) {
  if (p === null || p === undefined) return "—";
  return (p * 100).toFixed(0) + "%";
}

function fmtNumber(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

function cpToBarPct(cp) {
  if (Math.abs(cp) > 50000) return cp > 0 ? 100 : 0;
  const x = cp / 400;
  return Math.round(100 / (1 + Math.exp(-x)));
}

function uciToSquares(uci) {
  if (!uci || uci.length < 4) return null;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

function setMeter(name, cp, label) {
  document.getElementById(`m-${name}`).style.width = cpToBarPct(cp) + "%";
  document.getElementById(`m-${name}-val`).textContent = label;
}

function setAngerMeter(name, prob) {
  document.getElementById(`m-${name}`).style.width = Math.round(prob * 100) + "%";
  document.getElementById(`m-${name}-val`).textContent = fmtPct(prob);
}

function setHumanMeter(name, cp) {
  const pct = Math.max(0, Math.min(100, 50 + cp / 8));
  document.getElementById(`m-${name}`).style.width = pct + "%";
  document.getElementById(`m-${name}-val`).textContent = fmtCp(cp);
}

function sanFromUci(candidates, uci) {
  const c = candidates.find((c) => c.move_uci === uci);
  return c ? c.move_san : uci;
}

function renderCandidates(cands, objBest, trollBest) {
  const tbody = document.querySelector("#candidates tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < cands.length; i++) {
    const c = cands[i];
    const tr = document.createElement("tr");
    if (c.move_uci === trollBest) tr.classList.add("top-troll");
    tr.dataset.uci = c.move_uci;
    const empN = c.empirical_total || 0;
    const empWR = c.empirical_win_rate || 0;
    const empCell =
      empN >= 30
        ? `<span class="${empWR > 0.5 ? "good" : empWR < 0.5 ? "bad" : ""}">${(empWR * 100).toFixed(0)}% <span class="hint">(${empN})</span></span>`
        : empN > 0
        ? `<span class="hint">${empN}</span>`
        : "—";
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="move">${c.move_san}</td>
      <td>${c.objective_rank}</td>
      <td>${fmtCp(c.objective_eval)}</td>
      <td>${fmtCp(c.expected_eval)}</td>
      <td class="${c.expected_material >= 0 ? "good" : "bad"}">${(c.expected_material / 100).toFixed(2)}</td>
      <td>${fmtPct(c.anger_probability)}</td>
      <td class="sac ${c.is_sacrifice ? "yes" : ""}">${c.is_sacrifice ? "⚡" + (c.sacrifice_value / 100).toFixed(1) : ""}</td>
      <td class="${c.trap_potential_cp > 0 ? 'good' : ''}">${c.trap_potential_cp > 0 ? "🪤" + (c.trap_potential_cp / 100).toFixed(1) : ""}</td>
      <td>${empCell}</td>
      <td class="notes">${(c.notes || []).join("; ")}</td>
    `;
    tr.addEventListener("click", () => renderReplies(c));
    tr.addEventListener("dblclick", () => sendMove(c.move_uci));
    tbody.appendChild(tr);
  }
}

function renderReplies(c) {
  const panel = document.getElementById("reply-panel");
  document.getElementById("reply-move-name").textContent = `after ${c.move_san}`;
  const ul = document.getElementById("reply-list");
  ul.innerHTML = "";
  if (!c.replies || c.replies.length === 0) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "";
  for (const r of c.replies) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div>
        ${r.move_san} ${r.is_best_reply ? '<span style="color:var(--good)">★</span>' : ""}
        <div class="reply-bar"><div style="width:${Math.round(r.probability * 100)}%"></div></div>
      </div>
      <span>${fmtPct(r.probability)}</span>
      <span>${fmtCp(r.eval_after)}</span>
    `;
    ul.appendChild(li);
  }
}

function drawArrows(r) {
  const shapes = [];
  const obj = uciToSquares(r.objective_best_uci);
  if (obj) shapes.push({ orig: obj[0], dest: obj[1], brush: "blue" });
  if (r.troll_best_uci !== r.objective_best_uci) {
    const troll = uciToSquares(r.troll_best_uci);
    if (troll) shapes.push({ orig: troll[0], dest: troll[1], brush: "red" });
  }
  for (const c of r.candidates.slice(0, 4)) {
    if (c.move_uci === r.objective_best_uci) continue;
    if (c.move_uci === r.troll_best_uci) continue;
    const sq = uciToSquares(c.move_uci);
    if (sq) shapes.push({ orig: sq[0], dest: sq[1], brush: "yellow" });
  }
  cg.setAutoShapes(shapes);
}

function maybePromotion(dest) {
  return dest[1] === "8" || dest[1] === "1" ? "q" : "";
}

// --- UI controls --------------------------------------------------- //

document.getElementById("load-btn").addEventListener("click", () => {
  const v = document.getElementById("fen-input").value.trim();
  sendJson({ type: "reset", fen: v });
});

document.getElementById("flip-btn").addEventListener("click", () => {
  state.orientation = state.orientation === "white" ? "black" : "white";
  cg.set({ orientation: state.orientation });
});

document.getElementById("elo-select").addEventListener("change", (e) => {
  state.elo = parseInt(e.target.value, 10);
  sendJson({ type: "elo", elo: state.elo });
});

document.getElementById("style-select").addEventListener("change", (e) => {
  state.style = e.target.value;
  sendJson({ type: "style", style: state.style });
});

document.getElementById("mode-select").addEventListener("change", (e) => {
  state.mode = e.target.value;
  setStatus(`Precision: ${state.mode} — re-preparing position…`);
  sendJson({ type: "mode", mode: state.mode });
});

document.getElementById("playstyle-select").addEventListener("change", (e) => {
  state.playstyle = e.target.value;
  setStatus(`Playstyle: ${state.playstyle}`);
  sendJson({ type: "playstyle", playstyle: state.playstyle });
});

document.getElementById("autoplay-toggle").addEventListener("change", (e) => {
  state.autoplay = e.target.checked;
  sendJson({ type: "autoplay", on: state.autoplay });
});

document.getElementById("autostyle-toggle").addEventListener("change", (e) => {
  state.autostyle = e.target.checked;
  sendJson({ type: "autostyle", on: state.autostyle });
});

// Side selector — radio group; "" means observer.
for (const radio of document.querySelectorAll('input[name="side"]')) {
  radio.addEventListener("change", (e) => {
    const v = e.target.value;
    state.side = v || null;
    state.orientation = v === "black" ? "black" : "white";
    cg.set({ orientation: state.orientation });
    sendJson({ type: "side", color: state.side });
  });
}

// --- boot ---------------------------------------------------------- //

connect();
