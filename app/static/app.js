// ChessDorrie frontend.
//
// Communicates with the FastAPI backend at /api/*. Renders the board
// via chessground (Lichess's lightweight chess board lib) with arrows
// for the engine's top picks and the troll's top picks. Click a row
// in the candidate table to play that move and re-analyse.

import { Chessground } from "/static/vendor/chessground.min.js";

const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const state = {
  fen: STARTING_FEN,
  elo: 1500,
  style: "balanced",
  orientation: "white",
  lastResult: null,
  busy: false,
};

// --- chessground initialisation --------------------------------------

const boardEl = document.getElementById("board");
const cg = Chessground(boardEl, {
  fen: state.fen,
  orientation: state.orientation,
  turnColor: "white",
  draggable: { enabled: true, showGhost: true },
  movable: {
    free: false,                 // strict legal-only mode
    color: "both",
    showDests: true,
    dests: new Map(),            // populated by refreshLegalMoves()
    events: {
      after: (orig, dest, _metadata) => handleUserMove(orig, dest),
    },
  },
  drawable: { enabled: true, defaultSnapToValidMove: true },
  highlight: { lastMove: true, check: true },
});

window.addEventListener("resize", () => cg.redrawAll());

// Fetch legal destinations for the current state.fen and apply to the board.
async function refreshLegalMoves() {
  try {
    const res = await fetch("/api/legal-moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: state.fen }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const dests = new Map();
    for (const [from, to] of Object.entries(data.dests || {})) {
      dests.set(from, to);
    }
    cg.set({
      fen: data.fen,
      turnColor: data.turn,
      check: data.in_check,
      movable: { dests },
    });
    if (data.is_game_over) {
      setStatus(data.is_checkmate ? "Checkmate." : (data.is_stalemate ? "Stalemate." : "Game over."));
    }
  } catch (e) {
    /* ignore — analyse() will surface errors */
  }
}

// --- helpers ---------------------------------------------------------

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function fmtCp(cp) {
  if (cp === null || cp === undefined) return "—";
  // Mate scores live near ±100000 in the backend.
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

// Map a cp value to a 0..100 "bar fill" using a logistic squash so the
// bar fills smoothly between -8 and +8 pawns and clips beyond.
function cpToBarPct(cp) {
  if (Math.abs(cp) > 50000) return cp > 0 ? 100 : 0;
  const x = cp / 400; // 4 pawns -> ~73%, 8 pawns -> ~88%
  return Math.round(100 / (1 + Math.exp(-x)));
}

function uciToSquares(uci) {
  if (!uci || uci.length < 4) return null;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

// --- analysis flow ---------------------------------------------------

async function analyse() {
  if (state.busy) return;
  state.busy = true;
  setStatus("Thinking… Stockfish is plotting, Maia is sneering.");
  try {
    const res = await fetch("/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: state.fen, elo: state.elo, style: state.style }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus("Analyse error: " + (err.detail || res.status));
      return;
    }
    const data = await res.json();
    state.lastResult = data;
    renderResult(data);
  } catch (e) {
    setStatus("Network error: " + e.message);
  } finally {
    state.busy = false;
  }
}

function renderResult(r) {
  document.getElementById("elapsed").textContent = `${r.elapsed_ms} ms`;
  document.getElementById("elo-info").textContent = `Elo modelled: ${r.elo_assumed}`;

  const top = r.candidates[0] || null;
  if (!top) {
    setStatus("No candidates — terminal position?");
    document.querySelector("#candidates tbody").innerHTML = "";
    return;
  }

  // Update meters using the top troll candidate.
  setMeter("obj", top.objective_eval, fmtCp(top.objective_eval));
  setMeter("troll", top.expected_eval, fmtCp(top.expected_eval));
  setAngerMeter("anger", top.anger_probability);
  setHumanMeter("human", top.human_factor);

  // Update candidate table.
  renderCandidates(r.candidates, r.objective_best_uci, r.troll_best_uci);

  // Draw arrows on the board.
  drawArrows(r);

  // Auto-show replies for the top troll move.
  renderReplies(top);

  setStatus(
    `${r.side_to_move} to move · Stockfish: ${top.move_san === sanFromUci(r.candidates, r.objective_best_uci) ? "agrees" : "best=" + sanFromUci(r.candidates, r.objective_best_uci)} · Troll: ${top.move_san}`
  );
}

function sanFromUci(candidates, uci) {
  const c = candidates.find(c => c.move_uci === uci);
  return c ? c.move_san : uci;
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
  // Human factor is signed cp gain from exploiting humanness.
  // Map 0 → 50%, +400 → 100%, -400 → 0%.
  const pct = Math.max(0, Math.min(100, 50 + cp / 8));
  document.getElementById(`m-${name}`).style.width = pct + "%";
  document.getElementById(`m-${name}-val`).textContent = fmtCp(cp);
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
    const empCell = empN >= 30
      ? `<span class="${empWR > 0.5 ? 'good' : (empWR < 0.5 ? 'bad' : '')}">${(empWR*100).toFixed(0)}% <span class="hint">(${empN})</span></span>`
      : (empN > 0 ? `<span class="hint">${empN}</span>` : '—');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="move">${c.move_san}</td>
      <td>${c.objective_rank}</td>
      <td>${fmtCp(c.objective_eval)}</td>
      <td>${fmtCp(c.expected_eval)}</td>
      <td class="${c.expected_material >= 0 ? 'good' : 'bad'}">${(c.expected_material / 100).toFixed(2)}</td>
      <td>${fmtPct(c.anger_probability)}</td>
      <td class="sac ${c.is_sacrifice ? 'yes' : ''}">${c.is_sacrifice ? '⚡' + (c.sacrifice_value/100).toFixed(1) : ''}</td>
      <td>${empCell}</td>
      <td class="notes">${(c.notes || []).join('; ')}</td>
    `;
    tr.addEventListener("click", () => {
      // Single click → show replies; double click → play the move.
      renderReplies(c);
    });
    tr.addEventListener("dblclick", () => playMove(c.move_uci));
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
        ${r.move_san} ${r.is_best_reply ? '<span style="color:var(--good)">★</span>' : ''}
        <div class="reply-bar"><div style="width:${Math.round(r.probability*100)}%"></div></div>
      </div>
      <span>${fmtPct(r.probability)}</span>
      <span>${fmtCp(r.eval_after)}</span>
    `;
    ul.appendChild(li);
  }
}

function drawArrows(r) {
  const shapes = [];
  // Objective best — blue
  const obj = uciToSquares(r.objective_best_uci);
  if (obj) shapes.push({ orig: obj[0], dest: obj[1], brush: "blue" });
  // Troll best — orange (or the chessground default 'red' which we'll restyle)
  if (r.troll_best_uci !== r.objective_best_uci) {
    const troll = uciToSquares(r.troll_best_uci);
    if (troll) shapes.push({ orig: troll[0], dest: troll[1], brush: "red" });
  }
  // Other candidates — faint
  for (const c of r.candidates.slice(0, 4)) {
    if (c.move_uci === r.objective_best_uci) continue;
    if (c.move_uci === r.troll_best_uci) continue;
    const sq = uciToSquares(c.move_uci);
    if (sq) shapes.push({ orig: sq[0], dest: sq[1], brush: "yellow" });
  }
  cg.setAutoShapes(shapes);
}

// --- move handling ----------------------------------------------------

async function playMove(uciOrSan) {
  setStatus(`Playing ${uciOrSan}…`);
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: state.fen, move: uciOrSan }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus("Move error: " + (err.detail || res.status));
      return;
    }
    const data = await res.json();
    state.fen = data.fen;
    document.getElementById("fen-input").value = state.fen;
    await refreshLegalMoves();
    if (!data.is_game_over) {
      await analyse();
    }
  } catch (e) {
    setStatus("Network error: " + e.message);
  }
}

async function handleUserMove(orig, dest) {
  // chessground only fires `movable.events.after` for legal moves now,
  // so we can submit straight away. Underpromotion currently defaults
  // to queen (chessground's `promotion` hook is future work).
  await playMove(orig + dest + (isPromotion(orig, dest) ? "q" : ""));
}

function isPromotion(orig, dest) {
  // A move ending on rank 1 or 8 by a pawn is a promotion. We don't
  // know the moving piece here (chessground already moved it), so we
  // approximate: only pawns ever reach rank 1 or 8, and the backend
  // rejects spurious "q" suffixes on non-promotion moves.
  return (dest[1] === "8" || dest[1] === "1");
}

// --- buttons ---------------------------------------------------------

document.getElementById("load-btn").addEventListener("click", async () => {
  const v = document.getElementById("fen-input").value.trim();
  try {
    state.fen = v;
    cg.set({ fen: state.fen });
    await refreshLegalMoves();
    setStatus("Loaded.");
  } catch (e) {
    setStatus("Invalid FEN.");
  }
});

document.getElementById("analyse-btn").addEventListener("click", async () => {
  state.fen = document.getElementById("fen-input").value.trim() || state.fen;
  state.elo = parseInt(document.getElementById("elo-select").value, 10);
  state.style = document.getElementById("style-select").value;
  cg.set({ fen: state.fen });
  await refreshLegalMoves();
  await analyse();
});

document.getElementById("flip-btn").addEventListener("click", () => {
  state.orientation = state.orientation === "white" ? "black" : "white";
  cg.set({ orientation: state.orientation });
});

document.getElementById("elo-select").addEventListener("change", (e) => {
  state.elo = parseInt(e.target.value, 10);
});

document.getElementById("style-select").addEventListener("change", (e) => {
  state.style = e.target.value;
  if (state.lastResult) analyse();  // re-rank with the new style
});

// Auto-analyse on load (and prime the legal-moves map first so drag-drop
// doesn't go through a "no dests" window).
(async () => {
  await refreshLegalMoves();
  await analyse();
})();
