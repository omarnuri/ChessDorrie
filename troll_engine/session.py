"""Per-user game session with continuous ponder.

One `GameSession` owns:

  * the current `chess.Board` (the analyser's ground-truth position)
  * the user's selected side (white / black / observer)
  * the active style (greedy / balanced / cautious) and Elo
  * a long-lived Stockfish `Engine`, the human-move model, the
    Lichess Explorer client, the trap DB
  * a worker thread that continuously streams analysis on the current
    position, debounced ~2 Hz
  * a set of subscribed WebSocket clients
  * an LRU cache of recent `AnalysisResult`s keyed by
    `(fen, style, elo, bot_side)` so revisiting a position is instant

When the user changes the board (via `submit_move`, `reset_fen`, etc.)
the worker thread cancels its current streaming analysis, restarts on
the new position, and uses the cache to broadcast a first-paint
snapshot while it re-deepens.

Threading model
---------------

The worker is a plain `threading.Thread` (not asyncio). It owns the
Stockfish process; nothing else touches the engine. Updates to
WebSocket subscribers are scheduled onto the FastAPI event loop via
`loop.call_soon_threadsafe(asyncio.create_task, ...)`.

Inter-thread state is protected by `_state_lock` (Lock). The worker
takes a snapshot of (board, style, elo, bot_side) under the lock at
the start of each iteration; mutations from the API side push into
the next iteration on the next loop turn.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import chess

from .engine import Engine, material_balance, PIECE_CP, MATE_SCORE, _pov_score_to_cp
from .engine_pool import EnginePool, make_pool, DEFAULT_POOL_SIZE
from .human_model import HumanModel, get_human_model, PredictedMove
from .lc0_engine import get_engine
from .lichess_explorer import LichessExplorer
from .opponent_tracker import OpponentTracker
from .search import TrollSearch
from .trap_db import TrapDB, default_db, trap_score_for_move
from .types import AnalysisResult, Candidate, Reply, EngineMetrics
from .utility import TrollFeatures, troll_utility


log = logging.getLogger("troll_engine.session")


# Cap on the LRU per session — generous enough to cover a full game,
# tight enough to bound memory.
CACHE_LIMIT = 256

# Streaming-analysis caps. Set high; the worker preempts on board change
# so the engine rarely actually reaches these.
DEFAULT_STREAM_MAX_DEPTH = int(os.environ.get("CD_STREAM_MAX_DEPTH", "32"))
DEFAULT_MULTIPV = 8

# Broadcast cadence (seconds).
BROADCAST_INTERVAL = 0.5

# Idle timeout for an unused session (no subscribers).
IDLE_TIMEOUT_S = 60.0


@dataclass
class PreparedPosition:
    """Cheap per-position precompute used by the streaming hot path.

    Computed once when a new position arrives; consulted on every
    broadcast tick without touching the engine.

    `sub_evals` is populated only in `hybrid`/`deep` analysis modes —
    it maps `(candidate_uci, reply_uci) → cp_from_bot_pov` for the
    top-K candidates × top-N predicted replies. Built during prep
    while the engine is still free; the streaming phase reuses these
    values for every snapshot.
    """

    fen: str
    predicted_replies: dict[str, list[PredictedMove]]  # candidate_uci → top replies
    explorer_total: int
    explorer_by_uci: dict[str, dict]  # uci → {wins, draws, total, win_rate}
    sub_evals: dict[tuple[str, str], float] = field(default_factory=dict)
    # Setup-mode: per-candidate cp value of the largest sacrifice opportunity
    # that becomes available 2 plies in the future (after our move + opp's
    # likely reply).
    trap_potentials: dict[str, float] = field(default_factory=dict)


# Analysis-precision modes.
#
# These are CPU defaults — pleasant on a 4-core sandbox. On a beefier
# machine (Colab A100/H100 with Lc0+CUDA serving Maia) the engine can
# afford much deeper sub-evals. Override via env vars:
#
#   CD_PREP_SHALLOW_DEPTH=16     (default 12)
#   CD_HYBRID_DEPTH=12           (default 10)
#   CD_DEEP_DEPTH=18             (default 14)
#   CD_SETUP_DEEP_DEPTH=18       (default 16)
#   CD_STREAM_MAX_DEPTH=40       (default 32)
#
#   "lite"   — no per-reply sub-evals (expected_eval = objective_eval).
#              Fastest; pure trap-DB + Explorer + sacrifice-detection signal.
#   "hybrid" — sub-evals for the top 5 candidates × top 3 replies.
#              ~1-2 s prep cost on CPU; brings back the human_factor and
#              anger signals for the moves that matter most.
#   "deep"   — sub-evals for the top 8 candidates × top 5 replies, deeper.
#              ~5-10 s prep cost on CPU; high-fidelity troll utility.
ANALYSIS_MODES = ("lite", "hybrid", "deep")


@dataclass
class _ModeConfig:
    candidates: int     # top-K candidates we sub-evaluate
    replies: int        # top-N replies per candidate
    depth: int          # sub-eval depth


_HYBRID_DEPTH = int(os.environ.get("CD_HYBRID_DEPTH", "10"))
_DEEP_DEPTH = int(os.environ.get("CD_DEEP_DEPTH", "14"))
_PREP_SHALLOW_DEPTH = int(os.environ.get("CD_PREP_SHALLOW_DEPTH", "12"))
_SETUP_DEEP_DEPTH = int(os.environ.get("CD_SETUP_DEEP_DEPTH", "16"))

_MODE_CONFIGS: dict[str, _ModeConfig] = {
    "lite":   _ModeConfig(0, 0, 0),
    "hybrid": _ModeConfig(5, 3, _HYBRID_DEPTH),
    "deep":   _ModeConfig(8, 5, _DEEP_DEPTH),
}


CacheKey = tuple[str, str, int, "int | None", str, str]  # (fen_key, style, elo, bot_side, mode, playstyle)


def _fen_key(fen: str) -> str:
    """Strip halfmove clock + en-passant for cache matching."""
    parts = fen.split()
    return " ".join(parts[:3]) if len(parts) >= 3 else fen


class GameSession:
    """Persistent ponder session for one logical user/game."""

    def __init__(
        self,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
        *,
        elo: int = 1500,
        style: str = "balanced",
        bot_side: chess.Color | None = None,
        analysis_mode: str = "lite",
        playstyle: str = "direct",
        autoplay: bool = False,
        weights_dir: str = "weights",
        explorer: LichessExplorer | None = None,
        trap_db: TrapDB | None = None,
        engine_threads: int = 2,
        engine_type: str = "auto",
        use_explorer: bool = True,
        trap_model_path: str | None = None,
    ) -> None:
        self.session_id = session_id
        self._loop = loop
        self.board = chess.Board()
        self.elo = elo
        self.style = style
        self.bot_side = bot_side  # None = observer (think for side-to-move)
        self.analysis_mode = analysis_mode if analysis_mode in ANALYSIS_MODES else "lite"
        self.playstyle = playstyle if playstyle in ("direct", "setup", "setup_deep") else "direct"
        self.autoplay = bool(autoplay)
        self.weights_dir = weights_dir

        self._engine = get_engine(
            engine_type=engine_type, threads=engine_threads,
        )
        self.engine_type = (
            "lc0" if self._engine.__class__.__name__ == "Lc0Engine" else "stockfish"
        )
        self.engine_limit_kind = self._engine.limit_kind  # "depth" or "nodes"

        # Parallel helper pool for sub-evals. Lazily built on first use
        # because spawning N engines takes seconds — we don't want to
        # block session creation. Set CD_POOL_SIZE=0 to disable.
        self._pool: EnginePool | None = None
        self._pool_lock = threading.Lock()
        self._engine_type_for_pool = engine_type
        # Pool workers run lighter than the main engine: 1 thread each
        # so a pool-of-4 fits on a 4-core sandbox without contention,
        # and a pool-of-8 fits on a 12-core Colab. Override via
        # CD_POOL_THREADS.
        self._pool_threads = int(os.environ.get("CD_POOL_THREADS", "1"))
        self._explorer = explorer or (
            LichessExplorer(
                ratings=(max(1000, elo - 200), elo, min(2500, elo + 200)),
                speeds=("blitz", "rapid"),
            ) if use_explorer else None
        )
        self._human = get_human_model(
            elo, self._engine, weights_dir=weights_dir,
            use_explorer=use_explorer, explorer=self._explorer,
            trap_model_path=trap_model_path,
        )
        self._trap_db = trap_db or default_db()
        # Reuse the existing search for the prep phase + final scoring.
        # During streaming we bypass `analyse` and call lower-level helpers.
        self._search = TrollSearch(
            self._engine, self._human,
            candidate_count=DEFAULT_MULTIPV,
            candidate_depth=_PREP_SHALLOW_DEPTH,
            reply_count=5,
            subposition_depth=_HYBRID_DEPTH,
            elo=elo, explorer=self._explorer, trap_db=self._trap_db,
        )
        self._opp_tracker = OpponentTracker(self._engine)
        self._autostyle = False  # auto-apply detected style

        self._state_lock = threading.Lock()
        self._subscribers: set[Any] = set()  # WebSocket objects
        self._subscribers_lock = threading.Lock()
        self._last_subscriber_active = time.monotonic()

        self._cache: "OrderedDict[CacheKey, AnalysisResult]" = OrderedDict()

        self._stop_event = threading.Event()
        self._restart_event = threading.Event()  # signals "board changed"
        self._metrics_lock = threading.Lock()
        self._metrics = EngineMetrics()
        self._search_started_at: float = 0.0

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True,
            name=f"GameSession-{session_id[:8]}"
        )
        self._worker.start()

    # ----- public API (called from API/WS handlers) ----- #

    def add_subscriber(self, ws: Any) -> None:
        with self._subscribers_lock:
            self._subscribers.add(ws)
            self._last_subscriber_active = time.monotonic()

    def remove_subscriber(self, ws: Any) -> None:
        with self._subscribers_lock:
            self._subscribers.discard(ws)

    @property
    def has_subscribers(self) -> bool:
        with self._subscribers_lock:
            return bool(self._subscribers)

    @property
    def is_idle(self) -> bool:
        with self._subscribers_lock:
            if self._subscribers:
                return False
            return (time.monotonic() - self._last_subscriber_active) > IDLE_TIMEOUT_S

    def position_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return self._position_payload()

    def _position_payload(self) -> dict[str, Any]:
        """Build the `position` WS message body. Must be called under
        `_state_lock`."""
        b = self.board
        dests: dict[str, list[str]] = {}
        for m in b.legal_moves:
            dests.setdefault(chess.SQUARE_NAMES[m.from_square], []).append(
                chess.SQUARE_NAMES[m.to_square]
            )
        for k, v in dests.items():
            dests[k] = sorted(set(v))
        last_uci = b.move_stack[-1].uci() if b.move_stack else None
        return {
            "fen": b.fen(),
            "dests": dests,
            "turn": "white" if b.turn else "black",
            "in_check": b.is_check(),
            "is_game_over": b.is_game_over(),
            "is_checkmate": b.is_checkmate(),
            "is_stalemate": b.is_stalemate(),
            "last_move_uci": last_uci,
            "bot_side": ("white" if self.bot_side else "black") if self.bot_side is not None else None,
            "style": self.style,
            "elo": self.elo,
            "analysis_mode": self.analysis_mode,
            "playstyle": self.playstyle,
            "autoplay": self.autoplay,
            "autostyle": self._autostyle,
            "opp_detected": self._opp_tracker.snapshot().to_dict(),
            "engine_type": self.engine_type,
            "engine_limit_kind": self.engine_limit_kind,
        }

    def submit_move(self, uci: str) -> bool:
        """Validate + apply a move. Return True on success."""
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return False
        observe_args: tuple | None = None
        with self._state_lock:
            if mv not in self.board.legal_moves:
                return False
            # If the side moving is the OPPONENT (not the bot's side), feed
            # the tracker. bot_side is the colour the BOT analyses for, so
            # the opponent is the OTHER side from bot_side. When bot_side
            # is None (observer), don't track.
            if self.bot_side is not None and self.board.turn != self.bot_side:
                observe_args = (self.board.copy(stack=False), mv)
            self.board.push(mv)
        if observe_args is not None:
            try:
                self._opp_tracker.observe(*observe_args)
            except Exception:
                log.exception("opp tracker observe failed")
            # If autostyle is on, apply detected style now
            if self._autostyle:
                snap = self._opp_tracker.snapshot()
                if (snap.detected_style != self.style
                    and snap.confidence > 0.5):
                    with self._state_lock:
                        self.style = snap.detected_style
                    log.info("autostyle: applied %s (conf %.2f)",
                             snap.detected_style, snap.confidence)
        self._restart_event.set()
        return True

    def set_autostyle(self, on: bool) -> None:
        with self._state_lock:
            self._autostyle = bool(on)

    def reset_to(self, fen: str) -> bool:
        try:
            b = chess.Board(fen)
        except Exception:
            return False
        with self._state_lock:
            self.board = b
        self._restart_event.set()
        return True

    def set_side(self, color: str | None) -> None:
        with self._state_lock:
            if color == "white":
                self.bot_side = chess.BLACK   # bot thinks for the OTHER side
            elif color == "black":
                self.bot_side = chess.WHITE
            else:
                self.bot_side = None
        self._restart_event.set()

    def set_style(self, style: str) -> None:
        with self._state_lock:
            self.style = style
        self._restart_event.set()

    def set_elo(self, elo: int) -> None:
        with self._state_lock:
            self.elo = elo
        # Elo change requires rebuilding the human model — defer.
        # For now, mark the cache stale and the next iteration will
        # use the existing model (still functional, just maybe wrong rating).
        # TODO: rebuild self._human / self._search on Elo change.
        self._restart_event.set()

    def set_analysis_mode(self, mode: str) -> None:
        if mode not in ANALYSIS_MODES:
            return
        with self._state_lock:
            self.analysis_mode = mode
        self._restart_event.set()

    def set_playstyle(self, playstyle: str) -> None:
        if playstyle not in ("direct", "setup", "setup_deep"):
            return
        with self._state_lock:
            self.playstyle = playstyle
        self._restart_event.set()

    def set_autoplay(self, on: bool) -> None:
        with self._state_lock:
            self.autoplay = bool(on)
        # No need to invalidate cache — autoplay is purely an output knob.

    def close(self) -> None:
        self._stop_event.set()
        self._restart_event.set()
        try:
            self._human.close()
        except Exception:
            pass
        try:
            self._engine.close()
        except Exception:
            pass
        with self._pool_lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception:
                    pass
                self._pool = None

    def _get_pool(self) -> EnginePool | None:
        """Lazily build the helper pool the first time we need parallel work."""
        if DEFAULT_POOL_SIZE <= 0:
            return None
        with self._pool_lock:
            if self._pool is None:
                self._pool = make_pool(
                    engine_type=self._engine_type_for_pool,
                    size=DEFAULT_POOL_SIZE,
                    threads_per_engine=self._pool_threads,
                )
            return self._pool

    # ----- worker thread ----- #

    def _snapshot_state(self) -> tuple[chess.Board, str, int, "int | None", str, str]:
        with self._state_lock:
            return (
                self.board.copy(stack=False),
                self.style, self.elo, self.bot_side, self.analysis_mode,
                self.playstyle,
            )

    def _board_changed(self, snapshot_fen: str) -> bool:
        with self._state_lock:
            return self.board.fen() != snapshot_fen

    def _cache_get(self, key: CacheKey) -> AnalysisResult | None:
        # Touch for LRU
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: CacheKey, value: AnalysisResult) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_LIMIT:
            self._cache.popitem(last=False)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._iterate_one_position()
            except Exception as exc:
                log.exception("session %s worker iteration failed", self.session_id)
                self._broadcast_threadsafe({"type": "error", "msg": str(exc)})
                time.sleep(0.5)

    def _iterate_one_position(self) -> None:
        board, style, elo, bot_side, analysis_mode, playstyle = self._snapshot_state()

        # If the bot's side is set and it isn't the bot's turn to move,
        # we still analyse — but the search models the "opponent" as the
        # side to move. The UI uses bot_side mostly to constrain the
        # user's drag-drop, not to skip analysis.
        snapshot_fen = board.fen()
        ck: CacheKey = (
            _fen_key(snapshot_fen), style, elo,
            bot_side if bot_side is None else int(bot_side),
            analysis_mode, playstyle,
        )

        self._restart_event.clear()

        # 1) Cache hit — broadcast immediately (zero-latency first paint).
        cached = self._cache_get(ck)
        if cached is not None:
            self._broadcast_threadsafe({
                "type": "snapshot",
                "result": cached.to_dict(),
                "cache_hit": True,
            })

        # 2) Prep: predictions + Lichess Explorer + (if non-lite) per-reply sub-evals
        #    + (if setup playstyle) 2-ply lookahead for trap-setting potential.
        prep = self._prepare(board, style, elo, analysis_mode, playstyle)
        if self._restart_event.is_set() or self._stop_event.is_set():
            return

        # 3) Streaming phase — continuously update from the engine.
        self._search_started_at = time.monotonic()
        with self._metrics_lock:
            self._metrics = EngineMetrics()
        # Track troll_best stability for autoplay
        last_troll_best: str | None = None
        stable_count = 0
        try:
            with self._engine.stream_analysis(
                board, multipv=DEFAULT_MULTIPV, max_depth=DEFAULT_STREAM_MAX_DEPTH
            ) as analysis:
                last_broadcast = 0.0
                for info in analysis:
                    if self._stop_event.is_set() or self._restart_event.is_set():
                        break
                    self._update_metrics_from_info(info)
                    now = time.monotonic()
                    if now - last_broadcast >= BROADCAST_INTERVAL:
                        result = self._assemble_from_streaming(
                            board, list(analysis.multipv), prep, style, elo, bot_side,
                            analysis_mode, playstyle,
                        )
                        if result is not None:
                            self._cache_put(ck, result)
                            self._broadcast_threadsafe({
                                "type": "snapshot",
                                "result": result.to_dict(),
                                "cache_hit": False,
                            })
                            last_broadcast = now
                            # Autoplay: if enabled, it's bot's turn, and the
                            # troll-best has been stable for ≥ 2 ticks AND we've
                            # searched at least a moderate depth, play it.
                            current_tb = result.troll_best_uci
                            if current_tb == last_troll_best:
                                stable_count += 1
                            else:
                                stable_count = 1
                            last_troll_best = current_tb
                            if self._should_autoplay(bot_side, board, result, stable_count):
                                self._auto_apply_move(current_tb)
                                break
        except chess.engine.EngineTerminatedError:
            log.error("session %s engine terminated", self.session_id)
            return
        except Exception:
            log.exception("session %s streaming failed", self.session_id)
            raise

        # 4) Final snapshot once max_depth reached (loop exited cleanly).
        if not self._restart_event.is_set() and not self._stop_event.is_set():
            try:
                with self._metrics_lock:
                    self._metrics.is_final = True
                # Engine reached max depth — broadcast once more then idle.
                # The outer loop will spin and re-enter at the next state change.
                self._broadcast_threadsafe({"type": "metrics",
                                             "metrics": self._metrics_dict()})
                # Brief pause to avoid spinning when nothing changes.
                self._restart_event.wait(timeout=1.0)
            except Exception:
                pass

    # ----- autoplay ----- #

    def _should_autoplay(
        self,
        bot_side: chess.Color | None,
        snapshot_board: chess.Board,
        result: AnalysisResult,
        stable_count: int,
    ) -> bool:
        if not self.autoplay or bot_side is None:
            return False
        # Only autoplay when it's the bot's turn
        if snapshot_board.turn != bot_side:
            return False
        # Need enough thinking depth and stable best move
        meta = result.metrics
        if meta.depth < 14:
            return False
        if stable_count < 2:
            return False
        if not result.troll_best_uci:
            return False
        return True

    def _auto_apply_move(self, uci: str) -> None:
        """Push the bot's chosen move and broadcast the new position."""
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return
        with self._state_lock:
            if mv not in self.board.legal_moves:
                return
            self.board.push(mv)
        log.info("session %s autoplay: %s", self.session_id, uci)
        # Broadcast new position immediately (worker loop will pick up
        # the restart on its next iteration via _restart_event).
        payload = {"type": "position", **self.position_snapshot(),
                   "autoplayed": uci}
        self._broadcast_threadsafe(payload)
        self._restart_event.set()

    # ----- prep phase: precompute predictions once per position ----- #

    def _prepare(
        self, board: chess.Board, style: str, elo: int,
        analysis_mode: str, playstyle: str,
    ) -> PreparedPosition:
        # Predicted replies for top candidates: ask the human model on
        # the position AFTER each plausible candidate move. We don't
        # know the candidates yet here, so use a shallow multipv to get
        # them.
        cfg = _MODE_CONFIGS[analysis_mode]
        predicted: dict[str, list[PredictedMove]] = {}
        candidate_ucis: list[str] = []
        sub_boards: dict[str, chess.Board] = {}
        try:
            shallow = self._engine.multipv(board, k=DEFAULT_MULTIPV, depth=_PREP_SHALLOW_DEPTH)
            for var in shallow:
                if not var.pv:
                    continue
                cand_uci = var.move.uci()
                candidate_ucis.append(cand_uci)
                sub = board.copy()
                try:
                    sub.push(var.move)
                except Exception:
                    continue
                sub_boards[cand_uci] = sub
                try:
                    preds = self._human.predict(sub, top_k=max(5, cfg.replies))
                except Exception:
                    preds = []
                predicted[cand_uci] = preds
        except Exception:
            log.exception("session %s prep multipv failed", self.session_id)

        # Lichess Explorer for the root.
        explorer_by_uci: dict[str, dict] = {}
        explorer_total = 0
        if self._explorer is not None:
            try:
                resp = self._explorer.lookup(board.fen())
                explorer_total = resp.total
                for m in resp.moves:
                    if not m.move_uci:
                        continue
                    explorer_by_uci[m.move_uci] = {
                        "white": m.white, "draws": m.draws, "black": m.black,
                        "total": m.total,
                        "win_rate_white": m.win_rate_for(chess.WHITE),
                        "win_rate_black": m.win_rate_for(chess.BLACK),
                        "average_rating": m.average_rating,
                    }
            except Exception:
                pass

        # Per-reply sub-evals: in hybrid/deep modes we evaluate the
        # position after each (candidate, predicted reply) so the troll
        # utility can use a real expected_eval / worst_case rather than
        # collapsing onto the objective.
        sub_evals: dict[tuple[str, str], float] = {}
        bot_pov = board.turn
        if cfg.candidates > 0 and cfg.replies > 0:
            prep_start = time.monotonic()
            # Build the batch of all (cand, reply) sub-positions, then
            # dispatch via the helper pool. With pool_size=4 and 40
            # sub-positions, this is ~4x faster than the old serial loop
            # on a multi-core / multi-GPU box.
            jobs: list[tuple[chess.Board, bool, int]] = []
            keys: list[tuple[str, str]] = []
            for cand_uci in candidate_ucis[: cfg.candidates]:
                sub = sub_boards.get(cand_uci)
                if sub is None:
                    continue
                replies = predicted.get(cand_uci, [])[: cfg.replies]
                for pred in replies:
                    try:
                        sub2 = sub.copy()
                        sub2.push(pred.move)
                    except Exception:
                        continue
                    jobs.append((sub2, bot_pov, cfg.depth))
                    keys.append((cand_uci, pred.move.uci()))

            pool = self._get_pool()
            if pool is not None and len(jobs) > 1:
                results = pool.evaluate_batch(jobs)
                for k, v in zip(keys, results):
                    if v is not None:
                        sub_evals[k] = v
            else:
                # Fallback: serial on the main engine.
                for (b, pov, d), k in zip(jobs, keys):
                    if self._restart_event.is_set() or self._stop_event.is_set():
                        break
                    try:
                        sub_evals[k] = self._engine.evaluate_for(b, pov, depth=d)
                    except Exception:
                        continue
            log.debug(
                "session %s prep mode=%s sub_evals=%d in %.2fs (pool=%s)",
                self.session_id, analysis_mode, len(sub_evals),
                time.monotonic() - prep_start,
                pool.size if pool else "off",
            )

        # Setup-mode look-ahead: for the top-K candidates, simulate
        # opp's TOP-2 likely replies (weighted by Maia probability) and
        # run shallow multipv on each resulting position. If a
        # sacrifice with positive eval is available there, that's our
        # "trap potential" — playing this candidate sets up a future sac.
        #
        # We iterate over multiple replies (not just top-1) because the
        # whole point is that opp picks suboptimal moves: even when
        # Maia's #1 reply defends perfectly, #2/#3 might walk into the trap.
        trap_potentials: dict[str, float] = {}
        if playstyle == "setup_deep":
            trap_potentials = self._lookahead_setup_deep(
                board, candidate_ucis, sub_boards, predicted,
            )
        elif playstyle == "setup":
            tp_start = time.monotonic()
            tp_calls = 0
            for cand_uci in candidate_ucis[:5]:
                if self._restart_event.is_set() or self._stop_event.is_set():
                    break
                sub = sub_boards.get(cand_uci)
                if sub is None:
                    continue
                replies = predicted.get(cand_uci, [])
                if not replies:
                    continue

                weighted_quality = 0.0
                used_prob = 0.0
                for reply_pred in replies[:2]:  # top-2 most likely replies
                    if self._restart_event.is_set() or self._stop_event.is_set():
                        break
                    sub2 = sub.copy()
                    try:
                        sub2.push(reply_pred.move)
                    except Exception:
                        continue
                    if sub2.is_game_over():
                        continue
                    try:
                        future_vars = self._engine.multipv(sub2, k=3, depth=max(12, _DEEP_DEPTH - 2))
                    except Exception:
                        continue
                    tp_calls += 1
                    # Best sac we found from this branch.
                    # Walk each PV: if at any "after our move" position the
                    # material balance dipped vs the start of the lookahead
                    # AND the line's overall eval is winning, that's a
                    # sacrificial setup the engine is willing to play.
                    material_at_sub2 = material_balance(sub2, sub2.turn)
                    best_for_branch = 0.0
                    for fv in future_vars:
                        if not fv.pv:
                            continue
                        if fv.score_cp < 80:  # PV doesn't win enough to justify a sac
                            continue
                        test = sub2.copy()
                        max_drop = 0  # biggest material loss along the PV
                        for ply, mv in enumerate(fv.pv[:8]):
                            try:
                                test.push(mv)
                            except Exception:
                                break
                            # After every full move pair we've made another
                            # of our own moves; check material then.
                            if ply % 2 == 1:  # opp just replied → it's our pov + their last move
                                drop = material_at_sub2 - material_balance(test, sub2.turn)
                                if drop > max_drop:
                                    max_drop = drop
                        if max_drop > 100:
                            # Quality: how much sac × how confident the engine is
                            quality = min(max_drop, 800) * (min(fv.score_cp, 1000) / 1000.0)
                            best_for_branch = max(best_for_branch, quality)
                    weighted_quality += reply_pred.probability * best_for_branch
                    used_prob += reply_pred.probability

                if used_prob > 0:
                    # Normalize and store
                    trap_potentials[cand_uci] = weighted_quality / used_prob
            log.debug(
                "session %s setup lookahead %d cands, %d eval-calls, %.2fs",
                self.session_id, len(trap_potentials), tp_calls,
                time.monotonic() - tp_start,
            )

        return PreparedPosition(
            fen=board.fen(),
            predicted_replies=predicted,
            explorer_total=explorer_total,
            explorer_by_uci=explorer_by_uci,
            sub_evals=sub_evals,
            trap_potentials=trap_potentials,
        )

    # ----- setup_deep: 4-ply alternating SF/Maia lookahead ----- #

    def _lookahead_setup_deep(
        self,
        board: chess.Board,
        candidate_ucis: list[str],
        sub_boards: dict[str, chess.Board],
        predicted: dict[str, list[PredictedMove]],
    ) -> dict[str, float]:
        """Per-candidate 4-ply lookahead alternating SF (us) and Maia
        (opp). Catches trap patterns SF dis-prefers at modest depth
        (e.g. Fried Liver) because we trust Maia to walk into them on
        opp's behalf rather than asking SF for their best reply.

        For each top-5 candidate:

          1. Take our move (already chosen).
          2. Take Maia's top-1 opp reply.   ← sub2 (our turn)
          3. Take SF's best move at depth 14.  ← sub3 (opp turn)
          4. Take Maia's top-1 reply.       ← sub4 (our turn)
          5. Evaluate sub4 with SF depth 12.

        If material dropped along the chain and the final eval is
        winning, signal a trap.
        """
        out: dict[str, float] = {}
        bot_pov = board.turn  # the side we're analysing for
        material_at_root = material_balance(board, bot_pov)
        tp_start = time.monotonic()

        for cand_uci in candidate_ucis[:5]:
            if self._restart_event.is_set() or self._stop_event.is_set():
                break
            sub1 = sub_boards.get(cand_uci)
            if sub1 is None:
                continue
            replies = predicted.get(cand_uci, [])
            if not replies:
                continue

            weighted_quality = 0.0
            used_prob = 0.0
            for reply_pred in replies[:2]:
                if self._restart_event.is_set() or self._stop_event.is_set():
                    break
                sub2 = sub1.copy()
                try:
                    sub2.push(reply_pred.move)
                except Exception:
                    continue
                if sub2.is_game_over():
                    continue

                # 1) SF best for us at sub2
                try:
                    var = self._engine.multipv(sub2, k=1, depth=_SETUP_DEEP_DEPTH)
                    if not var or not var[0].pv:
                        continue
                    our_m = var[0].pv[0]
                except Exception:
                    continue
                sub3 = sub2.copy()
                try:
                    sub3.push(our_m)
                except Exception:
                    continue
                if sub3.is_game_over():
                    # Mate or stalemate after our move; check if winning
                    if sub3.is_checkmate() and sub3.turn != bot_pov:
                        # We just delivered mate — huge trap signal
                        weighted_quality += reply_pred.probability * 600.0
                        used_prob += reply_pred.probability
                    continue

                # 2) Maia top-1 reply for opp
                try:
                    opp_preds = self._human.predict(sub3, top_k=1)
                except Exception:
                    opp_preds = []
                if not opp_preds:
                    continue
                opp_m = opp_preds[0].move
                sub4 = sub3.copy()
                try:
                    sub4.push(opp_m)
                except Exception:
                    continue

                # 3) Eval the resulting position
                try:
                    eval4 = self._engine.evaluate_for(sub4, bot_pov, depth=max(12, _SETUP_DEEP_DEPTH - 2))
                except Exception:
                    continue

                # 4) Check material drop along the chain
                material_at_sub4 = material_balance(sub4, bot_pov)
                drop = material_at_root - material_at_sub4
                if drop > 100 and eval4 > 80:
                    # The bot wants to play (cand → reply → our_m → opp_m)
                    # which sacrifices material but ends up winning.
                    quality = min(drop, 800) * (min(eval4, 1000) / 1000.0)
                    weighted_quality += reply_pred.probability * quality
                used_prob += reply_pred.probability

            if used_prob > 0:
                out[cand_uci] = weighted_quality / used_prob

        log.debug(
            "session %s setup_deep %d cands in %.2fs",
            self.session_id, len(out), time.monotonic() - tp_start,
        )
        return out

    # ----- streaming assembly: build candidates from current multipv ----- #

    def _assemble_from_streaming(
        self,
        board: chess.Board,
        multipv: list[dict],
        prep: PreparedPosition,
        style: str,
        elo: int,
        bot_side: chess.Color | None,
        analysis_mode: str,
        playstyle: str,
    ) -> AnalysisResult | None:
        bot_pov = board.turn
        material_before = material_balance(board, bot_pov)
        if not multipv:
            return None

        # Normalize: each entry is an InfoDict from python-chess.
        # We expect score + pv + depth fields.
        candidates: list[Candidate] = []
        objective_best_eval: float | None = None
        objective_best_uci = ""
        for idx, info in enumerate(multipv):
            score_obj = info.get("score")
            pv = info.get("pv") or []
            if not pv or score_obj is None:
                continue
            score_cp = _pov_score_to_cp(score_obj, bot_pov)
            # Extract WDL if engine reports it (modern SF + Lc0 do).
            wdl_obj = info.get("wdl")
            wdl_tuple: tuple[float, float, float] | None = None
            if wdl_obj is not None:
                try:
                    pov_wdl = wdl_obj.pov(bot_pov)
                    total = pov_wdl.wins + pov_wdl.draws + pov_wdl.losses
                    if total > 0:
                        wdl_tuple = (
                            pov_wdl.wins / total,
                            pov_wdl.draws / total,
                            pov_wdl.losses / total,
                        )
                except Exception:
                    pass
            if objective_best_eval is None:
                objective_best_eval = score_cp
                objective_best_uci = pv[0].uci()

            cand = self._build_candidate(
                board, pv, score_cp, idx + 1,
                objective_best_eval, material_before, bot_pov,
                prep, style, elo, analysis_mode, playstyle,
                wdl=wdl_tuple,
            )
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.troll_score, reverse=True)
        troll_best_uci = candidates[0].move_uci

        with self._metrics_lock:
            metrics_copy = EngineMetrics(**self._metrics.__dict__)
        elapsed_ms = int((time.monotonic() - self._search_started_at) * 1000)
        metrics_copy.elapsed_ms = elapsed_ms

        return AnalysisResult(
            fen=board.fen(),
            side_to_move=("white" if bot_pov else "black"),
            candidates=candidates,
            objective_best_uci=objective_best_uci,
            troll_best_uci=troll_best_uci,
            elapsed_ms=elapsed_ms,
            elo_assumed=elo,
            metrics=metrics_copy,
        )

    def _build_candidate(
        self,
        board: chess.Board,
        pv: list[chess.Move],
        score_cp: float,
        rank: int,
        objective_best_eval: float,
        material_before: int,
        bot_pov: chess.Color,
        prep: PreparedPosition,
        style: str,
        elo: int,
        analysis_mode: str,
        playstyle: str,
        wdl: tuple[float, float, float] | None = None,
    ) -> Candidate | None:
        move = pv[0]
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()

        # Sacrifice detection from the PV: did we drop material when
        # the opponent plays SF's predicted best reply?
        after = board.copy()
        try:
            after.push(move)
        except Exception:
            return None
        is_sac = False
        sac_value = 0
        material_after = material_before  # no opponent move yet
        if len(pv) >= 2:
            reply = pv[1]
            after2 = after.copy()
            try:
                after2.push(reply)
            except Exception:
                after2 = None
            if after2 is not None:
                material_after = material_balance(after2, bot_pov)
                delta = material_after - material_before
                if delta < -100:
                    is_sac = True
                    sac_value = -delta

        cand_uci = move.uci()
        replies_predicted = prep.predicted_replies.get(cand_uci, [])

        # Per-reply data: if prep ran sub_evals (hybrid/deep), use those
        # for eval_after and material_after; otherwise fall back to the
        # PV-score approximation (lite).
        replies_out: list[Reply] = []
        weighted_eval = 0.0
        worst_case = float("inf")
        best_case = float("-inf")
        weighted_material = 0.0
        prob_total = 0.0
        for p in replies_predicted:
            try:
                r_san = after.san(p.move)
            except Exception:
                r_san = p.move.uci()
            reply_key = (cand_uci, p.move.uci())
            sub_eval = prep.sub_evals.get(reply_key)
            if sub_eval is not None:
                eval_after = sub_eval
                # Material after the actual reply
                try:
                    sub2 = after.copy()
                    sub2.push(p.move)
                    mat_after_reply = material_balance(sub2, bot_pov) - material_before
                except Exception:
                    mat_after_reply = 0
            else:
                # Lite fallback: PV-eval approximation, material only
                # accurate for the SF best reply.
                eval_after = score_cp
                mat_after_reply = (
                    material_after - material_before
                    if (len(pv) >= 2 and p.move == pv[1]) else 0
                )

            replies_out.append(Reply(
                move_uci=p.move.uci(),
                move_san=r_san,
                probability=p.probability,
                eval_after=eval_after,
                material_after=mat_after_reply,
                is_best_reply=(len(pv) >= 2 and p.move == pv[1]),
            ))
            weighted_eval += p.probability * eval_after
            weighted_material += p.probability * mat_after_reply
            worst_case = min(worst_case, eval_after)
            best_case = max(best_case, eval_after)
            prob_total += p.probability

        # If we got real per-reply evals, use the weighted aggregates.
        # Otherwise (lite mode or no predictions), collapse onto SF score.
        have_honest = analysis_mode != "lite" and any(
            (cand_uci, p.move.uci()) in prep.sub_evals for p in replies_predicted
        ) and prob_total > 0
        if have_honest:
            # Normalize in case the top-K probs didn't sum to 1.
            expected_eval = weighted_eval / prob_total
            expected_material_cp = weighted_material / prob_total
            worst_case_eval = worst_case
            best_case_eval = best_case
            # Anger: opponent gives us cp when they don't play their best reply.
            best_reply_eval_for_opp = min(r.eval_after for r in replies_out)
            prob_blunder = 0.0
            expected_loss = 0.0
            for r in replies_out:
                drop = r.eval_after - best_reply_eval_for_opp  # ≥0 in bot's POV
                if drop > 150:
                    prob_blunder += r.probability
                expected_loss += r.probability * drop
            prob_blunder = min(1.0, prob_blunder / prob_total)
            expected_loss = expected_loss / prob_total
        else:
            expected_eval = score_cp
            expected_material_cp = float(material_after - material_before)
            worst_case_eval = score_cp
            best_case_eval = score_cp
            prob_blunder = 0.0
            expected_loss = 0.0

        is_mate_for_us = score_cp > (MATE_SCORE - 1000)
        mate_in: int | None = None
        if is_mate_for_us:
            mate_in = max(1, MATE_SCORE - int(score_cp))

        trap_potential_cp = prep.trap_potentials.get(cand_uci, 0.0)

        features = TrollFeatures(
            objective_eval=score_cp,
            objective_rank=rank,
            objective_best_eval=objective_best_eval,
            expected_eval=expected_eval,
            worst_case_eval=worst_case_eval,
            best_case_eval=best_case_eval,
            expected_material_cp=expected_material_cp,
            reply_entropy=_entropy_of(replies_predicted),
            is_sacrifice=is_sac,
            sacrifice_value_cp=sac_value,
            expected_eval_loss_to_opponent=expected_loss,
            prob_opponent_blunders=prob_blunder,
            is_mate_for_us=is_mate_for_us,
            mate_in=mate_in,
            trap_potential_cp=trap_potential_cp,
            wdl=wdl,
        )
        util = troll_utility(features, elo=elo, style=style, playstyle=playstyle)

        cand = Candidate(
            move_uci=cand_uci,
            move_san=san,
            objective_eval=score_cp,
            objective_rank=rank,
            is_sacrifice=is_sac,
            sacrifice_value=sac_value,
            expected_eval=expected_eval,
            worst_case_eval=worst_case_eval,
            expected_material=expected_material_cp,
            replies=replies_out,
            troll_score=util.troll_score,
            anger_probability=util.anger_probability,
            human_factor=util.human_factor,
            trap_depth=0,
            notes=util.notes,
            trap_potential_cp=trap_potential_cp,
            wdl=wdl,
        )

        # Trap-DB bonus.
        bonus = trap_score_for_move(board.fen(), cand_uci, self._trap_db)
        if bonus > 0:
            cand.troll_score += bonus
            cand.notes.append(f"📚 known trap (+{bonus:.0f}cp prior)")

        # Lichess Explorer bonus + UI fields.
        exm = prep.explorer_by_uci.get(cand_uci)
        if exm is not None and exm.get("total", 0) >= 30:
            wr = exm["win_rate_white"] if bot_pov else exm["win_rate_black"]
            cand.empirical_total = exm["total"]
            cand.empirical_win_rate = wr
            if wr > 0.52:
                import math
                emp_bonus = 600 * (wr - 0.5) * math.sqrt(min(exm["total"], 1000) / 100)
                cand.troll_score += emp_bonus
                cand.notes.append(
                    f"📊 wins {wr*100:.0f}% in {exm['total']} Lichess games at this rating"
                )

        return cand

    # ----- metrics ----- #

    def _update_metrics_from_info(self, info: dict) -> None:
        with self._metrics_lock:
            d = info.get("depth")
            if d is not None:
                self._metrics.depth = max(self._metrics.depth, int(d))
            sd = info.get("seldepth")
            if sd is not None:
                self._metrics.seldepth = max(self._metrics.seldepth, int(sd))
            n = info.get("nodes")
            if n is not None:
                self._metrics.nodes = int(n)
            nps = info.get("nps")
            if nps is not None:
                self._metrics.nps = int(nps)
            # Time info from engine (in ms)
            t = info.get("time")
            if t is not None:
                self._metrics.elapsed_ms = int(t * 1000) if isinstance(t, float) else int(t)
            else:
                self._metrics.elapsed_ms = int(
                    (time.monotonic() - self._search_started_at) * 1000
                )

        # GPU monitor (best-effort)
        try:
            from .gpu_monitor import GpuMonitor
            r = GpuMonitor.instance().latest()
            with self._metrics_lock:
                self._metrics.gpu_util = r.util
                if r.vram_used_mb is not None:
                    self._metrics.vram_mb = r.vram_used_mb
        except Exception:
            pass

    def _metrics_dict(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "depth": self._metrics.depth,
                "seldepth": self._metrics.seldepth,
                "nodes": self._metrics.nodes,
                "nps": self._metrics.nps,
                "elapsed_ms": self._metrics.elapsed_ms,
                "gpu_util": self._metrics.gpu_util,
                "vram_mb": self._metrics.vram_mb,
                "is_final": self._metrics.is_final,
            }

    # ----- broadcast ----- #

    def _broadcast_threadsafe(self, payload: dict[str, Any]) -> None:
        """Schedule a broadcast onto the asyncio loop from the worker thread."""
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)
        except RuntimeError:
            # Loop already stopped — server shutting down.
            pass

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        import json
        msg = json.dumps(payload, default=_json_default)
        with self._subscribers_lock:
            subs = list(self._subscribers)
            self._last_subscriber_active = time.monotonic()
        if not subs:
            return
        for ws in subs:
            try:
                await ws.send_text(msg)
            except Exception:
                self.remove_subscriber(ws)


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #

def _entropy_of(predictions: list[PredictedMove]) -> float:
    import math
    if not predictions:
        return 0.0
    h = 0.0
    for p in predictions:
        if p.probability > 0:
            h -= p.probability * math.log(p.probability)
    return h


def _json_default(o):
    # Fallback for any non-trivially-serialisable types we may slip
    # through (e.g. numpy floats from third-party libs).
    if hasattr(o, "to_dict"):
        return o.to_dict()
    try:
        return float(o)
    except Exception:
        return str(o)
