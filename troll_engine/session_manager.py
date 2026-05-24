"""SessionManager — lifecycle for `GameSession` instances.

Generates session IDs, enforces a cap on concurrent sessions, and runs
an idle-cleanup thread that closes sessions with no subscribers after
a timeout (frees the Stockfish process).

The active Lichess Explorer instance is shared across sessions to
exploit its in-memory cache.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid

import chess

from .lichess_explorer import LichessExplorer
from .session import GameSession, IDLE_TIMEOUT_S
from .trap_db import default_db


log = logging.getLogger("troll_engine.session_manager")


MAX_SESSIONS = 8


class SessionManager:
    """Owns the set of live sessions."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        weights_dir: str = "weights",
        use_explorer: bool = True,
        trap_model_path: str | None = None,
    ) -> None:
        self._loop = loop
        self._weights_dir = weights_dir
        self._use_explorer = use_explorer
        self._trap_model_path = trap_model_path
        self._lock = threading.Lock()
        self._sessions: dict[str, GameSession] = {}
        self._explorer = (
            LichessExplorer(
                ratings=(1500, 1700, 1900),
                speeds=("blitz", "rapid"),
            ) if use_explorer else None
        )
        self._trap_db = default_db()
        self._stop_event = threading.Event()
        self._janitor = threading.Thread(
            target=self._janitor_loop, daemon=True, name="SessionJanitor"
        )
        self._janitor.start()

    def create(
        self,
        *,
        elo: int = 1500,
        style: str = "balanced",
        bot_side: chess.Color | None = None,
        analysis_mode: str = "lite",
        starting_fen: str | None = None,
    ) -> GameSession:
        with self._lock:
            # Enforce cap by evicting the oldest idle session.
            if len(self._sessions) >= MAX_SESSIONS:
                self._evict_oldest_idle_locked()
            sid = uuid.uuid4().hex[:16]
            session = GameSession(
                sid, self._loop,
                elo=elo, style=style, bot_side=bot_side,
                analysis_mode=analysis_mode,
                weights_dir=self._weights_dir,
                explorer=self._explorer,
                trap_db=self._trap_db,
                use_explorer=self._use_explorer,
                trap_model_path=self._trap_model_path,
            )
            if starting_fen:
                session.reset_to(starting_fen)
            self._sessions[sid] = session
            log.info("created session %s (total=%d)", sid, len(self._sessions))
            return session

    def get(self, sid: str) -> GameSession | None:
        with self._lock:
            return self._sessions.get(sid)

    def get_or_create(
        self,
        sid: str | None,
        **kwargs,
    ) -> GameSession:
        if sid:
            existing = self.get(sid)
            if existing is not None:
                return existing
        return self.create(**kwargs)

    def dispose(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s is not None:
            log.info("disposing session %s", sid)
            s.close()

    def _evict_oldest_idle_locked(self) -> None:
        # Find the oldest session with no subscribers; close it.
        candidates = [
            (s.session_id, s) for s in self._sessions.values() if not s.has_subscribers
        ]
        if not candidates:
            # Cap hit but everyone is active — refuse to evict.
            raise RuntimeError(
                f"session cap reached ({MAX_SESSIONS}) and no idle sessions to evict"
            )
        sid, s = candidates[0]
        del self._sessions[sid]
        log.info("evicted idle session %s to make room", sid)
        s.close()

    def _janitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(15.0)
            if self._stop_event.is_set():
                return
            to_close: list[GameSession] = []
            with self._lock:
                for sid, s in list(self._sessions.items()):
                    if s.is_idle:
                        del self._sessions[sid]
                        to_close.append(s)
            for s in to_close:
                log.info("session %s idle timeout, closing", s.session_id)
                try:
                    s.close()
                except Exception:
                    log.exception("failed to close session %s", s.session_id)

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            try:
                s.close()
            except Exception:
                pass
        self._janitor.join(timeout=2.0)
