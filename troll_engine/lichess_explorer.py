"""Lichess Opening Explorer client.

The Explorer aggregates **every standard game on Lichess** (billions
of them, going back to ~2013) into a per-position move table. Given a
FEN, the API returns the top-N moves humans actually played from that
position, with white/draw/black win counts, average rating of players
who chose each move, and the most-played continuation.

This is two signals for the troll engine, in one query:

1. **Empirical move probability** — used as a `HumanModel` for
   positions where the Explorer has enough samples (≥ ``min_samples``
   games). For unseen / novel positions we fall back to Maia or to the
   softmax-Stockfish approximation.

2. **Outcome statistics** — per-move win rates that we feed into the
   search as additional troll bonuses. If the side-to-move's win rate
   from a position after playing a particular move is >> 50% in
   ~thousands of games at the target rating, that's a known trap that
   *empirically* humiliates humans, regardless of what an engine thinks.

API docs: <https://lichess.org/api#tag/Opening-Explorer>

The service is free and unauthenticated; ratings/speeds filters narrow
the dataset to "your" opponents (e.g. 1500-1800 blitz). Be polite —
the public service is shared with the Lichess UI.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

import requests


EXPLORER_BASE = "https://explorer.lichess.ovh"

# Lichess rating buckets the API recognises.
_RATING_BUCKETS = (0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500)
_SPEEDS = ("ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence")


@dataclass
class ExplorerMove:
    """One move from the Explorer's response for a position."""

    move_uci: str
    move_san: str
    white: int          # games won by White from this move
    draws: int
    black: int          # games won by Black from this move
    average_rating: int | None

    @property
    def total(self) -> int:
        return self.white + self.draws + self.black

    def win_rate_for(self, side: bool) -> float:
        """Side-to-move's win rate (W = True for White), with draws as
        half-points. 0.5 = neutral."""
        if self.total == 0:
            return 0.5
        if side:
            return (self.white + 0.5 * self.draws) / self.total
        return (self.black + 0.5 * self.draws) / self.total


@dataclass
class ExplorerResponse:
    fen: str
    total: int
    moves: list[ExplorerMove]


def _rating_param(target_ranges: Iterable[int]) -> str:
    """Snap each target rating to the nearest Lichess bucket and dedupe."""
    buckets = set()
    for r in target_ranges:
        nearest = min(_RATING_BUCKETS, key=lambda b: abs(b - r))
        buckets.add(nearest)
    return ",".join(str(b) for b in sorted(buckets))


class LichessExplorer:
    """Thin client over the Lichess Opening Explorer.

    Caches by (FEN, ratings, speeds, moves) in memory. Polite by
    default — sleeps ``min_request_interval`` between requests to keep
    the public service happy.
    """

    def __init__(
        self,
        *,
        ratings: Iterable[int] = (1600, 1800, 2000),
        speeds: Iterable[str] = ("blitz", "rapid"),
        moves: int = 12,
        timeout_s: float = 6.0,
        min_request_interval: float = 0.05,  # 20 req/s ceiling
        cache_size: int = 4096,
    ) -> None:
        self._ratings = _rating_param(ratings)
        self._speeds = ",".join(s for s in speeds if s in _SPEEDS)
        self._moves = moves
        self._timeout = timeout_s
        self._min_interval = min_request_interval
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._cache: dict[str, ExplorerResponse] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._session = requests.Session()

    # -- cache plumbing ----------------------------------------------- #
    def _cache_get(self, key: str) -> ExplorerResponse | None:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: ExplorerResponse) -> None:
        if key in self._cache:
            return
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    # -- API ---------------------------------------------------------- #
    def lookup(self, fen: str, *, database: str = "lichess") -> ExplorerResponse:
        """Return the Explorer's response for `fen`.

        ``database`` is ``"lichess"`` (default — billions of online
        games) or ``"masters"`` (OTB master-game database, much
        smaller, no rating/speed filtering applied by the server).
        """
        key = f"{database}|{fen}|{self._ratings}|{self._speeds}|{self._moves}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        params = {
            "fen": fen,
            "moves": self._moves,
            "topGames": 0,
            "recentGames": 0,
        }
        if database == "lichess":
            params["ratings"] = self._ratings
            params["speeds"] = self._speeds

        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()

        try:
            url = f"{EXPLORER_BASE}/{database}?{urllib.parse.urlencode(params)}"
            r = self._session.get(url, timeout=self._timeout)
            r.raise_for_status()
            payload = r.json()
        except Exception:
            empty = ExplorerResponse(fen=fen, total=0, moves=[])
            self._cache_put(key, empty)
            return empty

        moves = []
        for m in payload.get("moves", []):
            moves.append(ExplorerMove(
                move_uci=m.get("uci", ""),
                move_san=m.get("san", ""),
                white=int(m.get("white", 0)),
                draws=int(m.get("draws", 0)),
                black=int(m.get("black", 0)),
                average_rating=m.get("averageRating"),
            ))
        total = int(payload.get("white", 0)) + int(payload.get("draws", 0)) + int(payload.get("black", 0))
        if total == 0:
            total = sum(m.total for m in moves)

        result = ExplorerResponse(fen=fen, total=total, moves=moves)
        self._cache_put(key, result)
        return result
