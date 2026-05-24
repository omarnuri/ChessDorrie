"""Process pool for parallel chess-engine evaluation.

The session's `_prepare` phase computes sub-evaluations for every
(candidate move × predicted reply) pair. With 8 candidates × 5
replies = 40 sub-positions, serialising those through a single
Stockfish or Lc0 process is the dominant cost in deep mode (~10 s
on CPU, ~5 s on GPU).

`EnginePool` spawns N independent engine processes (same factory
the main session uses) and dispatches a batch of jobs across them
via a thread pool. Each engine is single-tenant (chess engines
aren't reentrant), so concurrency = pool_size. The thread pool's
GIL doesn't matter because every worker thread spends its time
blocked on engine I/O.

On a Blackwell-class GPU (96 GB VRAM) you can comfortably run
4-8 parallel Lc0 instances with BT3 (~280 MB net each, ~1-2 GB
working set each). Set `CD_POOL_SIZE` to tune.

Memory budget rough estimate:
  * 4 Lc0 + BT3 ≈ 4 × 2 GB = 8 GB VRAM
  * 8 Lc0 + BT3 ≈ 8 × 2 GB = 16 GB VRAM
  * 4 Stockfish (CPU) ≈ negligible VRAM, ~1 GB RAM each

The pool exposes:
  * `evaluate_batch(jobs)` — list of (board, pov, depth)
  * `multipv_batch(jobs)`  — list of (board, k, depth)
  * `close()`              — terminate all workers
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import chess


LOG = logging.getLogger("troll_engine.pool")


DEFAULT_POOL_SIZE = int(os.environ.get("CD_POOL_SIZE", "4"))


class EnginePool:
    """Manages N independent engine workers for parallel evaluation."""

    def __init__(
        self,
        engine_factory: Callable[[], "object"],
        size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self._size = max(1, size)
        # Build engines synchronously; if one fails we shrink the pool.
        self._engines: list = []
        for i in range(self._size):
            try:
                self._engines.append(engine_factory())
            except Exception as e:
                LOG.warning("pool worker %d failed to start: %s", i, e)
        if not self._engines:
            raise RuntimeError("EnginePool: no workers could be started")
        self._available: queue.Queue = queue.Queue()
        for e in self._engines:
            self._available.put(e)
        self._executor = ThreadPoolExecutor(
            max_workers=len(self._engines), thread_name_prefix="enginepool"
        )
        LOG.info("EnginePool ready with %d workers", len(self._engines))

    @property
    def size(self) -> int:
        return len(self._engines)

    def evaluate_batch(
        self,
        jobs: list[tuple[chess.Board, bool, int]],
        timeout_per_job: float = 60.0,
    ) -> list[float | None]:
        """Each job is `(board, pov, depth)`. Returns list of cp evals
        (from pov), `None` on per-job failure. Same order as input."""
        if not jobs:
            return []

        def _do(idx_job):
            idx, (board, pov, depth) = idx_job
            engine = self._available.get()
            try:
                return idx, engine.evaluate_for(board, pov, depth=depth)
            except Exception as e:
                LOG.debug("evaluate_batch[%d] failed: %s", idx, e)
                return idx, None
            finally:
                self._available.put(engine)

        results: list[float | None] = [None] * len(jobs)
        futures = [
            self._executor.submit(_do, (i, j)) for i, j in enumerate(jobs)
        ]
        for f in futures:
            try:
                idx, val = f.result(timeout=timeout_per_job)
                results[idx] = val
            except Exception as e:
                LOG.warning("evaluate_batch future failed: %s", e)
        return results

    def multipv_batch(
        self,
        jobs: list[tuple[chess.Board, int, int]],
        timeout_per_job: float = 60.0,
    ) -> list[list]:
        """Each job is `(board, k, depth)`. Returns list of Variation lists."""
        if not jobs:
            return []

        def _do(idx_job):
            idx, (board, k, depth) = idx_job
            engine = self._available.get()
            try:
                return idx, engine.multipv(board, k=k, depth=depth)
            except Exception as e:
                LOG.debug("multipv_batch[%d] failed: %s", idx, e)
                return idx, []
            finally:
                self._available.put(engine)

        results: list[list] = [[] for _ in jobs]
        futures = [
            self._executor.submit(_do, (i, j)) for i, j in enumerate(jobs)
        ]
        for f in futures:
            try:
                idx, val = f.result(timeout=timeout_per_job)
                results[idx] = val
            except Exception as e:
                LOG.warning("multipv_batch future failed: %s", e)
        return results

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        for e in self._engines:
            try:
                e.close()
            except Exception:
                pass
        self._engines.clear()


def make_pool(
    engine_type: str = "auto",
    *,
    size: int | None = None,
    threads_per_engine: int = 2,
) -> EnginePool | None:
    """Build an EnginePool using the same factory the main session uses.

    Returns None on systems where spinning up extra engines isn't
    worthwhile (e.g. tiny CPU runtime). In that case callers fall
    back to single-engine sequential evaluation.
    """
    from .lc0_engine import get_engine

    n = size if size is not None else DEFAULT_POOL_SIZE
    if n <= 0:
        return None

    def factory():
        return get_engine(engine_type=engine_type, threads=threads_per_engine)

    try:
        return EnginePool(factory, size=n)
    except Exception as e:
        LOG.warning("EnginePool init failed: %s — falling back to single engine", e)
        return None
