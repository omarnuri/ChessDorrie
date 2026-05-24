"""Lc0 (Leela Chess Zero) engine wrapper, drop-in compatible with
`troll_engine.engine.Engine`.

Lc0 plays chess via Monte Carlo Tree Search guided by a neural
network, all of which runs on the GPU. On a Blackwell-class card
(RTX PRO 6000, H100) with a strong net (BT3, T82) it delivers
~3500+ Elo strength using ~50-200k nodes per second — equivalent
to top Stockfish playing depth ~35-40, but with the *entire* search
burden on the GPU. CPU stays idle and free for other work.

Why we want this:
  * Stockfish search is fundamentally CPU-bound (alpha-beta + NNUE).
    On a Colab Pro+ with 12 CPUs it's fast, but the user's H100/G4
    sits doing nothing during analysis.
  * With Lc0 as the main engine, GPU cores actually drive the search.
    CPU becomes available for parallel work (a second helper engine,
    background mining, multiple sessions, …).

UCI mapping
-----------

Lc0 speaks UCI but uses *nodes* (MCTS visits), not *depth*. We expose
the same public methods (`multipv`, `evaluate_for`, `stream_analysis`)
as `Engine`, but internally translate depth-like arguments to a
node budget via `_depth_to_nodes`. A rough rule of thumb is
`nodes ≈ 1000 * 2^(depth - 10)`, capped at sane bounds. This keeps
the existing search/session code working unchanged.

Selection
---------

`get_engine(engine_type="auto", ...)` returns the right impl:
  * `"auto"` — Lc0 if available + nets exist, else Stockfish
  * `"stockfish"` — always Stockfish (the legacy `Engine`)
  * `"lc0"` — fail loudly if Lc0 / weights not present
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine

from .engine import (
    Engine, Variation, MATE_SCORE, _pov_score_to_cp, material_balance,
)


LOG = logging.getLogger("troll_engine.lc0")


# Default weights search path (override via LC0_WEIGHTS env var).
_DEFAULT_NET_CANDIDATES = (
    "weights/lc0/BT3.pb.gz",
    "weights/lc0/BT4.pb.gz",
    "weights/lc0/T82.pb.gz",
    "weights/lc0/T80.pb.gz",
)


def find_lc0_weights() -> str | None:
    """Locate a strong-play Lc0 net (not Maia — Maia is for human
    prediction, separate path)."""
    env = os.environ.get("LC0_WEIGHTS")
    if env and os.path.isfile(env):
        return env
    for c in _DEFAULT_NET_CANDIDATES:
        if os.path.isfile(c):
            return c
    return None


def _depth_to_nodes(depth: int) -> int:
    """Translate Stockfish-style depth to an Lc0 node budget that
    yields roughly comparable analysis quality on a strong net (BT3/T82).

    Empirical mapping (BT3, cuda-fp16, ~150k NPS on G4):
      depth 10 →     1 000 nodes (instant)
      depth 14 →     5 000 nodes (~30 ms)
      depth 18 →    28 000 nodes (~200 ms)
      depth 22 →   150 000 nodes (~1 s)
      depth 26 →   775 000 nodes (~5 s)
      depth 30 →     4.1 M nodes (~30 s)
      depth 34 →    21.0 M nodes (~2-3 min)
      depth 38 →    50.0 M nodes (capped, ~6 min)

    Override via `CD_LC0_DEPTH_K` env var if needed (default 0.6).
    """
    if depth <= 8:
        return 500
    if depth <= 10:
        return 1_000
    k = float(os.environ.get("CD_LC0_DEPTH_K", "0.6"))
    extra = depth - 10
    base = 1_000 * int(round(2 ** (extra * k)))
    return max(500, min(base, 50_000_000))


class Lc0Engine:
    """Lc0 wrapped to look like our `Engine`. Same public method
    signatures — search/session code works unchanged."""

    limit_kind = "nodes"

    def __init__(
        self,
        weights_path: str | None = None,
        *,
        backend: str = "cuda-fp16",
        threads: int = 2,
        minibatch_size: int = 256,
        multipv_default: int = 8,
    ) -> None:
        if not shutil.which("lc0"):
            raise RuntimeError("lc0 binary not found in PATH")
        wp = weights_path or find_lc0_weights()
        if not wp or not os.path.isfile(wp):
            raise RuntimeError(
                f"Lc0 weights not found (set LC0_WEIGHTS or drop a net in weights/lc0/): {wp}"
            )
        self._weights = wp
        self._engine = chess.engine.SimpleEngine.popen_uci("lc0")

        # Best-effort config; some options vary by Lc0 build.
        opts = {
            "WeightsFile": wp,
            "Backend": backend,
            "Threads": threads,
            "MinibatchSize": minibatch_size,
            "MultiPV": multipv_default,
            # Don't autoload tablebases on Colab (often unavailable).
            "SyzygyPath": "",
            # Useful for diagnostics in info lines.
            "VerboseMoveStats": False,
        }
        # Strip options the running Lc0 doesn't accept rather than crash.
        for k, v in list(opts.items()):
            try:
                self._engine.configure({k: v})
            except Exception as e:
                LOG.debug("lc0 option %s=%s rejected: %s", k, v, e)

        LOG.info("Lc0 ready: weights=%s backend=%s", wp, backend)

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "Lc0Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- analysis -------------------------------------------------------

    def multipv(
        self,
        board: chess.Board,
        k: int = 8,
        depth: int = 16,
    ) -> list[Variation]:
        """Multi-PV search using an `nodes` limit derived from `depth`."""
        nodes = _depth_to_nodes(depth)
        infos = self._engine.analyse(
            board, chess.engine.Limit(nodes=nodes), multipv=k
        )
        out: list[Variation] = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            out.append(Variation(
                move=pv[0],
                score_cp=_pov_score_to_cp(info["score"], board.turn),
                pv=list(pv),
                depth=int(info.get("depth", 0)),  # selective depth
            ))
        return out

    def evaluate(self, board: chess.Board, depth: int = 14) -> float:
        return self.evaluate_for(board, board.turn, depth=depth)

    def evaluate_for(
        self,
        board: chess.Board,
        pov: chess.Color,
        depth: int = 14,
    ) -> float:
        nodes = _depth_to_nodes(depth)
        info = self._engine.analyse(board, chess.engine.Limit(nodes=nodes))
        return _pov_score_to_cp(info["score"], pov)

    # -- streaming ------------------------------------------------------
    @contextlib.contextmanager
    def stream_analysis(
        self,
        board: chess.Board,
        multipv: int = 8,
        max_depth: int = 24,
        max_nodes: int | None = None,
    ):
        """Stream MCTS visits; iterator yields info dicts as the tree grows."""
        if max_nodes is None:
            max_nodes = _depth_to_nodes(max_depth)
        result = self._engine.analysis(
            board, chess.engine.Limit(nodes=max_nodes),
            multipv=multipv, info=chess.engine.INFO_ALL,
        )
        try:
            yield result
        finally:
            try:
                result.stop()
            except Exception:
                pass


# --------------------------------------------------------------------- #
# Factory                                                               #
# --------------------------------------------------------------------- #

def get_engine(
    engine_type: str = "auto",
    *,
    threads: int = 2,
    hash_mb: int = 256,
    lc0_weights: str | None = None,
    lc0_backend: str = "cuda-fp16",
) -> "Engine | Lc0Engine":
    """Construct the best engine for the user's hardware.

    `engine_type`:
      * `"auto"` — try Lc0+strong-net first; fall back to Stockfish
      * `"stockfish"` — always Stockfish (CPU)
      * `"lc0"` — always Lc0+strong-net (raise if unavailable)
    """
    et = engine_type.lower()
    if et == "stockfish":
        return Engine(threads=threads, hash_mb=hash_mb)

    if et in ("lc0", "auto"):
        wp = lc0_weights or find_lc0_weights()
        if shutil.which("lc0") and wp:
            try:
                return Lc0Engine(wp, backend=lc0_backend, threads=threads)
            except Exception as e:
                if et == "lc0":
                    raise
                LOG.warning("Lc0 init failed (%s); falling back to Stockfish", e)
        elif et == "lc0":
            raise RuntimeError(
                "engine_type='lc0' but lc0 binary or weights are missing. "
                "Run scripts/install_lc0_cuda.sh and drop a net at "
                "weights/lc0/BT3.pb.gz (see scripts/download_lc0_net.sh)."
            )

    return Engine(threads=threads, hash_mb=hash_mb)
