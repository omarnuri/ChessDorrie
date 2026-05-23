"""High-level analysis façade.

Layer the web server (and tests) call. Owns a single long-lived
`Engine` and `HumanModel` and re-uses them across requests.
"""

from __future__ import annotations

from typing import Optional

import chess

from .engine import Engine
from .human_model import HumanModel, get_human_model
from .search import TrollSearch
from .types import AnalysisResult


class Analyzer:
    """Single-instance, thread-unsafe analyser.

    The Stockfish process and the human model are persistent — much
    cheaper than spinning them up per request.
    """

    def __init__(
        self,
        elo: int = 1500,
        weights_dir: str = "weights",
        candidate_count: int = 8,
        candidate_depth: int = 16,
        reply_count: int = 5,
        subposition_depth: int = 12,
        engine_threads: int = 2,
    ) -> None:
        self.elo = elo
        self._engine = Engine(threads=engine_threads)
        self._human = get_human_model(elo, self._engine, weights_dir=weights_dir)
        self._search = TrollSearch(
            self._engine,
            self._human,
            candidate_count=candidate_count,
            candidate_depth=candidate_depth,
            reply_count=reply_count,
            subposition_depth=subposition_depth,
            elo=elo,
        )

    def analyse(self, fen: str, style: str = "balanced") -> AnalysisResult:
        board = chess.Board(fen)
        return self._search.analyse(board, style=style)

    def close(self) -> None:
        self._human.close()
        self._engine.close()


# Convenience one-shot — exists mainly for quick tests.
def analyse_position(
    fen: str,
    elo: int = 1500,
    **kwargs,
) -> AnalysisResult:
    a = Analyzer(elo=elo, **kwargs)
    try:
        return a.analyse(fen)
    finally:
        a.close()
