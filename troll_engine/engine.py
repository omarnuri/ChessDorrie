"""Stockfish wrapper.

Owns a single long-lived Stockfish UCI process (via python-chess) and
exposes the operations the troll search actually needs: multi-PV
evaluation of the root, fast single-line evaluation of a sub-position,
mate detection.

Scores are always returned in **centipawns from the side-to-move's
POV**. Mates are mapped to a large finite value (±MATE_SCORE).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine


MATE_SCORE = 100_000  # cp value used to represent forced mate
PIECE_CP = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}


def _stockfish_path() -> str:
    """Find a Stockfish binary on the system."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.isfile(env):
        return env
    for cand in ("stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish"):
        which = shutil.which(cand) if not os.path.isabs(cand) else (
            cand if os.path.isfile(cand) else None
        )
        if which:
            return which
    raise RuntimeError("Could not locate the Stockfish binary. Set STOCKFISH_PATH.")


@dataclass
class Variation:
    """One principal variation returned by multi-PV analysis."""
    move: chess.Move
    score_cp: float          # centipawns from side-to-move's POV
    pv: list[chess.Move]     # principal variation, including the first move
    depth: int


def _pov_score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> float:
    """Convert python-chess PovScore to cp from `turn`'s POV.

    Mates become ±MATE_SCORE. We compress mate distance into the score
    so longer mates score slightly less than shorter ones — useful for
    ordering, and so the troll utility can choose "win material now"
    over "mate in 12".
    """
    s = score.pov(turn)
    if s.is_mate():
        m = s.mate()
        # m == 0 → already mated; positive → we mate, negative → we get mated
        if m is None:
            return 0.0
        if m > 0:
            return MATE_SCORE - m  # mate in 1 > mate in 5
        return -MATE_SCORE - m
    return float(s.score(mate_score=MATE_SCORE))


class Engine:
    """Long-lived Stockfish process. Use as a context manager or call
    `close()` explicitly."""

    def __init__(self, threads: int = 2, hash_mb: int = 256) -> None:
        path = _stockfish_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "Engine":
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
        """Return top-`k` candidates for `board` to depth `depth`.

        Scores are from the side-to-move's POV.
        """
        infos = self._engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=k
        )
        out: list[Variation] = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            out.append(
                Variation(
                    move=pv[0],
                    score_cp=_pov_score_to_cp(info["score"], board.turn),
                    pv=list(pv),
                    depth=int(info.get("depth", depth)),
                )
            )
        return out

    def evaluate(self, board: chess.Board, depth: int = 14) -> float:
        """Single-line evaluation, cp from side-to-move's POV."""
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
        return _pov_score_to_cp(info["score"], board.turn)

    def evaluate_for(
        self,
        board: chess.Board,
        pov: chess.Color,
        depth: int = 14,
    ) -> float:
        """Like `evaluate`, but in `pov`'s POV (not necessarily side-to-move)."""
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
        return _pov_score_to_cp(info["score"], pov)


# -- material helpers (used by search.py for sacrifice detection) -----

def material_balance(board: chess.Board, pov: chess.Color) -> int:
    """Net material in cp from `pov`'s perspective (excluding kings)."""
    total = 0
    for piece_type, value in PIECE_CP.items():
        if piece_type == chess.KING:
            continue
        total += value * len(board.pieces(piece_type, pov))
        total -= value * len(board.pieces(piece_type, not pov))
    return total
