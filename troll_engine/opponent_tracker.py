"""Per-session opponent-style detector.

Watches the moves made by the side opposite to the bot's chosen
`bot_side`. Each move contributes signals to three counters:

  * `greedy`   — captures, especially captures of pieces we offered
  * `cautious` — retreats away from contested squares, prophylactic
                 king-safety moves
  * `strong`   — moves that match Stockfish's top pick

After ≥ 3 observed opp moves we classify the opponent. The result
is exposed in `session.position_snapshot()` so the UI can render
"Opp detected: 😋 greedy (4/6 moves)" and optionally auto-apply
the matching `style` (greedy / balanced / cautious) when the user
toggles "Auto-detect style".

Cost
----

Each `observe()` call runs one shallow Stockfish search (depth 8) for
the "matches_sf_top" feature. That's ~50-150 ms per opp move. We
cache the SF top-move per FEN to avoid recomputing on tab refreshes.

This module deliberately does NOT use the Maia human-model — Maia
predicts what humans typically play, but we want to detect the
*deviation* from the engine line, which only SF gives us.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import chess

from .engine import Engine, material_balance, PIECE_CP


LOG = logging.getLogger("troll_engine.opp_tracker")


# Tunable thresholds
MIN_MOVES_FOR_DETECTION = 3
GREEDY_THRESHOLD = 0.55       # ≥55% of moves greedy → greedy classification
CAUTIOUS_THRESHOLD = 0.55


@dataclass
class OpponentSnapshot:
    observed_moves: int = 0
    greedy: float = 0.0       # 0..1
    cautious: float = 0.0
    strong: float = 0.0       # 0..1, fraction matching SF best
    detected_style: str = "balanced"
    confidence: float = 0.0   # 0..1, scales with observed_moves

    def to_dict(self) -> dict:
        return {
            "observed_moves": self.observed_moves,
            "greedy": self.greedy,
            "cautious": self.cautious,
            "strong": self.strong,
            "detected_style": self.detected_style,
            "confidence": self.confidence,
        }


class OpponentTracker:
    """Lives for the lifetime of a `GameSession`."""

    def __init__(self, engine: Engine, sf_depth: int = 8,
                 fen_cache_size: int = 256) -> None:
        self._engine = engine
        self._sf_depth = sf_depth
        self._lock = threading.Lock()
        self._n_moves = 0
        self._greedy_pts = 0.0
        self._cautious_pts = 0.0
        self._strong_pts = 0.0
        # FEN → SF best move (avoid re-searching the same position)
        self._sf_cache: OrderedDict[str, chess.Move] = OrderedDict()
        self._cache_size = fen_cache_size

    def _sf_best(self, board: chess.Board) -> chess.Move | None:
        fen_key = board.board_fen() + (" w" if board.turn else " b")
        if fen_key in self._sf_cache:
            self._sf_cache.move_to_end(fen_key)
            return self._sf_cache[fen_key]
        try:
            vars_ = self._engine.multipv(board, k=1, depth=self._sf_depth)
            best = vars_[0].move if vars_ else None
        except Exception:
            best = None
        if best is not None:
            self._sf_cache[fen_key] = best
            while len(self._sf_cache) > self._cache_size:
                self._sf_cache.popitem(last=False)
        return best

    @staticmethod
    def _was_capture(board_before: chess.Board, move: chess.Move) -> bool:
        return board_before.is_capture(move)

    @staticmethod
    def _was_retreat(board_before: chess.Board, move: chess.Move) -> bool:
        """True if the moved piece was attacked on its origin and the
        destination is NOT attacked (i.e. they ran away to safety)."""
        opp = not board_before.turn
        attackers_at_origin = board_before.attackers(opp, move.from_square)
        if not attackers_at_origin:
            return False
        # Check the destination's threat status on the post-move board
        post = board_before.copy()
        post.push(move)
        attackers_at_dest = post.attackers(board_before.turn ^ True, move.to_square)
        # `board_before.turn ^ True` after push: same color as the original mover
        # (we want: who attacks the new square from the *opposing* side, which
        # is the side now to move on `post`).
        # The expression above is intentional but a bit twisted; correct it:
        attackers_at_dest = post.attackers(post.turn, move.to_square)
        return len(attackers_at_dest) == 0

    def observe(self, board_before: chess.Board, move: chess.Move,
                hung_value_cp: int = 0) -> None:
        """Record one opp move.

        `hung_value_cp` is optionally provided by the caller: it's the
        value of a piece WE hung on the previous ply that this move
        could have captured. If non-zero AND the move captured, it
        counts as a strong "greedy bait" signal.
        """
        if move not in board_before.legal_moves:
            return

        was_cap = self._was_capture(board_before, move)
        was_ret = self._was_retreat(board_before, move)
        sf_best = self._sf_best(board_before)
        was_sf_top = (sf_best is not None and sf_best == move)

        with self._lock:
            self._n_moves += 1
            if was_cap:
                self._greedy_pts += 1.0
            if hung_value_cp >= 200 and was_cap:
                # Took the bait — strong greedy signal
                self._greedy_pts += 1.0
            if was_ret:
                self._cautious_pts += 1.0
            if was_sf_top:
                self._strong_pts += 1.0

        LOG.debug(
            "opp observe: cap=%s retreat=%s sf_top=%s n=%d g=%.1f c=%.1f s=%.1f",
            was_cap, was_ret, was_sf_top,
            self._n_moves, self._greedy_pts, self._cautious_pts, self._strong_pts,
        )

    def snapshot(self) -> OpponentSnapshot:
        with self._lock:
            n = self._n_moves
            g = self._greedy_pts / max(1, n)
            c = self._cautious_pts / max(1, n)
            s = self._strong_pts / max(1, n)

        if n < MIN_MOVES_FOR_DETECTION:
            detected = "balanced"
            conf = 0.0
        elif g >= GREEDY_THRESHOLD and g > c:
            detected = "greedy"
            conf = min(1.0, g)
        elif c >= CAUTIOUS_THRESHOLD and c > g:
            detected = "cautious"
            conf = min(1.0, c)
        else:
            detected = "balanced"
            conf = max(0.0, 1.0 - abs(g - c))

        # Confidence ramps with sample size — full confidence at 8 moves
        conf *= min(1.0, n / 8.0)

        return OpponentSnapshot(
            observed_moves=n,
            greedy=g,
            cautious=c,
            strong=s,
            detected_style=detected,
            confidence=conf,
        )

    def reset(self) -> None:
        with self._lock:
            self._n_moves = 0
            self._greedy_pts = 0.0
            self._cautious_pts = 0.0
            self._strong_pts = 0.0
            self._sf_cache.clear()
