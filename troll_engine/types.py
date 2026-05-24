"""Shared data classes for analysis output.

All numeric evaluations are in **centipawns from the side-to-move's POV**
unless stated otherwise. A positive score always means "good for the
side to move" — i.e. for the bot when it's about to play.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Reply:
    """A predicted opponent reply with the engine's evaluation of the
    resulting position."""

    move_uci: str
    move_san: str
    probability: float          # Maia's predicted probability, in [0, 1]
    eval_after: float           # cp, from the BOT's POV (i.e. side to move at the root)
    material_after: int         # net material in centipawns from bot's POV
    is_best_reply: bool         # whether this is also the engine's #1 reply


@dataclass
class Candidate:
    """One candidate move for the bot, with all the metrics needed to
    rank it on the troll axis."""

    move_uci: str
    move_san: str

    # Objective view (Stockfish, deep)
    objective_eval: float        # cp after this move, deep search, perfect play
    objective_rank: int          # 1 = engine's best move, 2 = second-best, ...
    is_sacrifice: bool           # did we just give up material on net?
    sacrifice_value: int         # cp value of material given up (0 if not a sac)

    # Human-aware view
    expected_eval: float         # cp, weighted by opponent reply probabilities
    worst_case_eval: float       # cp, the worst position from the top-N replies
    expected_material: float     # weighted material delta in cp
    replies: list[Reply] = field(default_factory=list)

    # Derived UI metrics
    troll_score: float = 0.0          # the composite ranking score
    anger_probability: float = 0.0    # P(opponent blunders) * size_of_blunder
    human_factor: float = 0.0         # |expected - objective|
    trap_depth: int = 0               # plies of refutation depth, if known
    notes: list[str] = field(default_factory=list)  # human-readable annotations

    # Empirical Lichess stats (from the Opening Explorer, if available)
    empirical_total: int = 0          # # of Lichess games from this position
    empirical_win_rate: float = 0.0   # side-to-move's score: (W + 0.5D)/N

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class EngineMetrics:
    """Engine telemetry surfaced to the UI live during a search."""
    depth: int = 0
    seldepth: int = 0
    nodes: int = 0
    nps: int = 0
    elapsed_ms: int = 0
    gpu_util: float | None = None       # 0..100, or None if no GPU monitor
    vram_mb: float | None = None        # MB used, or None
    lc0_nps: int | None = None          # if Maia/Lc0 reports its own NPS
    is_final: bool = False              # True only when search hit max depth or stopped


@dataclass
class AnalysisResult:
    """Top-level result for one analysed position."""

    fen: str
    side_to_move: str               # "white" or "black"
    candidates: list[Candidate]     # ranked by troll_score, descending
    objective_best_uci: str         # engine's #1 move
    troll_best_uci: str             # our re-ranked #1
    elapsed_ms: int                 # how long the analysis took
    elo_assumed: int                # which Maia model was used
    metrics: EngineMetrics = field(default_factory=EngineMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "candidates": [c.to_dict() for c in self.candidates],
            "objective_best_uci": self.objective_best_uci,
            "troll_best_uci": self.troll_best_uci,
            "elapsed_ms": self.elapsed_ms,
            "elo_assumed": self.elo_assumed,
            "metrics": asdict(self.metrics),
        }
