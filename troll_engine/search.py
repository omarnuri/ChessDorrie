"""Troll search orchestrator.

For every analysed position:

    1. Get top-K candidate moves from Stockfish (deep multi-PV).
    2. For each candidate, ask the human model what the opponent is
       likely to play in reply (probability distribution over top-N moves).
    3. For each (candidate, reply) pair, evaluate the resulting position
       with Stockfish at moderate depth — that's the position the bot
       *expects to be in* against a human.
    4. Compute per-candidate features (expected eval, expected material,
       whether it's a sacrifice, response entropy, …).
    5. Run `troll_utility` and rank.

All scoring inside this module is from the BOT's POV — the side to
move at the root position. Stockfish's PovScore is converted
accordingly.
"""

from __future__ import annotations

import math
import time

import chess

from .engine import Engine, material_balance, _pov_score_to_cp, MATE_SCORE, PIECE_CP
from .human_model import HumanModel, PredictedMove
from .lichess_explorer import LichessExplorer
from .trap_db import TrapDB, default_db, trap_score_for_move
from .types import AnalysisResult, Candidate, Reply
from .utility import TrollFeatures, troll_utility, shannon_entropy


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #

def _captured_value_cp(board: chess.Board, move: chess.Move) -> int:
    """cp value of the piece captured by `move` (0 if not a capture).
    Handles en passant."""
    if board.is_en_passant(move):
        return PIECE_CP[chess.PAWN]
    if not board.is_capture(move):
        return 0
    target = board.piece_at(move.to_square)
    return PIECE_CP[target.piece_type] if target else 0


def _detect_sacrifice(
    board_before: chess.Board,
    our_move: chess.Move,
    replies: list[tuple[chess.Move, float]],
) -> tuple[bool, int]:
    """Did `our_move` give material away?

    Returns (is_sacrifice, sac_value_cp). `sac_value` is the worst-case
    material loss for us across the supplied reply candidates.
    """
    pov = board_before.turn
    material_before = material_balance(board_before, pov)
    board_after = board_before.copy()
    board_after.push(our_move)

    worst_delta = 0  # negative = bad for us
    for reply, _prob in replies:
        b2 = board_after.copy()
        try:
            b2.push(reply)
        except Exception:
            continue
        delta = material_balance(b2, pov) - material_before
        if delta < worst_delta:
            worst_delta = delta

    sac_value = max(0, -worst_delta)
    # Threshold: must be giving up at least a pawn for it to count.
    return sac_value >= 100, sac_value


def _entropy(probs: list[float]) -> float:
    return shannon_entropy(probs)


# --------------------------------------------------------------------- #
# Search                                                                #
# --------------------------------------------------------------------- #

class TrollSearch:
    """Glues Stockfish + human model + utility into a position analyser."""

    def __init__(
        self,
        engine: Engine,
        human_model: HumanModel,
        *,
        candidate_count: int = 8,
        candidate_depth: int = 16,
        reply_count: int = 5,
        subposition_depth: int = 12,
        elo: int = 1500,
        trap_db: TrapDB | None = None,
        explorer: LichessExplorer | None = None,
    ) -> None:
        self._engine = engine
        self._human = human_model
        self._candidate_count = candidate_count
        self._candidate_depth = candidate_depth
        self._reply_count = reply_count
        self._subposition_depth = subposition_depth
        self._elo = elo
        self._trap_db = trap_db if trap_db is not None else default_db()
        self._explorer = explorer

    def analyse(self, board: chess.Board, style: str = "balanced") -> AnalysisResult:
        t0 = time.monotonic()
        bot_pov = board.turn

        # 1. Candidate generation (Stockfish multi-PV)
        variations = self._engine.multipv(
            board, k=self._candidate_count, depth=self._candidate_depth
        )
        if not variations:
            return AnalysisResult(
                fen=board.fen(),
                side_to_move=("white" if bot_pov else "black"),
                candidates=[],
                objective_best_uci="",
                troll_best_uci="",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                elo_assumed=self._elo,
            )

        objective_best_eval = variations[0].score_cp
        objective_best_uci = variations[0].move.uci()

        material_before = material_balance(board, bot_pov)

        # Pull empirical stats from Lichess for the root, if enabled.
        explorer_moves: dict[str, "object"] = {}
        explorer_total = 0
        if self._explorer is not None:
            try:
                resp = self._explorer.lookup(board.fen())
                explorer_total = resp.total
                explorer_moves = {m.move_uci: m for m in resp.moves}
            except Exception:
                pass

        candidates: list[Candidate] = []
        for rank, var in enumerate(variations, start=1):
            cand = self._evaluate_candidate(
                board, var, rank, objective_best_eval, material_before, bot_pov,
                style=style,
            )
            # Trap DB bonus — small, additive, only fires on known patterns.
            bonus = trap_score_for_move(board.fen(), cand.move_uci, self._trap_db)
            if bonus > 0:
                cand.troll_score += bonus
                cand.notes.append(f"📚 known trap (+{bonus:.0f}cp prior)")

            # Lichess-empirical bonus: if the side-to-move wins >> 50% of
            # historical games at this rating after playing this move, bump.
            exm = explorer_moves.get(cand.move_uci)
            if exm is not None and exm.total >= 30:
                wr = exm.win_rate_for(bot_pov)
                if wr > 0.52:
                    # Scale by sqrt(n) so a 200-game result beats a 30-game one.
                    bonus = 600 * (wr - 0.5) * math.sqrt(min(exm.total, 1000) / 100)
                    cand.troll_score += bonus
                    cand.notes.append(
                        f"📊 wins {wr*100:.0f}% in {exm.total} Lichess games at this rating (+{bonus:.0f}cp)"
                    )
                # Also expose raw sample size in the candidate's notes so
                # the UI can show it.
                cand.empirical_total = exm.total
                cand.empirical_win_rate = wr
            candidates.append(cand)

        # Rank by troll score (descending).
        candidates.sort(key=lambda c: c.troll_score, reverse=True)
        troll_best_uci = candidates[0].move_uci if candidates else ""

        return AnalysisResult(
            fen=board.fen(),
            side_to_move=("white" if bot_pov else "black"),
            candidates=candidates,
            objective_best_uci=objective_best_uci,
            troll_best_uci=troll_best_uci,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            elo_assumed=self._elo,
        )

    # ------------------------------------------------------------------ #
    def _evaluate_candidate(
        self,
        root: chess.Board,
        var,                     # engine.Variation
        rank: int,
        objective_best_eval: float,
        material_before: int,
        bot_pov: chess.Color,
        *,
        style: str = "balanced",
    ) -> Candidate:
        move = var.move
        san = root.san(move)
        after = root.copy()
        after.push(move)

        # 2. Predict opponent's reply distribution.
        predicted = self._human.predict(after, top_k=self._reply_count)
        # If the human model failed or position is terminal, fall back to
        # a single "engine best reply" sample so we still produce metrics.
        if not predicted:
            # No replies (mate or stalemate) — just evaluate the position.
            try:
                eval_after = self._engine.evaluate_for(after, bot_pov, depth=self._subposition_depth)
            except Exception:
                eval_after = var.score_cp
            return self._wrap_terminal_candidate(
                move, san, rank, var.score_cp, objective_best_eval,
                eval_after, after, root, material_before, bot_pov
            )

        # 3. Evaluate each (candidate, reply) sub-position.
        replies_out: list[Reply] = []
        replies_for_sac_check: list[tuple[chess.Move, float]] = []
        weighted_eval = 0.0
        weighted_material = 0.0
        worst_case = math.inf
        best_case = -math.inf
        probs = []

        for i, pred in enumerate(predicted):
            sub = after.copy()
            try:
                sub.push(pred.move)
            except Exception:
                continue
            try:
                sub_eval = self._engine.evaluate_for(
                    sub, bot_pov, depth=self._subposition_depth
                )
            except Exception:
                sub_eval = var.score_cp

            material_after = material_balance(sub, bot_pov)
            material_delta = material_after - material_before

            weighted_eval += pred.probability * sub_eval
            weighted_material += pred.probability * material_delta
            worst_case = min(worst_case, sub_eval)
            best_case = max(best_case, sub_eval)
            probs.append(pred.probability)
            replies_for_sac_check.append((pred.move, pred.probability))

            try:
                san_r = after.san(pred.move)
            except Exception:
                san_r = pred.move.uci()
            replies_out.append(
                Reply(
                    move_uci=pred.move.uci(),
                    move_san=san_r,
                    probability=pred.probability,
                    eval_after=sub_eval,
                    material_after=material_delta,
                    is_best_reply=(i == 0),
                )
            )

        # 4. Sacrifice detection (worst-case reply, only over PREDICTED replies —
        # we don't punish the bot for sacs against moves humans never play).
        is_sac, sac_value = _detect_sacrifice(root, move, replies_for_sac_check)

        # 5. Anger: P(reply that loses ≥150cp vs. their best reply).
        if replies_out:
            best_reply_eval_for_opp = min(r.eval_after for r in replies_out)
            # ^ from bot POV, opp's best reply MINIMIZES bot eval.
            prob_blunder = 0.0
            expected_eval_loss_to_opp = 0.0
            for r in replies_out:
                drop_for_opp = r.eval_after - best_reply_eval_for_opp  # ≥0
                if drop_for_opp > 150:
                    prob_blunder += r.probability
                expected_eval_loss_to_opp += r.probability * drop_for_opp
        else:
            prob_blunder = 0.0
            expected_eval_loss_to_opp = 0.0

        # 6. Mate info.
        is_mate_for_us = var.score_cp > (MATE_SCORE - 1000)
        mate_in: int | None = None
        if is_mate_for_us:
            mate_in = max(1, MATE_SCORE - int(var.score_cp))

        features = TrollFeatures(
            objective_eval=var.score_cp,
            objective_rank=rank,
            objective_best_eval=objective_best_eval,
            expected_eval=weighted_eval,
            worst_case_eval=worst_case if worst_case != math.inf else var.score_cp,
            best_case_eval=best_case if best_case != -math.inf else var.score_cp,
            expected_material_cp=weighted_material,
            reply_entropy=_entropy(probs),
            is_sacrifice=is_sac,
            sacrifice_value_cp=sac_value,
            expected_eval_loss_to_opponent=expected_eval_loss_to_opp,
            prob_opponent_blunders=prob_blunder,
            is_mate_for_us=is_mate_for_us,
            mate_in=mate_in,
        )
        util = troll_utility(features, elo=self._elo, style=style)

        return Candidate(
            move_uci=move.uci(),
            move_san=san,
            objective_eval=var.score_cp,
            objective_rank=rank,
            is_sacrifice=is_sac,
            sacrifice_value=sac_value,
            expected_eval=weighted_eval,
            worst_case_eval=features.worst_case_eval,
            expected_material=weighted_material,
            replies=replies_out,
            troll_score=util.troll_score,
            anger_probability=util.anger_probability,
            human_factor=util.human_factor,
            trap_depth=0,  # populated by a follow-up trap-depth probe (future)
            notes=util.notes,
        )

    # ------------------------------------------------------------------ #
    def _wrap_terminal_candidate(
        self,
        move: chess.Move,
        san: str,
        rank: int,
        objective_eval: float,
        objective_best_eval: float,
        eval_after: float,
        after: chess.Board,
        root: chess.Board,
        material_before: int,
        bot_pov: chess.Color,
    ) -> Candidate:
        """Candidate where opp has no replies (mate/stalemate after our move)."""
        material_after = material_balance(after, bot_pov)
        material_delta = material_after - material_before
        is_mate = after.is_checkmate()
        return Candidate(
            move_uci=move.uci(),
            move_san=san,
            objective_eval=objective_eval,
            objective_rank=rank,
            is_sacrifice=False,
            sacrifice_value=0,
            expected_eval=eval_after,
            worst_case_eval=eval_after,
            expected_material=material_delta,
            replies=[],
            troll_score=eval_after + (5000 if is_mate else 0),
            anger_probability=0.0,
            human_factor=0.0,
            notes=["Delivers mate" if is_mate else "Forces stalemate"],
        )
