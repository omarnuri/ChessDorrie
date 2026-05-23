"""Human-move predictor.

The bot's whole soul is in here. To find moves that crush *humans*
rather than perfect engines, we need a probability distribution over
what a human at a given Elo is likely to play.

Two implementations:

* `MaiaLc0Model` — the real deal. Loads a Maia weight file into Lc0 and
  asks for `nodes=1` so Lc0 returns the raw policy distribution. Requires
  Lc0 and the Maia weight files on disk.

* `SoftmaxStockfishModel` — fallback. Runs Stockfish at shallow depth,
  takes the top-N moves, and softmax-weights them by score. Captures the
  "humans usually pick one of the top few moves" property but misses the
  *patterns* Maia learned from millions of human games.

`get_human_model(elo)` picks the best available implementation.
"""

from __future__ import annotations

import math
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass

import chess

from .engine import Engine, _pov_score_to_cp
from .lichess_explorer import LichessExplorer


@dataclass
class PredictedMove:
    move: chess.Move
    probability: float


class HumanModel(ABC):
    """Predicts the distribution over an opponent's reply."""

    @abstractmethod
    def predict(
        self,
        board: chess.Board,
        top_k: int = 5,
    ) -> list[PredictedMove]:
        """Return up to `top_k` likely moves with their probabilities.

        Probabilities are renormalized to sum to 1 over the returned
        moves. The full policy may have more tail moves we drop.
        """
        ...

    def close(self) -> None:  # pragma: no cover
        pass


# --------------------------------------------------------------------- #
# Fallback: softmax over Stockfish multi-PV scores.                     #
# --------------------------------------------------------------------- #

class SoftmaxStockfishModel(HumanModel):
    """Stockfish-based approximation of human-move probabilities.

    For each position we ask Stockfish for top-N moves at shallow depth
    and softmax their evaluations (in pawns). Temperature controls how
    "peaked" the distribution is — high T (e.g. 1.5) gives a flat
    "any of these moves looks plausible" distribution typical of weaker
    players; low T (~0.3) gives a peaked "almost always the engine
    move" distribution typical of strong players.
    """

    def __init__(
        self,
        engine: Engine,
        elo: int = 1500,
        depth: int = 8,
        top_k_search: int = 8,
    ) -> None:
        self._engine = engine
        self._elo = elo
        self._depth = depth
        self._top_k_search = top_k_search
        # Empirical temperature curve: 1100 → 1.4, 1500 → 0.8, 1900 → 0.4
        # (humans of any strength still mostly pick from the top few SF moves)
        self._temperature = max(0.2, 2.0 - elo / 1000.0)

    def predict(self, board: chess.Board, top_k: int = 5) -> list[PredictedMove]:
        if board.is_game_over():
            return []
        try:
            variations = self._engine.multipv(
                board, k=min(self._top_k_search, max(top_k, 1)), depth=self._depth
            )
        except Exception:
            return []
        if not variations:
            return []

        # Convert cp scores to pawn-units for softmax stability
        scores = [v.score_cp / 100.0 for v in variations[:top_k]]
        T = self._temperature
        # Shift by max for numerical stability
        m = max(scores)
        exps = [math.exp((s - m) / T) for s in scores]
        Z = sum(exps) or 1.0
        probs = [e / Z for e in exps]

        return [
            PredictedMove(move=variations[i].move, probability=probs[i])
            for i in range(len(probs))
        ]


# --------------------------------------------------------------------- #
# Real Maia via Lc0.                                                    #
# --------------------------------------------------------------------- #

class MaiaLc0Model(HumanModel):
    """Maia neural network, served by Lc0 as a UCI engine.

    We use `nodes=1` so that Lc0 returns the raw policy network output
    (no MCTS). The visit counts from `VerboseMoveStats` then directly
    encode the policy distribution.
    """

    def __init__(self, weights_path: str, top_k_search: int = 8) -> None:
        import chess.engine as ce

        lc0_path = shutil.which("lc0")
        if not lc0_path:
            raise RuntimeError("lc0 binary not found in PATH")
        if not os.path.isfile(weights_path):
            raise RuntimeError(f"Maia weights not found: {weights_path}")

        self._engine = ce.SimpleEngine.popen_uci(lc0_path)
        self._engine.configure({
            "WeightsFile": weights_path,
            "MultiPV": top_k_search,
            "VerboseMoveStats": True,
        })

    def predict(self, board: chess.Board, top_k: int = 5) -> list[PredictedMove]:
        import chess.engine as ce
        if board.is_game_over():
            return []
        infos = self._engine.analyse(
            board, ce.Limit(nodes=1), multipv=top_k
        )
        out: list[PredictedMove] = []
        # With nodes=1, Lc0 returns policy as the only "search" output.
        # The `nodes` field of each line == the visit count proxy for policy.
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            visits = info.get("nodes", 1)
            out.append(PredictedMove(move=pv[0], probability=float(visits)))

        # Normalize
        total = sum(p.probability for p in out) or 1.0
        for p in out:
            p.probability /= total
        out.sort(key=lambda p: p.probability, reverse=True)
        return out[:top_k]

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


# --------------------------------------------------------------------- #
# Empirical: Lichess Opening Explorer.                                  #
# --------------------------------------------------------------------- #

class LichessExplorerModel(HumanModel):
    """Uses the Lichess Opening Explorer's *empirical* move frequencies
    as the move-probability distribution.

    For positions with very few historical games we return an empty
    list and let the caller fall back to a different model.
    """

    def __init__(
        self,
        explorer: LichessExplorer,
        *,
        min_samples: int = 30,
    ) -> None:
        self._explorer = explorer
        self._min_samples = min_samples

    def predict(self, board: chess.Board, top_k: int = 5) -> list[PredictedMove]:
        if board.is_game_over():
            return []
        resp = self._explorer.lookup(board.fen())
        if resp.total < self._min_samples or not resp.moves:
            return []
        # The Explorer already ranks moves by frequency; we normalize
        # the top-k counts to a probability distribution.
        top = resp.moves[:top_k]
        denom = sum(m.total for m in top) or 1
        out: list[PredictedMove] = []
        for m in top:
            try:
                mv = chess.Move.from_uci(m.move_uci)
            except Exception:
                continue
            if mv not in board.legal_moves:
                continue
            out.append(PredictedMove(move=mv, probability=m.total / denom))
        return out


class HybridHumanModel(HumanModel):
    """Explorer first; falls back to a secondary model for novel positions.

    The classic case is opening / early middlegame, where the Explorer
    has thousands of games and gives us the true empirical
    distribution. Once the game leaves book the Explorer dries up and
    Maia (or softmax-Stockfish) takes over — that's exactly the regime
    where pattern-matching neural nets shine.
    """

    def __init__(self, primary: LichessExplorerModel, fallback: HumanModel) -> None:
        self._primary = primary
        self._fallback = fallback

    def predict(self, board: chess.Board, top_k: int = 5) -> list[PredictedMove]:
        preds = self._primary.predict(board, top_k=top_k)
        if preds:
            return preds
        return self._fallback.predict(board, top_k=top_k)

    def close(self) -> None:
        self._primary.close()
        self._fallback.close()


# --------------------------------------------------------------------- #
# Factory.                                                              #
# --------------------------------------------------------------------- #

def get_human_model(
    elo: int,
    fallback_engine: Engine,
    weights_dir: str = "weights",
    *,
    use_explorer: bool = True,
    explorer: LichessExplorer | None = None,
) -> HumanModel:
    """Return the best human-model implementation available.

    Order of preference, all wrapped behind a hybrid when applicable:

      1. `LichessExplorerModel` for in-book positions (≥ 30 games at
         the target rating bucket on Lichess).
      2. `MaiaLc0Model` once we leave book, if lc0 + weights present.
      3. `SoftmaxStockfishModel` as the final fallback.

    Pass ``use_explorer=False`` to disable the API call (e.g. offline).
    """
    rungs = (1100, 1500, 1900)
    nearest = min(rungs, key=lambda r: abs(r - elo))
    weights = os.path.join(weights_dir, f"maia-{nearest}.pb.gz")

    # Choose the neural / engine fallback first.
    neural: HumanModel
    if shutil.which("lc0") and os.path.isfile(weights):
        try:
            neural = MaiaLc0Model(weights)
        except Exception:
            neural = SoftmaxStockfishModel(fallback_engine, elo=elo)
    else:
        neural = SoftmaxStockfishModel(fallback_engine, elo=elo)

    if not use_explorer:
        return neural

    exp = explorer or LichessExplorer(
        ratings=(max(1000, elo - 200), elo, min(2500, elo + 200)),
        speeds=("blitz", "rapid"),
    )
    return HybridHumanModel(LichessExplorerModel(exp), neural)
