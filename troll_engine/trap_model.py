"""PyTorch-backed `HumanModel`.

Wraps a Maia-shaped policy network checkpoint (either a converted
Maia or one fine-tuned on trap-rich games via
`scripts/train_trap_policy.py`) and exposes the standard
`HumanModel.predict()` interface, so the rest of the troll engine
(prep phase, lookahead, autoplay) consumes it identically to
`MaiaLc0Model` or `SoftmaxStockfishModel`.

Loads lazily on first `predict()` call to avoid paying the
load-time cost when the model isn't actually used.

Selection
---------
Selected by passing `trap_model_path=<file.pt>` to
`troll_engine.human_model.get_human_model`. The factory wraps the
returned `TrappyMaiaModel` in `HybridHumanModel` with the Lichess
Explorer, the same composition used for the Lc0 path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import chess
import numpy as np

from .human_model import HumanModel, PredictedMove


LOG = logging.getLogger("troll_engine.trap_model")


class TrappyMaiaModel(HumanModel):
    """A Maia-shaped policy network served from a PyTorch checkpoint."""

    def __init__(self, checkpoint_path: str,
                 blocks: int = 6, channels: int = 64,
                 device: str | None = None) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        self._path = checkpoint_path
        self._blocks = blocks
        self._channels = channels
        self._device = device
        self._model: Any = None  # loaded lazily

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "torch not installed; install it or use a different HumanModel"
            ) from e

        # Import the shared architecture from the train script.
        # Doing it at runtime keeps the import graph clean.
        from scripts.train_trap_policy import _make_model  # type: ignore

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        LOG.info("loading TrappyMaia checkpoint %s on %s",
                 self._path, self._device)

        model = _make_model(self._blocks, self._channels)
        ckpt = torch.load(self._path, map_location=self._device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)  # tolerate raw state_dict files

        # Allow partial loading — the converter may leave keys missing.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            LOG.warning("%d missing keys (training random init)", len(missing))
        if unexpected:
            LOG.warning("%d unexpected keys (ignored)", len(unexpected))

        model.to(self._device)
        model.eval()
        self._model = model
        self._torch = torch  # cache the module handle

    def predict(self, board: chess.Board, top_k: int = 5) -> list[PredictedMove]:
        if board.is_game_over():
            return []
        self._ensure_loaded()
        from .encoding import encode_board, legal_move_mask, index_to_move

        # Encode board → batch of 1
        tensor = encode_board(board).astype(np.float32) / 255.0
        x = self._torch.from_numpy(tensor).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            logits = self._model(x)[0].cpu().numpy()

        # Mask illegal moves, then softmax
        mask = legal_move_mask(board)
        logits = logits + mask
        # Stable softmax
        m = logits.max()
        exps = np.exp(logits - m)
        probs = exps / exps.sum()

        # Pick top-k legal moves
        order = np.argsort(probs)[::-1]
        out: list[PredictedMove] = []
        for idx in order:
            if probs[idx] <= 0:
                continue
            mv = index_to_move(int(idx), board)
            if mv is None:
                continue
            out.append(PredictedMove(move=mv, probability=float(probs[idx])))
            if len(out) >= top_k:
                break

        # Renormalise to top_k
        total = sum(p.probability for p in out)
        if total > 0:
            for p in out:
                p.probability /= total
        return out

    def close(self) -> None:
        self._model = None
