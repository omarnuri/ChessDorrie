"""FastAPI server for ChessDorrie.

Endpoints
---------
GET  /                        Single-page UI (static/index.html)
GET  /static/*                Frontend assets
POST /api/analyse             Analyse a FEN, return ranked candidates + metrics
POST /api/move                Apply a move to a FEN, return new FEN
GET  /api/health              Liveness check

The analyser is constructed lazily on first request and reused between
requests. The Stockfish process and human model are kept alive in the
background — the cost of spinning them up is significant.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from troll_engine import Analyzer
from troll_engine.gpu_monitor import GpuMonitor
from troll_engine.session_manager import SessionManager
from app.ws import router as ws_router


logging.basicConfig(
    level=os.environ.get("CD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


STATIC_DIR = Path(__file__).parent / "static"
WEIGHTS_DIR = os.environ.get("CHESSDORRIE_WEIGHTS_DIR", "weights")

# Analyser is shared across requests — Stockfish is not safe to call
# concurrently from multiple threads, so we serialise with a lock.
_analyzer: dict[int, Analyzer] = {}
_analyzer_lock = Lock()


def get_analyzer(elo: int) -> Analyzer:
    elo = int(elo)
    # Snap to nearest Maia rung so we don't construct 1000 analysers.
    rungs = (1100, 1500, 1900)
    elo = min(rungs, key=lambda r: abs(r - elo))
    with _analyzer_lock:
        if elo not in _analyzer:
            _analyzer[elo] = Analyzer(
                elo=elo,
                weights_dir=WEIGHTS_DIR,
                # Tunable defaults — push higher in Colab with GPU Lc0.
                candidate_count=int(os.environ.get("CD_CANDIDATES", "8")),
                candidate_depth=int(os.environ.get("CD_DEPTH", "16")),
                reply_count=int(os.environ.get("CD_REPLIES", "5")),
                subposition_depth=int(os.environ.get("CD_SUB_DEPTH", "12")),
                engine_threads=int(os.environ.get("CD_THREADS", "2")),
            )
        return _analyzer[elo]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the GPU monitor (singleton — harmless to nudge it).
    GpuMonitor.instance()
    # SessionManager owns the live ponder sessions.
    loop = asyncio.get_running_loop()
    app.state.session_manager = SessionManager(
        loop,
        weights_dir=WEIGHTS_DIR,
        use_explorer=os.environ.get("CD_USE_EXPLORER", "1") == "1",
    )
    try:
        yield
    finally:
        try:
            app.state.session_manager.shutdown()
        except Exception:
            pass
        with _analyzer_lock:
            for a in _analyzer.values():
                a.close()
            _analyzer.clear()
        try:
            GpuMonitor.instance().stop()
        except Exception:
            pass


app = FastAPI(title="ChessDorrie", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ws_router)


# --- request/response models ----------------------------------------- #

class AnalyseRequest(BaseModel):
    fen: str
    elo: int = Field(default=1500, ge=1000, le=2500)
    style: str = Field(default="balanced", pattern="^(greedy|balanced|cautious)$")


class MoveRequest(BaseModel):
    fen: str
    move: str  # UCI or SAN


class LegalMovesRequest(BaseModel):
    fen: str


def _legal_dests(board: chess.Board) -> dict[str, list[str]]:
    """Map from origin square to list of legal destination squares.

    Used by the chessground frontend to restrict drag-and-drop to
    legal moves only. Promotions collapse into the destination — the
    UI defaults underpromotions to queen.
    """
    out: dict[str, list[str]] = {}
    for m in board.legal_moves:
        out.setdefault(chess.SQUARE_NAMES[m.from_square], []).append(
            chess.SQUARE_NAMES[m.to_square]
        )
    # Deduplicate (a single move can produce multiple promotion-target
    # entries to the same square).
    for k, v in out.items():
        out[k] = sorted(set(v))
    return out


# --- endpoints ------------------------------------------------------- #

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/legal-moves")
def legal_moves(req: LegalMovesRequest):
    """Return the legal-move destination map for `fen`, plus a few
    derived flags useful to the UI.
    """
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")
    return {
        "fen": board.fen(),
        "dests": _legal_dests(board),
        "turn": "white" if board.turn else "black",
        "in_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
    }


@app.post("/api/analyse")
def analyse(req: AnalyseRequest):
    # Validate FEN early — Stockfish is happy to crash on garbage.
    try:
        chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    # Stockfish isn't thread-safe and the analyser holds one process,
    # so serialise.
    analyzer = get_analyzer(req.elo)
    with _analyzer_lock:
        result = analyzer.analyse(req.fen, style=req.style)
    return result.to_dict()


@app.post("/api/move")
def move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    mv = req.move.strip()
    parsed = None
    # Try UCI first, then SAN.
    try:
        parsed = chess.Move.from_uci(mv)
        if parsed not in board.legal_moves:
            parsed = None
    except Exception:
        parsed = None
    if parsed is None:
        try:
            parsed = board.parse_san(mv)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Illegal move: {mv}")

    board.push(parsed)
    return {
        "fen": board.fen(),
        "san": board.san(parsed) if False else "",  # board.san is for the position BEFORE the push
        "is_game_over": board.is_game_over(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "turn": "white" if board.turn else "black",
    }


# --- static UI -------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host=os.environ.get("CD_HOST", "0.0.0.0"),
        port=int(os.environ.get("CD_PORT", "8000")),
        reload=False,
    )
