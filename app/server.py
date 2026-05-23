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
    yield
    with _analyzer_lock:
        for a in _analyzer.values():
            a.close()
        _analyzer.clear()


app = FastAPI(title="ChessDorrie", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response models ----------------------------------------- #

class AnalyseRequest(BaseModel):
    fen: str
    elo: int = Field(default=1500, ge=1000, le=2500)


class MoveRequest(BaseModel):
    fen: str
    move: str  # UCI or SAN


# --- endpoints ------------------------------------------------------- #

@app.get("/api/health")
def health():
    return {"ok": True}


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
        result = analyzer.analyse(req.fen)
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
