"""End-to-end smoke test for the WebSocket ponder session.

Spins the FastAPI app via TestClient (in-process), opens a websocket
connection, asserts that:
  1. The server greets us with `session` + `position` messages.
  2. The position's `dests` matches python-chess `board.legal_moves`.
  3. We receive ≥ 2 `snapshot` messages within ~6 seconds (proving the
     worker thread is doing iterative deepening).
  4. After we submit a legal move, the position payload updates.
  5. The snapshot includes engine metrics (depth > 0 after a few sec).
"""

import json
import time

import chess
import pytest
from fastapi.testclient import TestClient

from app.server import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _read_until(ws, target_type: str, timeout_s: float = 8.0):
    """Read messages from `ws` until we see one of `target_type`."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        # TestClient's WebSocket doesn't have a timeout argument; receive_text
        # blocks. We rely on the worker producing messages quickly.
        try:
            raw = ws.receive_text(timeout=max(0.1, end - time.monotonic()))
        except TypeError:
            raw = ws.receive_text()
        m = json.loads(raw)
        if m["type"] == "ping":
            ws.send_text(json.dumps({"type": "pong"}))
            continue
        if m["type"] == target_type:
            return m
    raise AssertionError(f"timed out waiting for {target_type}")


def _collect(ws, total_seconds: float):
    """Collect all messages received within `total_seconds`."""
    out = []
    end = time.monotonic() + total_seconds
    while time.monotonic() < end:
        try:
            raw = ws.receive_text(timeout=max(0.1, end - time.monotonic()))
        except TypeError:
            raw = ws.receive_text()
        except Exception:
            break
        m = json.loads(raw)
        if m["type"] == "ping":
            ws.send_text(json.dumps({"type": "pong"}))
            continue
        out.append(m)
    return out


def test_session_basics(client):
    with client.websocket_connect("/ws/session/new") as ws:
        sess = _read_until(ws, "session", timeout_s=2.0)
        assert sess["session_id"]
        pos = _read_until(ws, "position", timeout_s=2.0)
        # Validate dests against python-chess
        board = chess.Board(pos["fen"])
        expected_origins = set()
        for m in board.legal_moves:
            expected_origins.add(chess.SQUARE_NAMES[m.from_square])
        actual_origins = set(pos["dests"].keys())
        assert actual_origins == expected_origins, (
            f"dests origins differ: {expected_origins - actual_origins=}, "
            f"{actual_origins - expected_origins=}"
        )

        # Wait for snapshots to flow
        messages = _collect(ws, 6.0)
        snapshots = [m for m in messages if m["type"] == "snapshot"]
        assert len(snapshots) >= 2, f"expected ≥2 snapshots in 6s, got {len(snapshots)}"

        # Metrics should include a positive depth on the latest snapshot
        last = snapshots[-1]
        metrics = last["result"].get("metrics", {})
        assert metrics.get("depth", 0) > 0, f"expected depth>0, got {metrics}"


def test_move_updates_position(client):
    with client.websocket_connect("/ws/session/new") as ws:
        _read_until(ws, "session", timeout_s=2.0)
        pos = _read_until(ws, "position", timeout_s=2.0)
        assert pos["turn"] == "white"

        # Drain initial snapshots briefly
        _collect(ws, 1.0)

        # Submit e2e4
        ws.send_text(json.dumps({"type": "move", "uci": "e2e4"}))
        new_pos = _read_until(ws, "position", timeout_s=4.0)
        assert new_pos["last_move_uci"] == "e2e4"
        assert new_pos["turn"] == "black"
        # FEN should reflect the move: pawn moved from e2 to e4.
        new_board = chess.Board(new_pos["fen"])
        assert new_board.piece_at(chess.E4) is not None
        assert new_board.piece_at(chess.E2) is None
