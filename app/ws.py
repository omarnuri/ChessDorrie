"""WebSocket endpoint for live ponder sessions.

URL: `/ws/session/{session_id}`

On connect, the session is created if it doesn't exist (the
session_id from the URL is the key — pass "new" for a fresh session,
or reuse an existing id to reconnect).

Server → client messages (JSON):

  * `position` — emitted on every board change; carries fen, legal
    `dests` (chessground-ready), turn, in_check, last_move_uci
  * `snapshot` — emitted ~2 Hz by the worker; carries the full
    `AnalysisResult` (candidates, troll/objective/empirical/etc.) and
    a live `metrics` block (depth, nps, gpu_util...)
  * `metrics` — light-weight engine-only update if a snapshot didn't
    change but the metrics did
  * `error` — diagnostic; non-fatal

Client → server messages:

  * `move`     — submit a UCI move (validated against current board)
  * `side`     — set the user's side: "white" | "black" | null
  * `style`    — "greedy" | "balanced" | "cautious"
  * `elo`      — int
  * `reset`    — load a FEN
  * `pong`     — heartbeat ack
"""

from __future__ import annotations

import asyncio
import json
import logging

import chess
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


log = logging.getLogger("app.ws")

router = APIRouter()


@router.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str):
    await ws.accept()
    mgr = ws.app.state.session_manager

    # "new" or an unknown ID → create. A reused ID → reattach.
    if session_id == "new":
        session = mgr.create()
    else:
        session = mgr.get(session_id)
        if session is None:
            session = mgr.create()

    session.add_subscriber(ws)
    log.info("ws connected to session %s", session.session_id)

    # Send the session ID (client should remember it for reconnects)
    # plus initial position so the UI can render even before the first
    # snapshot arrives.
    try:
        await ws.send_text(json.dumps({
            "type": "session",
            "session_id": session.session_id,
        }))
        await ws.send_text(json.dumps({
            "type": "position",
            **session.position_snapshot(),
        }))
    except Exception:
        session.remove_subscriber(ws)
        return

    heartbeat_task = asyncio.create_task(_heartbeat(ws))

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "msg": "bad json"}))
                continue
            await _dispatch(session, ws, msg)
    finally:
        heartbeat_task.cancel()
        session.remove_subscriber(ws)
        log.info("ws disconnected from session %s", session.session_id)


async def _heartbeat(ws: WebSocket) -> None:
    """Send a ping every 30 s so Cloudflare's tunnel doesn't close us."""
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({"type": "ping"}))
    except Exception:
        pass


async def _dispatch(session, ws: WebSocket, msg: dict) -> None:
    """Handle one client → server message."""
    t = msg.get("type")
    try:
        if t == "move":
            uci = str(msg.get("uci", ""))
            ok = session.submit_move(uci)
            if not ok:
                await ws.send_text(json.dumps({"type": "error", "msg": f"illegal move: {uci}"}))
                return
            # Broadcast the new position to all subscribers.
            await _broadcast_position(session)

        elif t == "side":
            color = msg.get("color")
            session.set_side(color if color in ("white", "black") else None)
            await _broadcast_position(session)

        elif t == "style":
            style = msg.get("style", "balanced")
            if style in ("greedy", "balanced", "cautious"):
                session.set_style(style)
                await _broadcast_position(session)

        elif t == "elo":
            try:
                elo = int(msg.get("elo", 1500))
            except Exception:
                return
            session.set_elo(max(1000, min(2500, elo)))
            await _broadcast_position(session)

        elif t == "reset":
            fen = str(msg.get("fen", chess.STARTING_FEN))
            if session.reset_to(fen):
                await _broadcast_position(session)
            else:
                await ws.send_text(json.dumps({"type": "error", "msg": "invalid FEN"}))

        elif t == "pong":
            # heartbeat ack — no-op
            pass

        else:
            await ws.send_text(json.dumps({"type": "error", "msg": f"unknown type: {t}"}))
    except Exception as e:
        log.exception("dispatch error")
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass


async def _broadcast_position(session) -> None:
    """Broadcast the current position to all subscribers."""
    payload = {"type": "position", **session.position_snapshot()}
    msg = json.dumps(payload)
    # Reuse the session's broadcast machinery via direct send
    subs = list(session._subscribers)  # type: ignore[attr-defined]
    for s in subs:
        try:
            await s.send_text(msg)
        except Exception:
            session.remove_subscriber(s)
