"""Lichess monthly-dump trap miner.

Downloads (or streams) a Lichess monthly dump in PGN.zst format and
extracts positions where the **lower-rated** player won material in the
following ≤6 plies after introducing a tactical sequence (a capture
that gives back material). These are the positions that historically
trick humans.

The output is `data/mined_traps.json` in the schema understood by
`troll_engine.trap_db`.

This is a long-running batch job — a single month of Lichess Standard
games is on the order of 10 GB compressed. Run on Colab with a fast
disk, or pass `--max-games` to cap.

Usage
-----
    python -m data.mine_lichess --month 2024-09 --max-games 500000

If `python-zstandard` is not available, decompress with:

    zstd -d lichess_db_standard_rated_2024-09.pgn.zst -o games.pgn

and pass `--pgn games.pgn` directly.

Dependencies (optional, will skip if missing):
    pip install zstandard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import IO, Iterator

try:
    import zstandard as zstd  # type: ignore
except ImportError:
    zstd = None

import chess
import chess.pgn
import requests

from troll_engine.engine import material_balance, PIECE_CP
from troll_engine.trap_db import fen_to_match_key


LICHESS_URL_TEMPLATE = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)

# A "trap" candidate: position from which (within MIN_TRAP_HORIZON plies),
# the higher-rated side LOST ≥ TRAP_MATERIAL_THRESHOLD cp of material to
# the lower-rated side.
MIN_RATING_GAP = 100         # Elo gap between trap-setter (low) and victim (high)
TRAP_MATERIAL_THRESHOLD = 200  # cp gained by the lower-rated side
MIN_TRAP_HORIZON = 1         # plies until the swing
MAX_TRAP_HORIZON = 6
MIN_OBSERVATIONS = 5         # require this many independent games to keep a trap


def _open_dump(path: str) -> IO[bytes]:
    """Open a .pgn or .pgn.zst file for binary reading."""
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; can't decompress .zst")
        f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(f)
    return open(path, "rb")


def _download(month: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    url = LICHESS_URL_TEMPLATE.format(month=month)
    dest = os.path.join(target_dir, f"lichess_db_standard_rated_{month}.pgn.zst")
    if os.path.exists(dest):
        print(f"[mine_lichess] already have {dest}")
        return dest
    print(f"[mine_lichess] downloading {url} → {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def _iter_games(stream: IO[bytes]) -> Iterator[chess.pgn.Game]:
    """Yield games from a binary PGN stream. The python-chess PGN parser
    is text-based, so we wrap the binary stream in a text wrapper."""
    import io
    text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    while True:
        g = chess.pgn.read_game(text)
        if g is None:
            return
        yield g


def _scan_game(game: chess.pgn.Game) -> Iterator[tuple[str, str, str, int]]:
    """Yield (fen_key, trap_move_uci, trap_move_san, victim_elo) for
    every trap-like swing in `game`.

    Heuristic:
      * The side that just moved is the "setter".
      * The setter must be ≥ MIN_RATING_GAP Elo below the opponent.
      * Within MAX_TRAP_HORIZON plies, the setter's material balance
        must improve by ≥ TRAP_MATERIAL_THRESHOLD cp from this position.
    """
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", "0") or 0)
        black_elo = int(headers.get("BlackElo", "0") or 0)
    except ValueError:
        return
    if min(white_elo, black_elo) < 1000 or max(white_elo, black_elo) < 1400:
        return

    board = game.board()
    moves = list(game.mainline_moves())
    for idx, move in enumerate(moves):
        setter = board.turn
        setter_elo = white_elo if setter == chess.WHITE else black_elo
        victim_elo = black_elo if setter == chess.WHITE else white_elo
        if victim_elo - setter_elo < MIN_RATING_GAP:
            board.push(move)
            continue

        fen_key = fen_to_match_key(board.fen())
        move_uci = move.uci()
        try:
            move_san = board.san(move)
        except Exception:
            move_san = move_uci

        # Track setter's material from THIS ply onward, peek ahead.
        mat_start = material_balance(board, setter)
        peek = board.copy()
        peek.push(move)

        best_gain = 0
        for j in range(min(MAX_TRAP_HORIZON, len(moves) - idx - 1)):
            peek.push(moves[idx + 1 + j])
            gain = material_balance(peek, setter) - mat_start
            best_gain = max(best_gain, gain)

        if best_gain >= TRAP_MATERIAL_THRESHOLD:
            yield (fen_key, move_uci, move_san, victim_elo)

        board.push(move)


def mine(pgn_path: str, output_path: str, max_games: int | None = None) -> None:
    counts: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"san": "", "wins": 0, "n": 0, "victim_elo_sum": 0}
    )

    with _open_dump(pgn_path) as stream:
        for n, game in enumerate(_iter_games(stream), 1):
            for fen_key, move_uci, move_san, victim_elo in _scan_game(game):
                entry = counts[(fen_key, move_uci)]
                entry["san"] = move_san
                entry["wins"] += 1
                entry["n"] += 1
                entry["victim_elo_sum"] += victim_elo
            if n % 5000 == 0:
                print(f"[mine_lichess] scanned {n} games, {len(counts)} trap candidates", file=sys.stderr)
            if max_games and n >= max_games:
                break

    # Filter & convert.
    out = []
    for (fen_key, move_uci), e in counts.items():
        if e["n"] < MIN_OBSERVATIONS:
            continue
        out.append({
            "fen_key": fen_key,
            "move_uci": move_uci,
            "move_san": e["san"],
            "name": "mined",
            "win_rate": e["wins"] / e["n"],  # always 1.0 by construction; useful when we add losses
            "sample_size": e["n"],
            "source": os.path.basename(pgn_path),
            "follow_up": [],
            "avg_victim_elo": e["victim_elo_sum"] / e["n"],
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mine_lichess] wrote {len(out)} traps → {output_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="Lichess month, e.g. 2024-09")
    ap.add_argument("--pgn", help="path to already-downloaded pgn(.zst) file")
    ap.add_argument("--out", default="data/mined_traps.json")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--download-dir", default="data/dumps")
    args = ap.parse_args()

    if args.pgn:
        pgn_path = args.pgn
    elif args.month:
        pgn_path = _download(args.month, args.download_dir)
    else:
        ap.error("pass --month or --pgn")

    mine(pgn_path, args.out, args.max_games)


if __name__ == "__main__":
    main()
