"""Stream a Lichess monthly PGN and keep only games that decided via
a sacrifice — that's our "Tal-style" subset for fine-tuning a trap
policy network.

Heuristic: scan each game's PGN-extracted move sequence with a
shallow internal material counter. A game is KEPT if:

  * Result is decisive (1-0 or 0-1)
  * The eventual WINNER (lower-rated or otherwise) gave up at least
    one minor piece (≥ 300 cp) at some point AND was at least 200 cp
    DOWN in material at that moment, AND ended up winning anyway.

The first time a "behind by a minor piece" moment occurs we mark the
game and emit it. The output is a single PGN file the next stage can
tensorize.

Run
---
    python scripts/dataset_filter_traps.py \
        --pgn data/dumps/lichess_2024-09.pgn.zst \
        --out data/trap_games.pgn \
        --max-games 5000000

Expect 0.5-3% of input games to survive depending on rating mix.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import IO

import chess
import chess.pgn

try:
    import zstandard as zstd  # type: ignore
except ImportError:
    zstd = None


_PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def _material(board: chess.Board, color: bool) -> int:
    total = 0
    for piece_type, val in _PIECE_CP.items():
        total += val * len(board.pieces(piece_type, color))
        total -= val * len(board.pieces(piece_type, not color))
    return total


def _open(path: str) -> IO[bytes]:
    if path.endswith(".zst"):
        if zstd is None:
            raise SystemExit("install zstandard: pip install zstandard")
        return zstd.ZstdDecompressor().stream_reader(open(path, "rb"))
    return open(path, "rb")


def _is_trap_game(game: chess.pgn.Game, min_material_deficit: int = 200) -> bool:
    result = game.headers.get("Result", "*")
    if result not in ("1-0", "0-1"):
        return False
    winner = chess.WHITE if result == "1-0" else chess.BLACK
    board = game.board()
    saw_deficit = False
    for move in game.mainline_moves():
        board.push(move)
        # From the winner's POV
        bal = _material(board, winner)
        if bal <= -min_material_deficit:
            saw_deficit = True
            break  # stop early — we know
    return saw_deficit


def filter_games(in_path: str, out_path: str, max_games: int) -> None:
    f_in = _open(in_path)
    text_in = io.TextIOWrapper(f_in, encoding="utf-8", errors="replace")
    kept = 0
    seen = 0
    with open(out_path, "w") as f_out:
        exporter = chess.pgn.FileExporter(f_out)
        while seen < max_games:
            game = chess.pgn.read_game(text_in)
            if game is None:
                break
            seen += 1
            try:
                if _is_trap_game(game):
                    game.accept(exporter)
                    kept += 1
            except Exception:
                continue
            if seen % 10000 == 0:
                print(f"[filter] scanned {seen}, kept {kept}", file=sys.stderr)
    print(f"[filter] done: scanned {seen}, kept {kept} ({kept/max(1,seen)*100:.1f}%)",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=10_000_000)
    args = ap.parse_args()
    filter_games(args.pgn, args.out, args.max_games)


if __name__ == "__main__":
    main()
