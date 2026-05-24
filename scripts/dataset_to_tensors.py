"""PGN → (board tensor, move index) shards on disk.

Streams a PGN file (plain or .pgn.zst), encodes each
`(position before move, move played)` pair using
`troll_engine.encoding`, and writes them in fixed-size shards to a
directory. The shards are the input format expected by
`scripts/train_trap_policy.py`.

Schema per shard (`train_<NNNNN>.pt` or `val_<NNNNN>.pt`):

    {
      "boards": torch.uint8  tensor of shape (N, 20, 8, 8),
      "moves":  torch.int32  tensor of shape (N,),
    }

Memory: peak ~ shard_size × 20 × 8 × 8 bytes ≈ 25 MB at shard_size=200k.
The PGN itself is streamed game-by-game so files of any size are fine.

Usage
-----
    python scripts/dataset_to_tensors.py \
        --pgn data/trap_games.pgn \
        --out data/tensors/ \
        --max-positions 5000000 \
        --shard-size 200000 \
        --val-frac 0.02

Time
----
~70 k positions per minute on a single CPU core. A typical fine-tune
run on 5 M positions therefore costs ~75 min of tensorisation —
substantially less than training itself. Adding `--workers N` for
multiprocessing is a follow-up.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import chess
import chess.pgn

try:
    import zstandard as zstd  # type: ignore
except ImportError:
    zstd = None

from troll_engine.encoding import encode_board, move_to_index, NUM_CHANNELS


def _open(path: str):
    """Open a .pgn or .pgn.zst file and return a text stream."""
    if path.endswith(".zst"):
        if zstd is None:
            raise SystemExit("install zstandard: pip install zstandard")
        raw = zstd.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _flush_shard(out_dir: Path, name: str, idx: int,
                  boards: list[np.ndarray], moves: list[int]) -> None:
    """Stack and save a shard. Returns nothing; clears the input lists
    via the caller (we don't reach in here)."""
    import torch
    arr_boards = np.stack(boards, axis=0)             # (N, 20, 8, 8) uint8
    arr_moves = np.array(moves, dtype=np.int32)       # (N,) int32
    path = out_dir / f"{name}_{idx:05d}.pt"
    torch.save(
        {
            "boards": torch.from_numpy(arr_boards),
            "moves": torch.from_numpy(arr_moves),
        },
        path,
    )
    print(f"  wrote {path} ({len(moves)} positions)", file=sys.stderr)


def tensorize(pgn_path: str, out_dir: str, max_positions: int,
              shard_size: int, val_frac: float) -> None:
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0xCDC0DE)
    stream = _open(pgn_path)

    train_buf_b: list[np.ndarray] = []
    train_buf_m: list[int] = []
    val_buf_b: list[np.ndarray] = []
    val_buf_m: list[int] = []
    train_shard_idx = 0
    val_shard_idx = 0

    total = 0
    games = 0
    while total < max_positions:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games += 1
        board = game.board()
        for move in game.mainline_moves():
            try:
                idx = move_to_index(move, board.turn)
            except Exception:
                # Skip moves we can't encode (very rare on legal PGN).
                board.push(move)
                continue
            tensor = encode_board(board)
            if rng.random() < val_frac:
                val_buf_b.append(tensor)
                val_buf_m.append(idx)
            else:
                train_buf_b.append(tensor)
                train_buf_m.append(idx)
            board.push(move)
            total += 1
            if total >= max_positions:
                break

            # Flush shards when full
            if len(train_buf_m) >= shard_size:
                _flush_shard(od, "train", train_shard_idx,
                             train_buf_b, train_buf_m)
                train_shard_idx += 1
                train_buf_b.clear(); train_buf_m.clear()
            if len(val_buf_m) >= shard_size:
                _flush_shard(od, "val", val_shard_idx,
                             val_buf_b, val_buf_m)
                val_shard_idx += 1
                val_buf_b.clear(); val_buf_m.clear()

        if games % 1000 == 0:
            print(f"[tensorize] games={games}, positions={total}", file=sys.stderr)

    # Flush the residuals
    if train_buf_m:
        _flush_shard(od, "train", train_shard_idx,
                     train_buf_b, train_buf_m)
    if val_buf_m:
        _flush_shard(od, "val", val_shard_idx, val_buf_b, val_buf_m)

    print(f"[tensorize] done: scanned {games} games, wrote {total} positions, "
          f"{train_shard_idx + 1} train shards, "
          f"{val_shard_idx + 1 if val_buf_m else val_shard_idx} val shards",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="input PGN (plain or .pgn.zst)")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--max-positions", type=int, default=10_000_000)
    ap.add_argument("--shard-size", type=int, default=200_000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is required: pip install torch")

    tensorize(args.pgn, args.out, args.max_positions,
              args.shard_size, args.val_frac)


if __name__ == "__main__":
    main()
