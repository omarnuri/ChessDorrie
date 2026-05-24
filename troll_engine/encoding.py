"""Board → tensor + move ↔ index encoders, shared between the
tensorizer (`scripts/dataset_to_tensors.py`), the converter
(`scripts/convert_maia_to_torch.py`), and the inference wrapper
(`troll_engine/trap_model.py`).

This is one source of truth for the shape contract:

  * Boards are encoded as `(20, 8, 8)` uint8 (channels-first, square
    a1 → index 0 on the H axis, square h8 → 63), matching the
    Maia / Lc0 input convention.
  * Moves are encoded as a single integer in `[0, 4672)`, matching
    Lc0's standard policy head. The 4672 number comes from
    `64 squares × 73 move planes`: 56 queen-like rays + 8 knight
    jumps + 9 underpromotions per origin.

Why this exact layout: we want any model trained against these
tensors to be drop-in compatible with Maia, so warm-starting from
the open Maia weights is a state-dict copy (in
`scripts/convert_maia_to_torch.py`), not a layer-by-layer translation.

The Lc0 move-plane mapping is documented at
<https://lczero.org/dev/wiki/training-network/#policy-output> and
implemented faithfully here.
"""

from __future__ import annotations

import numpy as np
import chess


# --------------------------------------------------------------------- #
# Board encoder                                                         #
# --------------------------------------------------------------------- #

# Channel index ranges
_PIECE_CHANNELS_WHITE = {  # offsets within the 12-channel piece block
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}
_PIECE_CHANNELS_BLACK = {
    chess.PAWN:   6,
    chess.KNIGHT: 7,
    chess.BISHOP: 8,
    chess.ROOK:   9,
    chess.QUEEN:  10,
    chess.KING:   11,
}
# 12: side-to-move (uniform 1.0 if white to move else 0.0)
# 13-16: castling rights KQkq
# 17: en-passant target square (one-hot)
# 18: halfmove clock / 100  (normalised)
# 19: fullmove number / 200 (normalised, soft-clipped)
NUM_CHANNELS = 20
BOARD_SHAPE = (NUM_CHANNELS, 8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Return a `(20, 8, 8)` uint8 tensor for `board`.

    All channels live in {0, 1, 2, ..., 255}. The two normalised
    channels (clock + move number) are quantised to that range.
    """
    out = np.zeros(BOARD_SHAPE, dtype=np.uint8)

    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if piece.color == chess.WHITE:
            ch = _PIECE_CHANNELS_WHITE[piece.piece_type]
        else:
            ch = _PIECE_CHANNELS_BLACK[piece.piece_type]
        out[ch, rank, file] = 1

    if board.turn == chess.WHITE:
        out[12].fill(1)

    cr = board.castling_rights
    if cr & chess.BB_H1: out[13].fill(1)   # white kingside
    if cr & chess.BB_A1: out[14].fill(1)   # white queenside
    if cr & chess.BB_H8: out[15].fill(1)   # black kingside
    if cr & chess.BB_A8: out[16].fill(1)   # black queenside

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        out[17, rank, file] = 1

    out[18].fill(min(255, int(board.halfmove_clock * 2.55)))  # ~0..100 → 0..255
    out[19].fill(min(255, int(board.fullmove_number * 1.275)))  # ~0..200 → 0..255

    return out


# --------------------------------------------------------------------- #
# Move ↔ Lc0 policy index                                               #
# --------------------------------------------------------------------- #

# 73 planes per origin square:
#   0..55  : "queen-like" rays — 8 directions × 7 distances
#   56..63 : 8 knight jumps
#   64..72 : 9 underpromotions (3 directions × 3 piece types: knight/bishop/rook)
#            (queen-promotion is encoded as the matching queen-ray move)
#
# Directions in canonical Lc0 order (queen rays):
#   N, NE, E, SE, S, SW, W, NW

_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Underpromotions are encoded for pawns reaching the last rank with
# direction "straight / capture-left / capture-right" × piece N/B/R.
_UNDERPROMO_DIRS = [(-1, 0), (0, 0), (1, 0)]   # file delta: capture-left / straight / capture-right
_UNDERPROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

POLICY_SIZE = 64 * 73  # 4672


def _square_rc(sq: int) -> tuple[int, int]:
    return (chess.square_rank(sq), chess.square_file(sq))


def move_to_index(move: chess.Move, perspective: bool = chess.WHITE) -> int:
    """Encode `move` into `[0, 4672)`.

    `perspective` is the side-to-move; Maia / Lc0 always look at the
    board from the side-to-move's POV (board is flipped vertically for
    black). The move encoding follows the same convention: ranks are
    mirrored when `perspective == chess.BLACK`.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    fr_r, fr_f = _square_rc(from_sq)
    to_r, to_f = _square_rc(to_sq)
    if perspective == chess.BLACK:
        fr_r = 7 - fr_r
        to_r = 7 - to_r

    dr = to_r - fr_r
    df = to_f - fr_f

    plane = None

    # Underpromotion — pawn moves to last rank with non-queen promotion
    if move.promotion and move.promotion != chess.QUEEN:
        # df ∈ {-1, 0, 1}; piece ∈ {N, B, R}
        try:
            dir_idx = _UNDERPROMO_DIRS.index((df, 0))
        except ValueError:
            dir_idx = 1  # straight fallback (shouldn't happen on legal moves)
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + dir_idx * 3 + piece_idx

    elif (dr, df) in _KNIGHT_OFFSETS:
        plane = 56 + _KNIGHT_OFFSETS.index((dr, df))

    else:
        # Queen-like ray
        if dr == 0 and df == 0:
            raise ValueError(f"null move {move}")
        # Reduce to unit direction + distance
        if dr == 0:
            dist = abs(df)
            unit = (0, 1 if df > 0 else -1)
        elif df == 0:
            dist = abs(dr)
            unit = (1 if dr > 0 else -1, 0)
        elif abs(dr) == abs(df):
            dist = abs(dr)
            unit = (1 if dr > 0 else -1, 1 if df > 0 else -1)
        else:
            raise ValueError(f"non-queen-like move {move} ({dr},{df})")
        dir_idx = _QUEEN_DIRS.index(unit)
        plane = dir_idx * 7 + (dist - 1)

    # Origin square is from the perspective POV too.
    from_r_p = fr_r
    from_f_p = fr_f
    origin = from_r_p * 8 + from_f_p
    return plane * 64 + origin


def index_to_move(idx: int, board: chess.Board) -> chess.Move | None:
    """Reverse `move_to_index`. Returns the legal move at `idx`, or
    None if the index does not decode to a legal move on `board`.

    Generally callers should mask logits by legal moves first
    (`legal_move_mask`) rather than calling this function repeatedly.
    """
    plane = idx // 64
    origin = idx % 64
    perspective = board.turn
    fr_r = origin // 8
    fr_f = origin % 8

    if plane < 56:
        # Queen ray
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        unit = _QUEEN_DIRS[dir_idx]
        to_r = fr_r + unit[0] * dist
        to_f = fr_f + unit[1] * dist
    elif plane < 64:
        # Knight
        offs = _KNIGHT_OFFSETS[plane - 56]
        to_r = fr_r + offs[0]
        to_f = fr_f + offs[1]
    else:
        # Underpromotion
        p = plane - 64
        dir_idx = p // 3
        piece_idx = p % 3
        df = _UNDERPROMO_DIRS[dir_idx][0]
        to_r = fr_r + (1 if perspective == chess.WHITE else -1) * 1  # ignored: pawn moves to last rank
        # Actually the underpromotion target rank is fixed by colour.
        # For white promotion, fr_r is rank 6 → to_r is rank 7.
        # For black (perspective-flipped), fr_r is "rank 6 from black" → original rank 1.
        # We just push one rank forward in perspective-flipped coords:
        to_r = fr_r + 1
        to_f = fr_f + df
        # Reverse perspective flip:
        if perspective == chess.BLACK:
            from_sq = chess.square(fr_f, 7 - fr_r)
            to_sq = chess.square(to_f, 7 - to_r)
        else:
            from_sq = chess.square(fr_f, fr_r)
            to_sq = chess.square(to_f, to_r)
        promotion = _UNDERPROMO_PIECES[piece_idx]
        m = chess.Move(from_sq, to_sq, promotion=promotion)
        return m if m in board.legal_moves else None

    if not (0 <= to_r < 8 and 0 <= to_f < 8):
        return None

    # Reverse perspective flip for queen / knight planes
    if perspective == chess.BLACK:
        from_sq = chess.square(fr_f, 7 - fr_r)
        to_sq = chess.square(to_f, 7 - to_r)
    else:
        from_sq = chess.square(fr_f, fr_r)
        to_sq = chess.square(to_f, to_r)

    # Auto-queen promotion for pawn-to-last-rank moves via queen rays
    moving = board.piece_at(from_sq)
    if (moving is not None and moving.piece_type == chess.PAWN
        and chess.square_rank(to_sq) in (0, 7)):
        m = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    else:
        m = chess.Move(from_sq, to_sq)
    return m if m in board.legal_moves else None


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a `(4672,)` float32 mask where legal moves are 0 and
    illegal moves are `-inf`. Add to logits before softmax."""
    mask = np.full(POLICY_SIZE, -np.inf, dtype=np.float32)
    perspective = board.turn
    for mv in board.legal_moves:
        try:
            idx = move_to_index(mv, perspective)
            mask[idx] = 0.0
        except ValueError:
            continue
    return mask
