"""Eyeball-the-output demo.

Run a couple of famous tactical positions through the analyser and dump
the candidate moves with all their metrics. Use this to sanity-check
that the troll utility is doing something interesting.

Limitations: with the SoftmaxStockfish fallback (no real Maia weights)
the bot can't model opponents who play obviously bad moves Stockfish
would never suggest. So traps like Legal's mate — which rely on the
human grabbing the queen — won't fully light up until Maia is wired in.
"""

import sys

from troll_engine import analyse_position


POSITIONS = [
    (
        "Starting position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        # After 1.e4 e5 2.Nf3 d6 3.Bc4 Bg4 4.Nc3 g6 — white to play Nxe5!
        "Legal's mate setup (white to move 5.Nxe5!?)",
        "rn1qkbnr/ppp2p1p/3p2p1/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 5",
    ),
    (
        # White is up material; black to play looking for swindle.
        "Endgame: white up exchange, black to swindle",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
    ),
]


def format_replies(c):
    parts = []
    for r in c.replies[:3]:
        parts.append(f"{r.move_san}({r.probability*100:.0f}% → {r.eval_after/100:+.2f})")
    return ", ".join(parts)


def main():
    for title, fen in POSITIONS:
        print()
        print("=" * 78)
        print(title)
        print(f"FEN: {fen}")
        try:
            res = analyse_position(
                fen, elo=1500,
                candidate_count=6, candidate_depth=14,
                reply_count=5, subposition_depth=10,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print(f"Side to move: {res.side_to_move}")
        print(f"Stockfish best: {res.objective_best_uci}")
        print(f"Troll best:     {res.troll_best_uci}")
        print(f"Elapsed: {res.elapsed_ms} ms")
        print()
        print(f"{'#':>2} {'move':>6} {'rank':>5} "
              f"{'obj':>7} {'troll':>7} {'exp':>7} {'worst':>7} "
              f"{'mat':>6} {'anger':>6} {'sac':>5} notes")
        for i, c in enumerate(res.candidates[:6], 1):
            print(f"{i:>2} {c.move_san:>6} {c.objective_rank:>5} "
                  f"{c.objective_eval/100:>+7.2f} "
                  f"{c.troll_score/100:>+7.2f} "
                  f"{c.expected_eval/100:>+7.2f} "
                  f"{c.worst_case_eval/100:>+7.2f} "
                  f"{c.expected_material/100:>+6.2f} "
                  f"{c.anger_probability*100:>5.0f}% "
                  f"{('Y' if c.is_sacrifice else '-'):>5} "
                  f"{'; '.join(c.notes)[:50]}")
            if c.replies:
                print(f"     replies: {format_replies(c)}")


if __name__ == "__main__":
    main()
