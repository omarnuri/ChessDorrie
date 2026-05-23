"""Smoke tests: does the pipeline run end-to-end and produce sensible numbers?

These do not assert specific moves — engine output varies with version
and parameters. They just check structural invariants.
"""

import chess

from troll_engine import analyse_position
from troll_engine.engine import Engine
from troll_engine.human_model import SoftmaxStockfishModel


def test_starting_position_runs():
    res = analyse_position(chess.STARTING_FEN, elo=1500,
                           candidate_depth=10, subposition_depth=8)
    assert res.candidates, "expected at least one candidate"
    assert res.objective_best_uci, "expected an objective best move"
    assert res.troll_best_uci, "expected a troll best move"
    for c in res.candidates:
        assert c.move_uci
        assert c.move_san
        # Replies may be empty if mate; on starting pos they should be present.
        assert c.replies, "expected predicted replies for non-terminal position"
        # Probability of replies sums to ~1.
        total = sum(r.probability for r in c.replies)
        assert abs(total - 1.0) < 0.05, f"reply probs sum to {total}"


def test_softmax_model_returns_distribution():
    with Engine() as e:
        m = SoftmaxStockfishModel(e, elo=1500, depth=8, top_k_search=5)
        b = chess.Board()
        preds = m.predict(b, top_k=5)
        assert len(preds) > 0
        assert abs(sum(p.probability for p in preds) - 1.0) < 1e-6
        # Temperature is moderate at 1500, so top move shouldn't be > 90%.
        assert preds[0].probability < 0.95


if __name__ == "__main__":
    test_softmax_model_returns_distribution()
    print("✓ softmax model works")
    test_starting_position_runs()
    print("✓ end-to-end analysis works")
