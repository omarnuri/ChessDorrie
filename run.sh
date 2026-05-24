#!/usr/bin/env bash
# Launches the ChessDorrie web server.
# Activates the venv created by setup.sh if it exists.

set -euo pipefail
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Engine tuning — defaults are CPU-friendly. On Colab GPU + Lc0 bump:
#   CD_STREAM_MAX_DEPTH=40         (default 32; main multipv goes that deep)
#   CD_PREP_SHALLOW_DEPTH=16       (default 12; how deep to pick candidates)
#   CD_HYBRID_DEPTH=14             (default 10; sub-evals in hybrid mode)
#   CD_DEEP_DEPTH=18               (default 14; sub-evals in deep mode)
#   CD_SETUP_DEEP_DEPTH=20         (default 16; setup_deep alternating lookahead)
#   CD_POOL_SIZE=8                 (default 4; parallel sub-eval workers)
#   CD_POOL_THREADS=2              (default 1; threads per pool worker)
#   CHESS_ENGINE=lc0               (default auto; force Lc0+GPU as main engine)
# Legacy POST endpoint tuning:
export CD_THREADS="${CD_THREADS:-8}"
export CD_DEPTH="${CD_DEPTH:-18}"
export CD_SUB_DEPTH="${CD_SUB_DEPTH:-14}"
export CD_CANDIDATES="${CD_CANDIDATES:-10}"
export CD_REPLIES="${CD_REPLIES:-6}"
export CD_HOST="${CD_HOST:-0.0.0.0}"
export CD_PORT="${CD_PORT:-8000}"
export PYTHONPATH="${PYTHONPATH:-.}"

# Optional fine-tuned model: TRAP_MODEL=/path/to/checkpoint.pt
# (overrides Maia/Lc0 for the human-move predictor)
if [ -n "${TRAP_MODEL:-}" ]; then
    export TRAP_MODEL
fi

echo "==> ChessDorrie on http://${CD_HOST}:${CD_PORT}"
echo "    Stockfish: $(command -v stockfish || echo /usr/games/stockfish)"
echo "    Lc0:       $(command -v lc0 2>/dev/null || echo '(missing — softmax fallback)')"
echo "    Weights:   $(ls weights/maia-*.pb.gz 2>/dev/null | wc -l) Maia file(s)"
echo

exec python3 -m app.server
