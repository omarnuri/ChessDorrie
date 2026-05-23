#!/usr/bin/env bash
# Local install + first-run for ChessDorrie.
# Detects your OS and GPU, installs Stockfish + Lc0 with the right
# backend, downloads Maia weights, sets up a venv.
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

OS=$(uname -s)
HAVE_NVIDIA=$(command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

echo "==> Detected: $OS  |  NVIDIA GPU: $([ "$HAVE_NVIDIA" = "1" ] && echo yes || echo no)"

# --- Stockfish -------------------------------------------------------- #
echo "==> Stockfish..."
if ! command -v stockfish >/dev/null && ! [ -x /usr/games/stockfish ]; then
    case "$OS" in
        Linux)  sudo apt-get update -qq && sudo apt-get install -y stockfish ;;
        Darwin) brew install stockfish ;;
        *)      echo "  ! please install Stockfish manually for $OS"; exit 1 ;;
    esac
fi
(command -v stockfish || echo /usr/games/stockfish) | head -1

# --- Lc0 -------------------------------------------------------------- #
echo "==> Lc0 (Maia inference)..."
if ! command -v lc0 >/dev/null; then
    case "$OS" in
        Linux)
            if [ "$HAVE_NVIDIA" = "1" ]; then
                echo "  ! NVIDIA GPU detected — apt's Lc0 is CPU-only."
                echo "    Grab the CUDA build from:"
                echo "      https://github.com/LeelaChessZero/lc0/releases/latest"
                echo "    Pick the *linux-cuda* asset, extract to ~/lc0/, then:"
                echo "      sudo ln -s \$HOME/lc0/lc0 /usr/local/bin/lc0"
                echo "    (skipping apt install — would clobber GPU build)"
            else
                sudo apt-get install -y lc0 2>&1 | tail -2 || \
                    echo "  ! lc0 not in your apt sources — install manually"
            fi
            ;;
        Darwin)
            brew install lc0 || echo "  ! brew install failed — try building from source"
            ;;
    esac
fi
command -v lc0 && lc0 --help 2>&1 | head -1 || echo "  (no lc0 — softmax fallback will be used)"

# --- Python venv + deps ---------------------------------------------- #
echo "==> Python deps..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt

# --- Maia weights ---------------------------------------------------- #
echo "==> Maia weights (1100, 1500, 1900)..."
mkdir -p weights
for elo in 1100 1500 1900; do
    f="weights/maia-${elo}.pb.gz"
    if [ ! -f "$f" ]; then
        url="https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-${elo}.pb.gz"
        echo "  -> $f"
        curl -fsSL "$url" -o "$f" || echo "  ! failed: $url"
    else
        echo "  ✓ have $f"
    fi
done

# --- GPU backend hint ------------------------------------------------ #
if command -v lc0 >/dev/null; then
    if [ "$OS" = "Darwin" ]; then
        echo "==> Mac detected — Lc0 will use Metal backend automatically."
    elif [ "$HAVE_NVIDIA" = "1" ]; then
        echo "==> NVIDIA detected — Lc0 should pick up CUDA backend. Test with:"
        echo "      lc0 --backend=cuda-fp16  (or cuda-auto)"
    fi
fi

cat <<EOF

✓ Setup complete.

To launch:
    ./run.sh

Then open http://localhost:8000

EOF
