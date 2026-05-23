#!/usr/bin/env bash
# Local install script for ChessDorrie.
# Idempotent — safe to run repeatedly.

set -euo pipefail

echo "==> Installing Stockfish..."
if ! command -v stockfish >/dev/null && ! [ -x /usr/games/stockfish ]; then
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y stockfish
    elif command -v brew >/dev/null; then
        brew install stockfish
    else
        echo "  ! Please install stockfish manually."
        exit 1
    fi
fi

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

mkdir -p weights

echo "==> Downloading Maia weights (1100, 1500, 1900)..."
for elo in 1100 1500 1900; do
    f="weights/maia-${elo}.pb.gz"
    if [ ! -f "$f" ]; then
        url="https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-${elo}.pb.gz"
        echo "  -> ${f}"
        curl -fL "$url" -o "$f" || echo "  ! failed to download $url (network policy?)"
    fi
done

echo "==> Installing Lc0 (for Maia inference)..."
if ! command -v lc0 >/dev/null; then
    if command -v apt-get >/dev/null; then
        sudo apt-get install -y lc0 || echo "  ! lc0 not in apt — install manually if you want GPU Maia."
    fi
fi

echo "Done. Launch the server with:  python -m app.server"
