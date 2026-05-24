#!/usr/bin/env bash
# Download a strong-play Lc0 neural network.
#
# We support three options, in descending order of strength + size:
#   * BT4 (~600 MB) — biggest transformer, top current strength
#   * BT3 (~280 MB) — strong transformer, recommended default
#   * T82 ( ~50 MB) — smaller test net, still ~3500 Elo
#
# Maia weights (used for the human-move predictor) are downloaded
# separately by scripts/colab_setup.sh — they're tiny (~11 MB) and
# tuned for human prediction, not strength.
#
# Usage:
#   bash scripts/download_lc0_net.sh           # downloads BT3 (recommended)
#   bash scripts/download_lc0_net.sh BT4
#   bash scripts/download_lc0_net.sh T82
#   bash scripts/download_lc0_net.sh maia-2000 # smaller Maia for opponent modelling

set -euo pipefail
cd "$(dirname "$0")/.."

NET="${1:-BT3}"
DEST_DIR="weights/lc0"
mkdir -p "$DEST_DIR"

# Lc0's training repo + lczero.org publish releases under a few URLs;
# the canonical ones move occasionally. We pin known-good URLs here
# and fall back to a discovery hop.
case "$NET" in
    BT3)
        URL="https://storage.lczero.org/files/networks-contrib/BT3-768x15smolgen-12h-do-swa-onnx-1500000.pb.gz"
        OUT="$DEST_DIR/BT3.pb.gz"
        ;;
    BT4)
        URL="https://storage.lczero.org/files/networks-contrib/BT4-1024x15smolgen-128h-swa-3000000.pb.gz"
        OUT="$DEST_DIR/BT4.pb.gz"
        ;;
    T82)
        URL="https://training.lczero.org/get_network?sha=t82-768x15x24h-distill-swa-3850000.pb.gz"
        OUT="$DEST_DIR/T82.pb.gz"
        ;;
    T80)
        URL="https://training.lczero.org/get_network?sha=t80-distilled-swa-3170000.pb.gz"
        OUT="$DEST_DIR/T80.pb.gz"
        ;;
    *)
        echo "unknown net: $NET (choices: BT3 BT4 T82 T80)" >&2
        exit 1
        ;;
esac

if [ -s "$OUT" ]; then
    echo "Already have $OUT ($(du -h "$OUT" | cut -f1))"
    exit 0
fi

echo "Downloading $NET from $URL"
echo "  → $OUT"
echo "  (this is ~50-600 MB depending on net; takes 1-5 min on Colab)"

if curl -fL --max-time 600 --retry 3 -C - "$URL" -o "$OUT"; then
    echo "OK — $OUT ($(du -h "$OUT" | cut -f1))"
    echo
    echo "To use it, launch with:"
    echo "    CHESS_ENGINE=lc0 LC0_WEIGHTS=$OUT bash run.sh"
    echo "or just leave CHESS_ENGINE=auto and the engine picks it up."
else
    echo "FAILED. Try a different net (BT3/BT4/T82/T80) or check the URL." >&2
    rm -f "$OUT"
    exit 1
fi
