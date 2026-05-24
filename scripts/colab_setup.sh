#!/usr/bin/env bash
# One-shot Colab installer for ChessDorrie.
#
# Bundles everything the user needs to go from a fresh Colab kernel to
# "ready to ponder or train" in a single Run-all cell:
#
#   * apt deps (stockfish, zstd, curl)
#   * Python deps (requirements.txt + torch + zstandard)
#   * Lc0 with the CUDA backend, via scripts/install_lc0_cuda.sh
#   * Maia weights (1100 / 1500 / 1900)
#   * (optional) one Lichess monthly dump for training, ~28 GB
#   * smoke verification that prints "ALL READY" or a precise diagnostic
#
# Usage:
#   bash scripts/colab_setup.sh                     # inference-only setup
#   bash scripts/colab_setup.sh --with-data 2024-09 # also pull the training dump
#   bash scripts/colab_setup.sh --skip-lc0          # skip Lc0 install (CPU-only)
#   bash scripts/colab_setup.sh --skip-weights      # skip Maia weights download
#
# Idempotent: re-running picks up where the last run left off (skips
# things already installed/downloaded).

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ------------- argument parsing -------------
WITH_DATA=""
SKIP_LC0=0
SKIP_WEIGHTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --with-data) WITH_DATA="$2"; shift 2;;
        --with-data=*) WITH_DATA="${1#*=}"; shift;;
        --skip-lc0) SKIP_LC0=1; shift;;
        --skip-weights) SKIP_WEIGHTS=1; shift;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 1;;
    esac
done

say() { printf '\n[colab_setup] %s\n' "$*"; }
ok() { printf '  ✓ %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
die() { printf '\n[colab_setup] ABORT: %s\n' "$*" >&2; exit 1; }

# Use sudo only if we don't already have root (Colab kernels run as root).
if [ "$(id -u)" = "0" ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# ------------- 1. apt deps -------------
say "1/6 system packages..."
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq 2>&1 | tail -2 || warn "apt-get update had warnings"
    $SUDO apt-get install -y -qq stockfish zstd curl ca-certificates 2>&1 | tail -2 \
        || die "apt-get install failed"
    ok "stockfish: $(command -v stockfish || echo /usr/games/stockfish)"
else
    warn "apt-get not available — assuming you installed stockfish/zstd manually"
fi

# ------------- 2. Python deps -------------
say "2/6 python packages..."
PIP_FLAGS="--quiet"
# Colab kernels already have torch; outside of Colab fall back to break-system.
if [ -n "${COLAB_GPU:-}${COLAB_RELEASE_TAG:-}" ]; then
    PIP_INSTALL="pip install $PIP_FLAGS"
else
    PIP_INSTALL="pip install --break-system-packages $PIP_FLAGS"
fi

$PIP_INSTALL -r requirements.txt 2>&1 | tail -3 || die "requirements.txt install failed"
$PIP_INSTALL zstandard websockets pytest httpx 2>&1 | tail -2 || warn "extras install had warnings"

# torch is heavy; skip if already importable
if ! python3 -c "import torch" 2>/dev/null; then
    say "  installing torch (this can take a few minutes)..."
    $PIP_INSTALL torch 2>&1 | tail -3 || warn "torch install failed (training will fail; inference still works)"
fi

if python3 -c "import torch" 2>/dev/null; then
    ok "torch: $(python3 -c 'import torch; print(torch.__version__)')"
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        ok "cuda: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
    else
        warn "torch installed but no CUDA — training will be slow"
    fi
else
    warn "torch missing — training scripts won't run"
fi

# ------------- 3. Lc0 CUDA -------------
say "3/6 lc0 (Maia inference backend)..."
if [ "$SKIP_LC0" = 1 ]; then
    warn "--skip-lc0 set; using softmax-Stockfish fallback for the human model"
elif command -v lc0 >/dev/null 2>&1 && lc0 --help 2>&1 | grep -qi cuda; then
    ok "lc0 (CUDA) already installed at $(command -v lc0)"
elif command -v nvidia-smi >/dev/null 2>&1; then
    bash scripts/install_lc0_cuda.sh 2>&1 | tail -8
    if command -v lc0 >/dev/null 2>&1; then
        ok "lc0 ready: $(command -v lc0)"
    else
        warn "lc0 install failed; falling back to softmax-Stockfish"
    fi
else
    warn "no NVIDIA GPU detected — skipping lc0 (softmax fallback will be used)"
fi

# ------------- 4. Maia weights -------------
say "4/6 Maia weights (1100/1500/1900)..."
mkdir -p weights
if [ "$SKIP_WEIGHTS" = 1 ]; then
    warn "--skip-weights set; engine will use the softmax fallback"
else
    for elo in 1100 1500 1900; do
        f="weights/maia-${elo}.pb.gz"
        if [ -s "$f" ]; then
            ok "have $f ($(du -h "$f" | cut -f1))"
            continue
        fi
        url="https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-${elo}.pb.gz"
        say "  downloading $url"
        if curl -fsSL --retry 3 --max-time 60 "$url" -o "$f"; then
            ok "$f ($(du -h "$f" | cut -f1))"
        else
            warn "failed to download $url — softmax fallback will be used for $elo"
            rm -f "$f"
        fi
    done
fi

# ------------- 5. Training dataset (optional) -------------
say "5/6 Lichess training dataset..."
if [ -z "$WITH_DATA" ]; then
    ok "skipped (pass --with-data 2024-09 to download a monthly dump)"
else
    mkdir -p data/dumps
    target="data/dumps/lichess_${WITH_DATA}.pgn.zst"
    if [ -s "$target" ]; then
        ok "have $target ($(du -h "$target" | cut -f1))"
    else
        url="https://database.lichess.org/standard/lichess_db_standard_rated_${WITH_DATA}.pgn.zst"
        # Probe size so we can give the user an ETA.
        size_bytes="$(curl -sIL --max-time 15 "$url" | awk '/[Cc]ontent-[Ll]ength:/ {print $2}' | tr -d '\r\n' | tail -1)"
        if [ -n "$size_bytes" ] && [ "$size_bytes" -gt 0 ]; then
            size_human=$(numfmt --to=iec "$size_bytes" 2>/dev/null || echo "${size_bytes}B")
            # Crude ETA: 50 MB/s sustained on Colab → seconds = bytes / 50e6
            eta_s=$(( size_bytes / 50000000 ))
            say "  $url → $target"
            say "  size: $size_human  ETA at 50 MB/s: ~$((eta_s/60)) min"
        else
            warn "couldn't determine size — proceeding anyway"
        fi
        if curl -fL --max-time 7200 --retry 3 -C - "$url" -o "$target"; then
            ok "$target ($(du -h "$target" | cut -f1))"
        else
            warn "download failed — re-run with the same flag to resume (-C -)"
        fi
    fi
fi

# ------------- 6. Smoke verification -------------
say "6/6 smoke verification..."

# Use a here-doc so the python block doesn't shell-escape oddly.
VERIFY_PY=$(cat <<'PY'
import sys, os
errors = []

# Imports
try:
    import chess, chess.engine
    import fastapi, uvicorn, websockets, requests, numpy
    from troll_engine import Analyzer
    from troll_engine.trap_db import default_db
except Exception as e:
    errors.append(f"import failed: {e}")
    sys.exit("ERR " + "; ".join(errors))

# Stockfish
sf = "/usr/games/stockfish"
if not os.path.exists(sf):
    sf = "stockfish"
try:
    e = chess.engine.SimpleEngine.popen_uci(sf)
    info = e.analyse(chess.Board(), chess.engine.Limit(depth=8))
    e.quit()
    print(f"  stockfish ok (eval at depth 8)")
except Exception as ex:
    errors.append(f"stockfish: {ex}")

# Trap DB
try:
    db = default_db()
    print(f"  trap DB: {len(db)} entries loaded")
except Exception as ex:
    errors.append(f"trap_db: {ex}")

# End-to-end Analyzer
try:
    a = Analyzer(elo=1500, candidate_count=4, candidate_depth=10, reply_count=3,
                 subposition_depth=8, use_explorer=False)
    r = a.analyse(chess.STARTING_FEN)
    a.close()
    assert r.candidates, "no candidates"
    print(f"  Analyzer ok: top moves = {[c.move_san for c in r.candidates[:3]]}")
except Exception as ex:
    errors.append(f"analyzer: {ex}")

# Optional torch / cuda
try:
    import torch
    print(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}")
except ImportError:
    print(f"  torch missing — training scripts unavailable")

# Optional weights
import os.path
have_weights = [e for e in (1100, 1500, 1900)
                if os.path.exists(f"weights/maia-{e}.pb.gz")]
print(f"  Maia weights present: {have_weights or 'none'}")

if errors:
    print("ERR " + "; ".join(errors))
    sys.exit(1)
print("OK")
PY
)

set +e
result="$(cd "$REPO_ROOT" && PYTHONPATH=. python3 -c "$VERIFY_PY")"
status=$?
set -e
echo "$result"

if [ "$status" -ne 0 ] || ! echo "$result" | grep -q '^OK$'; then
    echo
    echo "=================================================="
    echo "  SETUP INCOMPLETE — see ERR line(s) above        "
    echo "=================================================="
    exit 1
fi

cat <<EOF

==================================================
  ALL READY — ChessDorrie installation complete.

  Launch the server:    bash run.sh
  (or in Colab, run the launch cell)

  If you passed --with-data, the dataset is at:
    data/dumps/lichess_${WITH_DATA:-<month>}.pgn.zst

  Mine traps from it:
    python -m data.mine_lichess --pgn data/dumps/lichess_${WITH_DATA:-MONTH}.pgn.zst --max-games 500000

==================================================
EOF
