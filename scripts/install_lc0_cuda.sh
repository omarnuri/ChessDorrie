#!/usr/bin/env bash
# Install Lc0 with the CUDA backend on a Linux machine (designed for
# Colab GPU runtimes). Resolves the latest release via the GitHub API
# with a pinned fallback, verifies the GPU backend works via a quick
# benchmark, and prints the achieved NPS so the user knows it's live.
#
# Exit codes:
#   0  success — Lc0 running on the GPU
#   2  fallback to CPU (--backend=blas) — GPU benchmark failed
#   1  hard failure (download/extract error, missing tools, no GPU)
#
# Overrides:
#   LC0_VERSION   — pin a specific release (e.g. v0.31.2)
#   LC0_TARGET    — install prefix (default /opt/lc0)
#   LC0_FORCE     — set non-empty to re-install even if already present

set -euo pipefail

LC0_VERSION="${LC0_VERSION:-}"
LC0_TARGET="${LC0_TARGET:-/opt/lc0}"
LC0_FORCE="${LC0_FORCE:-}"
LC0_FALLBACK_VERSION="v0.31.2"

say() { printf '[install_lc0] %s\n' "$*"; }
die() { printf '[install_lc0] ERROR: %s\n' "$*" >&2; exit 1; }

# 1. Required tools
for tool in curl tar xz; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found in PATH"
done

# 2. Detect GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found — no NVIDIA GPU available. Use the CPU lc0 (apt install lc0) instead."
fi
gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
say "GPU: ${gpu_info:-unknown}"

# 3. Already installed?
if [ -z "$LC0_FORCE" ] && command -v lc0 >/dev/null 2>&1; then
    if lc0 --help 2>&1 | grep -qi cuda; then
        say "lc0 with CUDA backend already installed at $(command -v lc0)"
        say "(set LC0_FORCE=1 to reinstall)"
        exit 0
    fi
fi

# 4. Resolve the download URL
url=""
if [ -n "$LC0_VERSION" ]; then
    url="https://github.com/LeelaChessZero/lc0/releases/download/${LC0_VERSION}/lc0-${LC0_VERSION}-linux-cuda.tar.xz"
    say "Using pinned version: $LC0_VERSION"
else
    say "Resolving latest release via GitHub API..."
    api_json="$(curl -fsSL --max-time 10 \
        https://api.github.com/repos/LeelaChessZero/lc0/releases/latest 2>/dev/null || true)"
    url="$(printf '%s' "$api_json" | grep -oE 'https://[^"]*linux-cuda[^"]*\.tar\.xz' | head -1 || true)"
    if [ -z "$url" ]; then
        say "GitHub API lookup failed, falling back to pinned $LC0_FALLBACK_VERSION"
        url="https://github.com/LeelaChessZero/lc0/releases/download/${LC0_FALLBACK_VERSION}/lc0-${LC0_FALLBACK_VERSION}-linux-cuda.tar.xz"
    fi
fi
say "Download URL: $url"

# 5. Remove prior CPU build (best-effort)
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get remove -y lc0 >/dev/null 2>&1 || true
fi

# 6. Download + extract
tmpfile="/tmp/lc0-cuda.tar.xz"
say "Downloading..."
curl -fL --retry 3 --max-time 180 -o "$tmpfile" "$url" \
    || die "download failed: $url"

sudo mkdir -p "$LC0_TARGET"
say "Extracting to $LC0_TARGET..."
sudo tar -xJf "$tmpfile" -C "$LC0_TARGET" --strip-components=1 \
    || die "extraction failed"
rm -f "$tmpfile"

sudo chmod +x "$LC0_TARGET/lc0"
sudo ln -sf "$LC0_TARGET/lc0" /usr/local/bin/lc0
hash -r 2>/dev/null || true

# 7. Verify GPU backend with a quick benchmark
say "Verifying GPU backend..."
bench_out="$(lc0 benchmark --backend=cuda-fp16 --nodes=10000 2>&1 || true)"
nps_line="$(printf '%s' "$bench_out" | grep -oE '[0-9]+ nodes per second' | head -1 || true)"
nps="$(printf '%s' "$nps_line" | grep -oE '^[0-9]+' || true)"

if [ -n "$nps" ] && [ "$nps" -gt 1000 ]; then
    say "OK — Lc0 GPU ready: ${nps} nodes/sec on cuda-fp16"
    exit 0
fi

# 8. Fall back to CPU backend with warning
say "WARNING: cuda-fp16 benchmark did not return a sane NPS."
say "Benchmark output (last 10 lines):"
printf '%s\n' "$bench_out" | tail -10 | sed 's/^/  | /'
say "Trying CPU backend (blas) as a fallback..."
bench_cpu="$(lc0 benchmark --backend=blas --nodes=5000 2>&1 || true)"
nps_cpu="$(printf '%s' "$bench_cpu" | grep -oE '[0-9]+ nodes per second' | head -1 | grep -oE '^[0-9]+' || true)"
if [ -n "$nps_cpu" ] && [ "$nps_cpu" -gt 100 ]; then
    say "CPU fallback works: ${nps_cpu} nodes/sec on blas. Maia will be slow."
    exit 2
fi
die "Neither GPU nor CPU backend works. Lc0 is installed but unusable."
