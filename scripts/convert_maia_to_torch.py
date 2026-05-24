"""Convert Lc0 / Maia .pb.gz weights to a PyTorch state_dict.

The Maia network is a small ResNet exposed by Lc0 as a protobuf-encoded
checkpoint. To convert it we need (a) the protobuf schema and (b) a
layer-by-layer mapping into the PyTorch architecture defined in
`scripts/train_trap_policy.py`.

This script ships with two paths:

  * **proto path** (recommended) — uses the Lc0 protobuf schema
    `net.proto`. The schema isn't bundled here because it tracks
    upstream Lc0 changes; the script auto-generates the `_pb2.py` on
    first run if `protoc` is installed, or prints instructions if not.
  * **lc0-describe path** (fallback, lossy) — spawns `lc0 describenet`
    and parses its textual layer dump. Used when the proto schema
    cannot be generated. Recovers shapes but not all metadata.

Usage
-----
    python scripts/convert_maia_to_torch.py weights/maia-1500.pb.gz \
        --out converted/maia-1500.pt

The output is `{ "state_dict": ..., "meta": { "elo": 1500, "source": ... } }`
ready to load via:

    torch.load("converted/maia-1500.pt")["state_dict"]

Status
------
**Skeleton**: the proto download + parse logic is implemented as a
clear, well-documented function chain. The actual layer-by-layer
weight copy needs to match Lc0's net format exactly (BHWC vs BCHW
convention, residual-block ordering, etc). For a definitive
implementation see the `maia_chess/convert_weights` utility upstream
at github.com/CSSLab/maia-chess, or copy from Lc0's own
`tools/net_dump.cc` reference.

The runtime path (`troll_engine/trap_model.py`) is wired against the
*same* PyTorch architecture, so once this script produces a valid
state_dict the integration works end-to-end. Until then, train from
scratch (slower) or use Maia via Lc0 directly (unchanged behaviour).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("convert_maia")


# ------------------------------------------------------------------ #
# Lc0 protobuf schema fetch & compile                                #
# ------------------------------------------------------------------ #

NET_PROTO_URL = (
    "https://raw.githubusercontent.com/LeelaChessZero/lc0/master/"
    "libs/lczero-common/proto/net.proto"
)


def _ensure_pb2() -> "module | None":
    """Make sure `net_pb2` is importable. Returns the module or None
    on failure (with diagnostics logged)."""
    proto_dir = Path(__file__).parent / "_proto"
    pb2_path = proto_dir / "net_pb2.py"
    proto_path = proto_dir / "net.proto"

    if pb2_path.exists():
        sys.path.insert(0, str(proto_dir))
        try:
            import net_pb2  # type: ignore
            return net_pb2
        except Exception as e:
            LOG.warning("net_pb2 present but won't import: %s", e)

    proto_dir.mkdir(exist_ok=True)
    if not proto_path.exists():
        LOG.info("fetching net.proto from %s", NET_PROTO_URL)
        try:
            import urllib.request
            urllib.request.urlretrieve(NET_PROTO_URL, proto_path)
        except Exception as e:
            LOG.error("failed to download net.proto: %s", e)
            return None

    # Compile
    if not (protoc := _find_protoc()):
        LOG.error(
            "protoc not found. Install it:\n"
            "  Debian/Ubuntu/Colab:  sudo apt install -y protobuf-compiler\n"
            "  Mac:                  brew install protobuf\n"
            "Then re-run this script."
        )
        return None
    LOG.info("compiling net.proto with protoc")
    res = subprocess.run(
        [protoc, "--python_out", str(proto_dir), "--proto_path", str(proto_dir),
         str(proto_path)],
        capture_output=True,
    )
    if res.returncode != 0:
        LOG.error("protoc failed: %s", res.stderr.decode())
        return None

    sys.path.insert(0, str(proto_dir))
    try:
        import net_pb2  # type: ignore
        return net_pb2
    except Exception as e:
        LOG.error("net_pb2 still won't import: %s", e)
        return None


def _find_protoc() -> str | None:
    import shutil
    return shutil.which("protoc")


# ------------------------------------------------------------------ #
# Weight extraction                                                  #
# ------------------------------------------------------------------ #

def load_maia_proto(weights_path: str):
    """Decompress + parse a .pb.gz Maia checkpoint.

    Returns the protobuf Net message, or None if loading fails.
    """
    pb2 = _ensure_pb2()
    if pb2 is None:
        return None
    with gzip.open(weights_path, "rb") as fh:
        data = fh.read()
    net = pb2.Net()
    try:
        net.ParseFromString(data)
    except Exception as e:
        LOG.error("protobuf parse failed: %s", e)
        return None
    return net


def proto_to_state_dict(net) -> dict:
    """Map an Lc0 `Net` protobuf into a PyTorch state_dict shaped like
    `scripts/train_trap_policy.py::_make_model`.

    This is the part that is genuinely fiddly — Lc0 stores conv
    weights as flat arrays with a specific [output, input, h, w]
    ordering, batch-norm params with a particular permutation, and so
    on. The exact mapping is documented in Lc0's own
    `tools/net_dump.cc` and in the maia_chess repo.

    For now this returns an empty dict with a warning so callers can
    flow-test the rest of the pipeline. Replace with a real
    implementation when the proto schema is settled.
    """
    LOG.warning(
        "proto_to_state_dict is a stub — returning empty state_dict. "
        "Implement layer mapping against your local Maia checkpoint; "
        "Lc0's net format is documented at "
        "https://github.com/LeelaChessZero/lc0/blob/master/src/neural/"
        "loader.cc"
    )
    return {}


# ------------------------------------------------------------------ #
# Lc0 describenet fallback                                           #
# ------------------------------------------------------------------ #

def describe_maia(weights_path: str) -> dict | None:
    """Parse `lc0 describenet --weights=...` output as a fallback when
    proto isn't available. Recovers layer shapes only, not actual
    weights — useful for sanity-checking the architecture but not for
    warm-starting training.
    """
    import shutil
    lc0 = shutil.which("lc0")
    if not lc0:
        LOG.error("lc0 binary not found, can't use describenet fallback")
        return None
    try:
        out = subprocess.check_output(
            [lc0, "describenet", f"--weights={weights_path}"],
            stderr=subprocess.STDOUT, timeout=30,
        ).decode()
    except Exception as e:
        LOG.error("lc0 describenet failed: %s", e)
        return None
    # Very rough shape extraction
    shapes = []
    for line in out.splitlines():
        m = re.search(r'\[(\d+(?:,\s*\d+)*)\]', line)
        if m:
            shapes.append(tuple(int(x) for x in m.group(1).split(",")))
    return {"raw": out, "shapes": shapes}


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="path to maia-*.pb.gz")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--elo", type=int, default=None,
                    help="rating tag (auto-detected from filename if not given)")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"weights not found: {args.weights}")

    elo = args.elo
    if elo is None:
        m = re.search(r'(\d{3,4})', os.path.basename(args.weights))
        if m:
            elo = int(m.group(1))

    LOG.info("loading %s", args.weights)
    net = load_maia_proto(args.weights)

    if net is not None:
        state_dict = proto_to_state_dict(net)
        meta = {
            "elo": elo,
            "source": "maia",
            "weights_path": args.weights,
            "via": "protobuf",
        }
    else:
        LOG.warning("falling back to lc0 describenet (shapes only, no weights)")
        info = describe_maia(args.weights)
        if info is None:
            sys.exit("could not extract weights — see errors above")
        state_dict = {}
        meta = {
            "elo": elo,
            "source": "maia",
            "weights_path": args.weights,
            "via": "describenet_shapes_only",
            "describenet": info,
        }

    try:
        import torch  # type: ignore
    except ImportError:
        sys.exit("torch is required: pip install torch")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": meta}, args.out)
    LOG.info("wrote %s", args.out)
    LOG.info("state_dict keys: %d", len(state_dict))
    LOG.info("meta: %s", json.dumps(meta, default=str)[:200])


if __name__ == "__main__":
    main()
