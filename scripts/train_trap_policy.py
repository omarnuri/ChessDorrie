"""Fine-tune a Maia-style policy network on trap-rich Lichess games.

Pipeline (run order):

    1. scripts/dataset_filter_traps.py — stream a Lichess monthly dump,
       keep only games where a sacrifice (≥1 minor piece given up by
       the eventual winner) decided the game. Output: a single PGN
       containing ~1-5% of the input, focused on Tal-style trap
       outcomes.
    2. scripts/dataset_to_tensors.py — convert filtered PGN → (board
       tensor, move target) pairs in shards on disk.
    3. scripts/train_trap_policy.py (this file) — load a pretrained
       Maia checkpoint, fine-tune the policy head on the trap dataset
       with a low learning rate.

Hardware budget
---------------

  * T4 / V100: not really enough — train batch fits but you'll wait
    20+ hours for ~1 epoch on 1M positions.
  * **A100 40GB (Colab Pro+)**: 6-12 hours for 3 epochs on ~5M
    positions. Use `--batch-size 256 --epochs 3`.
  * **H100 80GB (paid runtime)**: ~2-4 hours for the same. Use
    `--batch-size 512 --epochs 3`.

The pretrained Maia weights are in Tensorflow protobuf format. We
convert once via Lc0's `net describe` then load into PyTorch via the
`maia_chess` repo's converter (see `scripts/convert_maia_to_torch.py`).

This file is a SKELETON. The user is expected to:

  * Have a Colab GPU runtime selected (A100/H100 preferred).
  * Have the filtered+tensorized dataset on disk or Google Drive.
  * Run `python scripts/train_trap_policy.py --data DATA_DIR --out OUT_DIR`.

The skeleton declares the model + optimiser + train loop but does
NOT bundle a specific dataset format — you wire that to your own
shards.

References
----------

  * Maia paper: McIlroy-Young et al. 2020 (KDD).
  * Maia weights: <https://github.com/CSSLab/maia-chess/tree/master/maia_weights>
  * Lichess data: <https://database.lichess.org/>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Defer torch imports so the script can be `--help`-ed without a GPU.

LOG = logging.getLogger("train_trap_policy")


def _make_model(num_residual_blocks: int = 6, channels: int = 64):
    """Maia-style policy network.

    Architecture:
        * Input: 13 piece channels × 8 × 8 + 7 metadata channels = 20 × 8 × 8
        * Stem: 3×3 conv → channels
        * `num_residual_blocks` residual blocks (3×3 conv + BN + ReLU twice + skip)
        * Policy head: 1×1 conv → channels → flatten → linear → 4672 logits
          (4672 = the standard Lc0 move-space encoding)

    Returns
    -------
    torch.nn.Module
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, ch: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            out = nn.functional.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return nn.functional.relu(out + x)

    class TrapPolicyNet(nn.Module):
        def __init__(self, blocks: int, ch: int) -> None:
            super().__init__()
            self.stem = nn.Conv2d(20, ch, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(ch)
            self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
            self.policy_conv = nn.Conv2d(ch, 73, 1)  # 73 = move-plane count
            self.policy_fc = nn.Linear(73 * 64, 4672)

        def forward(self, x):
            x = nn.functional.relu(self.stem_bn(self.stem(x)))
            x = self.blocks(x)
            x = self.policy_conv(x)
            x = x.flatten(1)
            return self.policy_fc(x)

    return TrapPolicyNet(num_residual_blocks, channels)


def _load_pretrained_maia(model, weights_path: str) -> None:
    """Initialise from a Maia checkpoint.

    This is a stub. In practice you'd use the converter at
    `scripts/convert_maia_to_torch.py` (not yet written) which reads
    the Lc0 protobuf weights and copies them into our PyTorch model
    layer-by-layer. The matching is non-trivial because Lc0's move
    encoding differs slightly from python-chess's.
    """
    if not weights_path or not os.path.isfile(weights_path):
        LOG.warning("no pretrained weights — training from scratch (slow)")
        return
    LOG.warning(
        "pretrained-weights path supplied (%s) but converter not "
        "implemented in this skeleton; training from scratch",
        weights_path,
    )


def _build_dataset(data_dir: str, batch_size: int):
    """Return train/val DataLoader pair.

    Expects `data_dir/train_*.pt` and `data_dir/val_*.pt` shards in the
    format produced by `scripts/dataset_to_tensors.py`. Each shard is
    a dict with keys `boards` (N, 20, 8, 8) uint8 and `moves` (N,) int32.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset, ConcatDataset

    class ShardDataset(Dataset):
        def __init__(self, path: str) -> None:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            self.boards = shard["boards"]
            self.moves = shard["moves"]

        def __len__(self) -> int:
            return len(self.moves)

        def __getitem__(self, idx: int):
            return self.boards[idx].float() / 255.0, self.moves[idx].long()

    train_shards = sorted(Path(data_dir).glob("train_*.pt"))
    val_shards = sorted(Path(data_dir).glob("val_*.pt"))
    if not train_shards:
        raise SystemExit(
            f"no train shards in {data_dir} — run dataset_to_tensors.py first"
        )
    train_ds = ConcatDataset([ShardDataset(p) for p in train_shards])
    val_ds = ConcatDataset([ShardDataset(p) for p in val_shards]) if val_shards else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
        if val_ds is not None else None
    )
    return train_loader, val_loader


def train(args) -> None:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device: %s", device)
    if device.type == "cuda":
        LOG.info("GPU: %s, %.1f GB", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    model = _make_model(args.blocks, args.channels).to(device)
    _load_pretrained_maia(model, args.init_from)

    train_loader, val_loader = _build_dataset(args.data, args.batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.monotonic()
        running = 0.0
        n_seen = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            if step % 200 == 0:
                LOG.info("epoch %d step %d  loss=%.4f  rate=%.0f pos/s",
                         epoch, step, loss.item(),
                         n_seen / max(0.1, time.monotonic() - t0))

        # Val
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast(enabled=(device.type == "cuda")):
                        logits = model(x)
                        loss = crit(logits, y)
                    v_loss += loss.item() * x.size(0)
                    v_n += x.size(0)
            v_loss /= max(1, v_n)
            LOG.info("epoch %d val_loss=%.4f", epoch, v_loss)
            if v_loss < best_val:
                best_val = v_loss
                ckpt = out_dir / "best.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
                LOG.info("  saved best to %s", ckpt)
        else:
            ckpt = out_dir / f"epoch_{epoch:02d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
            LOG.info("  saved %s", ckpt)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir with train_*.pt / val_*.pt shards")
    ap.add_argument("--out", default="trained/", help="checkpoint dir")
    ap.add_argument("--init-from", default="", help="pretrained Maia .pb.gz to warm-start from")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--blocks", type=int, default=6, help="residual blocks (Maia uses 6)")
    ap.add_argument("--channels", type=int, default=64, help="conv channels (Maia uses 64)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        LOG.error("PyTorch is not installed. In Colab:  !pip install torch")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
