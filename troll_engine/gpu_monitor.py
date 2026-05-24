"""Background nvidia-smi poller.

Single thread, polls GPU utilization and VRAM every ~500ms. The
session worker reads the latest cached values when it broadcasts a
metrics snapshot — no per-tick subprocess overhead.

Gracefully degrades to all-None readings when nvidia-smi is absent
(CPU-only or Mac machines).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class GpuReading:
    util: float | None = None     # percent
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    ts: float = 0.0


class GpuMonitor:
    """Singleton — start once per app, read from anywhere."""

    _instance: "GpuMonitor | None" = None

    @classmethod
    def instance(cls) -> "GpuMonitor":
        if cls._instance is None:
            cls._instance = GpuMonitor()
        return cls._instance

    def __init__(self) -> None:
        self._reading = GpuReading()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        if self._has_nvidia:
            self._thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="GpuMonitor"
            )
            self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=2.0,
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                # First line covers GPU 0; if multi-GPU, we ignore the rest.
                first = out.splitlines()[0] if out else ""
                parts = [p.strip() for p in first.split(",")]
                if len(parts) >= 3:
                    with self._lock:
                        self._reading = GpuReading(
                            util=float(parts[0]),
                            vram_used_mb=float(parts[1]),
                            vram_total_mb=float(parts[2]),
                            ts=time.monotonic(),
                        )
            except Exception:
                # Transient nvidia-smi failures shouldn't propagate.
                pass
            # Poll at 2 Hz.
            self._stop.wait(0.5)

    def latest(self) -> GpuReading:
        with self._lock:
            return self._reading

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
