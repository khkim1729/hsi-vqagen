"""Runtime measurement records with explicit unavailable states."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


class VramMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    peak_vram_bytes: int | None = Field(default=None, ge=0)
    method: str
    unavailable_reason: str | None = None


def peak_vram_measurement(
    reader: Callable[[], int | None] | None = None,
    method: str = "torch.cuda.max_memory_allocated",
) -> VramMeasurement:
    if reader is None:
        try:
            import torch

            reader = lambda: int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        except ImportError:
            reader = lambda: None
    try:
        value = reader()
    except Exception as exc:  # Telemetry must not destroy a completed generation.
        return VramMeasurement(
            method=method,
            unavailable_reason=f"measurement failed: {type(exc).__name__}",
        )
    if value is None:
        return VramMeasurement(
            method=method,
            unavailable_reason="peak VRAM is unavailable from this backend/process",
        )
    return VramMeasurement(peak_vram_bytes=int(value), method=method)


def nvidia_smi_gpu_memory_bytes(gpu_index: int) -> int:
    """Return total used memory for one visible physical GPU."""

    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    mib = int(result.stdout.strip().splitlines()[0])
    return mib * 1024 * 1024


class PeakMemorySampler:
    """Poll a memory reader during a request and retain the observed maximum."""

    def __init__(self, reader: Callable[[], int], interval_seconds: float = 0.1):
        self.reader = reader
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            value = int(self.reader())
            self.peak_bytes = value if self.peak_bytes is None else max(self.peak_bytes, value)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "PeakMemorySampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._sample()
