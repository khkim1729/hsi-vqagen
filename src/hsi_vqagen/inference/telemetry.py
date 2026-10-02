"""Runtime measurement records with explicit unavailable states."""

from __future__ import annotations

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
