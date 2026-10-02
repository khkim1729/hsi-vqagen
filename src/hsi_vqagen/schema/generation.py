"""Shared request, raw-response, and normalized-output contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hsi_vqagen.config import GenerationSettings


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hsi-vqagen.request.v1"] = "hsi-vqagen.request.v1"
    experiment_id: str
    case_id: str
    sample_id: str
    config_id: str
    model: str
    revision: str
    backend: Literal["vllm", "transformers"]
    input_condition: Literal["multimodal", "text_only"]
    prompt_version: str
    prompt_sha256: str
    generation: GenerationSettings
    messages: list[dict[str, Any]]
    provenance: dict[str, Any]
    rgb_path: Path | None = None
    template_controls: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prohibit_text_only_images(self) -> "GenerationRequest":
        image_parts = [
            part
            for message in self.messages
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}
        ]
        if self.input_condition == "text_only" and (self.rgb_path is not None or image_parts):
            raise ValueError("text-only request cannot contain image input")
        if self.input_condition == "multimodal" and (
            self.rgb_path is None or len(image_parts) != 1
        ):
            raise ValueError("multimodal request must contain exactly one image input")
        return self


class RawGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    backend: Literal["vllm", "transformers"]
    model: str
    revision: str
    latency_seconds: float = Field(ge=0)
    model_load_seconds: float = Field(ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    backend_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class GeneratedPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    question_type: str = Field(min_length=1)


class NormalizedCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hsi-vqagen.normalized.v1"] = "hsi-vqagen.normalized.v1"
    experiment_id: str
    case_id: str
    sample_id: str
    config_id: str
    model: str
    revision: str
    backend: Literal["vllm", "transformers"]
    input_condition: Literal["multimodal", "text_only"]
    provenance: dict[str, Any]
    prompt_version: str
    prompt_sha256: str
    generation_settings: dict[str, Any]
    latency_seconds: float
    model_load_seconds: float
    peak_vram_bytes: int | None
    validation_status: Literal["valid", "repaired", "failed"]
    validation_errors: tuple[str, ...] = ()
    raw_attempts: tuple[str, ...]
    pairs: tuple[GeneratedPair, ...] = ()

    def jsonl_rows(self) -> list[dict[str, Any]]:
        """Expand a valid case to the common one-row-per-pair JSONL schema."""

        common = self.model_dump(
            mode="json", exclude={"pairs", "raw_attempts", "validation_errors"}
        )
        common["validation_errors"] = list(self.validation_errors)
        return [
            {
                **common,
                "pair_index": index,
                **pair.model_dump(mode="json"),
            }
            for index, pair in enumerate(self.pairs, 1)
        ]
