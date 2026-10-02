"""Validated experiment registry for the feasibility study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


EXPECTED_IDS = tuple(f"C{number:02d}" for number in range(1, 12))
ABLATION_PAIRS = (("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06"))


class GenerationSettings(BaseModel):
    """Generation controls held fixed wherever a backend supports them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qa_count: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    seed: int
    enable_thinking: bool


class ExperimentConfig(BaseModel):
    """One input condition applied to one pinned model checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    output_dir: str = Field(pattern=r"^[a-z0-9_]+$")
    checkpoint: str
    revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    condition: Literal["multimodal", "text_only"]
    allow_image: bool
    preferred_backend: Literal["vllm", "transformers"]
    fallback_backend: Literal["vllm", "transformers"] | None = None
    prompt_version: str
    generation: GenerationSettings
    template_controls: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    preprocessing_exceptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_condition(self) -> "ExperimentConfig":
        if self.condition == "text_only" and self.allow_image:
            raise ValueError("text-only configurations must set allow_image=false")
        if self.condition == "multimodal" and not self.allow_image:
            raise ValueError("multimodal configurations must set allow_image=true")
        return self


class ExperimentRegistry(BaseModel):
    """The complete, preregistered C01--C11 comparison matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_version: str
    sample_manifest_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    configurations: tuple[ExperimentConfig, ...]

    @property
    def by_id(self) -> dict[str, ExperimentConfig]:
        return {item.id: item for item in self.configurations}

    @model_validator(mode="after")
    def enforce_study_contract(self) -> "ExperimentRegistry":
        ids = tuple(item.id for item in self.configurations)
        if ids != EXPECTED_IDS:
            raise ValueError(f"configuration IDs/order must be exactly {EXPECTED_IDS}")
        output_dirs = [item.output_dir for item in self.configurations]
        if len(set(output_dirs)) != len(output_dirs):
            raise ValueError("configuration output directories must be unique")
        if len({item.checkpoint for item in self.configurations}) != 7:
            raise ValueError("the registry must contain exactly seven checkpoints")

        by_id = self.by_id
        for multimodal_id, text_only_id in ABLATION_PAIRS:
            multimodal = by_id[multimodal_id]
            text_only = by_id[text_only_id]
            controlled = (
                multimodal.checkpoint,
                multimodal.revision,
                multimodal.prompt_version,
                multimodal.generation,
            )
            comparison = (
                text_only.checkpoint,
                text_only.revision,
                text_only.prompt_version,
                text_only.generation,
            )
            if controlled != comparison:
                raise ValueError(
                    f"same-checkpoint pair {multimodal_id}/{text_only_id} has control drift"
                )
        return self


def load_experiment_registry(path: Path) -> ExperimentRegistry:
    """Load and validate an experiment registry from YAML."""

    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    # This YAML-only key gives every condition one identical generation mapping.
    payload.pop("shared_generation", None)
    return ExperimentRegistry.model_validate(payload)
