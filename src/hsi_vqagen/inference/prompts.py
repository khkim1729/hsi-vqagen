"""Build semantically shared multimodal and text-only requests."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Literal

from hsi_vqagen.config import ExperimentConfig
from hsi_vqagen.data.records import DatasetRecord
from hsi_vqagen.schema.generation import GenerationRequest


ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATHS = {
    "vqa-generation-v1": ROOT / "prompts" / "vqa_generation_v1.txt",
    "vqa-generation-ko-v1": ROOT / "prompts" / "vqa_generation_ko_v1.txt",
}
REPAIR_PROMPT_PATH = ROOT / "prompts" / "json_repair_v1.txt"
EXPERIMENT_IDS = {
    "vqa-generation-v1": "hsi-vqagen-feasibility-v1",
    "vqa-generation-ko-v1": "hsi-vqagen-feasibility-ko-v1",
}


def _prompt_path(prompt_version: str) -> Path:
    try:
        return PROMPT_PATHS[prompt_version]
    except KeyError as exc:
        raise ValueError(f"unknown prompt version: {prompt_version}") from exc


def _template(prompt_version: str) -> str:
    return _prompt_path(prompt_version).read_text(encoding="utf-8")


def prompt_sha256(prompt_version: str = "vqa-generation-v1") -> str:
    return hashlib.sha256(_prompt_path(prompt_version).read_bytes()).hexdigest()


def repair_prompt_sha256() -> str:
    return hashlib.sha256(REPAIR_PROMPT_PATH.read_bytes()).hexdigest()


def build_validation_retry_request(request, raw_text: str, validation_error: str):
    """Build one provenance-visible corrective turn without changing task evidence."""

    instruction = REPAIR_PROMPT_PATH.read_text(encoding="utf-8").strip()
    correction = f"{instruction}\n\nValidation error: {validation_error}"
    messages = [
        *request.messages,
        {"role": "assistant", "content": [{"type": "text", "text": raw_text}]},
        {"role": "user", "content": [{"type": "text", "text": correction}]},
    ]
    return request.model_copy(update={"messages": messages})


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(
    record: DatasetRecord,
    config: ExperimentConfig,
    image_record: DatasetRecord | None = None,
) -> list[dict]:
    """Create a request with one shared task body and a controlled evidence label."""

    evidence_mode = "RGB + description" if config.allow_image else "description only"
    user_text = _template(config.prompt_version).format(
        evidence_mode=evidence_mode,
        description=record.description,
    )
    content: list[dict] = []
    if config.allow_image:
        source = image_record or record
        content.append(
            {"type": "image_url", "image_url": {"url": _data_uri(source.rgb_path)}}
        )
    content.append({"type": "text", "text": user_text})
    system_message = (
        "한국어 초분광 원격탐사 VQA 데이터를 생성하고 엄격한 JSON만 반환한다."
        if config.prompt_version == "vqa-generation-ko-v1"
        else "You generate grounded VQA data and return strict JSON only."
    )
    return [
        {
            "role": "system",
            "content": system_message,
        },
        {"role": "user", "content": content},
    ]


def build_generation_request(
    record: DatasetRecord,
    config: ExperimentConfig,
    backend: Literal["vllm", "transformers"] | None = None,
    image_record: DatasetRecord | None = None,
    experiment_id: str | None = None,
) -> GenerationRequest:
    if config.image_variant == "shuffled":
        if image_record is None or image_record.sample_id == record.sample_id:
            raise ValueError("shuffled RGB requires an explicit different image record")
    elif image_record is not None and image_record.sample_id != record.sample_id:
        raise ValueError("only shuffled RGB may use another sample's image")
    if not config.allow_image and image_record is not None:
        raise ValueError("text-only request cannot receive an image record")
    selected_backend = backend or config.preferred_backend
    image_source = image_record or record
    provenance = {
        "dataset_shard": record.shard,
        "description": record.description,
        "description_prompt_version": record.prompt_version,
        "description_prompt_hash": record.prompt_hash,
        "hsi_shape": list(record.hsi_shape),
        "rgb_size": list(record.rgb_size),
        "description_sha256": hashlib.sha256(record.description.encode("utf-8")).hexdigest(),
        "rgb_sha256": hashlib.sha256(record.rgb_path.read_bytes()).hexdigest(),
    }
    if config.image_variant is not None:
        provenance["image_variant"] = config.image_variant
        provenance["image_source_sample_id"] = (
            image_source.sample_id if config.allow_image else None
        )
        provenance["image_rgb_sha256"] = (
            hashlib.sha256(image_source.rgb_path.read_bytes()).hexdigest()
            if config.allow_image else None
        )
    return GenerationRequest(
        experiment_id=experiment_id or EXPERIMENT_IDS[config.prompt_version],
        case_id=f"{config.id}:{record.sample_id}",
        sample_id=record.sample_id,
        config_id=config.id,
        model=config.checkpoint,
        revision=config.revision,
        backend=selected_backend,
        input_condition=config.condition,
        prompt_version=config.prompt_version,
        prompt_sha256=prompt_sha256(config.prompt_version),
        generation=config.generation,
        messages=build_messages(record, config, image_record=image_record),
        provenance=provenance,
        rgb_path=image_source.rgb_path if config.allow_image else None,
        template_controls=config.template_controls,
    )
