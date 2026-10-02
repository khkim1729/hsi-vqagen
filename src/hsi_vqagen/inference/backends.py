"""Backend adapters that emit one common raw-generation contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from hsi_vqagen.schema.generation import GenerationRequest, RawGeneration
from hsi_vqagen.inference.telemetry import PeakMemorySampler, nvidia_smi_gpu_memory_bytes


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> RawGeneration: ...


class VLLMBackend:
    """Adapter for a local vLLM OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        client,
        load_seconds: float = 0.0,
        server_version: str | None = None,
        gpu_index: int | None = None,
    ):
        self.client = client
        self.load_seconds = load_seconds
        self.server_version = server_version
        self.gpu_index = gpu_index

    @classmethod
    def from_endpoint(
        cls, base_url: str, api_key: str = "EMPTY", **metadata
    ) -> "VLLMBackend":
        from openai import OpenAI

        return cls(OpenAI(base_url=base_url, api_key=api_key), **metadata)

    def generate(self, request: GenerationRequest) -> RawGeneration:
        settings = request.generation
        kwargs = {
            "model": request.model,
            "messages": self._messages_for_template(request),
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "seed": settings.seed,
        }
        extra_body = {
            key: value
            for key, value in {"top_k": settings.top_k, "min_p": settings.min_p}.items()
            if value is not None
        }
        if request.template_controls.get("tokenizer_mode") != "mistral":
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": settings.enable_thinking
            }
        kwargs["extra_body"] = extra_body
        sampler = (
            PeakMemorySampler(lambda: nvidia_smi_gpu_memory_bytes(self.gpu_index))
            if self.gpu_index is not None
            else None
        )
        started = time.perf_counter()
        if sampler is None:
            response = self.client.chat.completions.create(**kwargs)
        else:
            with sampler:
                response = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        content = response.choices[0].message.content or ""
        return RawGeneration(
            text=content,
            backend="vllm",
            model=request.model,
            revision=request.revision,
            latency_seconds=latency,
            model_load_seconds=self.load_seconds,
            peak_vram_bytes=sampler.peak_bytes if sampler is not None else None,
            backend_metadata={
                "server_version": self.server_version,
                "measurement_method": (
                    "nvidia-smi 100ms GPU-used-memory polling"
                    if sampler is not None and sampler.peak_bytes is not None
                    else "unavailable"
                ),
                "measurement_error": sampler.error if sampler is not None else None,
                "template_controls": request.template_controls,
                "image_order": "image_before_description" if request.rgb_path else "no_image",
            },
        )

    @staticmethod
    def _messages_for_template(request: GenerationRequest) -> list[dict]:
        """Apply explicitly recorded model-template compatibility controls."""

        if request.template_controls.get("system_message_mode") != "inline_user":
            return request.messages

        system_text = "\n\n".join(
            str(message["content"])
            for message in request.messages
            if message.get("role") == "system"
        )
        messages = [
            {**message, "content": [dict(part) for part in message["content"]]}
            for message in request.messages
            if message.get("role") != "system"
        ]
        if not messages or messages[0].get("role") != "user":
            raise ValueError("inline_user requires a user message after the system message")
        text_part = next(
            (part for part in messages[0]["content"] if part.get("type") == "text"),
            None,
        )
        if text_part is None:
            raise ValueError("inline_user requires a text content part")
        text_part["text"] = f"{system_text}\n\n{text_part['text']}"
        return messages


class TransformersBackend:
    """Adapter around a model-specific Transformers runner.

    `runner` is intentionally injectable because InternVL remote-code chat and the
    standard AutoProcessor models expose different generation call shapes. The
    smoke-test loader binds the appropriate runner while this class centralizes
    timing, provenance, and CUDA memory reporting.
    """

    def __init__(
        self,
        runner: Callable[[GenerationRequest], str],
        load_seconds: float,
        preprocessing_class: str,
        peak_memory_reader: Callable[[], int | None] | None = None,
        library_version: str | None = None,
    ):
        self.runner = runner
        self.load_seconds = load_seconds
        self.preprocessing_class = preprocessing_class
        self.peak_memory_reader = peak_memory_reader or (lambda: None)
        self.library_version = library_version

    def generate(self, request: GenerationRequest) -> RawGeneration:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        started = time.perf_counter()
        text = self.runner(request)
        latency = time.perf_counter() - started
        peak = self.peak_memory_reader()
        return RawGeneration(
            text=text,
            backend="transformers",
            model=request.model,
            revision=request.revision,
            latency_seconds=latency,
            model_load_seconds=self.load_seconds,
            peak_vram_bytes=peak,
            backend_metadata={
                "library_version": self.library_version,
                "preprocessing_class": self.preprocessing_class,
                "template_controls": request.template_controls,
                "image_order": "image_before_description" if request.rgb_path else "no_image",
            },
        )
