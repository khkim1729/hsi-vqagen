"""Backend adapters that emit one common raw-generation contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from hsi_vqagen.schema.generation import GenerationRequest, RawGeneration


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> RawGeneration: ...


class VLLMBackend:
    """Adapter for a local vLLM OpenAI-compatible chat endpoint."""

    def __init__(self, client, load_seconds: float = 0.0, server_version: str | None = None):
        self.client = client
        self.load_seconds = load_seconds
        self.server_version = server_version

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
            "messages": request.messages,
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
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": settings.enable_thinking
        }
        kwargs["extra_body"] = extra_body
        started = time.perf_counter()
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
            backend_metadata={
                "server_version": self.server_version,
                "template_controls": request.template_controls,
                "image_order": "image_before_description" if request.rgb_path else "no_image",
            },
        )


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
