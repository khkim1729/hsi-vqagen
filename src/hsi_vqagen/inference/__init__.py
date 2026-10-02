"""Condition-safe generation interfaces."""

from .backends import GenerationBackend, TransformersBackend, VLLMBackend
from .normalize import normalize_response
from .prompts import build_generation_request, build_messages

__all__ = [
    "GenerationBackend",
    "TransformersBackend",
    "VLLMBackend",
    "build_generation_request",
    "build_messages",
    "normalize_response",
]
