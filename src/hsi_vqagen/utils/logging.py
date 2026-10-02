"""Log sanitization helpers."""

from __future__ import annotations

import re


_TOKEN = re.compile(r"\b(?:hf|ghp|olp)_[A-Za-z0-9_-]{10,}\b")
_AUTHORIZATION = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[^\s,;]+"
)


def redact_secrets(value: str) -> str:
    value = _AUTHORIZATION.sub(r"\1[REDACTED]", str(value))
    return _TOKEN.sub("[REDACTED]", value)
