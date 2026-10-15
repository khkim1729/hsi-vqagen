#!/usr/bin/env python3
"""Verify a local OpenAI-compatible model server and minimal generation."""

import argparse
import json

import httpx
from openai import OpenAI


def verify_model_revision(model_card: dict, expected_revision: str) -> tuple[bool, str | None]:
    """A model ID alone cannot prove a weights revision."""
    observed = model_card.get("revision") or model_card.get("model_revision")
    if observed is None:
        return False, None
    if observed != expected_revision:
        raise ValueError(f"served model revision mismatch: {observed}")
    return True, observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    root = args.base_url.removesuffix("/v1")
    response = httpx.get(f"{root}/health", timeout=5)
    response.raise_for_status()
    if args.health_only:
        return
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120)
    models = client.models.list().data
    available = {item.id for item in models}
    if args.model not in available:
        raise SystemExit(f"served model mismatch: expected {args.model}, got {sorted(available)}")
    card = next(item for item in models if item.id == args.model)
    revision_verified, observed_revision = verify_model_revision(card.model_dump(), args.revision)
    result = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        temperature=0,
        max_tokens=8,
    )
    if not (result.choices[0].message.content or "").strip():
        raise SystemExit("health generation returned empty content")
    print(json.dumps({
        "model": args.model, "configured_revision": args.revision,
        "served_revision_verified": revision_verified,
        "observed_served_revision": observed_revision,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
