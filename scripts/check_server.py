#!/usr/bin/env python3
"""Verify a local OpenAI-compatible model server and minimal generation."""

import argparse

import httpx
from openai import OpenAI


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
    available = {item.id for item in client.models.list().data}
    if args.model not in available:
        raise SystemExit(f"served model mismatch: expected {args.model}, got {sorted(available)}")
    result = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        temperature=0,
        max_tokens=8,
    )
    if not (result.choices[0].message.content or "").strip():
        raise SystemExit("health generation returned empty content")
    print(f"ready model={args.model} revision={args.revision}")


if __name__ == "__main__":
    main()
