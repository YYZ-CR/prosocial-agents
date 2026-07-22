"""Confirm which OpenRouter backend provider actually serves a model.

Why this exists: for a controlled experiment you want every request to hit the
same upstream (same weights / quantization). We pin routing with
``provider = {"order": [<name>], "allow_fallbacks": False}`` (see
simulation/utils/models.py:_provider_extra_body). This script fires one cheap
request with that pin and prints the provider OpenRouter reports back, so you can
confirm the pin works *before* spending real budget. With fallbacks off, a
provider that cannot serve the model errors loudly here instead of silently
routing elsewhere.

Usage (from repo root, venv python):
    .venv/Scripts/python.exe scripts/check_provider.py
    .venv/Scripts/python.exe scripts/check_provider.py --model qwen/qwen3-235b-a22b --provider amazon-bedrock
    .venv/Scripts/python.exe scripts/check_provider.py --list      # show providers OpenRouter offers for the model
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
except Exception:
    pass

BASE = "https://openrouter.ai/api/v1"


def list_providers(model: str, key: str) -> int:
    """Print the providers OpenRouter lists for this model (endpoints API)."""
    author, _, slug = model.partition("/")
    url = f"{BASE}/models/{author}/{slug}/endpoints"
    resp = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text[:400]}", file=sys.stderr)
        return 1
    data = resp.json().get("data", {})
    endpoints = data.get("endpoints", [])
    print(f"OpenRouter lists {len(endpoints)} provider endpoint(s) for {model}:")
    for ep in endpoints:
        tag = ep.get("provider_name") or ep.get("name")
        quant = ep.get("quantization") or "?"
        ctx = ep.get("context_length") or "?"
        print(f"  - {tag:24s} quant={quant:8s} context={ctx}")
    print("\nUse the exact provider slug (lowercased, spaces->hyphens) in --provider,")
    print("e.g. 'Amazon Bedrock' -> amazon-bedrock.")
    return 0


def check(model: str, provider: str, key: str) -> int:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 5,
        "temperature": 0,
        "provider": {"order": [provider], "allow_fallbacks": False},
    }
    print(f"Pinning model={model} -> provider={provider} (allow_fallbacks=False)\n")
    try:
        resp = httpx.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
    except httpx.HTTPError as e:
        print(f"REQUEST FAILED: {e}", file=sys.stderr)
        return 1

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text[:600]}", file=sys.stderr)
        print(
            "\nA 404 / 'no allowed providers' here means this provider does not serve\n"
            "this model. Either pick another provider (--list) or clear OPENROUTER_PROVIDER.",
            file=sys.stderr,
        )
        return 1

    data = resp.json()
    served_by = data.get("provider", "<not reported>")
    served_model = data.get("model", model)
    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass
    usage = data.get("usage", {})

    print("SUCCESS")
    print(f"  served by provider : {served_by}")
    print(f"  served model       : {served_model}")
    print(f"  reply              : {content!r}")
    print(f"  tokens (in/out)    : {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")
    if isinstance(served_by, str) and provider.replace("-", " ").lower() not in served_by.lower():
        print(
            f"\nNOTE: reported provider '{served_by}' does not obviously match the pin "
            f"'{provider}'. Double-check the slug with --list.",
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Confirm the OpenRouter provider pin works.")
    p.add_argument("--model", default="qwen/qwen3-235b-a22b")
    p.add_argument(
        "--provider",
        default=os.getenv("OPENROUTER_PROVIDER", "amazon-bedrock"),
        help="provider slug to pin (default: $OPENROUTER_PROVIDER or amazon-bedrock)",
    )
    p.add_argument("--list", action="store_true", help="list providers for the model and exit")
    args = p.parse_args()

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set (put it in .env).", file=sys.stderr)
        return 1

    if args.list:
        return list_providers(args.model, key)
    return check(args.model, args.provider, key)


if __name__ == "__main__":
    raise SystemExit(main())
