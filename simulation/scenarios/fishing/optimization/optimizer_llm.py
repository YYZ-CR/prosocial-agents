"""Thin OpenRouter chat client for the optimizers' own reasoning.

This is separate from the sim's model wrapper: TextGrad/ACE use it to write
critiques, revise norms, and curate the playbook. Honors the same provider pin
(OPENROUTER_PROVIDER) as the sim so all traffic goes to one backend.
"""
from __future__ import annotations

import os

import httpx

BASE = "https://openrouter.ai/api/v1"


class OptimizerLLM:
    def __init__(self, model: str, *, temperature: float = 0.4, max_tokens: int = 1200):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._key = os.getenv("OPENROUTER_API_KEY")

    def _extra(self) -> dict:
        provider = os.getenv("OPENROUTER_PROVIDER", "").strip()
        if not provider:
            return {}
        return {"provider": {"order": [provider], "allow_fallbacks": False}}

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        if not self._key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            **self._extra(),
        }
        resp = httpx.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json=body, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self.calls += 1
        usage = data.get("usage", {})
        self.tokens_in += usage.get("prompt_tokens", 0) or 0
        self.tokens_out += usage.get("completion_tokens", 0) or 0
        return data["choices"][0]["message"]["content"] or ""
