from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 512
    audio_part_format: str = "audio_url"
    extra_body: dict[str, Any] = field(default_factory=dict)


def load_provider_config(name: str | None = None) -> ProviderConfig:
    provider_name = name or os.getenv("OAJ_PROVIDER", "qwen")
    prefix = "OAJ_"

    if provider_name == "qwen":
        default_base_url = "http://localhost:8091/v1"
        default_model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        default_api_key = "EMPTY"
    elif provider_name == "gemini":
        default_base_url = "https://generativelanguage.googleapis.com/v1beta"
        default_model = "gemini-3.5-flash"
        default_api_key = os.getenv("GEMINI_API_KEY", "")
    else:
        default_base_url = "http://localhost:8000/v1"
        default_model = "audio-judge"
        default_api_key = "EMPTY"

    extra_body_raw = os.getenv(f"{prefix}EXTRA_BODY_JSON", "{}")
    try:
        extra_body = json.loads(extra_body_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OAJ_EXTRA_BODY_JSON is not valid JSON: {exc}") from exc

    return ProviderConfig(
        name=provider_name,
        base_url=os.getenv(f"{prefix}BASE_URL", default_base_url),
        api_key=os.getenv(f"{prefix}API_KEY", default_api_key),
        model=os.getenv(f"{prefix}MODEL", default_model),
        timeout_seconds=float(os.getenv(f"{prefix}TIMEOUT_SECONDS", "120")),
        temperature=float(os.getenv(f"{prefix}TEMPERATURE", "0")),
        max_tokens=int(os.getenv(f"{prefix}MAX_TOKENS", "512")),
        audio_part_format=os.getenv(f"{prefix}AUDIO_PART_FORMAT", "audio_url"),
        extra_body=extra_body,
    )
