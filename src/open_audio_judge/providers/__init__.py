from __future__ import annotations

from open_audio_judge.config import load_provider_config
from open_audio_judge.providers.base import JudgeProvider
from open_audio_judge.providers.gemini import GeminiProvider
from open_audio_judge.providers.mock import MockProvider
from open_audio_judge.providers.openai_compatible import OpenAICompatibleProvider


def build_provider(name: str | None = None) -> JudgeProvider:
    provider_name = name or "qwen"
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "gemini":
        return GeminiProvider(load_provider_config(provider_name))
    if provider_name in {"qwen", "openai-compatible"}:
        return OpenAICompatibleProvider(load_provider_config(provider_name))
    raise ValueError(f"Unknown provider: {provider_name}")


__all__ = ["GeminiProvider", "JudgeProvider", "MockProvider", "OpenAICompatibleProvider", "build_provider"]
