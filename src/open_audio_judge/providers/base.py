from __future__ import annotations

from typing import Protocol

from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt


class JudgeProvider(Protocol):
    name: str

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        """Return raw judge text for a rendered prompt and case."""
