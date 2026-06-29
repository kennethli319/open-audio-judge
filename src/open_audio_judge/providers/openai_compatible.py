from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from open_audio_judge.config import ProviderConfig
from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        payload = self._build_payload(case, prompt)
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = _extract_content(data)
        return ProviderResponse(content=content, raw=data)

    def _build_payload(self, case: EvaluationCase, prompt: RenderedPrompt) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = []
        audio_part = self._audio_part(case)
        if audio_part:
            user_content.append(audio_part)
        user_content.append({"type": "text", "text": prompt.user})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": prompt.system}]},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "modalities": ["text"],
        }
        payload.update(self.config.extra_body)
        return payload

    def _audio_part(self, case: EvaluationCase) -> dict[str, Any] | None:
        if not case.audio_url and not case.audio_path:
            return None

        if self.config.audio_part_format == "qwen_path":
            if not case.audio_path:
                raise ValueError("qwen_path audio format requires audio_path.")
            return {"type": "audio", "audio": case.audio_path}

        if self.config.audio_part_format == "input_audio":
            if not case.audio_path:
                raise ValueError("input_audio format currently requires audio_path.")
            audio_path = Path(case.audio_path)
            encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            audio_format = audio_path.suffix.lstrip(".") or "wav"
            return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}

        url = case.audio_url or _data_url(Path(case.audio_path or ""))
        return {"type": "audio_url", "audio_url": {"url": url}}


def _data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Provider response has no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        joined = "\n".join(part for part in text_parts if part)
        if joined.strip():
            return joined
    raise ValueError("Provider response did not include text content.")
