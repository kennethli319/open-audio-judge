from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from open_audio_judge.case_contract import require_audio_and_text
from open_audio_judge.config import ProviderConfig
from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt


class GeminiProvider:
    def __init__(self, config: ProviderConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.name = config.name
        self.transport = transport

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        require_audio_and_text(case)
        payload = self._build_payload(case, prompt)
        url = f"{self.config.base_url.rstrip('/')}/interactions"
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.config.timeout_seconds, transport=self.transport) as client:
            response = client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(_format_http_error(exc.response)) from exc
            data = response.json()

        return ProviderResponse(content=_extract_output_text(data), raw=_sanitize_raw_response(data))

    def _build_payload(self, case: EvaluationCase, prompt: RenderedPrompt) -> dict[str, Any]:
        input_parts: list[dict[str, Any]] = [
            {"type": "text", "text": prompt.system},
            *self._audio_parts(case),
            {"type": "text", "text": prompt.user},
        ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_parts,
        }
        if prompt.response_schema:
            payload["response_format"] = prompt.response_schema
        payload.update(self.config.extra_body)
        return payload

    def _audio_parts(self, case: EvaluationCase) -> list[dict[str, Any]]:
        if case.audio_url:
            return [
                {
                    "type": "audio",
                    "uri": case.audio_url,
                    "mime_type": _mime_type_from_name(case.audio_url),
                }
            ]
        if case.audio_path:
            path = Path(case.audio_path)
            audio_bytes = path.read_bytes()
            return [
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "mime_type": _mime_type_from_name(path.name),
                }
            ]
        return []


def _extract_output_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = data.get("output")
    if isinstance(output, list):
        joined = "\n".join(_text_parts_from_items(output))
        if joined.strip():
            return joined

    steps = data.get("steps")
    if isinstance(steps, list):
        model_outputs = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("type") in {None, "model_output"}
        ]
        joined = "\n".join(_text_parts_from_items(model_outputs))
        if joined.strip():
            return joined

    raise ValueError("Gemini response did not include output_text.")


def _text_parts_from_items(items: list[Any]) -> list[str]:
    text_parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            text_parts.extend(part.get("text", "") for part in content if isinstance(part, dict))
        elif isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return [part for part in text_parts if part]


def _mime_type_from_name(name: str) -> str:
    mime_type = mimetypes.guess_type(name)[0] or "audio/wav"
    if mime_type == "audio/x-wav":
        return "audio/wav"
    return mime_type


def _sanitize_raw_response(data: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "id",
        "status",
        "model",
        "object",
        "service_tier",
        "created",
        "updated",
        "usage",
    }
    return {key: value for key, value in data.items() if key in allowed}


def _format_http_error(response: httpx.Response) -> str:
    body = response.text.strip().replace("\n", " ")
    if len(body) > 500:
        body = f"{body[:500]}..."
    if body:
        return f"Gemini HTTP {response.status_code}: {body}"
    return f"Gemini HTTP {response.status_code}"
