import json

import httpx

from open_audio_judge.config import ProviderConfig
from open_audio_judge.models import EvaluationCase, RenderedPrompt
from open_audio_judge.providers.gemini import GeminiProvider


def test_gemini_payload_uses_interactions_audio_shape(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVE")
    prompt = RenderedPrompt(
        judge_id="asr_error",
        judge_version="0.2.0",
        system="system rubric",
        user="user case",
    )
    provider = GeminiProvider(
        ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3.5-flash",
        )
    )

    payload = provider._build_payload(
        EvaluationCase(id="case", task="asr_error", audio_path=str(audio_path)),
        prompt,
    )

    assert payload["model"] == "gemini-3.5-flash"
    assert payload["input"][0] == {"type": "text", "text": "system rubric"}
    assert payload["input"][1]["type"] == "audio"
    assert payload["input"][1]["mime_type"] == "audio/wav"
    assert payload["input"][1]["data"]
    assert payload["input"][2] == {"type": "text", "text": "user case"}


def test_gemini_provider_extracts_output_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["model"] == "gemini-3.5-flash"
        return httpx.Response(
            200,
            json={
                "output_text": '{"overall_score": 90, "reason": "Clean transcript."}',
                "usage": {"total_tokens": 12},
                "steps": [{"signature": "large-opaque-value"}],
            },
        )

    provider = GeminiProvider(
        ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3.5-flash",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = provider.generate(
        EvaluationCase(
            id="case",
            task="asr_error",
            audio_url="https://example.test/a.wav",
            reference_text="Clean transcript.",
        ),
        RenderedPrompt(
            judge_id="asr_error",
            judge_version="0.2.0",
            system="system rubric",
            user="user case",
        ),
    )

    assert response.content == '{"overall_score": 90, "reason": "Clean transcript."}'
    assert response.raw == {"usage": {"total_tokens": 12}}


def test_gemini_provider_extracts_steps_model_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [
                    {"type": "thinking", "content": [{"text": "internal"}]},
                    {
                        "type": "model_output",
                        "content": [
                            {"text": '{"overall_score": 82, "reason": "Meaning preserved."}'}
                        ],
                    },
                ],
            },
        )

    provider = GeminiProvider(
        ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3.5-flash",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = provider.generate(
        EvaluationCase(
            id="case",
            task="asr_error",
            audio_url="https://example.test/a.wav",
            reference_text="Meaning preserved.",
        ),
        RenderedPrompt(
            judge_id="asr_error",
            judge_version="0.2.0",
            system="system rubric",
            user="user case",
        ),
    )

    assert response.content == '{"overall_score": 82, "reason": "Meaning preserved."}'


def test_gemini_provider_includes_bounded_http_error_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            text='{"error":{"message":"Could not fetch audio URI."}}',
        )

    provider = GeminiProvider(
        ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3.5-flash",
        ),
        transport=httpx.MockTransport(handler),
    )

    try:
        provider.generate(
            EvaluationCase(
                id="case",
                task="tts_naturalness",
                audio_url="https://example.test/a.ogg",
                reference_text="hello",
            ),
            RenderedPrompt(
                judge_id="tts_naturalness",
                judge_version="0.1.0",
                system="system rubric",
                user="user case",
            ),
        )
    except RuntimeError as exc:
        assert str(exc) == 'Gemini HTTP 400: {"error":{"message":"Could not fetch audio URI."}}'
    else:
        raise AssertionError("Expected Gemini HTTP failure")


def test_gemini_provider_requires_audio_and_text() -> None:
    provider = GeminiProvider(
        ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3.5-flash",
        )
    )
    prompt = RenderedPrompt(
        judge_id="asr_error",
        judge_version="0.2.0",
        system="system rubric",
        user="user case",
    )

    for case in [
        EvaluationCase(id="no-audio", task="asr_error", reference_text="hello"),
        EvaluationCase(id="no-text", task="asr_error", audio_url="https://example.test/a.wav"),
    ]:
        try:
            provider.generate(case, prompt)
        except ValueError as exc:
            assert "Audio judge cases require" in str(exc)
        else:
            raise AssertionError("Expected audio+text contract failure")
