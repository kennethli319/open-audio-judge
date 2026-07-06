from pathlib import Path

from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt
from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers.mock import MockProvider
from open_audio_judge.runner import evaluate_case, evaluate_cases, load_cases


class MalformedJsonProvider:
    name = "diagnostic-provider"

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        return ProviderResponse(
            content="not json",
            raw={"usage": {"total_tokens": 42}, "status": "completed"},
        )


class PartialJsonProvider:
    name = "partial-json-provider"

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        return ProviderResponse(
            content='{"overall_score": 80, "reason": "Natural enough."}',
            raw={"usage": {"total_tokens": 17}},
        )


class FailingProvider:
    name = "failing-provider"

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        raise RuntimeError(f"temporary outage\n{'x' * 600}")


def test_evaluate_cases_with_mock(tmp_path: Path) -> None:
    cases = load_cases(Path("examples/asr_cases.jsonl"))
    prompt = load_prompt("asr_error")
    results = evaluate_cases(cases, prompt, MockProvider(), tmp_path)

    assert len(results) == len(cases)
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "report.html").exists()
    assert all(result.status == "ok" for result in results)
    by_id = {result.case_id: result for result in results}
    assert by_id["asr-demo-002"].overall_score <= 55
    assert "number_error" in by_id["asr-demo-002"].error_categories
    assert by_id["asr-calibration-negation-001"].overall_score <= 40
    assert "negation_error" in by_id["asr-calibration-negation-001"].error_categories
    assert "entity_error" in by_id["asr-calibration-entity-001"].error_categories
    assert "date_time_error" in by_id["asr-calibration-date-001"].error_categories
    assert "date_time_error" in by_id["asr-calibration-time-001"].error_categories
    assert "unit_error" in by_id["asr-calibration-unit-001"].error_categories


def test_evaluate_tts_case_with_mock_matches_tts_schema() -> None:
    prompt = load_prompt("tts_naturalness")
    result = evaluate_case(
        EvaluationCase(
            id="tts-mock",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="hello",
        ),
        prompt,
        MockProvider(),
    )

    assert result.status == "ok"
    assert result.overall_score == 75
    assert result.semantic_error_summary == (
        "Mock provider does not listen to audio or evaluate perceptual quality."
    )
    assert result.error_categories == ["mock"]


def test_parse_error_preserves_provider_raw_response() -> None:
    prompt = load_prompt("asr_error")
    result = evaluate_case(
        EvaluationCase(
            id="bad-json",
            task="asr_error",
            audio_url="https://example.test/audio.wav",
            reference_text="hello",
        ),
        prompt,
        MalformedJsonProvider(),
    )

    assert result.status == "parse_error"
    assert result.raw_response == {"usage": {"total_tokens": 42}, "status": "completed"}
    assert "No JSON object found" in (result.error or "")


def test_parse_error_uses_prompt_response_schema_for_diagnostics() -> None:
    prompt = load_prompt("tts_naturalness")
    result = evaluate_case(
        EvaluationCase(
            id="partial-json",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="hello",
        ),
        prompt,
        PartialJsonProvider(),
    )

    assert result.status == "parse_error"
    assert result.raw_response == {"usage": {"total_tokens": 17}}
    assert "semantic_error_summary" in (result.error or "")


def test_provider_error_preserves_bounded_diagnostic_metadata() -> None:
    prompt = load_prompt("tts_naturalness")
    result = evaluate_case(
        EvaluationCase(
            id="provider-error",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="hello",
        ),
        prompt,
        FailingProvider(),
    )

    assert result.status == "provider_error"
    assert result.raw_response["error_type"] == "RuntimeError"
    assert result.raw_response["message"].startswith("temporary outage x")
    assert "\n" not in result.raw_response["message"]
    assert len(result.raw_response["message"]) <= 503
