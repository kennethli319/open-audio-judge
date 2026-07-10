import json
from pathlib import Path

from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt
from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers.mock import MockProvider
from open_audio_judge.runner import (
    evaluate_case,
    evaluate_case_with_sampling,
    evaluate_cases,
    load_cases,
    load_results_jsonl,
)


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


class SequencedTtsProvider:
    name = "sequenced-tts-provider"

    def __init__(self, scores: list[int]) -> None:
        self.scores = scores
        self.index = 0

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        score = self.scores[self.index]
        self.index += 1
        return ProviderResponse(
            content=json.dumps(
                {
                    "overall_score": score,
                    "reason": f"Attempt scored {score}.",
                    "semantic_error_summary": "Naturalness varied across samples.",
                    "key_differences": [],
                    "error_categories": ["no_error"] if score >= 80 else ["awkward_pacing"],
                    "researcher_notes": ["Inspect sampling variance."],
                }
            ),
            raw={"attempt_score": score},
        )


class PartiallyFailingTtsProvider:
    name = "partially-failing-tts-provider"

    def __init__(self) -> None:
        self.index = 0

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        self.index += 1
        if self.index == 3:
            raise RuntimeError("temporary judge outage")
        return ProviderResponse(
            content=json.dumps(
                {
                    "overall_score": 100,
                    "reason": "Transcript is accurate.",
                    "semantic_error_summary": "Meaning is preserved.",
                    "key_differences": [],
                    "error_categories": ["no_error"],
                    "researcher_notes": [],
                }
            )
        )


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


def test_evaluate_case_with_sampling_averages_scores_and_preserves_attempts() -> None:
    prompt = load_prompt("tts_naturalness")
    result = evaluate_case_with_sampling(
        EvaluationCase(
            id="sampled-tts",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="Read this naturally.",
            metadata={"eval_category": "storytelling"},
        ),
        prompt,
        SequencedTtsProvider([70, 80, 90]),
        judge_samples=3,
    )

    assert result.overall_score == 80
    assert result.label == "needs_review"
    assert result.metadata["judge_sample_count"] == 3
    assert result.metadata["judge_sample_scores"] == [70, 80, 90]
    assert result.metadata["judge_sample_average"] == 80.0
    assert result.metadata["eval_category"] == "storytelling"
    assert "Average of 3 judge samples" in result.reason
    assert result.raw_response["judge_samples"][0]["overall_score"] == 70


def test_evaluate_case_with_sampling_excludes_failed_attempts_from_score() -> None:
    prompt = load_prompt("tts_naturalness")
    result = evaluate_case_with_sampling(
        EvaluationCase(
            id="sampled-tts-with-outage",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="Read this naturally.",
        ),
        prompt,
        PartiallyFailingTtsProvider(),
        judge_samples=3,
    )

    assert result.status == "ok"
    assert result.overall_score == 100
    assert result.label == "accurate"
    assert result.metadata["judge_sample_count"] == 3
    assert result.metadata["judge_sample_success_count"] == 2
    assert result.metadata["judge_sample_failure_count"] == 1
    assert result.metadata["judge_sample_scores"] == [100, 100]
    assert result.metadata["judge_sample_statuses"] == ["ok", "ok", "provider_error"]
    assert (
        result.metadata["judge_sample_score_policy"]
        == "successful_attempts_only_with_all_failed_fallback_v1"
    )
    assert "1 failed attempt excluded" in result.reason
    assert result.raw_response["judge_samples"][2]["status"] == "provider_error"


def test_load_results_reconciles_legacy_failed_sample_score(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "legacy-partial-outage",
                "task": "asr_error",
                "judge_id": "asr_error",
                "judge_version": "0.2.0",
                "provider": "gemini",
                "overall_score": 67,
                "reason": (
                    "Average of 3 judge samples: 67.00 (scores: 100, 100, 1). "
                    "Representative reason: Transcript is accurate."
                ),
                "meaning_preservation": "preserved",
                "error_categories": ["formatting_only"],
                "label": "needs_review",
                "status": "ok",
                "metadata": {
                    "judge_sample_count": 3,
                    "judge_sample_scores": [100, 100, 1],
                    "judge_sample_average": 67.0,
                    "judge_sample_statuses": ["ok", "ok", "provider_error"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_results_jsonl(path)[0]

    assert result.overall_score == 100
    assert result.label == "accurate"
    assert result.metadata["judge_sample_scores"] == [100, 100]
    assert result.metadata["judge_sample_success_count"] == 2
    assert result.metadata["judge_sample_failure_count"] == 1
    assert (
        result.metadata["judge_sample_score_policy"]
        == "successful_attempts_only_with_all_failed_fallback_v1"
    )
    assert "1 failed attempt excluded" in result.reason


def test_load_results_applies_current_label_threshold_to_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "case_id": "legacy-unsampled-80",
            "task": "asr_error",
            "judge_id": "asr_error",
            "judge_version": "0.2.0",
            "provider": "gemini",
            "overall_score": 80,
            "reason": "Legacy threshold labeled this accurate.",
            "label": "accurate",
            "status": "ok",
        },
        {
            "case_id": "legacy-sampled-80",
            "task": "asr_error",
            "judge_id": "asr_error",
            "judge_version": "0.2.0",
            "provider": "gemini",
            "overall_score": 80,
            "reason": "Average of 2 judge samples: 80.00.",
            "label": "accurate",
            "status": "ok",
            "metadata": {
                "judge_sample_count": 2,
                "judge_sample_scores": [80, 80],
                "judge_sample_statuses": ["ok", "ok"],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    results = load_results_jsonl(path)

    assert [result.label for result in results] == ["needs_review", "needs_review"]


def test_evaluate_cases_accepts_judge_samples(tmp_path: Path) -> None:
    cases = [
        EvaluationCase(
            id="sampled-tts",
            task="tts_naturalness",
            audio_url="https://example.test/audio.wav",
            reference_text="Read this naturally.",
        )
    ]
    prompt = load_prompt("tts_naturalness")
    results = evaluate_cases(
        cases,
        prompt,
        SequencedTtsProvider([61, 62, 63]),
        tmp_path,
        judge_samples=3,
    )

    assert results[0].overall_score == 62
    written = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert written["metadata"]["judge_sample_scores"] == [61, 62, 63]
    assert "judge samples: 61, 62, 63; avg 62.00" in (tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
