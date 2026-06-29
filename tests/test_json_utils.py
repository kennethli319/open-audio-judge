import json

import pytest

from open_audio_judge.json_utils import extract_json_object, parse_judge_output


def test_extract_json_from_code_fence() -> None:
    parsed = extract_json_object('```json\n{"overall_score": 91, "reason": "Clean."}\n```')
    assert parsed["overall_score"] == 91


def test_extract_json_from_surrounding_text() -> None:
    parsed = extract_json_object('Result:\n{"overall_score": 42, "reason": "Major entity error."}\nDone')
    assert parsed["reason"] == "Major entity error."


def test_parse_alternate_score_key() -> None:
    output = parse_judge_output('{"score": 88, "explanation": "Meaning preserved."}')
    assert output.overall_score == 88
    assert output.reason == "Meaning preserved."


def test_parse_asr_diagnostics() -> None:
    output = parse_judge_output(
        """{
          "overall_score": 52,
          "reason": "A key amount changed.",
          "judge_transcript": "Please transfer fifteen dollars.",
          "meaning_preservation": "partial_loss",
          "semantic_error_summary": "The amount is wrong.",
          "key_differences": ["fifteen became fifty"],
          "error_categories": ["number_error", "substitution"],
          "researcher_notes": ["Improve numeric robustness."]
        }"""
    )

    assert output.judge_transcript == "Please transfer fifteen dollars."
    assert output.meaning_preservation == "partial_loss"
    assert output.key_differences == ["fifteen became fifty"]
    assert output.error_categories == ["number_error", "substitution"]


def test_parse_tts_naturalness_diagnostics() -> None:
    output = parse_judge_output(
        """{
          "overall_score": 64,
          "reason": "The sample is intelligible but has choppy rhythm.",
          "semantic_error_summary": "Naturalness is reduced by pacing artifacts.",
          "key_differences": ["pause before the final word sounds inserted"],
          "error_categories": ["awkward_pacing", "artifact"],
          "researcher_notes": ["Improve pause prediction around short phrases."]
        }"""
    )

    assert output.semantic_error_summary == "Naturalness is reduced by pacing artifacts."
    assert output.key_differences == ["pause before the final word sounds inserted"]
    assert output.error_categories == ["awkward_pacing", "artifact"]
    assert output.researcher_notes == ["Improve pause prediction around short phrases."]


def test_parse_with_response_schema_rejects_missing_diagnostic_fields() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overall_score",
            "reason",
            "semantic_error_summary",
            "key_differences",
            "error_categories",
            "researcher_notes",
        ],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "semantic_error_summary": {"type": "string"},
            "key_differences": {"type": "array", "items": {"type": "string"}},
            "error_categories": {"type": "array", "items": {"type": "string"}},
            "researcher_notes": {"type": "array", "items": {"type": "string"}},
        },
    }

    with pytest.raises(ValueError, match="missing required field"):
        parse_judge_output('{"overall_score": 88, "reason": "Mostly natural."}', schema)


def test_parse_with_response_schema_rejects_extra_fields() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
        },
    }

    with pytest.raises(ValueError, match="unsupported field"):
        parse_judge_output(
            '{"overall_score": 88, "reason": "Mostly natural.", "hidden_reasoning": "secret"}',
            schema,
        )


def test_parse_with_response_schema_rejects_wrong_diagnostic_types() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason", "key_differences"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "key_differences": {"type": "array", "items": {"type": "string"}},
        },
    }

    with pytest.raises(ValueError, match="key_differences must match array"):
        parse_judge_output(
            '{"overall_score": 88, "reason": "Mostly natural.", "key_differences": "none"}',
            schema,
        )


def test_parse_with_response_schema_allows_nullable_judge_transcript() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason", "judge_transcript"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "judge_transcript": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }

    output = parse_judge_output(
        '{"overall_score": 44, "reason": "Audio was unclear.", "judge_transcript": null}',
        schema,
    )

    assert output.judge_transcript is None


def test_parse_with_response_schema_rejects_nested_missing_required_fields() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason", "diagnostics"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "diagnostics": {
                "type": "object",
                "additionalProperties": False,
                "required": ["artifact_count"],
                "properties": {
                    "artifact_count": {"type": "integer", "minimum": 0, "maximum": 10},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }

    with pytest.raises(ValueError, match="diagnostics is missing required field"):
        parse_judge_output(
            '{"overall_score": 88, "reason": "Mostly natural.", "diagnostics": {}}',
            schema,
        )


def test_parse_with_response_schema_rejects_nested_extra_fields() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason", "diagnostics"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "diagnostics": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"artifact_count": {"type": "integer"}},
            },
        },
    }

    with pytest.raises(ValueError, match="diagnostics includes unsupported field"):
        parse_judge_output(
            (
                '{"overall_score": 88, "reason": "Mostly natural.", '
                '"diagnostics": {"artifact_count": 1, "hidden": "x"}}'
            ),
            schema,
        )


def test_parse_with_response_schema_rejects_integer_bounds() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "reason", "confidence"],
        "properties": {
            "overall_score": {"type": "integer"},
            "reason": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 5},
        },
    }

    with pytest.raises(ValueError, match="confidence must be at most 5"):
        parse_judge_output(
            '{"overall_score": 88, "reason": "Mostly natural.", "confidence": 6}',
            schema,
        )


def test_parse_rejects_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason must not be empty"):
        parse_judge_output('{"overall_score": 88, "reason": "   "}')


def test_parse_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="overall_score must be an integer from 1 to 100"):
        parse_judge_output('{"overall_score": 101, "reason": "Too high."}')


def test_parse_rejects_non_integer_score() -> None:
    with pytest.raises(ValueError, match="overall_score must be an integer from 1 to 100"):
        parse_judge_output('{"overall_score": "high", "reason": "Not numeric."}')


@pytest.mark.parametrize("score", ["88.5", 88.5, True])
def test_parse_rejects_fractional_and_bool_scores(score: object) -> None:
    with pytest.raises(ValueError, match="overall_score must be an integer from 1 to 100"):
        parse_judge_output(json.dumps({"overall_score": score, "reason": "Not an integer."}))
