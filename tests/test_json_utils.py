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


def test_parse_rejects_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason must not be empty"):
        parse_judge_output('{"overall_score": 88, "reason": "   "}')


def test_parse_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="overall_score must be an integer from 1 to 100"):
        parse_judge_output('{"overall_score": 101, "reason": "Too high."}')


def test_parse_rejects_non_integer_score() -> None:
    with pytest.raises(ValueError, match="overall_score must be an integer from 1 to 100"):
        parse_judge_output('{"overall_score": "high", "reason": "Not numeric."}')
