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
